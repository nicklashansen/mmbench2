# interactive.py
"""Browser interface for open-ended, interactive rollouts of the trained world
model. Serves a local web UI (default http://localhost:7860). Run from inside
``src/`` (flat imports), e.g. ``python interactive.py``.
"""
import os
import math
import json
import time
import argparse
import asyncio
import concurrent.futures
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, List, Set, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from aiohttp import web, WSMsgType
from PIL import Image
import io

from task_set import TASK_SET, UNSEEN_TASK_SET


# Metadata fallback for the unseen test tasks. Maps each entry of
# UNSEEN_TASK_SET to a TASK_SET task whose tasks_json entry (language
# embedding, action_dim) should be borrowed when the UNSEEN task itself
# isn't in tasks.json. Keys are kept in lockstep with UNSEEN_TASK_SET
# (10 entries).
TEST_TASK_SET: Dict[str, str] = {
    # DMControl visual variants — same dynamics as the named base.
    'cup-catch-var1':            'cup-catch',
    'finger-turn-easy-var1':     'finger-turn-easy',
    # ManiSkill object swap.
    'ms-push-banana':            'ms-push-cube',
    # OGBench layout swap.
    'og-point-var1':             'og-point-maze',
    'og-point-var2':             'og-point-maze',
    # PyGame point-maze layout swap.
    'pygame-point-maze-var4':    'pygame-point-maze-var3',
    # PyGame "completely unseen" entries borrow from the closest analog.
    'pygame-reacher-easy':       'pygame-air-hockey',
    'pygame-dungeon-explorer1':  'pygame-point-maze-var1',
    'pygame-foraging':           'pygame-rocket-collect',
    'pygame-whirlpool':          'pygame-rocket-collect',
}

from model import (
    Encoder, Decoder, Tokenizer, Dynamics,
    temporal_patchify, temporal_unpatchify,
    RewardHeadMTP, PolicyHeadMTP, symexp,
)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def pack_bottleneck_to_spatial(z_btLd: torch.Tensor, *, n_spatial: int, k: int) -> torch.Tensor:
    # (B,T,L,Db) -> (B,T,n_spatial,k*Db) with L == n_spatial*k
    B, T, L, Db = z_btLd.shape
    assert L == n_spatial * k, f"L={L} != n_spatial*k={n_spatial*k}"
    return z_btLd.view(B, T, n_spatial, k, Db).reshape(B, T, n_spatial, k * Db)


def unpack_spatial_to_bottleneck(z_packed: torch.Tensor, *, k: int, d_bottleneck: int) -> torch.Tensor:
    # (B,T,n_spatial,k*Db) -> (B,T,n_spatial*k,Db)
    B, T, n_spatial, Dz = z_packed.shape
    assert Dz == k * d_bottleneck, f"Dz={Dz} != k*Db={k*d_bottleneck}"
    return z_packed.view(B, T, n_spatial, k, d_bottleneck).reshape(B, T, n_spatial * k, d_bottleneck)


def _as_2d_packed(z: torch.Tensor) -> torch.Tensor:
    # ensure (n_spatial, d_spatial)
    if z.dim() == 2:
        return z
    if z.dim() == 3 and z.shape[0] == 1:
        return z[0]
    raise RuntimeError(f"Unexpected packed latent shape: {tuple(z.shape)}")


def _is_pow2_frac(x: float) -> bool:
    if x <= 0 or x > 1:
        return False
    inv = round(1.0 / x)
    return abs(1.0 / inv - x) < 1e-8 and (inv & (inv - 1)) == 0


def make_tau_schedule(*, k_max: int, schedule: str = "finest", d: Optional[float] = None) -> Dict[str, Any]:
    """
    Returns:
      K: Euler steps
      e: log2(K) (rounded)
      dt: step size
      tau: [i/K]
      tau_idx: discrete indices on k_max grid
    """
    schedule = str(schedule)
    if schedule == "finest":
        K = int(k_max)
        dt = 1.0 / float(K)
    elif schedule == "shortcut":
        assert d is not None and _is_pow2_frac(float(d)), "shortcut requires d = 1/(power of two)"
        dt = float(d)
        K = int(round(1.0 / dt))
        if dt < 1.0 / float(k_max):
            raise ValueError(f"shortcut d={dt} is finer than finest 1/k_max={1.0/k_max}")
    else:
        raise ValueError(f"Unknown schedule: {schedule}")

    e = int(round(math.log2(K)))
    tau = [i / float(K) for i in range(K)]
    stride = k_max // K
    if stride <= 0:
        raise ValueError(f"k_max={k_max} must be >= K={K}")
    tau_idx = [i * stride for i in range(K)]
    return {"K": K, "e": e, "dt": dt, "tau": tau, "tau_idx": tau_idx}


def reward_from_reward_head_output(logits_lk: torch.Tensor, centers_symlog: torch.Tensor) -> float:
    """
    logits_lk: (L,K) or (1,L,K)
    centers_symlog: (K,) (RewardHeadMTP centers_log)
    Returns: scalar reward in original reward space (inverse symlog)
    """
    if logits_lk.dim() == 3:
        logits_lk = logits_lk[0]
    logits_k = logits_lk[0]  # l=0 head
    probs = logits_k.float().softmax(dim=-1)
    symlog_hat = (probs * centers_symlog.float()).sum(dim=-1)
    return float(symexp(symlog_hat).item())


@torch.inference_mode()
def sample_one_timestep_packed(
    dyn: Dynamics,
    *,
    past_packed: torch.Tensor,                 # (B,t,n_spatial,d_spatial)
    k_max: int,
    sched: Dict[str, Any],
    actions: Optional[torch.Tensor] = None,    # (B,t+1,A) (action[0]=0)
    act_mask: Optional[torch.Tensor] = None,   # (B,t+1,A) or (A,)
    use_amp: bool = True,
    return_h: bool = False,
    tau_ctx: float = 0.0,               # context corruption level
    lang_emb: Optional[torch.Tensor] = None,   # (B,lang_dim) task embedding
    z_prev: Optional[torch.Tensor] = None,     # (B,n_spatial,d_spatial) previous latent for warm start
    tau_init: float = 0.0,                      # warm-start noise level (0 = pure noise)
    use_kv_cache: bool = False,                 # enable KV caching for context tokens
) -> Union[Tuple[torch.Tensor, float], Tuple[torch.Tensor, torch.Tensor, float]]:
    """
    Generate next packed latent z_{t}: (B,n_spatial,d_spatial) given past length t.
    Always returns a trailing `instability` scalar (float): the mean RMS change in
    x1_hat across the tail half of executed Euler steps. Low = confident denoising
    (x1_hat stabilizes); high = the model keeps revising its prediction and is
    likely hallucinating / in OOD territory.

    If return_h=True, also returns h_last for the *new* timestep only: (B,1,...)
    aligned with z_t.

    When z_prev is provided and tau_init > 0, the denoising is warm-started by
    initializing z as a blend of noise and z_prev at level tau_init, then skipping
    denoising steps below that level.  This reduces frame-to-frame jitter by
    anchoring the initial state to the previous prediction.

    When use_kv_cache=True and t > 0, the context tokens (positions 0..t-1) are
    processed once in a prefill pass and their time-attention K,V are cached. Each
    denoising step then only runs the single new token through the transformer,
    attending to the cached context. This reduces per-step attention cost from
    O(t+1) to O(1).
    """
    device = past_packed.device
    dtype = past_packed.dtype
    B, t = past_packed.shape[:2]
    n_spatial, d_spatial = past_packed.shape[2], past_packed.shape[3]

    K = int(sched["K"])
    e = int(sched["e"])
    tau = sched["tau"]
    tau_idx = sched["tau_idx"]
    dt = float(sched["dt"])

    # Initialize from noise, optionally warm-started toward z_prev
    z = torch.randn((B, 1, n_spatial, d_spatial), device=device, dtype=dtype)
    if z_prev is not None and tau_init > 0.0:
        zp = z_prev.unsqueeze(1) if z_prev.dim() == 3 else z_prev  # (B,1,n_spatial,d_spatial)
        z = ((1.0 - tau_init) * z.float() + tau_init * zp.float()).to(dtype)

    emax = int(round(math.log2(int(k_max))))

    # Slightly corrupt past context tokens for robustness to autoregressive errors.
    if tau_ctx > 0.0 and t > 0:
        z0_ctx = torch.randn_like(past_packed)
        past_input = ((1.0 - tau_ctx) * past_packed.float() + tau_ctx * z0_ctx.float()).to(dtype)
        ctx_sig_idx = min(int(round((1.0 - tau_ctx) * k_max)), k_max)
    else:
        past_input = past_packed
        ctx_sig_idx = k_max

    if act_mask is not None and act_mask.dim() == 1:
        act_mask = act_mask.view(1, 1, -1).expand(B, t + 1, -1)

    actions_in = None if actions is None else actions[:, : t + 1]
    actmask_in = None if act_mask is None else act_mask[:, : t + 1]

    # --- KV cache: prefill context tokens once ---
    kv_cache = None
    if use_kv_cache and t > 0:
        ctx_step_idxs = torch.full((B, t), emax, device=device, dtype=torch.long)
        ctx_signal_idxs = torch.full((B, t), ctx_sig_idx, device=device, dtype=torch.long)
        ctx_actions = None if actions_in is None else actions_in[:, :t]
        ctx_actmask = None if actmask_in is None else actmask_in[:, :t]

        with torch.autocast(device_type=device.type, enabled=(use_amp and device.type == "cuda")):
            _, _, kv_cache = dyn(
                ctx_actions,
                ctx_step_idxs,
                ctx_signal_idxs,
                past_input,
                act_mask=ctx_actmask,
                agent_tokens=None,
                lang_emb=lang_emb,
                return_kv_cache=True,
            )

    h_last_full = None

    # Track denoising-trajectory instability: step-to-step RMS change in x1_hat.
    x1_hat_prev: Optional[torch.Tensor] = None
    step_deltas: List[torch.Tensor] = []

    for i in range(K):
        tau_i = float(tau[i])
        if tau_i + dt <= tau_init:
            continue  # skip steps below warm-start level
        sig_i = int(tau_idx[i])

        with torch.autocast(device_type=device.type, enabled=(use_amp and device.type == "cuda")):
            if kv_cache is not None:
                # Decode mode: only process the new token (position t)
                new_step_idxs = torch.full((B, 1), e, device=device, dtype=torch.long)
                new_signal_idxs = torch.full((B, 1), sig_i, device=device, dtype=torch.long)
                new_actions = None if actions_in is None else actions_in[:, -1:]
                new_actmask = None if actmask_in is None else actmask_in[:, -1:]

                x1_hat, h_t_full = dyn(
                    new_actions,
                    new_step_idxs,
                    new_signal_idxs,
                    z,
                    act_mask=new_actmask,
                    agent_tokens=None,
                    lang_emb=lang_emb,
                    kv_cache=kv_cache,
                )
            else:
                # Full sequence mode (no cache or t == 0)
                step_idxs_full = torch.full((B, t + 1), emax, device=device, dtype=torch.long)
                step_idxs_full[:, -1] = e
                signal_idxs_full = torch.full((B, t + 1), ctx_sig_idx, device=device, dtype=torch.long)
                signal_idxs_full[:, -1] = sig_i
                packed_seq = torch.cat([past_input, z], dim=1)  # (B,t+1,...)

                x1_hat_full, h_t_full = dyn(
                    actions_in,
                    step_idxs_full,
                    signal_idxs_full,
                    packed_seq,
                    act_mask=actmask_in,
                    agent_tokens=None,
                    lang_emb=lang_emb,
                )
                x1_hat = x1_hat_full[:, -1:, :, :]

        if return_h:
            h_last_full = h_t_full

        x1_hat_f = x1_hat.float()
        if x1_hat_prev is not None:
            step_deltas.append((x1_hat_f - x1_hat_prev).pow(2).mean().sqrt())
        x1_hat_prev = x1_hat_f

        denom = max(1e-4, 1.0 - tau_i)
        b = (x1_hat_f - z.float()) / denom
        z = (z.float() + b * dt).to(dtype)

    # Instability score = mean RMS(x1_hat_i - x1_hat_{i-1}) over the tail half of
    # executed steps. Tail-only: the model is always "uncertain" at high noise
    # levels; only late-step flux reflects true disagreement.
    if step_deltas:
        all_deltas = torch.stack(step_deltas)
        tail = all_deltas[len(all_deltas) // 2 :] if len(all_deltas) > 1 else all_deltas
        instability = float(tail.mean().item())
    else:
        instability = 0.0

    z_next = z[:, 0]  # (B,n_spatial,d_spatial)

    if not return_h:
        return z_next, instability

    if h_last_full is None:
        raise RuntimeError("return_h=True but dyn returned h_t_full=None (check n_agent / dyn impl).")

    # Return representation for the *new* timestep only (the appended position).
    h_new = h_last_full[:, -1:]  # (B,1,...)  e.g. (B,1,n_agent,D) or (B,1,D)
    return z_next, h_new, instability


class _EnvCfg:
    """Minimal config object satisfying envs.make_env(cfg) requirements.

    Seeds each episode's initial frame from a live Gymnasium env's reset().
    """

    def __init__(self, task: str, img_size: int = 224, seed: int = 0):
        self.task = task
        self.obs = 'rgb'
        self.seed = seed
        self.child_env = True
        self.num_envs = 1
        self.save_video = False
        self.rank = 0
        self.render_size = img_size
        self.obs_shape = None
        self.action_dim = None
        self.episode_length = None

    def get(self, key, default=None):
        return getattr(self, key, default)


def env_obs_to_frame_chw01(obs: Any, *, H: int, W: int) -> torch.Tensor:
    """Convert an env observation (dict with 'rgb' or raw array) to (C,H,W) float32 in [0,1].

    Bilinearly resizes if the rendered frame doesn't match (H, W), matching
    the preprocess_dataset.py recipe.
    """
    if isinstance(obs, dict):
        frame = obs.get('rgb', obs)
    else:
        frame = obs
    arr = np.asarray(frame)
    if arr.ndim == 3 and arr.shape[2] == 3 and arr.shape[0] != 3:
        arr = np.transpose(arr, (2, 0, 1))
    if arr.dtype != np.uint8:
        if float(arr.max()) <= 1.5:
            arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
        else:
            arr = arr.clip(0, 255).astype(np.uint8)
    t = torch.from_numpy(arr).float() / 255.0
    if t.shape[-2] != H or t.shape[-1] != W:
        t = F.interpolate(
            t.unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False
        )[0].clamp(0.0, 1.0)
    return t.contiguous()


def load_task_action_dim(tasks_json: str, task: str, *, default_dim: int = 16) -> int:
    try:
        with open(tasks_json, "r") as f:
            meta = json.load(f)
        if task in meta and "action_dim" in meta[task]:
            return int(meta[task]["action_dim"])
    except Exception:
        pass
    return int(default_dim)


def _strip_prefix(sd: dict, prefix: str) -> dict:
    if not any(k.startswith(prefix) for k in sd.keys()):
        return sd
    return {k[len(prefix):]: v for k, v in sd.items()}


def _looks_like_state_dict(d: dict) -> bool:
    if not isinstance(d, dict) or len(d) == 0:
        return False
    k0 = next(iter(d.keys()))
    v0 = d[k0]
    return isinstance(k0, str) and (torch.is_tensor(v0) or isinstance(v0, torch.nn.Parameter))


def _get_state_dict(ckpt: dict) -> dict:
    if _looks_like_state_dict(ckpt):
        sd = ckpt
    else:
        for k in ("dynamics", "dyn_model", "model", "dyn", "state_dict"):
            v = ckpt.get(k, None)
            if isinstance(v, dict):
                if "state_dict" in v and isinstance(v["state_dict"], dict) and _looks_like_state_dict(v["state_dict"]):
                    v = v["state_dict"]
                if _looks_like_state_dict(v):
                    sd = v
                    break
        else:
            raise KeyError(f"Could not find state dict in checkpoint keys={list(ckpt.keys())}")

    for pfx in ("_orig_mod.", "module.", "dynamics.", "dyn."):
        sd = _strip_prefix(sd, pfx)
    return sd


def load_tokenizer_from_ckpt(tokenizer_ckpt: str, device: torch.device):
    ckpt = torch.load(tokenizer_ckpt, map_location="cpu")
    a = ckpt.get("args", {}) or {}

    H = int(a.get("H", 224))
    W = int(a.get("W", 224))
    C = int(a.get("C", 3))
    patch = int(a.get("patch", 4))
    d_model = int(a.get("d_model", 256))
    n_heads = int(a.get("n_heads", 4))
    depth = int(a.get("depth", 6))
    n_latents = int(a.get("n_latents", 16))
    d_bottleneck = int(a.get("d_bottleneck", 32))
    dropout = float(a.get("dropout", 0.0))
    mlp_ratio = float(a.get("mlp_ratio", 4.0))
    time_every = int(a.get("time_every", 1))

    assert H % patch == 0 and W % patch == 0
    n_patches = (H // patch) * (W // patch)
    d_patch = patch * patch * C

    enc = Encoder(
        patch_dim=d_patch,
        d_model=d_model,
        n_latents=n_latents,
        n_patches=n_patches,
        n_heads=n_heads,
        depth=depth,
        d_bottleneck=d_bottleneck,
        dropout=dropout,
        mlp_ratio=mlp_ratio,
        time_every=time_every,
        mae_p_min=0.0,
        mae_p_max=0.0,
    )
    dec = Decoder(
        d_bottleneck=d_bottleneck,
        d_model=d_model,
        n_heads=n_heads,
        depth=depth,
        n_latents=n_latents,
        n_patches=n_patches,
        d_patch=d_patch,
        dropout=dropout,
        mlp_ratio=mlp_ratio,
        time_every=time_every,
    )
    tok = Tokenizer(enc, dec).to(device)
    tok.load_state_dict(_get_state_dict(ckpt), strict=True)
    tok.eval()
    for p in tok.parameters():
        p.requires_grad_(False)

    info = dict(H=H, W=W, C=C, patch=patch, n_latents=n_latents, d_bottleneck=d_bottleneck)
    return tok, info


def _get_rew_head_state_dict(ckpt: dict) -> dict:
    for k in ("rew_head", "reward_head"):
        v = ckpt.get(k, None)
        if isinstance(v, dict):
            if "state_dict" in v and isinstance(v["state_dict"], dict) and _looks_like_state_dict(v["state_dict"]):
                v = v["state_dict"]
            if _looks_like_state_dict(v):
                sd = v
                break
    else:
        raise KeyError(f"Could not find reward head state dict in ckpt keys={list(ckpt.keys())}")

    for pfx in ("module.", "rew_head.", "reward_head."):
        sd = _strip_prefix(sd, pfx)
    return sd


def _get_policy_head_state_dict(ckpt: dict) -> dict:
    for k in ("policy_head", "bc_head"):
        v = ckpt.get(k, None)
        if isinstance(v, dict):
            if "state_dict" in v and isinstance(v["state_dict"], dict) and _looks_like_state_dict(v["state_dict"]):
                v = v["state_dict"]
            if _looks_like_state_dict(v):
                sd = v
                break
    else:
        raise KeyError(f"Could not find policy head state dict in ckpt keys={list(ckpt.keys())}")

    for pfx in ("module.", "policy_head.", "bc_head."):
        sd = _strip_prefix(sd, pfx)
    return sd


def load_dynamics_from_ckpt(
    dynamics_ckpt: str,
    *,
    device: torch.device,
    d_bottleneck: int,
    n_latents: int,
    packing_factor: int,
):
    ckpt = torch.load(dynamics_ckpt, map_location="cpu")
    a = ckpt.get("args", {}) or {}

    # dynamics
    d_model = int(a.get("d_model_dyn", a.get("dyn_d_model", a.get("d_model", 256))))
    n_heads = int(a.get("n_heads", 4))
    depth = int(a.get("dyn_depth", a.get("depth", 8)))
    dropout = float(a.get("dropout", 0.0))
    mlp_ratio = float(a.get("mlp_ratio", 4.0))
    time_every = int(a.get("time_every", 4))
    k_max = int(a.get("k_max", 8))
    n_register = int(a.get("n_register", 4))
    n_agent = int(a.get("n_agent", 0))
    lang_dim = int(a.get("lang_dim", 0))

    # reward
    reward_L = int(a.get("reward_L", 8))
    reward_num_bins = int(a.get("reward_num_bins", 101))
    reward_log_low = float(a.get("reward_log_low", -8.0))
    reward_log_high = float(a.get("reward_log_high", 8.0))
    reward_mlp_ratio = float(a.get("reward_mlp_ratio", 2.0))
    reward_pool_agent = str(a.get("reward_pool_agent", "attn"))

    # bc policy
    bc_L = int(a.get("bc_L", 8))
    bc_act_dim_max = int(a.get("bc_act_dim", 16))
    bc_mlp_ratio = float(a.get("bc_mlp_ratio", 2.0))
    bc_pool_agent = str(a.get("bc_pool_agent", "attn"))

    assert n_latents % packing_factor == 0
    n_spatial = n_latents // packing_factor
    d_spatial = d_bottleneck * packing_factor

    dyn = Dynamics(
        d_model=d_model,
        d_bottleneck=d_bottleneck,
        d_spatial=d_spatial,
        n_spatial=n_spatial,
        n_register=n_register,
        n_agent=n_agent,
        n_heads=n_heads,
        depth=depth,
        k_max=k_max,
        dropout=dropout,
        mlp_ratio=mlp_ratio,
        time_every=time_every,
        lang_dim=lang_dim,
    ).to(device)
    dyn.load_state_dict(_get_state_dict(ckpt), strict=True)
    dyn.eval()

    # reward head (optional — may not be present in older checkpoints)
    rew_head = None
    try:
        rew_sd = _get_rew_head_state_dict(ckpt)
        rew_head = RewardHeadMTP(
            d_model=d_model,
            L=int(reward_L),
            num_bins=int(reward_num_bins),
            log_low=float(reward_log_low),
            log_high=float(reward_log_high),
            mlp_ratio=float(reward_mlp_ratio),
            dropout=0.0,
            pool_agent=str(reward_pool_agent),
        ).to(device)
        rew_head.load_state_dict(rew_sd, strict=True)
        rew_head.eval()
        for p in rew_head.parameters():
            p.requires_grad_(False)
    except KeyError:
        pass

    # policy head (optional — BC-finetuned ckpts have this; earlier ckpts do not)
    policy_head = None
    try:
        pol_sd = _get_policy_head_state_dict(ckpt)
        policy_head = PolicyHeadMTP(
            d_model=d_model,
            L=int(bc_L),
            act_dim_max=int(bc_act_dim_max),
            mlp_ratio=float(bc_mlp_ratio),
            dropout=0.0,
            pool_agent=str(bc_pool_agent),
        ).to(device)
        policy_head.load_state_dict(pol_sd, strict=True)
        policy_head.eval()
        for p in policy_head.parameters():
            p.requires_grad_(False)
    except KeyError:
        pass

    return dyn, rew_head, policy_head, {"k_max": k_max, "n_spatial": n_spatial, "d_spatial": d_spatial, "d_model": d_model, "lang_dim": lang_dim}


@torch.inference_mode()
def decode_single_packed_frame(
    decoder: Decoder,
    *,
    z_packed: torch.Tensor,   # (n_spatial,d_spatial) or (1,n_spatial,d_spatial)
    H: int, W: int, C: int, patch: int,
    packing_factor: int,
    d_bottleneck: int,
) -> torch.Tensor:
    z2 = _as_2d_packed(z_packed)
    z_bt = z2.unsqueeze(0).unsqueeze(0)  # (1,1,n_spatial,d_spatial)
    z_btLd = unpack_spatial_to_bottleneck(z_bt, k=packing_factor, d_bottleneck=d_bottleneck)
    patches = decoder(z_btLd)  # (1,1,Np,Dp)
    frames = temporal_unpatchify(patches, H, W, C, patch)  # (1,1,C,H,W)
    return frames[0, 0].clamp(0, 1)


def frame_to_jpeg_bytes(frame_chw_01: torch.Tensor, *, quality: int = 85) -> bytes:
    fr_u8 = (frame_chw_01.clamp(0, 1) * 255.0).to(torch.uint8).detach().cpu().numpy()
    hwc = np.transpose(fr_u8, (1, 2, 0))
    im = Image.fromarray(hwc, mode="RGB")
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=int(quality), optimize=True)
    return buf.getvalue()


def frame_to_uint8_hwc(frame_chw_01: torch.Tensor) -> np.ndarray:
    """(C,H,W) float [0,1] -> (H,W,3) uint8 — used to buffer recorded frames."""
    return (
        frame_chw_01.clamp(0, 1).float().permute(1, 2, 0).detach().cpu().numpy() * 255.0
    ).astype(np.uint8)


def save_recording_mp4(path: Path, frames_hwc: List[np.ndarray], fps: float) -> None:
    """Write a list of (H,W,3) uint8 frames to ``path`` as an mp4 (libx264).

    macro_block_size=1 avoids
    silent padding for 224x224 frames, quality=8 is visually lossless-ish.
    """
    if not frames_hwc:
        return
    import imageio.v2 as imageio
    imageio.mimwrite(
        str(path), frames_hwc,
        fps=max(1, int(round(float(fps)))),
        codec="libx264",
        quality=8,
        macro_block_size=1,
    )


# Key-pair -> action-dimension bindings: single source of truth server-side
# (the client's shouldCapture list in interactive.html mirrors it).
KEY_BINDINGS: List[Tuple[str, str]] = [
    ("ArrowRight", "ArrowLeft"),  # dim 0
    ("ArrowUp", "ArrowDown"),     # dim 1
    ("d", "a"),                   # dim 2
    ("w", "s"),                   # dim 3
]

# The only keys a client can legitimately hold down (derived from the
# bindings, plus uppercase variants). Other keydown values are ignored
# server-side so junk can't grow session state.
ACTION_KEYS: Set[str] = (
    {k for pair in KEY_BINDINGS for k in pair}
    | {k.upper() for pair in KEY_BINDINGS for k in pair if len(k) == 1}
)


def build_action_from_keys(keys_down: Set[str], *, act_dim: int, A: int = 16) -> torch.Tensor:
    a = torch.zeros(A, dtype=torch.float32)
    if act_dim <= 0:
        return a
    # Each pair of keys maps to one action dimension; opposing keys cancel out.
    # Signs chosen so that visual direction matches key direction for common tasks.
    for dim, (pos_key, neg_key) in enumerate(KEY_BINDINGS):
        if dim >= act_dim:
            break
        pos = (pos_key in keys_down) or (pos_key.upper() in keys_down if len(pos_key) == 1 else False)
        neg = (neg_key in keys_down) or (neg_key.upper() in keys_down if len(neg_key) == 1 else False)
        if pos and not neg:
            a[dim] = +1.0
        elif neg and not pos:
            a[dim] = -1.0
    return a


def classify_uncertainty(
    value: float,
    samples: List[float],
    *,
    z_yellow: float = 1.0,
    z_red: float = 2.5,
    mad_rel_floor: float = 0.05,
) -> str:
    """Classify `value` as green/yellow/red against a median/MAD baseline fitted
    on `samples`. Robust to a single transient spike inside the calibration
    window (median is unaffected; MAD is not inflated the way std is).

    If MAD is too small relative to the median (nearly-constant calibration), we
    fall back to a multiplicative threshold so we don't hair-trigger on tiny
    deviations: red when value > (1 + z_red * mad_rel_floor) * median.
    """
    if not samples:
        return "green"
    arr = np.asarray(samples, dtype=np.float64)
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med))) * 1.4826  # Gaussian-consistent scale
    floor = max(abs(med) * mad_rel_floor, 1e-8)
    scale = max(mad, floor)
    z = (float(value) - med) / scale
    if z < z_yellow:
        return "green"
    if z < z_red:
        return "yellow"
    return "red"


def merge_u_states(*states: str) -> str:
    """Combine per-signal u_states into a single worst-of-N state."""
    order = {"off": 0, "green": 1, "calibrating": 2, "yellow": 3, "red": 4}
    worst = "off"
    for s in states:
        if order.get(s, 0) > order.get(worst, 0):
            worst = s
    return worst


def load_html(path: Optional[str], *, fallback: str = "") -> str:
    if not path:
        return fallback
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return fallback


@dataclass
class SessionState:
    task: str
    keys_down: Set[str]
    paused: bool
    reset_requested: bool
    step: int
    cum_reward: float
    last_reward_pred: float
    last_u_f: float
    last_u_r: float

    calib_f_samples: List[float]
    calib_r_samples: List[float]
    calib_done: bool

    z0_packed: torch.Tensor
    z_hist: List[torch.Tensor]
    a_hist: List[torch.Tensor]

    act_dim: int
    act_mask_1d: torch.Tensor
    lang_emb: Optional[torch.Tensor]   # (1, lang_dim) or None
    action_beta: float
    a_smooth: torch.Tensor   # (16,)
    ctx_window: int
    fps: float

    cached_frame_id: int
    cached_jpeg: Optional[bytes]

    # --- Recording (only populated when --record is passed). One mp4 is
    # written per episode boundary (reset / task switch / disconnect);
    # `recorded_frames` accumulates uint8 HWC frames between flushes.
    session_ts: str = ""
    episode_id: int = 0
    recorded_frames: List[np.ndarray] = field(default_factory=list)


class InteractiveServer:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.use_amp = (not bool(args.no_amp)) and (self.device.type == "cuda")
        self.infer_lock = asyncio.Lock()
        self.session_seq = 0

        # Live-env seeding: each episode's initial frame comes from a Gymnasium
        # env's reset(). The rollout itself is pure world model (no in-rollout
        # env stepping) — the env only supplies the starting frame, so no
        # offline dataset is needed. MuJoCo's EGL contexts are thread-local, so
        # all env touches are pinned to a single dedicated worker thread to
        # avoid EGL_BAD_ACCESS across threads.
        os.environ.setdefault('MUJOCO_GL', 'egl')
        self.envs: Dict[str, Any] = {}
        self.env_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="env-thread"
        )

        # HTML
        self.html = load_html(args.html, fallback="<html><body>missing html</body></html>")

        # Dropdown task list = 200 training tasks + 10 UNSEEN tasks. UNSEEN
        # tasks borrow lang_emb / action_dim metadata via TEST_TASK_SET; only
        # the live env is fresh.
        self.tasks = list(TASK_SET) + list(UNSEEN_TASK_SET)
        self.initial_task = args.task if args.task in self.tasks else self.tasks[0]

        # tokenizer
        tok, tok_info = load_tokenizer_from_ckpt(args.tokenizer_ckpt, self.device)
        self.encoder: Encoder = tok.encoder
        self.decoder: Decoder = tok.decoder

        self.d_bottleneck = int(tok_info["d_bottleneck"])
        self.n_latents = int(tok_info["n_latents"])
        self.H = int(tok_info["H"])
        self.W = int(tok_info["W"])
        self.C = int(tok_info["C"])
        self.patch = int(tok_info["patch"])

        # dynamics
        self.dyn, self.rew_head, self.policy_head, dyn_info = load_dynamics_from_ckpt(
            args.dynamics_ckpt,
            device=self.device,
            d_bottleneck=self.d_bottleneck,
            n_latents=self.n_latents,
            packing_factor=args.packing_factor,
        )

        self.k_max = int(dyn_info["k_max"])
        self.n_spatial = int(dyn_info["n_spatial"])
        self.d_spatial = int(dyn_info["d_spatial"])

        self.sched = make_tau_schedule(
            k_max=self.k_max,
            schedule=args.schedule,
            d=(args.eval_d if args.schedule == "shortcut" else None),
        )
        self.tau_ctx = float(args.tau_ctx)
        self.tau_init = float(args.tau_init)

        self.use_kv_cache = bool(args.kv_cache)

        # Recording: one mp4 per episode, written to recordings_dir on
        # reset / task switch / ws disconnect. No-op when --record is off.
        self.recordings_dir = Path(args.recordings_dir)
        if args.record:
            self.recordings_dir.mkdir(parents=True, exist_ok=True)
            print(f"[record] enabled — writing per-episode mp4s to {self.recordings_dir}")

        if args.compile:
            print("[compile] compiling dynamics and decoder (first few frames will be slow)...")
            self.dyn = torch.compile(self.dyn, mode="default")
            self.decoder = torch.compile(self.decoder, mode="default")

        # task metadata (language embeddings)
        self.task_meta = None
        if args.tasks_json and os.path.exists(args.tasks_json):
            try:
                with open(args.tasks_json, "r") as f:
                    self.task_meta = json.load(f)
            except Exception:
                pass
        self.lang_dim = int(dyn_info["lang_dim"])

        # The initial latent is deferred to the first new_session() call (which
        # runs on env_executor) — creating the env here on the main thread would
        # bind MuJoCo's EGL context to MainThread and then fail when
        # env_executor later tries to render.
        self.z0_packed = torch.zeros(
            (self.n_spatial, self.d_spatial), device=self.device,
        )
        self.act_dim, self.act_mask_1d = self._compute_act_mask(self.initial_task)

    def _get_or_make_env(self, task: str):
        """Lazy per-task env cache. Imports envs lazily so importing this
        module stays cheap."""
        env = self.envs.get(task)
        if env is not None:
            return env
        from envs import make_env as _make_env  # local import: env mode only
        cfg = _EnvCfg(task, img_size=self.W, seed=int(self.args.seed))
        print(f"[env] creating env for task={task!r} at {self.W}x{self.H}")
        env = _make_env(cfg)
        self.envs[task] = env
        return env

    @torch.inference_mode()
    def _encode_initial_latent(self, task: str) -> torch.Tensor:
        env = self._get_or_make_env(task)
        obs, _info = env.reset()
        frame0 = env_obs_to_frame_chw01(obs, H=self.H, W=self.W).to(self.device)
        return self._encode_frame_to_packed(frame0)

    @torch.inference_mode()
    def _encode_frame_to_packed(self, frame_chw_01: torch.Tensor) -> torch.Tensor:
        """Encode a single (C,H,W) frame in [0,1] to a packed latent matching z_next."""
        patches = temporal_patchify(
            frame_chw_01.view(1, 1, self.C, self.H, self.W), self.patch
        )
        with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
            z_btLd, _ = self.encoder(patches)
        z_packed = pack_bottleneck_to_spatial(
            z_btLd, n_spatial=self.n_spatial, k=self.args.packing_factor
        )[0, 0]
        return z_packed.to(torch.float32).detach()

    def _get_lang_emb(self, task: str) -> Optional[torch.Tensor]:
        """Returns (1, lang_dim) language embedding for a task, or None.

        For UNSEEN tasks (TEST_TASK_SET), borrow the embedding from the
        registered SEEN base task — most UNSEEN tasks aren't in tasks.json.
        """
        if self.task_meta is None:
            return None
        lookup = TEST_TASK_SET.get(task, task)
        if lookup not in self.task_meta:
            return None
        te = self.task_meta[lookup].get("text_embedding", None)
        if te is None:
            return None
        emb = torch.tensor(te, dtype=torch.float32).to(self.device)
        if emb.numel() != self.lang_dim:
            return None
        return emb.unsqueeze(0)  # (1, lang_dim)

    def _compute_act_mask(self, task: str, env=None) -> Tuple[int, torch.Tensor]:
        """Return (act_dim, mask) for `task`.

        - SEEN tasks: read from tasks_json (matches what the WM was trained on).
        - UNSEEN tasks: prefer the live env's action_space when available;
          fall back to the registered SEEN base task's tasks_json entry. The
          UNSEEN task's own tasks_json entry can have a different action_dim
          than the borrowed lang_emb base (e.g. pygame-dungeon-explorer1 is
          mapped to pygame-point-maze-var1 for lang_emb but is itself a
          different action arity).
        """
        act_dim: Optional[int] = None
        if task in TEST_TASK_SET:
            if env is not None:
                try:
                    act_dim = int(env.action_space.shape[0])
                except Exception:
                    pass
            if act_dim is None:
                base = TEST_TASK_SET[task]
                act_dim = int(load_task_action_dim(
                    self.args.tasks_json, base, default_dim=16,
                ))
        else:
            act_dim = int(load_task_action_dim(
                self.args.tasks_json, task, default_dim=16,
            ))
        act_dim = max(0, min(16, int(act_dim)))
        mask = torch.zeros(16, dtype=torch.float32)
        if act_dim > 0:
            mask[:act_dim] = 1.0
        return act_dim, mask.to(self.device)

    def new_session(self) -> SessionState:
        task = self.initial_task
        # Compute the act mask off the live env so UNSEEN tasks pick up the
        # right action_dim from the env's action space.
        env = self._get_or_make_env(task)
        act_dim, act_mask = self._compute_act_mask(task, env=env)
        z0 = _as_2d_packed(self._encode_initial_latent(task))
        beta = float(self.args.action_smooth_beta)
        a0 = torch.zeros(16, device=self.device, dtype=torch.float32)

        return SessionState(
            task=task,
            keys_down=set(),
            paused=False,
            reset_requested=False,
            step=0,
            cum_reward=0.0,
            last_reward_pred=0.0,
            last_u_f=0.0,
            last_u_r=0.0,
            calib_f_samples=[],
            calib_r_samples=[],
            calib_done=False,
            z0_packed=z0,
            z_hist=[z0],
            a_hist=[torch.zeros(16, device=self.device, dtype=torch.float32)],
            act_dim=act_dim,
            act_mask_1d=act_mask,
            lang_emb=self._get_lang_emb(task),
            action_beta=beta,
            a_smooth=a0,
            ctx_window=int(self.args.ctx_window),
            fps=float(self.args.fps),
            cached_frame_id=-1,
            cached_jpeg=None,
            session_ts=time.strftime("%Y%m%d_%H%M%S"),
            episode_id=0,
            recorded_frames=[],
        )

    def _flush_recording(self, st: SessionState) -> Optional[Path]:
        """Write the buffered episode to mp4 and clear the buffer.

        Called at every episode boundary — reset, task switch, and ws
        shutdown. Silently no-ops when --record is off or the buffer is
        empty (e.g. a reset with no preceding steps).
        """
        if not self.args.record or not st.recorded_frames:
            if st.recorded_frames:
                st.recorded_frames = []
            return None
        filename = f"{st.task}_{st.session_ts}_ep{st.episode_id:03d}.mp4"
        out_path = self.recordings_dir / filename
        n = len(st.recorded_frames)
        save_recording_mp4(out_path, st.recorded_frames, st.fps)
        st.recorded_frames = []
        print(f"[record] wrote {n} frames @ {st.fps:.1f} fps -> {out_path}")
        return out_path

    def _reset_session(self, st: SessionState):
        self._flush_recording(st)
        st.episode_id += 1
        st.z0_packed = self._encode_initial_latent(st.task)

        z0 = _as_2d_packed(st.z0_packed.detach())
        st.z_hist = [z0]
        st.a_hist = [torch.zeros(16, device=self.device, dtype=torch.float32)]
        st.a_smooth = torch.zeros(16, device=self.device, dtype=torch.float32)

        st.keys_down.clear()
        st.paused = False
        st.reset_requested = False
        st.step = 0
        st.cum_reward = 0.0
        st.last_reward_pred = 0.0
        st.last_u_f = 0.0
        st.last_u_r = 0.0
        st.calib_f_samples = []
        st.calib_r_samples = []
        st.calib_done = False
        st.cached_frame_id = -1
        st.cached_jpeg = None

    def _switch_task_sync(self, st: SessionState, new_task: str):
        if new_task not in self.tasks:
            return

        self._flush_recording(st)
        st.episode_id += 1
        st.task = new_task
        env = self._get_or_make_env(new_task)
        st.act_dim, st.act_mask_1d = self._compute_act_mask(new_task, env=env)
        st.lang_emb = self._get_lang_emb(new_task)

        st.z0_packed = _as_2d_packed(self._encode_initial_latent(new_task))

        st.z_hist = [st.z0_packed]
        st.a_hist = [torch.zeros(16, device=self.device, dtype=torch.float32)]
        st.a_smooth = torch.zeros(16, device=self.device, dtype=torch.float32)
        st.keys_down.clear()
        st.reset_requested = False
        st.step = 0
        st.cum_reward = 0.0
        st.last_reward_pred = 0.0
        st.last_u_f = 0.0
        st.last_u_r = 0.0
        st.calib_f_samples = []
        st.calib_r_samples = []
        st.calib_done = False

        st.cached_frame_id = -1
        st.cached_jpeg = None

    def _build_local_window(self, st: SessionState) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
          past: (1,t,n_spatial,d_spatial)
          actions_local: (1,t+1,16) — actions_local[:,k] = action that produced past[k]
          actmask_local: (1,t+1,16)

        Backward convention (matches dynamics training): actions_local[:,k] is the action
        that produced frame k.  actions_local[:,0] = 0 (first context frame has no
        producing action), actions_local[:,t] = current action (will produce the new frame).
        """
        g = len(st.z_hist)  # next frame index
        s = max(0, g - int(st.ctx_window))

        past_list = st.z_hist[s:g]  # list of (n_spatial,d_spatial)
        if len(past_list) == 0:
            past = torch.empty((1, 0, self.n_spatial, self.d_spatial),
                               device=self.device, dtype=st.z_hist[-1].dtype)
        else:
            past = torch.stack(past_list, dim=0).unsqueeze(0)  # (1,t,...)
        t = past.shape[1]

        actions_local = torch.zeros((1, t + 1, 16), device=self.device, dtype=torch.float32)
        if t >= 1:
            # Backward convention: actions_local[k] = action that produced past[k].
            # a_hist[s+k] produced z_hist[s+k] = past[k], so:
            #   actions_local[0..t-1] = a_hist[s..s+t-1] (actions that produced past[0..t-1])
            #   actions_local[t] = a_hist[-1] = current action (will produce the new frame)
            # Note: a_hist[0] = 0, so when s=0 actions_local[0] is correctly zero.
            actions_local[0, 0:t] = torch.stack(st.a_hist[s: s + t], dim=0)
            actions_local[0, t] = st.a_hist[-1]

        actmask_local = st.act_mask_1d.view(1, 1, 16).expand(1, t + 1, 16).contiguous()
        return past, actions_local, actmask_local

    def _render_step_sync(self, st: SessionState) -> Tuple[bytes, Dict[str, Any]]:
        """
        Runs at most one WM step (if not paused), then decodes the current frame.
        Called via asyncio.to_thread.
        """
        if st.reset_requested:
            self._reset_session(st)

        # action (raw from keys)
        a_raw = build_action_from_keys(
            st.keys_down, act_dim=st.act_dim, A=16
        ).to(self.device)

        a_raw = (a_raw.clamp(-1, 1) * st.act_mask_1d).to(torch.float32)

        # EMA smoothing
        beta = float(st.action_beta)
        if beta > 0.0:
            beta = min(max(beta, 0.0), 0.999)
            st.a_smooth = (beta * st.a_smooth + (1.0 - beta) * a_raw).to(torch.float32)
            a = st.a_smooth
        else:
            a = a_raw

        # Decoded frame for the current z_next; shared between display and u_r.
        frame_cur: Optional[torch.Tensor] = None
        stepped: bool = False

        if not st.paused and st.act_dim >= 0:
            stepped = True
            st.a_hist.append(a)

            past, actions_local, actmask_local = self._build_local_window(st)

            need_h = self.rew_head is not None
            z_prev = st.z_hist[-1].unsqueeze(0) if self.tau_init > 0.0 else None
            result = sample_one_timestep_packed(
                self.dyn,
                past_packed=past,
                k_max=self.k_max,
                sched=self.sched,
                actions=actions_local,
                act_mask=actmask_local,
                use_amp=self.use_amp,
                return_h=need_h,
                tau_ctx=self.tau_ctx,
                lang_emb=st.lang_emb,
                z_prev=z_prev,
                tau_init=self.tau_init,
                use_kv_cache=self.use_kv_cache,
            )
            if need_h:
                z_next, h, instability = result
            else:
                z_next, instability = result
            st.last_u_f = float(instability)
            st.z_hist.append(_as_2d_packed(z_next.detach()))
            st.step += 1

            # Cap the history so a long-lived session can't grow GPU memory
            # without bound: inference only ever reads the last ctx_window
            # frames (see _build_local_window), so anything older is dead weight.
            # z_hist and a_hist stay index-aligned, so trim both equally.
            cap = int(st.ctx_window) + 1
            if len(st.z_hist) > cap:
                st.z_hist = st.z_hist[-cap:]
                st.a_hist = st.a_hist[-cap:]

            # Tokenizer round-trip residual: decode z_next, re-encode, compare.
            # Motion-invariant: off-manifold latents produce persistent residual
            # even when the dynamics is "confidently" predicting no change.
            # --u_every N amortizes the extra encoder pass across N steps (the
            # display border updating at a few Hz is indistinguishable; u_f
            # still updates every step for free from the denoising loop).
            if self.args.uncertainty_overlay and (st.step % max(1, int(self.args.u_every)) == 0):
                z_cur = st.z_hist[-1]
                frame_cur = decode_single_packed_frame(
                    self.decoder,
                    z_packed=z_cur,
                    H=self.H, W=self.W, C=self.C, patch=self.patch,
                    packing_factor=self.args.packing_factor,
                    d_bottleneck=self.d_bottleneck,
                )
                z_recon = self._encode_frame_to_packed(frame_cur)
                diff = z_cur.to(torch.float32) - z_recon
                st.last_u_r = float(diff.pow(2).mean().sqrt().item())

                # Collect calibration samples over the first N computed values
                # post-reset (with u_every > 1 the wall-clock window stretches
                # accordingly).
                if not st.calib_done:
                    st.calib_f_samples.append(st.last_u_f)
                    st.calib_r_samples.append(st.last_u_r)
                    if len(st.calib_f_samples) >= int(self.args.calibration_steps):
                        st.calib_done = True

            # reward for *current* state (after stepping)
            if need_h:
                logits_btlk, centers = self.rew_head(h[:, -1:])  # (1,1,L,K)
                st.last_reward_pred = reward_from_reward_head_output(logits_btlk[0, 0], centers)
                st.cum_reward += st.last_reward_pred

        frame_id = st.step  # stable, monotonic id for "current displayed frame" (survives history cap)
        need_encode = (st.cached_jpeg is None) or (st.cached_frame_id != frame_id)

        jpeg: Optional[bytes] = None
        if need_encode:
            if frame_cur is None:
                frame_cur = decode_single_packed_frame(
                    self.decoder,
                    z_packed=st.z_hist[-1],
                    H=self.H, W=self.W, C=self.C, patch=self.patch,
                    packing_factor=self.args.packing_factor,
                    d_bottleneck=self.d_bottleneck,
                )
            st.cached_jpeg = frame_to_jpeg_bytes(frame_cur, quality=int(self.args.jpeg_quality))
            st.cached_frame_id = frame_id
            jpeg = st.cached_jpeg

            # Buffer the freshly decoded WM frame for the per-episode mp4.
            # Only stepped frames go in; paused ticks would just dupe the
            # last image and stretch the recording.
            if self.args.record and stepped:
                st.recorded_frames.append(frame_to_uint8_hwc(frame_cur))
        else:
            # Frame unchanged: don't resend bytes; the client keeps the last image.
            jpeg = None

        # Classify each signal against its own median/MAD baseline, then merge to
        # the worst. Rendering on the client already reads `u_state` and only
        # cares about the overall severity.
        u_state = "off"
        u_suffix = ""
        if self.args.uncertainty_overlay:
            if not st.calib_done:
                u_state = "calibrating"
                u_suffix = f" [cal {len(st.calib_f_samples)}/{int(self.args.calibration_steps)}]"
            else:
                state_f = classify_uncertainty(st.last_u_f, st.calib_f_samples)
                state_r = classify_uncertainty(st.last_u_r, st.calib_r_samples)
                u_state = merge_u_states(state_f, state_r)

        rec_suffix = (
            f" | rec={len(st.recorded_frames)} (ep{st.episode_id})"
            if self.args.record else ""
        )
        status = {
            "type": "status",
            "task": st.task,
            "paused": bool(st.paused),
            "act_dim": int(st.act_dim),  # lets clients show only the bindable keys
            "u_state": u_state,
            "text": (
                f"step={st.step} | "
                f"r={st.last_reward_pred:+.2f} | "
                f"R={st.cum_reward:+.2f} | "
                f"u_r={st.last_u_r:.3f} u_f={st.last_u_f:.3f}{u_suffix} | "
                f"fps={st.fps:.1f}"
                f"{rec_suffix}"
            ),
        }
        return jpeg, status

    async def status(self, request: web.Request) -> web.Response:
        return web.json_response(
            {
                "tasks": len(self.tasks),
                "task_list": self.tasks,
                "initial_task": self.initial_task,
            },
            headers={"Access-Control-Allow-Origin": "*"},
        )

    async def healthz(self, request: web.Request) -> web.Response:
        return web.Response(text="ok")

    async def index(self, request: web.Request) -> web.Response:
        html = self.html
        html = html.replace("__TASK_SET__", json.dumps(self.tasks))
        html = html.replace("__INITIAL_TASK__", self.initial_task)
        return web.Response(text=html, content_type="text/html")

    async def _run_blocking(self, fn, *args):
        """Run a sync method on the env worker thread.

        MuJoCo's EGL contexts are pinned to a single thread, so anything that
        may eventually touch the env (env.reset on session start / reset / task
        switch — which can happen inside _render_step_sync when reset_requested
        is true) must run on env_executor.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self.env_executor, fn, *args)

    async def ws_handler(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)

        self.session_seq += 1
        sid = self.session_seq

        async def _send_json(obj: Dict[str, Any]) -> bool:
            try:
                await asyncio.wait_for(ws.send_str(json.dumps(obj)), timeout=2.0)
                return True
            except Exception:
                return False

        end_reason = "disconnect"
        junk_count = 0
        last_set_task = 0.0
        last_reset = 0.0

        st: Optional[SessionState] = None
        try:
            # new_session() may touch the env (env.reset, render); run on the
            # env worker thread when in env mode.
            st = await self._run_blocking(self.new_session)
            print(f"[{time.strftime('%F %T')}] [session {sid}] open task={st.task}")
            # Best-effort: the client may have vanished between grant and here;
            # the loops below exit promptly on a closed socket.
            await _send_json({
                "type": "status",
                "task": st.task,
                "paused": bool(st.paused),
                "text": "connected",
            })

            async def recv_loop():
                nonlocal end_reason, junk_count, last_set_task, last_reset

                def junk() -> bool:
                    nonlocal junk_count, end_reason
                    junk_count += 1
                    if junk_count >= 20:
                        end_reason = "junk"
                        return True
                    return False

                def try_reset(now: float):
                    # One debounce for both reset paths (key and button).
                    nonlocal last_reset
                    if now - last_reset >= 0.3:
                        last_reset = now
                        st.reset_requested = True

                async for msg in ws:
                    if msg.type in (WSMsgType.CLOSE, WSMsgType.CLOSING, WSMsgType.ERROR):
                        return
                    now = time.monotonic()
                    if msg.type != WSMsgType.TEXT:
                        if junk():
                            return
                        continue

                    try:
                        data = json.loads(msg.data)
                        assert isinstance(data, dict)
                    except Exception:
                        if junk():
                            return
                        continue

                    t = str(data.get("type", ""))

                    if t == "keydown":
                        k = str(data.get("key", ""))

                        if k == "Space":
                            st.paused = not st.paused
                        elif k in ("r", "R"):
                            try_reset(now)
                        elif k in ("q", "Q", "Escape"):
                            await ws.close()
                            return
                        elif k in ACTION_KEYS:
                            st.keys_down.add(k)

                    elif t == "keyup":
                        k = str(data.get("key", ""))
                        st.keys_down.discard(k)

                    elif t == "set_task":
                        # Min interval: switching re-seeds under the GPU lock;
                        # mashing the dropdown must not starve other sessions.
                        # (The client gates only its dropdown sync on the ack,
                        # so a dropped switch is cosmetic, not a wedge.)
                        if now - last_set_task < 1.0:
                            continue
                        last_set_task = now
                        new_task = str(data.get("task", ""))[:128]
                        async with self.infer_lock:
                            await self._run_blocking(self._switch_task_sync, st, new_task)

                    elif t == "toggle_pause":
                        st.paused = not st.paused

                    elif t == "reset":
                        try_reset(now)

                    elif t == "disconnect":
                        await ws.close()
                        return

                    else:
                        if junk():
                            return

            async def send_loop():
                nonlocal end_reason
                dt = 1.0 / max(1e-6, float(st.fps))
                next_t = time.monotonic()

                while not ws.closed:
                    now = time.monotonic()
                    if now < next_t:
                        await asyncio.sleep(min(next_t - now, 0.25))
                        continue
                    next_t += dt
                    if now - next_t > 1.0:
                        # Resync after long stalls (lock contention, warm
                        # starts) instead of bursting to catch up.
                        next_t = now

                    step_t0 = time.monotonic()
                    async with self.infer_lock:
                        jpeg, status = await self._run_blocking(self._render_step_sync, st)
                    step_ms = (time.monotonic() - step_t0) * 1000.0

                    if ws.closed:
                        break
                    status["ms"] = round(step_ms, 1)
                    try:
                        # Send timeouts guard against slow readers ballooning
                        # the write buffer.
                        await asyncio.wait_for(ws.send_str(json.dumps(status)), timeout=2.0)
                        if jpeg is not None:
                            await asyncio.wait_for(ws.send_bytes(jpeg), timeout=2.0)
                    except Exception:
                        end_reason = "slow"
                        return

            recv = asyncio.create_task(recv_loop())
            send = asyncio.create_task(send_loop())
            done, pending = await asyncio.wait({recv, send}, return_when=asyncio.FIRST_COMPLETED)
            for p in pending:
                p.cancel()
            try:
                await ws.close()
            except Exception:
                pass
        finally:
            steps = st.step if st is not None else 0
            task = st.task if st is not None else "?"
            print(f"[{time.strftime('%F %T')}] [session {sid}] closed reason={end_reason} "
                  f"steps={steps} task={task}")

        # Final flush: write whatever was buffered for the in-progress episode.
        if st is not None:
            try:
                await asyncio.to_thread(self._flush_recording, st)
            except Exception as e:
                print(f"[record] final flush failed: {e}")
        return ws


def build_parser() -> argparse.ArgumentParser:
    """The full CLI surface, importable so tools (benchmarks, renderers) can
    construct a defaults-accurate args namespace without mirroring it."""
    p = argparse.ArgumentParser()

    # task + metadata
    p.add_argument("--task", type=str, default="og-point-maze",
                   help="initial task (any of the 210); switchable live in the UI")
    p.add_argument("--tasks_json", type=str, default="../tasks.json",
                   help="task metadata (language embeddings, action dims)")

    # checkpoints
    p.add_argument("--tokenizer_ckpt", type=str,
                   default="./logs/tokenizer_ckpts/latest.pt")
    p.add_argument("--dynamics_ckpt", type=str,
                   default="./logs/dynamics_ckpts/latest.pt")

    # rollout
    p.add_argument("--fps", type=float, default=10.0)
    p.add_argument("--packing_factor", type=int, default=2)
    p.add_argument("--ctx_window", type=int, default=24)
    p.add_argument("--schedule", type=str, default="shortcut", choices=["finest", "shortcut"])
    p.add_argument("--eval_d", type=float, default=0.125)
    p.add_argument("--no_amp", action="store_true", help="disable mixed-precision inference")
    p.add_argument("--jpeg_quality", type=int, default=90)
    p.add_argument("--action_smooth_beta", type=float, default=0.817)
    p.add_argument("--tau_ctx", type=float, default=0.01)  # context corruption at inference
    p.add_argument("--tau_init", type=float, default=0.125)  # warm-start denoising toward previous frame (0 = pure noise)

    # web server
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--html", type=str, default="interactive.html")

    # uncertainty overlay
    p.add_argument("--uncertainty_overlay", action="store_true",
                   help="color-code the frame border by per-step denoising instability (off by default)")
    p.add_argument("--calibration_steps", type=int, default=50,
                   help="number of steps used to calibrate the per-episode uncertainty baseline")
    p.add_argument("--u_every", type=int, default=1,
                   help="compute the u_r tokenizer round-trip every N stepped frames "
                        "(amortizes its encoder pass; 1 = every step)")

    # misc
    p.add_argument("--compile", action="store_true", help="torch.compile dynamics and decoder for faster inference")
    p.add_argument("--kv_cache", action="store_true", help="cache context KV in time-attention during denoising")
    p.add_argument("--seed", type=int, default=0)

    # recording
    p.add_argument("--record", action="store_true",
                   help="Buffer every WM-rollout frame and save one mp4 per "
                        "episode boundary (reset, task switch, disconnect). "
                        "Files are named <task>_<session_ts>_ep<NNN>.mp4.")
    p.add_argument("--recordings_dir", type=str,
                   default="./logs/interactive_recordings",
                   help="Where to write per-episode mp4s when --record is set.")

    return p


def main():
    args = build_parser().parse_args()

    server = InteractiveServer(args)

    app = web.Application()
    app.router.add_get("/", server.index)
    app.router.add_get("/ws", server.ws_handler)
    app.router.add_get("/status", server.status)
    app.router.add_get("/healthz", server.healthz)

    print(f"[web] serving on http://{args.host}:{args.port}  (task={args.task})")
    web.run_app(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
