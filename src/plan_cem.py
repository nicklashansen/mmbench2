# plan_cem.py
"""
Minimal WM evaluation: CEM planning with latent-L2-to-goal as the cost.

Validation protocol (single task):
  - Sample an episode from the expert partition uniformly at random.
  - Use the last observation in that episode as the goal.
  - Sample the initial state uniformly from the first 20% of the episode.
  - CEM in action space over horizon H = (L - 1 - start_idx). Terminal cost:
        ||z_{start + H} - z_goal||_2
  - Baselines: expert (GT) action slice, random actions, shuffled actions.
    Expert demos are near-optimal (slight-noise expert policies), so their
    cost under this metric is the reference quality bar.
  - Save per-plan visualization: real-GT frames / WM rollout under GT actions /
    WM rollout under CEM plan.

Typical usage:
    python plan_cem.py --task og-point-maze \
        --n_episodes 4 --n_starts_per_ep 3 \
        --cem_samples 256 --cem_iters 4 \
        --viz_dir ./logs/plan_cem_viz \
        --output ./logs/plan_cem_og-point-maze.csv
"""
import argparse
import glob
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from interactive import (
    load_tokenizer_from_ckpt,
    load_dynamics_from_ckpt,
    make_tau_schedule,
    pack_bottleneck_to_spatial,
    unpack_spatial_to_bottleneck,
    sample_one_timestep_packed,
)
from model import temporal_patchify, temporal_unpatchify, symexp

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# -------------------- data loading (single episode) -------------------- #

@dataclass
class Episode:
    ep_id: int
    frames: torch.Tensor   # (L, 3, H, W) float in [0,1]
    actions: torch.Tensor  # (L, 16) float; actions[0] is nan by convention
    rewards: torch.Tensor  # (L,)     float; rewards[0] is nan by convention


def load_episode(
    task: str, data_dir: str, frames_dir: str, ep_id: int, *, shard_size: int = 4096,
) -> Episode:
    td = torch.load(os.path.join(data_dir, f"{task}.pt"), map_location="cpu", weights_only=False)
    ep = td["episode"].to(torch.int64)
    idxs = (ep == ep_id).nonzero(as_tuple=False).flatten().tolist()
    if not idxs:
        raise ValueError(f"episode {ep_id} not found in {task}")

    shard_paths = sorted(glob.glob(os.path.join(frames_dir, task, "*shard*.pt")))
    if not shard_paths:
        raise FileNotFoundError(f"no shards under {frames_dir}/{task}")

    frames_out: List[Optional[torch.Tensor]] = [None] * len(idxs)
    by_shard: dict[int, list[tuple[int, int]]] = {}
    for out_i, raw_i in enumerate(idxs):
        s, off = raw_i // shard_size, raw_i % shard_size
        by_shard.setdefault(s, []).append((out_i, off))
    for s_idx, picks in by_shard.items():
        sd = torch.load(shard_paths[s_idx], map_location="cpu", weights_only=False)
        fr = sd["frames"]
        if fr.ndim == 4 and fr.shape[-1] == 3 and fr.shape[1] != 3:
            fr = fr.permute(0, 3, 1, 2).contiguous()
        for out_i, off in picks:
            frames_out[out_i] = fr[off]

    frames = torch.stack(frames_out).to(torch.float32) / 255.0
    actions = td["action"][idxs].to(torch.float32)
    rewards = td["reward"][idxs].to(torch.float32)
    return Episode(ep_id=int(ep_id), frames=frames, actions=actions, rewards=rewards)


def list_episode_ids(task: str, data_dir: str) -> List[int]:
    td = torch.load(os.path.join(data_dir, f"{task}.pt"), map_location="cpu", weights_only=False)
    ep = td["episode"].to(torch.int64)
    return ep.unique().tolist()


# -------------------- encoding / decoding -------------------- #

@torch.inference_mode()
def encode_frames_to_packed(
    encoder, frames: torch.Tensor, *, patch: int, n_spatial: int, packing_factor: int,
    use_amp: bool,
) -> torch.Tensor:
    """frames: (N, 3, H, W) float [0,1] on device -> (N, n_spatial, d_spatial)."""
    N, C, H, W = frames.shape
    patches = temporal_patchify(frames.view(N, 1, C, H, W), patch)
    device = frames.device
    with torch.autocast(device_type=device.type, enabled=(use_amp and device.type == "cuda")):
        z_btLd, _ = encoder(patches)
    z_packed = pack_bottleneck_to_spatial(z_btLd, n_spatial=n_spatial, k=packing_factor)
    return z_packed[:, 0].to(torch.float32)


@torch.inference_mode()
def decode_packed_sequence(
    decoder, z_packed_seq: torch.Tensor, *,
    H: int, W: int, C: int, patch: int, packing_factor: int, d_bottleneck: int,
    use_amp: bool,
) -> torch.Tensor:
    """z_packed_seq: (T, n_spatial, d_spatial) -> (T, C, H, W) float [0,1]."""
    T = z_packed_seq.shape[0]
    z_bt = z_packed_seq.unsqueeze(0)  # (1, T, n_spatial, d_spatial)
    z_btLd = unpack_spatial_to_bottleneck(z_bt, k=packing_factor, d_bottleneck=d_bottleneck)
    device = z_packed_seq.device
    with torch.autocast(device_type=device.type, enabled=(use_amp and device.type == "cuda")):
        patches = decoder(z_btLd)
    frames = temporal_unpatchify(patches, H, W, C, patch)  # (1, T, C, H, W)
    return frames[0].clamp(0, 1).to(torch.float32)


# -------------------- batched rollout -------------------- #

@torch.inference_mode()
def imagine_batch(
    dyn, *, z_start: torch.Tensor, action_plan: torch.Tensor, horizon: int,
    sched: dict, k_max: int, act_mask_1d: torch.Tensor,
    lang_emb: Optional[torch.Tensor], use_amp: bool,
    return_h: bool = False,
    use_kv_cache: bool = True,
):
    """
    Inputs:
      z_start:     (B, n_spatial, d_spatial)
      action_plan: (B, H, 16); action_plan[:,t] produces z_{t+1}.
    Returns:
      return_h=False (default): z_seq: (B, 1+H, n_spatial, d_spatial) with
        z_seq[:,0] = z_start.
      return_h=True: (z_seq, h_seq) where h_seq is (B, H, n_agent, D) or
        (B, H, D). h_seq[:,t] is h at rollout position t+1 (the one that
        produced z_{t+1}), which aligns with the reward head's MTP l=0
        convention: at position t+1 with context (z_0..z_{t+1}, a_0..a_t),
        l=0 predicts r_t = the reward received from applying action_plan[t].
        Summing these H predictions gives the total predicted return over
        the plan.

    use_kv_cache=True (default): at each rollout step, prefill the
    transformer's K,V for the past context tokens once and reuse them
    across the K=sched["K"] denoising iterations — so each denoising iter
    only runs the single new noisy token through attention instead of the
    full context+new sequence. Changes per-rollout compute from O(K·H³)
    to O(H³ + K·H²) with no change to the output. Set to False to disable
    caching.
    """
    B = z_start.shape[0]
    device = z_start.device

    # Prepend the zero action at position 0 (backward convention: action[k] produced z[k]).
    zero_a = torch.zeros((B, 1, 16), device=device, dtype=action_plan.dtype)
    full_actions = torch.cat([zero_a, action_plan], dim=1)
    full_mask = act_mask_1d.view(1, 1, -1).expand(B, 1 + horizon, -1).contiguous()

    past = z_start.unsqueeze(1).contiguous()
    z_outs = [z_start]
    h_outs: List[torch.Tensor] = []
    for t in range(horizon):
        out = sample_one_timestep_packed(
            dyn,
            past_packed=past,
            k_max=k_max,
            sched=sched,
            actions=full_actions[:, : past.shape[1] + 1],
            act_mask=full_mask[:, : past.shape[1] + 1],
            use_amp=use_amp,
            return_h=return_h,
            tau_ctx=0.0,
            lang_emb=lang_emb.expand(B, -1) if lang_emb is not None else None,
            use_kv_cache=use_kv_cache,
        )
        if return_h:
            z_next, h_new, _inst = out
            h_outs.append(h_new[:, 0])  # (B, n_agent, D) or (B, D)
        else:
            z_next, _inst = out
        z_outs.append(z_next)
        past = torch.cat([past, z_next.unsqueeze(1)], dim=1)

    z_seq = torch.stack(z_outs, dim=1).to(torch.float32)
    if not return_h:
        return z_seq
    h_seq = torch.stack(h_outs, dim=1)
    return z_seq, h_seq


LATENT_COST_KINDS = ("terminal",)
REWARD_COST_KINDS = ("reward",)


def latent_cost(
    z_seq: torch.Tensor, z_goal: torch.Tensor, *, kind: str = "terminal",
) -> torch.Tensor:
    """z_seq: (B, 1+H, n_spatial, d_spatial); scores z_seq[:, 1:] vs z_goal."""
    z_goal_b = z_goal.view(1, 1, *z_goal.shape).to(z_seq.dtype)
    diff = (z_seq[:, 1:] - z_goal_b).flatten(2)
    per_step_l2 = diff.pow(2).sum(dim=-1).sqrt()
    if kind == "terminal":
        return per_step_l2[:, -1]
    raise ValueError(f"unknown latent cost kind {kind}")


@torch.inference_mode()
def reward_cost(
    rew_head, h_seq: torch.Tensor, *, use_amp: bool, discount: float = 1.0,
) -> torch.Tensor:
    """Cost = -sum_t γ^t r_hat_t, where r_hat_t is the predicted reward at
    rollout step t (t = 0..H-1 within the plan) read from the reward head's
    MTP l=0 position. Sign convention matches the latent-cost modes (CEM
    minimizes), so CEM picking the lowest `reward_cost` is equivalent to
    picking the highest predicted (discounted) return.

    discount=1.0 (default) gives the undiscounted cumulative reward used by
    open-loop planning. discount<1.0 applies per-step γ weighting to the
    predicted return.

    h_seq: (B, H, n_agent, D) or (B, H, D), one hidden state per rollout step.
    """
    device = h_seq.device
    with torch.autocast(device_type=device.type, enabled=(use_amp and device.type == "cuda")):
        logits_btlk, centers_log = rew_head(h_seq)          # (B, H, L, K), (K,)
    logits_l0 = logits_btlk[..., 0, :].float()              # (B, H, K); head l=0 -> r at that step
    probs = logits_l0.softmax(dim=-1)
    expected_symlog = (probs * centers_log.float().view(1, 1, -1)).sum(dim=-1)  # (B, H)
    r_hat = symexp(expected_symlog)                         # (B, H), predicted reward per step
    if float(discount) < 1.0:
        H = r_hat.shape[-1]
        gammas = torch.pow(
            torch.tensor(float(discount), device=r_hat.device, dtype=r_hat.dtype),
            torch.arange(H, device=r_hat.device, dtype=r_hat.dtype),
        )                                                   # (H,)
        return -(r_hat * gammas.view(1, H)).sum(dim=-1)     # (B,)
    return -r_hat.sum(dim=-1)                               # (B,)


# -------------------- CEM -------------------- #

def colored_noise(
    n_samples: int, horizon: int, act_dim: int, beta: float,
    device: torch.device, generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Power-law-spectrum Gaussian noise, shape (n_samples, horizon, act_dim).

    Power spectral density scales as 1 / f^beta along the time axis (iCEM
    convention, Pinneri et al. 2020). beta=0 reproduces white noise; beta=1
    is pink; beta=2 is Brownian (random-walk actions, high temporal
    correlation). The DC component is zeroed so each sampled plan has zero
    mean, and the output is rescaled to unit per-dim std across
    (samples, time) to keep `init_std` interpretable across beta settings.
    """
    w = torch.randn((n_samples, horizon, act_dim), device=device, generator=generator)
    if beta == 0.0:
        return w
    W = torch.fft.rfft(w, dim=1)  # (n_samples, horizon//2 + 1, act_dim)
    freqs = torch.arange(W.shape[1], device=device, dtype=torch.float32)
    scale = torch.zeros_like(freqs)
    scale[1:] = freqs[1:].pow(-beta / 2.0)
    W = W * scale.view(1, -1, 1).to(W.dtype)
    colored = torch.fft.irfft(W, n=horizon, dim=1).to(torch.float32)
    std = colored.flatten(0, 1).std(dim=0, keepdim=True).clamp_min(1e-8)  # (1, act_dim)
    return colored / std.view(1, 1, -1)


@torch.inference_mode()
def cem_plan(
    dyn, *, z_start: torch.Tensor, z_goal: torch.Tensor,
    horizon: int, act_dim: int, act_mask_1d: torch.Tensor,
    sched: dict, k_max: int, lang_emb: Optional[torch.Tensor],
    cost_kind: str,
    n_samples: int, n_elites: int, n_iters: int,
    init_std: float, min_std: float, rollout_batch: int,
    noise_beta: float,
    device: torch.device, use_amp: bool,
    generator: Optional[torch.Generator] = None,
    init_mu: Optional[torch.Tensor] = None,
    n_rollouts_elite: int = 1,
    rew_head=None,
    discount: float = 1.0,
) -> dict:
    """Diagonal-Gaussian CEM over (horizon, act_dim) plans, clipped to [-1,1].
    `noise_beta` controls the time-axis noise color (0 = white, 2 = Brownian).
    `init_mu` optionally seeds the mean (e.g. from a BC warm-start rollout).
    `n_rollouts_elite` scores each sample by averaging over k denoising
    rollouts before elite selection (guards against single-rollout overfit)."""
    if init_mu is not None:
        assert init_mu.shape == (horizon, act_dim), \
            f"init_mu shape {tuple(init_mu.shape)} != (horizon={horizon}, act_dim={act_dim})"
        mu = init_mu.detach().to(device=device, dtype=torch.float32).clamp(-1.0, 1.0).clone()
    else:
        mu = torch.zeros((horizon, act_dim), device=device, dtype=torch.float32)
    std = torch.full((horizon, act_dim), float(init_std), device=device, dtype=torch.float32)

    best_cost = float("inf")
    best_plan = mu.clone()
    history = []

    z_start_b = z_start.unsqueeze(0)

    k_sel = max(1, int(n_rollouts_elite))

    for it in range(n_iters):
        noise = colored_noise(
            n_samples, horizon, act_dim, float(noise_beta),
            device=device, generator=generator,
        )
        samples = (mu + std * noise).clamp(-1.0, 1.0)

        full = torch.zeros((n_samples, horizon, 16), device=device, dtype=torch.float32)
        full[:, :, :act_dim] = samples
        full = full * act_mask_1d.view(1, 1, 16)

        # Replicate each plan k_sel times; each replica gets an independent
        # denoising seed inside the sampler, so averaging their costs guards
        # against single-rollout luck.
        total = n_samples * k_sel
        full_rep = full.unsqueeze(1).expand(-1, k_sel, -1, -1).reshape(total, horizon, 16).contiguous()

        use_reward_cost = cost_kind in REWARD_COST_KINDS
        if use_reward_cost and rew_head is None:
            raise ValueError(f"cost_kind={cost_kind!r} requires a reward head; got rew_head=None")

        costs_flat = torch.empty(total, device=device, dtype=torch.float32)
        for s in range(0, total, rollout_batch):
            e = min(s + rollout_batch, total)
            z0_b = z_start_b.expand(e - s, -1, -1).contiguous()
            if use_reward_cost:
                _z_seq, h_seq = imagine_batch(
                    dyn, z_start=z0_b, action_plan=full_rep[s:e], horizon=horizon,
                    sched=sched, k_max=k_max, act_mask_1d=act_mask_1d,
                    lang_emb=lang_emb, use_amp=use_amp,
                    return_h=True,
                )
                costs_flat[s:e] = reward_cost(rew_head, h_seq, use_amp=use_amp, discount=discount)
            else:
                z_seq = imagine_batch(
                    dyn, z_start=z0_b, action_plan=full_rep[s:e], horizon=horizon,
                    sched=sched, k_max=k_max, act_mask_1d=act_mask_1d,
                    lang_emb=lang_emb, use_amp=use_amp,
                )
                costs_flat[s:e] = latent_cost(z_seq, z_goal, kind=cost_kind)

        costs = costs_flat.view(n_samples, k_sel).mean(dim=1)

        elite_idx = torch.topk(costs, k=n_elites, largest=False).indices
        elites = samples[elite_idx]
        new_mu = elites.mean(dim=0)
        new_std = elites.std(dim=0).clamp_min(float(min_std))

        iter_best_cost = float(costs[elite_idx[0]].item())
        if iter_best_cost < best_cost:
            best_cost = iter_best_cost
            best_plan = samples[elite_idx[0]].clone()

        history.append({
            "iter": it,
            "mean_cost": float(costs.mean().item()),
            "elite_mean_cost": float(costs[elite_idx].mean().item()),
            "best_cost": float(iter_best_cost),
            "running_best_cost": float(best_cost),
            "n_rollouts_elite": int(k_sel),
        })
        mu, std = new_mu, new_std

    return {"best_plan": best_plan, "best_cost": float(best_cost), "history": history}


# -------------------- reference-plan scoring (with returned latents) -------------------- #

@torch.inference_mode()
def score_plan(
    dyn, *, z_start: torch.Tensor, z_goal: torch.Tensor,
    action_plan: torch.Tensor,   # (H, act_dim) in [-1,1]
    act_dim: int, act_mask_1d: torch.Tensor, horizon: int,
    sched: dict, k_max: int, lang_emb: Optional[torch.Tensor],
    cost_kind: str, use_amp: bool, device: torch.device,
    n_rollouts: int = 1, return_first_latents: bool = False,
    rew_head=None,
    discount: float = 1.0,
) -> Tuple[float, Optional[torch.Tensor]]:
    """Averages cost over n_rollouts stochastic denoising seeds. If
    return_first_latents, also returns the first rollout's (1+H, ...) latents
    for visualization."""
    use_reward_cost = cost_kind in REWARD_COST_KINDS
    if use_reward_cost and rew_head is None:
        raise ValueError(f"cost_kind={cost_kind!r} requires a reward head; got rew_head=None")

    full = torch.zeros((n_rollouts, horizon, 16), device=device, dtype=torch.float32)
    full[:, :, :act_dim] = action_plan.unsqueeze(0).expand(n_rollouts, -1, -1)
    full = full * act_mask_1d.view(1, 1, 16)

    z_start_b = z_start.unsqueeze(0).expand(n_rollouts, -1, -1).contiguous()
    if use_reward_cost:
        z_seq, h_seq = imagine_batch(
            dyn, z_start=z_start_b, action_plan=full, horizon=horizon, sched=sched,
            k_max=k_max, act_mask_1d=act_mask_1d, lang_emb=lang_emb, use_amp=use_amp,
            return_h=True,
        )
        costs = reward_cost(rew_head, h_seq, use_amp=use_amp, discount=discount)
    else:
        z_seq = imagine_batch(
            dyn, z_start=z_start_b, action_plan=full, horizon=horizon, sched=sched,
            k_max=k_max, act_mask_1d=act_mask_1d, lang_emb=lang_emb, use_amp=use_amp,
        )
        costs = latent_cost(z_seq, z_goal, kind=cost_kind)
    z0 = z_seq[0].detach() if return_first_latents else None
    return float(costs.mean().item()), z0


# -------------------- visualization -------------------- #

def _pick_viz_indices(T: int, max_frames: int) -> List[int]:
    """Linearly-spaced indices into a T-length sequence, capped at max_frames.
    Always includes 0 and T-1 (start and terminal)."""
    if T <= max_frames:
        return list(range(T))
    return sorted(set(np.linspace(0, T - 1, num=max_frames, dtype=int).tolist()))


def _tile_row(frames_tchw: torch.Tensor, gap_px: int = 4) -> torch.Tensor:
    """(T, C, H, W) -> (C, H, T*W + (T-1)*gap_px) with black gaps between frames."""
    T, C, Hp, Wp = frames_tchw.shape
    if gap_px <= 0 or T == 1:
        return frames_tchw.permute(1, 2, 0, 3).contiguous().view(C, Hp, T * Wp)
    panels = []
    for i in range(T):
        panels.append(frames_tchw[i])
        if i < T - 1:
            panels.append(torch.zeros((C, Hp, gap_px), dtype=frames_tchw.dtype,
                                      device=frames_tchw.device))
    return torch.cat([p if p.dim() == 3 else p for p in panels], dim=-1)


def save_viz(
    *, path: str,
    real_gt: torch.Tensor,   # (T_viz, C, H, W)
    wm_gt: torch.Tensor,     # (T_viz, C, H, W)
    wm_cem: torch.Tensor,    # (T_viz, C, H, W)
    row_gap_px: int = 8,
) -> None:
    assert real_gt.shape == wm_gt.shape == wm_cem.shape, "row shapes must match"
    C, Hp, Wp = real_gt.shape[1:]
    r1 = _tile_row(real_gt)
    r2 = _tile_row(wm_gt)
    r3 = _tile_row(wm_cem)
    gap = torch.zeros((C, row_gap_px, r1.shape[-1]), dtype=r1.dtype, device=r1.device)
    big = torch.cat([r1, gap, r2, gap, r3], dim=1)  # (C, 3H + 2*gap, W*T + gaps)
    big_u8 = (big.clamp(0, 1) * 255.0).to(torch.uint8).cpu().numpy()
    big_hwc = np.transpose(big_u8, (1, 2, 0))
    Image.fromarray(big_hwc, mode="RGB").save(path)


# -------------------- model / task setup -------------------- #

@torch.inference_mode()
def bc_rollout_integrated(
    policy_head, dyn, *,
    z_start: torch.Tensor, horizon: int, act_dim: int,
    act_mask_1d: torch.Tensor, sched: dict, k_max: int,
    lang_emb: Optional[torch.Tensor], use_amp: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Closed-loop BC rollout using the *integrated* policy head.

    The integrated policy is `policy_head(h_t)` where h_t is the dynamics
    transformer's task-conditioned agent-token state at rollout slot t. Task
    conditioning is handled inside the dynamics via `task_proj(lang_emb)`, so
    this rollout is zero-shot across all 200 tasks.

    Convention recap: training shifts BC targets so at slot t (context z_0..z_t,
    a_0..a_t in led-to convention, a_t led to z_t), head l=0 predicts a_{t+1}
    (the action to apply at z_t producing z_{t+1}). To predict the first action
    a_1 at z_0 we need h_0 — obtained here via a one-shot prefill dynamics pass
    on past=[z_0] before the rollout loop starts.

    Returns (z_seq, bc_plan):
      z_seq: (1 + horizon, n_spatial, d_spatial) with z_seq[0] = z_start
      bc_plan: (horizon, act_dim) of the post-clip, post-mask actions taken
    """
    device = z_start.device
    emax = int(round(math.log2(int(k_max))))
    n_spatial, d_spatial = z_start.shape
    lang_emb_1 = lang_emb.expand(1, -1) if lang_emb is not None else None

    past = z_start.unsqueeze(0).unsqueeze(0).contiguous()   # (1, 1, n_spatial, d_spatial)
    full_actions = torch.zeros((1, horizon + 1, 16), device=device, dtype=torch.float32)
    full_mask = act_mask_1d.view(1, 1, -1).expand(1, horizon + 1, -1).contiguous()

    # Prefill: one clean dynamics pass on past=[z_0] to extract h_0 (the agent-
    # token state at slot 0). step_idx=emax + signal_idx=k_max mark the slot as
    # fully denoised (matches how context slots are flagged in the denoising
    # sampler, so behavior is consistent with the learned training distribution).
    with torch.autocast(device_type=device.type, enabled=(use_amp and device.type == "cuda")):
        step_idx_pre = torch.full((1, 1), emax, device=device, dtype=torch.long)
        sig_idx_pre = torch.full((1, 1), int(k_max), device=device, dtype=torch.long)
        _, h_full = dyn(
            full_actions[:, :1],
            step_idx_pre,
            sig_idx_pre,
            past,
            act_mask=full_mask[:, :1],
            agent_tokens=None,
            lang_emb=lang_emb_1,
        )
    if h_full is None:
        raise RuntimeError("bc_rollout_integrated: dynamics returned h_t=None; need n_agent>0 ckpt.")
    h_cur = h_full[:, -1:]  # (1, 1, n_agent, D) — h at slot 0

    bc_plan = torch.zeros((horizon, act_dim), device=device, dtype=torch.float32)
    for t in range(horizon):
        with torch.autocast(device_type=device.type, enabled=(use_amp and device.type == "cuda")):
            a_all = policy_head(h_cur)     # (1, 1, L, A)  -- tanh-squashed in [-1,1]
        a_pred = a_all[:, 0, 0, :act_dim].float().clamp(-1.0, 1.0).squeeze(0)
        full_actions[0, t + 1, :act_dim] = a_pred * act_mask_1d[:act_dim]
        bc_plan[t] = a_pred

        z_next, h_new, _inst = sample_one_timestep_packed(
            dyn,
            past_packed=past,
            k_max=k_max,
            sched=sched,
            actions=full_actions[:, : past.shape[1] + 1],
            act_mask=full_mask[:, : past.shape[1] + 1],
            use_amp=use_amp,
            return_h=True,
            tau_ctx=0.0,
            lang_emb=lang_emb_1,
            use_kv_cache=True,
        )
        past = torch.cat([past, z_next.unsqueeze(1)], dim=1)
        h_cur = h_new  # (1, 1, n_agent, D) — h at newly predicted slot t+1

    return past[0], bc_plan


def build_models(args, device):
    tok, tok_info = load_tokenizer_from_ckpt(args.tokenizer_ckpt, device)
    dyn, rew_head, policy_head, dyn_info = load_dynamics_from_ckpt(
        args.dynamics_ckpt, device=device,
        d_bottleneck=int(tok_info["d_bottleneck"]),
        n_latents=int(tok_info["n_latents"]),
        packing_factor=int(args.packing_factor),
    )
    sched = make_tau_schedule(
        k_max=int(dyn_info["k_max"]),
        schedule=args.schedule,
        d=(args.eval_d if args.schedule == "shortcut" else None),
    )
    info = {**tok_info, **dyn_info, "packing_factor": int(args.packing_factor)}
    return tok.encoder, tok.decoder, dyn, rew_head, policy_head, sched, info


def load_lang_emb(tasks_json: str, task: str, lang_dim: int, device) -> Optional[torch.Tensor]:
    if not tasks_json or not os.path.exists(tasks_json):
        return None
    with open(tasks_json, "r") as f:
        meta = json.load(f)
    te = meta.get(task, {}).get("text_embedding")
    if te is None:
        return None
    emb = torch.tensor(te, dtype=torch.float32, device=device)
    if emb.numel() != lang_dim:
        return None
    return emb.view(1, -1)


def load_act_meta(tasks_json: str, task: str, device=None):
    with open(tasks_json, "r") as f:
        meta = json.load(f)
    act_dim = int(meta.get(task, {}).get("action_dim", 16))
    act_dim = max(0, min(16, act_dim))
    mask = torch.zeros(16, dtype=torch.float32)
    if act_dim > 0:
        mask[:act_dim] = 1.0
    if device is not None:
        mask = mask.to(device)
    return act_dim, mask


# -------------------- per-plan evaluation -------------------- #

def plan_one(
    *, dyn, encoder, decoder, info, sched,
    args, device, use_amp,
    episode: Episode, start_idx: int,
    act_dim: int, act_mask_1d: torch.Tensor,
    lang_emb: Optional[torch.Tensor],
    rng: torch.Generator,
    rew_head=None,
    policy_head=None,
) -> Optional[dict]:
    H_img, W_img, C, patch = int(info["H"]), int(info["W"]), int(info["C"]), int(info["patch"])
    n_spatial, packing_factor = int(info["n_spatial"]), int(info["packing_factor"])
    d_bottleneck, k_max = int(info["d_bottleneck"]), int(info["k_max"])

    frames = episode.frames.to(device)
    actions_stored = episode.actions.to(device)
    L = frames.shape[0]

    max_h_configured = (args.max_horizon if args.max_horizon and args.max_horizon > 0 else L - 1)
    horizon = min(L - 1 - start_idx, max_h_configured)
    if horizon < 2:
        return None

    # Encode start, goal, and real-GT viz frames.
    viz_idx_real = _pick_viz_indices(horizon + 1, args.n_viz_frames)
    viz_abs_idx = [start_idx + i for i in viz_idx_real]
    # Include both endpoints; also ensure we encode them.
    to_encode_idx = sorted(set([start_idx, L - 1] + viz_abs_idx))
    frames_to_encode = frames[to_encode_idx]
    z_encoded = encode_frames_to_packed(
        encoder, frames_to_encode, patch=patch, n_spatial=n_spatial,
        packing_factor=packing_factor, use_amp=use_amp,
    )
    idx_to_z = {i: z for i, z in zip(to_encode_idx, z_encoded)}
    z_start = idx_to_z[start_idx]
    z_goal = idx_to_z[L - 1]

    # GT action slice: actions[start_idx+1 : start_idx+1+horizon] under the
    # stored convention (action[k] produced obs[k]). These produce z_{start+1..start+horizon}.
    gt_plan_full = actions_stored[start_idx + 1: start_idx + 1 + horizon]
    gt_plan = gt_plan_full[:, :act_dim].clamp(-1, 1)
    if torch.isnan(gt_plan).any():
        return None

    # Baselines.
    rand_plan = torch.rand((horizon, act_dim), device=device, generator=rng) * 2.0 - 1.0
    ep_acts = actions_stored[1:, :act_dim]
    perm = torch.randperm(ep_acts.shape[0], generator=rng, device=device)
    shuf_plan = ep_acts[perm[:horizon]].clamp(-1, 1)

    def _sync():
        if device.type == "cuda":
            torch.cuda.synchronize()

    # Warm-start: BC rollout -> action sequence used to seed CEM's mu.
    bc_init_plan = None
    cost_bc_init = float("nan")
    warm_start = str(getattr(args, "warm_start", "none"))
    cem_init_std_eff = float(args.cem_init_std)
    if warm_start == "bc":
        assert policy_head is not None, "warm_start=bc requires a loaded policy head"
        _sync(); t_bcr = time.time()
        _z_bc, bc_init_plan = bc_rollout_integrated(
            policy_head, dyn, z_start=z_start, horizon=horizon, act_dim=act_dim,
            act_mask_1d=act_mask_1d, sched=sched, k_max=k_max,
            lang_emb=lang_emb, use_amp=use_amp,
        )
        bc_init_plan = bc_init_plan.clamp(-1.0, 1.0)
        _sync(); dt_bcr = time.time() - t_bcr
        t_bcs = time.time()
        cost_bc_init, _ = score_plan(
            dyn, z_start=z_start, z_goal=z_goal, action_plan=bc_init_plan,
            act_dim=act_dim, act_mask_1d=act_mask_1d, horizon=horizon,
            sched=sched, k_max=k_max, lang_emb=lang_emb, cost_kind=args.cost,
            use_amp=use_amp, device=device, n_rollouts=args.baseline_rollouts,
            rew_head=rew_head,
        )
        _sync(); dt_bcs = time.time() - t_bcs
        print(f"  [bc-warm] rollout={dt_bcr:.2f}s  rescore={dt_bcs:.2f}s  "
              f"cost_bc_init={cost_bc_init:.3f}", flush=True)
        cem_init_std_eff = float(args.warm_start_init_std)

    # CEM.
    _sync(); t_cem = time.time()
    out = cem_plan(
        dyn, z_start=z_start, z_goal=z_goal,
        horizon=horizon, act_dim=act_dim, act_mask_1d=act_mask_1d,
        sched=sched, k_max=k_max, lang_emb=lang_emb,
        cost_kind=args.cost,
        n_samples=args.cem_samples, n_elites=args.cem_elites,
        n_iters=args.cem_iters, init_std=cem_init_std_eff,
        min_std=args.cem_min_std, rollout_batch=args.rollout_batch,
        noise_beta=args.noise_beta,
        device=device, use_amp=use_amp, generator=rng,
        init_mu=bc_init_plan,
        n_rollouts_elite=int(args.cem_n_rollouts),
        rew_head=rew_head,
    )
    _sync(); dt_cem = time.time() - t_cem
    print(f"  [cem] {dt_cem:.2f}s  n_samples={args.cem_samples} iters={args.cem_iters} "
          f"n_rollouts_elite={args.cem_n_rollouts} rollout_batch={args.rollout_batch}  "
          f"best={out['best_cost']:.3f}", flush=True)

    # Score all plans with matched rollout budget; keep latents for viz.
    _sync(); t_sc = time.time()
    cost_gt, z_seq_gt = score_plan(
        dyn, z_start=z_start, z_goal=z_goal, action_plan=gt_plan,
        act_dim=act_dim, act_mask_1d=act_mask_1d, horizon=horizon,
        sched=sched, k_max=k_max, lang_emb=lang_emb, cost_kind=args.cost,
        use_amp=use_amp, device=device, n_rollouts=args.baseline_rollouts,
        return_first_latents=True,
        rew_head=rew_head,
    )
    cost_rand, _ = score_plan(
        dyn, z_start=z_start, z_goal=z_goal, action_plan=rand_plan,
        act_dim=act_dim, act_mask_1d=act_mask_1d, horizon=horizon,
        sched=sched, k_max=k_max, lang_emb=lang_emb, cost_kind=args.cost,
        use_amp=use_amp, device=device, n_rollouts=args.baseline_rollouts,
        rew_head=rew_head,
    )
    cost_shuf, _ = score_plan(
        dyn, z_start=z_start, z_goal=z_goal, action_plan=shuf_plan,
        act_dim=act_dim, act_mask_1d=act_mask_1d, horizon=horizon,
        sched=sched, k_max=k_max, lang_emb=lang_emb, cost_kind=args.cost,
        use_amp=use_amp, device=device, n_rollouts=args.baseline_rollouts,
        rew_head=rew_head,
    )
    cost_cem, z_seq_cem = score_plan(
        dyn, z_start=z_start, z_goal=z_goal, action_plan=out["best_plan"],
        act_dim=act_dim, act_mask_1d=act_mask_1d, horizon=horizon,
        sched=sched, k_max=k_max, lang_emb=lang_emb, cost_kind=args.cost,
        use_amp=use_amp, device=device, n_rollouts=args.baseline_rollouts,
        return_first_latents=True,
        rew_head=rew_head,
    )
    _sync(); dt_sc = time.time() - t_sc
    print(f"  [score] {dt_sc:.2f}s  gt={cost_gt:.3f} rand={cost_rand:.3f} "
          f"shuf={cost_shuf:.3f} cem={cost_cem:.3f}", flush=True)

    row = {
        "task": str(args.task),
        "ep_id": int(episode.ep_id),
        "start": int(start_idx),
        "ep_len": int(L),
        "horizon": int(horizon),
        "cost_kind": args.cost,
        "cost_gt": cost_gt,
        "cost_random": cost_rand,
        "cost_shuffled": cost_shuf,
        "cost_bc_init": cost_bc_init,
        "cost_cem_best": cost_cem,
        "cem_seconds": dt_cem,
        "cem_iters": int(args.cem_iters),
        "cem_samples": int(args.cem_samples),
        "noise_beta": float(args.noise_beta),
        "warm_start": warm_start,
        "cem_init_std_eff": float(cem_init_std_eff),
        "cem_n_rollouts": int(args.cem_n_rollouts),
        "iters_running_best": ",".join(f"{h['running_best_cost']:.3f}" for h in out["history"]),
    }

    # Visualization + per-plan JSON (incremental, crash-safe). Both live in
    # viz_dir with a shared basename so they can be inspected together.
    if args.viz_dir:
        os.makedirs(args.viz_dir, exist_ok=True)
        base = f"{args.task}_ep{episode.ep_id:03d}_s{start_idx:03d}"

        z_gt_viz = z_seq_gt[viz_idx_real]
        z_cem_viz = z_seq_cem[viz_idx_real]
        wm_gt_frames = decode_packed_sequence(
            decoder, z_gt_viz, H=H_img, W=W_img, C=C, patch=patch,
            packing_factor=packing_factor, d_bottleneck=d_bottleneck, use_amp=use_amp,
        )
        wm_cem_frames = decode_packed_sequence(
            decoder, z_cem_viz, H=H_img, W=W_img, C=C, patch=patch,
            packing_factor=packing_factor, d_bottleneck=d_bottleneck, use_amp=use_amp,
        )
        real_gt_frames = frames[viz_abs_idx]
        save_viz(path=os.path.join(args.viz_dir, base + ".png"),
                 real_gt=real_gt_frames, wm_gt=wm_gt_frames, wm_cem=wm_cem_frames)

        detail = dict(row)
        detail["task"] = str(args.task)
        detail["seed"] = int(args.seed)
        detail["cem_history"] = out["history"]
        detail["plans"] = {
            "gt":       gt_plan.detach().cpu().tolist(),
            "random":   rand_plan.detach().cpu().tolist(),
            "shuffled": shuf_plan.detach().cpu().tolist(),
            "cem_best": out["best_plan"].detach().cpu().tolist(),
            "bc_init":  (bc_init_plan.detach().cpu().tolist() if bc_init_plan is not None else None),
        }
        json_path = os.path.join(args.viz_dir, base + ".json")
        tmp_path = json_path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(detail, f, indent=2)
        os.replace(tmp_path, json_path)

    return row


# -------------------- main -------------------- #

def evaluate(args):
    rank = int(args.rank)
    world_size = int(args.world_size)
    assert world_size >= 1 and 0 <= rank < world_size, \
        f"invalid rank/world_size: rank={rank} world_size={world_size}"
    tag = f"[rank {rank}/{world_size}] " if world_size > 1 else ""

    # Auto-suffix shard outputs so concurrent ranks don't collide.
    if world_size > 1:
        if args.output:
            base, ext = os.path.splitext(args.output)
            args.output = f"{base}.rank{rank}{ext}"
        if args.viz_dir:
            args.viz_dir = f"{args.viz_dir}.rank{rank}"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    py_rng = np.random.default_rng(args.seed)
    torch_rng = torch.Generator(device=device).manual_seed(args.seed)

    encoder, decoder, dyn, rew_head, policy_head, sched, info = build_models(args, device)
    use_amp = bool(args.amp) and device.type == "cuda"
    if str(args.cost) in REWARD_COST_KINDS and rew_head is None:
        raise SystemExit(
            f"--cost={args.cost!r} requires a reward head in the dynamics ckpt. "
            f"Loaded ckpt {args.dynamics_ckpt!r} has none; use a ckpt "
            f"trained with --reward_weight > 0 and --n_agent > 0, or switch "
            f"--cost back to {'|'.join(LATENT_COST_KINDS)}."
        )
    if str(args.warm_start) == "bc" and policy_head is None:
        raise SystemExit(
            f"--warm_start=bc requires a policy head in the dynamics ckpt. "
            f"Loaded ckpt {args.dynamics_ckpt!r} has none; use a ckpt "
            f"trained with --bc_weight > 0 and --n_agent > 0, or switch "
            f"--warm_start to 'none'."
        )

    act_dim, act_mask_1d = load_act_meta(args.tasks_json, args.task, device=device)
    lang_emb = load_lang_emb(args.tasks_json, args.task, int(args.lang_dim), device)
    if act_dim == 0:
        raise SystemExit(f"task {args.task} has act_dim=0; nothing to plan")
    if str(args.warm_start) == "bc":
        print(f"{tag}[plan_cem] BC warm-start via integrated policy head; "
              f"init_std={args.warm_start_init_std}")

    all_ep_ids = list_episode_ids(args.task, args.data_dir)
    n_eps = min(args.n_episodes, len(all_ep_ids))
    chosen_eps = py_rng.choice(all_ep_ids, size=n_eps, replace=False).tolist()
    print(f"{tag}[plan_cem] task={args.task} act_dim={act_dim} "
          f"n_spatial={info['n_spatial']} k_max={info['k_max']} sched.K={sched['K']}")
    print(f"{tag}[plan_cem] sampled episodes: {chosen_eps}")

    all_rows = []
    t0 = time.time()
    plan_idx = 0  # global counter across all ranks; partition via plan_idx % world_size

    for ep_id in chosen_eps:
        epi = load_episode(args.task, args.data_dir, args.frames_dir, ep_id,
                           shard_size=int(args.shard_size))
        L = epi.frames.shape[0]
        upper = max(1, min(int(args.max_start_frame), L - 1))
        # Draw starts identically on every rank (same seed -> same py_rng state),
        # so the plan_idx-based partition below is consistent across the cluster.
        starts = py_rng.integers(0, upper, size=int(args.n_starts_per_ep)).tolist()
        starts = sorted(set(int(s) for s in starts))
        for s in starts:
            if plan_idx % world_size != rank:
                plan_idx += 1
                continue
            row = plan_one(
                dyn=dyn, encoder=encoder, decoder=decoder, info=info, sched=sched,
                args=args, device=device, use_amp=use_amp,
                episode=epi, start_idx=int(s),
                act_dim=act_dim, act_mask_1d=act_mask_1d, lang_emb=lang_emb,
                rng=torch_rng,
                rew_head=rew_head, policy_head=policy_head,
            )
            plan_idx += 1
            if row is None:
                print(f"{tag}[plan_cem] ep {ep_id} s={s}: skipped (short horizon or nan actions)")
                continue
            all_rows.append(row)
            bc_str = f"  bc_init={row['cost_bc_init']:.3f}" if row.get("warm_start") == "bc" else ""
            print(f"{tag}[ep {ep_id:3d} s={s:3d} H={row['horizon']:3d}]  "
                  f"gt={row['cost_gt']:.3f}  rand={row['cost_random']:.3f}  "
                  f"shuf={row['cost_shuffled']:.3f}{bc_str}  cem={row['cost_cem_best']:.3f}  "
                  f"(iters=[{row['iters_running_best']}], {row['cem_seconds']:.1f}s)")

    if not all_rows:
        print(f"{tag}[plan_cem] no rows collected; exiting")
        return

    def arr(k): return np.array([r[k] for r in all_rows], dtype=np.float64)
    gt, rnd, shf, cem = arr("cost_gt"), arr("cost_random"), arr("cost_shuffled"), arr("cost_cem_best")
    horizons = arr("horizon")
    scope = "this rank only" if world_size > 1 else "all plans"
    print(f"\n{tag}========== aggregate over {len(all_rows)} plans "
          f"({scope}; H range {int(horizons.min())}..{int(horizons.max())}, "
          f"mean {horizons.mean():.1f}) ==========")
    print(f"  cost_gt       mean={gt.mean():.3f}  median={np.median(gt):.3f}")
    print(f"  cost_random   mean={rnd.mean():.3f}  median={np.median(rnd):.3f}")
    print(f"  cost_shuffled mean={shf.mean():.3f}  median={np.median(shf):.3f}")
    if any(r.get("warm_start") == "bc" for r in all_rows):
        bci = arr("cost_bc_init")
        print(f"  cost_bc_init  mean={bci.mean():.3f}  median={np.median(bci):.3f}  "
              f"(CEM warm-started from BC plan)")
    print(f"  cost_cem_best mean={cem.mean():.3f}  median={np.median(cem):.3f}")
    # Validation framing depends on cost semantics. Latent distances are
    # non-negative: GT is near-zero under good WM, so gt*1.1 is a meaningful
    # "match-expert" tolerance and (cem-gt)/gt is a signed relative error.
    # Reward-cost is negated predicted return and typically spans both signs;
    # gt*1.1 flips meaning and relative error blows up near zero. Report
    # reward-appropriate statistics instead.
    if str(args.cost) in REWARD_COST_KINDS:
        r_gt, r_rnd, r_cem = -gt, -rnd, -cem
        print(f"  reward_gt     mean={r_gt.mean():+.3f}  median={np.median(r_gt):+.3f}")
        print(f"  reward_random mean={r_rnd.mean():+.3f}  median={np.median(r_rnd):+.3f}")
        print(f"  reward_cem    mean={r_cem.mean():+.3f}  median={np.median(r_cem):+.3f}")
        print(f"  (reward_gt > reward_random) rate : {(r_gt > r_rnd).mean():.1%}    <- validation #1 (predicted reward is informative)")
        print(f"  (reward_cem >= reward_gt)   rate : {(r_cem >= r_gt).mean():.1%}  <- validation #2 (CEM >= GT predicted return)")
        print(f"  reward_cem - reward_gt : mean={(r_cem - r_gt).mean():+.3f}  median={np.median(r_cem - r_gt):+.3f}  (positive = CEM improves on GT)")
    else:
        print(f"  (gt < random)        rate : {(gt < rnd).mean():.1%}    <- validation #1 (cost is informative)")
        print(f"  (cem <= gt * 1.1)    rate : {(cem <= gt * 1.1).mean():.1%}  <- validation #2 (CEM recovers expert-quality plans)")
        print(f"  (cem < random)       rate : {(cem < rnd).mean():.1%}")
        rel = (cem - gt) / np.maximum(gt, 1e-6)
        print(f"  (cem - gt)/gt        : mean={rel.mean():+.1%}  median={np.median(rel):+.1%}")
    print(f"  total wallclock: {time.time() - t0:.1f}s")

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        import csv
        with open(args.output, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            w.writeheader()
            w.writerows(all_rows)
        print(f"[plan_cem] wrote {args.output}")
    if args.viz_dir:
        print(f"[plan_cem] wrote visualizations to {args.viz_dir}")


def main():
    p = argparse.ArgumentParser()

    # task / data
    p.add_argument("--task", type=str, default="og-point-maze")
    p.add_argument("--data_dir", type=str, default="./data/expert")
    p.add_argument("--frames_dir", type=str, default="./data/expert-shards")
    p.add_argument("--tasks_json", type=str, default="../tasks.json")
    p.add_argument("--shard_size", type=int, default=4096)
    p.add_argument("--lang_dim", type=int, default=512)

    # checkpoints
    p.add_argument("--tokenizer_ckpt", type=str,
                   default="./logs/tokenizer_ckpts/latest.pt")
    p.add_argument("--dynamics_ckpt", type=str,
                   default="./logs/dynamics_ckpts/latest.pt")

    # rollout / denoising schedule
    p.add_argument("--packing_factor", type=int, default=2)
    p.add_argument("--schedule", type=str, default="shortcut", choices=["finest", "shortcut"])
    p.add_argument("--eval_d", type=float, default=0.125)
    p.add_argument("--amp", action="store_true", default=True)

    # planning problem
    p.add_argument(
        "--cost", type=str, default="terminal",
        choices=list(LATENT_COST_KINDS) + list(REWARD_COST_KINDS),
        help=(
            "CEM cost. Latent modes ({LATENT}) score z_seq vs z_goal "
            "(reward-head-free). 'reward' uses the dynamics ckpt's reward head: "
            "cost = -sum_t r_hat_t (l=0 MTP slot at each rollout step). Requires "
            "a ckpt trained with n_agent>0 and reward_weight>0."
        ).format(LATENT="|".join(LATENT_COST_KINDS)),
    )
    p.add_argument("--n_episodes", type=int, default=4)
    p.add_argument("--n_starts_per_ep", type=int, default=3)
    p.add_argument("--max_start_frame", type=int, default=5,
                   help="Sample each plan's start index uniformly from [0, max_start_frame). "
                        "Default 5 keeps starts close to the episode beginning so horizons are "
                        "comparable across tasks with different episode lengths.")
    p.add_argument("--max_horizon", type=int, default=0,
                   help="0 = use full remaining length (L - 1 - start). Cap for compute if needed.")

    # CEM
    p.add_argument("--cem_samples", type=int, default=256)
    p.add_argument("--cem_elites", type=int, default=32)
    p.add_argument("--cem_iters", type=int, default=4)
    p.add_argument("--cem_init_std", type=float, default=1.0)
    p.add_argument("--cem_min_std", type=float, default=0.05)
    p.add_argument("--rollout_batch", type=int, default=128)
    p.add_argument("--cem_n_rollouts", type=int, default=1,
                   help="Rollouts per sample during CEM elite selection. "
                        ">1 averages denoising-seed noise so the selector can't "
                        "overfit to a lucky single rollout.")
    p.add_argument("--noise_beta", type=float, default=0.0,
                   help="Action-noise power spectrum 1/f^beta. "
                        "0 = white (default); "
                        "1 = pink; 2 = Brownian (high temporal correlation, "
                        "recommended for navigation / smooth-velocity tasks).")

    # baselines
    p.add_argument("--baseline_rollouts", type=int, default=4,
                   help="stochastic rollouts to average for each reference plan's cost")

    # BC warm-start
    p.add_argument("--warm_start", type=str, default="none", choices=["none", "bc"],
                   help=(
                       "If 'bc', seed CEM's mu with a closed-loop rollout of the 200-task "
                       "language-conditioned policy head baked into the dynamics "
                       "ckpt (reads dynamics h_t agent tokens, zero-shot across tasks), "
                       "and use --warm_start_init_std instead of --cem_init_std."
                   ))
    p.add_argument("--warm_start_init_std", type=float, default=0.3,
                   help="CEM init_std when warm-starting from BC (should be < cem_init_std).")

    # outputs
    p.add_argument("--output", type=str, default="./logs/plan_cem_og-point-maze.csv")
    p.add_argument("--viz_dir", type=str, default="./logs/plan_cem_viz")
    p.add_argument("--n_viz_frames", type=int, default=12,
                   help="number of timesteps to display in each visualization row")

    # misc
    p.add_argument("--seed", type=int, default=0)

    # multi-GPU sharding (plans are embarrassingly parallel; no torch.distributed needed).
    # Launch N processes with CUDA_VISIBLE_DEVICES=i and --rank i --world_size N;
    # each rank owns plans whose global index satisfies `i % world_size == rank`.
    # --output and --viz_dir are auto-suffixed with `.rank{i}` when world_size > 1.
    # Concatenate the per-rank shards to obtain the global aggregate.
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--world_size", type=int, default=1)

    args = p.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
