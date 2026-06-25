# train_dynamics.py
"""Train the Dreamer 4 dynamics model: a block-causal Transformer over frozen
tokenizer latents, trained with shortcut flow-matching and action conditioning
(plus optional reward and behavior-cloning heads and coverage-aware sampling).
Run from inside ``src/`` (flat imports), e.g.
``torchrun --nproc_per_node=8 train_dynamics.py``.
"""
import os
import time
import math
import random
import argparse
from contextlib import nullcontext
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.amp import autocast
from torch.utils.data import DataLoader

import wandb

from task_set import TASK_SET, DOMAINS, UNSEEN_TASK_SET, task_to_domain, compute_task_weights

from model import (
    Encoder, Decoder, Tokenizer,
    temporal_patchify, temporal_unpatchify,
    pack_bottleneck_to_spatial,
    unpack_spatial_to_bottleneck,
    Dynamics, RewardHeadMTP, PolicyHeadMTP,
    symlog, dist_cross_entropy_from_symlog,
    EmaRms,
)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class PerDomainAccumulator:
    """Buckets per-sample losses by domain and aggregates across DDP ranks.

    Each rank keeps running sums/counts of losses between log flushes,
    bucketed by the sample's domain index (0..n_domains-1). flush()
    optionally all-reduces across ranks and returns per-domain means.
    """
    def __init__(self, n_domains: int, device: torch.device):
        self.n_domains = int(n_domains)
        self.device = device
        self.loss_sum = torch.zeros(self.n_domains, device=device, dtype=torch.float64)
        self.count = torch.zeros(self.n_domains, device=device, dtype=torch.float64)

    @torch.no_grad()
    def update(self, loss_per_sample: torch.Tensor, domain_ids: torch.Tensor):
        # loss_per_sample: (B,) float; domain_ids: (B,) int in [0, n_domains).
        l = loss_per_sample.detach().to(device=self.device, dtype=torch.float64)
        d = domain_ids.to(device=self.device, dtype=torch.long)
        self.loss_sum.index_add_(0, d, l)
        self.count.index_add_(0, d, torch.ones_like(l))

    @torch.no_grad()
    def flush(self, ddp: bool) -> tuple[torch.Tensor, torch.Tensor]:
        loss_sum = self.loss_sum.clone()
        count = self.count.clone()
        if ddp:
            dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(count, op=dist.ReduceOp.SUM)
        means = torch.where(count > 0, loss_sum / count.clamp_min(1.0), torch.full_like(loss_sum, float("nan")))
        self.loss_sum.zero_()
        self.count.zero_()
        return means, count


def get_dist_info():
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return rank, world_size, local_rank


def is_rank0() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def seed_everything(seed: int):
    s = int(seed) % (2**32)
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def worker_init_fn(worker_id: int):
    info = torch.utils.data.get_worker_info()
    seed_everything(info.seed)


def init_distributed() -> tuple[bool, int, int, int]:
    rank, world_size, local_rank = get_dist_info()
    ddp = world_size > 1
    if ddp:
        # 60-min collective timeout (default is 10 min). Needed for runs that do
        # long single-rank work like multi-episode env evaluation, where rank 0
        # stays in eval while the other ranks block at the next gradient
        # all-reduce.
        import datetime
        dist.init_process_group(
            backend="nccl", init_method="env://",
            timeout=datetime.timedelta(minutes=60),
        )
        torch.cuda.set_device(local_rank)
    return ddp, rank, world_size, local_rank


def _unwrap_model(model):
    """Strip DDP and torch.compile wrappers to get the raw nn.Module."""
    target = model
    if hasattr(target, "module"):       # DDP wrapper
        target = target.module
    if hasattr(target, "_orig_mod"):    # torch.compile wrapper
        target = target._orig_mod
    return target


def save_ckpt(path: Path, *, step: int, epoch: int, dyn_model, rew_head, policy_head, opt, args: argparse.Namespace, rms_state: dict = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    rew_head_state = _unwrap_model(rew_head).state_dict() if rew_head is not None else None
    policy_head_state = _unwrap_model(policy_head).state_dict() if policy_head is not None else None
    obj = {
        "step": step,
        "epoch": epoch,
        "dynamics": _unwrap_model(dyn_model).state_dict(),
        "rew_head": rew_head_state,
        "policy_head": policy_head_state,
        "opt": opt.state_dict(),
        "args": vars(args),
    }
    if rms_state is not None:
        obj["rms_state"] = rms_state
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def load_ckpt(path: Path, *, dyn_model, rew_head, policy_head, opt, rms_objects: dict = None, strict: bool = True) -> tuple[int, int]:
    ckpt = torch.load(path, map_location="cpu")
    state = ckpt["dynamics"]
    result = _unwrap_model(dyn_model).load_state_dict(state, strict=strict)
    if not strict and is_rank0():
        if result.missing_keys:
            print(f"[rank0][info] Missing keys (randomly initialized): {result.missing_keys}")
        if result.unexpected_keys:
            print(f"[rank0][info] Unexpected keys (ignored): {result.unexpected_keys}")

    # Reward head is optional (may be absent from older checkpoints)
    if rew_head is not None and ckpt.get("rew_head") is not None:
        _unwrap_model(rew_head).load_state_dict(ckpt["rew_head"], strict=True)
    # Policy (BC) head is optional (may be absent from older checkpoints)
    if policy_head is not None and ckpt.get("policy_head") is not None:
        _unwrap_model(policy_head).load_state_dict(ckpt["policy_head"], strict=True)
    # Optimizer state may be incompatible when enabling rewards/BC mid-run; fall back gracefully.
    try:
        opt.load_state_dict(ckpt["opt"])
    except Exception as e:
        if is_rank0():
            print(f"[rank0][warning] Could not load optimizer state (likely param mismatch after enabling rewards/BC): {e}")
            print("[rank0][warning] Continuing with freshly-initialized optimizer state.")

    if rms_objects is not None:
        for k, v in rms_objects.items():
            if k in ckpt.get("rms_state", {}):
                v.load_state_dict(ckpt["rms_state"][k])
    return int(ckpt.get("step", 0)), int(ckpt.get("epoch", 0))


@torch.no_grad()
def load_frozen_tokenizer_from_pt_ckpt(
    ckpt_path: str,
    *,
    device: torch.device,
    override: Optional[Dict[str, Any]] = None,
) -> tuple[Encoder, Decoder, Dict[str, Any]]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    tok_args = dict(ckpt.get("args", {}))
    if override:
        tok_args.update(override)

    # Required keys (fall back to defaults if missing)
    H = int(tok_args.get("H", 224))
    W = int(tok_args.get("W", 224))
    C = int(tok_args.get("C", 3))
    patch = int(tok_args.get("patch", 4))
    n_patches = (H // patch) * (W // patch)
    d_patch = patch * patch * C

    enc = Encoder(
        patch_dim=d_patch,
        d_model=int(tok_args.get("d_model", 256)),
        n_latents=int(tok_args.get("n_latents", 16)),
        n_patches=n_patches,
        n_heads=int(tok_args.get("n_heads", 4)),
        depth=int(tok_args.get("depth", 8)),
        d_bottleneck=int(tok_args.get("d_bottleneck", 32)),
        dropout=0.0,
        mlp_ratio=float(tok_args.get("mlp_ratio", 4.0)),
        time_every=int(tok_args.get("time_every", 1)),
        latents_only_time=bool(tok_args.get("latents_only_time", True)),
        mae_p_min=0.0,
        mae_p_max=0.0,
    )
    dec = Decoder(
        d_bottleneck=int(tok_args.get("d_bottleneck", 32)),
        d_model=int(tok_args.get("d_model", 256)),
        n_heads=int(tok_args.get("n_heads", 4)),
        depth=int(tok_args.get("depth", 8)),
        n_latents=int(tok_args.get("n_latents", 16)),
        n_patches=n_patches,
        d_patch=d_patch,
        dropout=0.0,
        mlp_ratio=float(tok_args.get("mlp_ratio", 4.0)),
        time_every=int(tok_args.get("time_every", 1)),
        latents_only_time=bool(tok_args.get("latents_only_time", True)),
    )

    tok = Tokenizer(enc, dec)
    state = ckpt["model"]
    # Strip _orig_mod. prefix added by torch.compile
    state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
    tok.load_state_dict(state, strict=True)

    tok = tok.to(device)
    tok.eval()
    for p in tok.parameters():
        p.requires_grad_(False)

    return tok.encoder, tok.decoder, tok_args


def _emax_from_kmax(k_max: int) -> int:
    emax = int(round(math.log2(k_max)))
    assert (1 << emax) == k_max, "k_max must be power of two"
    return emax


def _sample_step_excluding_dmin(device: torch.device, B: int, T: int, k_max: int) -> tuple[torch.Tensor, torch.Tensor]:
    emax = _emax_from_kmax(k_max)
    # step_idx in [0, emax) i.e. excludes emax (d_min)
    step_idx = torch.randint(low=0, high=max(1, emax), size=(B, T), device=device, dtype=torch.long)
    d = 1.0 / (1 << step_idx).to(torch.float32)
    return d, step_idx


def _sample_tau_for_step(device: torch.device, B: int, T: int, k_max: int, step_idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # K = 2^step_idx
    K = (1 << step_idx).to(torch.long)  # (B,T)
    u = torch.rand((B, T), device=device, dtype=torch.float32)
    j_idx = torch.floor(u * K.to(torch.float32)).to(torch.long)  # (B,T) in [0,K)
    tau = j_idx.to(torch.float32) / K.to(torch.float32)          # (B,T)
    scale = torch.div(torch.tensor(k_max, device=device), K, rounding_mode="floor")  # (B,T)
    tau_idx = j_idx * scale                                      # (B,T) <= k_max-1
    return tau, tau_idx


def dynamics_pretrain_loss(
    dynamics: torch.nn.Module,
    *,
    z1: torch.Tensor,                    # (B,T,Sz,Dz) packed clean targets
    actions: Optional[torch.Tensor],     # (B,T,A) led-to convention: actions[t] led to obs[t]; actions[:,0]=0
    act_mask: Optional[torch.Tensor],    # (B,T,A) per-task per-dim validity, same led-to shift as actions
    k_max: int,
    B_self: int,
    step: int,
    bootstrap_start: int,
    lang_emb: Optional[torch.Tensor] = None,         # (B,lang_dim) task embedding
    rew_head: Optional[torch.nn.Module] = None,
    rewards: Optional[torch.Tensor] = None,          # (B,T) aligned to frames/actions (see train loop)
    reward_weight: float = 0.0,
    policy_head: Optional[torch.nn.Module] = None,
    bc_weight: float = 0.0,
    rms_flow: Optional[EmaRms] = None,
    rms_rew: Optional[EmaRms] = None,
    rms_bc: Optional[EmaRms] = None,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    device = z1.device
    B, T = z1.shape[:2]
    assert 0 <= B_self < B
    B_emp = B - B_self
    emax = _emax_from_kmax(k_max)

    # action mask slices
    act_mask_full = act_mask
    act_mask_self = None if act_mask_full is None else act_mask_full[B_emp:]

    # step idx: empirical rows are finest (d_min), self rows sample coarser
    step_idx_emp = torch.full((B_emp, T), emax, device=device, dtype=torch.long)
    if B_self > 0:
        d_self, step_idx_self = _sample_step_excluding_dmin(device, B_self, T, k_max)
        step_idx_full = torch.cat([step_idx_emp, step_idx_self], dim=0)
    else:
        d_self = torch.zeros((0, T), device=device, dtype=torch.float32)
        step_idx_self = torch.zeros((0, T), device=device, dtype=torch.long)
        step_idx_full = step_idx_emp

    # sigma/tau per row/time
    sigma_full, sigma_idx_full = _sample_tau_for_step(device, B, T, k_max, step_idx_full)
    sigma_emp = sigma_full[:B_emp]
    sigma_self = sigma_full[B_emp:]
    sigma_idx_self = sigma_idx_full[B_emp:]

    # Corrupt inputs
    z0_full = torch.randn_like(z1)
    z_tilde_full = (1.0 - sigma_full)[..., None, None] * z0_full + sigma_full[..., None, None] * z1
    z_tilde_self = z_tilde_full[B_emp:]

    # Weights
    w_emp = 0.9 * sigma_emp + 0.1
    w_self = 0.9 * sigma_self + 0.1

    lang_emb_self = lang_emb[B_emp:] if lang_emb is not None else None

    # Main forward
    z1_hat_full, h_t_full = dynamics(actions, step_idx_full, sigma_idx_full, z_tilde_full, act_mask=act_mask_full, agent_tokens=None, lang_emb=lang_emb)
    z1_hat_emp = z1_hat_full[:B_emp]
    z1_hat_self = z1_hat_full[B_emp:]

    flow_per = (z1_hat_emp.float() - z1[:B_emp].float()).pow(2).mean(dim=(2, 3))  # (B_emp,T)
    loss_emp = (flow_per * w_emp).mean()

    boot_mse = torch.zeros((), device=device, dtype=torch.float32)
    loss_self = torch.zeros((), device=device, dtype=torch.float32)

    do_boot = (B_self > 0) and (step >= bootstrap_start)
    if do_boot:
        d_half = d_self / 2.0
        step_idx_half = step_idx_self + 1
        sigma_plus = sigma_self + d_half
        sigma_idx_plus = (sigma_idx_self + (torch.tensor(k_max, device=device, dtype=torch.float32) * d_half).to(torch.long)).clamp(0, k_max)

        with torch.no_grad():
            z1_hat_half1, _ = dynamics(actions[B_emp:] if actions is not None else None, step_idx_half, sigma_idx_self, z_tilde_self, act_mask=act_mask_self, agent_tokens=None, lang_emb=lang_emb_self)
            b_prime = (z1_hat_half1.float() - z_tilde_self.float()) / (1.0 - sigma_self).clamp_min(1e-6)[..., None, None]
            z_prime = z_tilde_self.float() + b_prime * d_half[..., None, None]

            z1_hat_half2, _ = dynamics(actions[B_emp:] if actions is not None else None, step_idx_half, sigma_idx_plus, z_prime.to(z_tilde_self.dtype), act_mask=act_mask_self, agent_tokens=None, lang_emb=lang_emb_self)
            b_doubleprime = (z1_hat_half2.float() - z_prime.float()) / (1.0 - sigma_plus).clamp_min(1e-6)[..., None, None]

        vhat_sigma = (z1_hat_self.float() - z_tilde_self.float()) / (1.0 - sigma_self).clamp_min(1e-6)[..., None, None]
        vbar_target = (b_prime + b_doubleprime) / 2.0

        boot_per = (1.0 - sigma_self).pow(2) * (vhat_sigma - vbar_target).pow(2).mean(dim=(2, 3))  # (B_self,T)
        loss_self = (boot_per * w_self).mean()
        boot_mse = boot_per.mean()

    # Combine losses
    loss_flow = ((loss_emp * (B - B_self)) + (loss_self * B_self)) / B

    # -----------------------
    # Reward modeling loss (optional)
    # -----------------------
    loss_rew = torch.zeros((), device=device, dtype=torch.float32)

    if (rew_head is not None) and (rewards is not None) and (reward_weight > 0.0):
        if h_t_full is None:
            raise RuntimeError("Reward modeling requested but dynamics returned h_t_full=None (set n_agent>0).")

        # logits: (B,T,L,K), centers_log: (K,)
        logits_btlk, centers_log = rew_head(h_t_full)
        K = logits_btlk.shape[-1]
        L = logits_btlk.shape[-2]

        # Align targets for MTP:
        # head l predicts reward at time (t + l). Mask invalid shifts.
        # Shift rewards to led-to convention: rewards_shift[t] = r_{t-1} = reward
        # received arriving at state s_t via action a_{t-1}. This matches the
        # led-to action convention already used by the dynamics, so head l=0 at
        # time t predicts a reward deterministic in h_t[t]'s causal context
        # (s_0..s_t, a_0..a_{t-1}). r_{-1} is undefined at t=0.
        rewards_shift = torch.zeros_like(rewards)
        rewards_shift[:, 1:] = rewards[:, :-1]
        rew_symlog = symlog(rewards_shift.float())  # (B,T)
        valid_bt = torch.ones((B, T), device=device, dtype=torch.bool)
        valid_bt[:, 0] = False

        # Build (B,T,L) shifted targets + masks with pad+unfold
        # pad on the right with L-1 zeros/False so unfold yields exactly L windows.
        rew_pad = F.pad(rew_symlog, (0, max(0, L - 1)))              # (B,T+L-1)
        msk_pad = F.pad(valid_bt,   (0, max(0, L - 1)), value=False) # (B,T+L-1)
        tgt_blt = rew_pad.unfold(dimension=1, size=T, step=1)        # (B,L,T)
        msk_blt = msk_pad.unfold(dimension=1, size=T, step=1)        # (B,L,T)
        tgt_btl = tgt_blt.permute(0, 2, 1).contiguous()              # (B,T,L)
        msk_btl = msk_blt.permute(0, 2, 1).contiguous()              # (B,T,L)

        loss_rew = dist_cross_entropy_from_symlog(
            logits=logits_btlk.float().reshape(-1, K),
            target_symlog=tgt_btl.reshape(-1),
            centers_log=centers_log,  # (K,)
            mask=msk_btl.reshape(-1),
        )

    # -----------------------
    # BC (policy) loss (optional, gradient-isolated from dynamics)
    # -----------------------
    # h_t.detach() severs the graph upstream: BC gradients update only
    # policy_head's parameters, not dynamics / agent-token init.
    #
    # Target alignment (mirror of the reward head but shifted the opposite way):
    # at slot t, the causal context has seen obs[0..t] + actions[0..t] (led-to
    # convention). The natural BC target is the NEXT action to take from obs[t],
    # which in the led-to array is actions[t+1]. Head l at slot t predicts
    # actions[t+1+l]. Invalid when t+1+l > T-1. act_mask is also shifted to the
    # "next position" to get the correct per-dim validity for the target action.
    loss_bc = torch.zeros((), device=device, dtype=torch.float32)

    if (policy_head is not None) and (bc_weight > 0.0) and (actions is not None) and (act_mask is not None):
        if h_t_full is None:
            raise RuntimeError("BC loss requested but dynamics returned h_t_full=None (set n_agent>0).")

        A = actions.shape[-1]

        # Shift actions / per-dim mask forward by 1: actions_next[t] = actions[t+1]
        actions_next = torch.zeros_like(actions)
        actions_next[:, :-1] = actions[:, 1:]
        mask_next = torch.zeros_like(act_mask)
        mask_next[:, :-1] = act_mask[:, 1:]

        # Time-position validity: actions[t+1] exists iff t < T-1.
        valid_bt = torch.ones((B, T), device=device, dtype=torch.bool)
        valid_bt[:, -1] = False

        L_bc = _unwrap_model(policy_head).L

        # pad + unfold along the T axis, producing (B, T, L, A) targets and masks.
        # F.pad order for (B, T, A): (last_dim_left, last_dim_right, T_left, T_right)
        act_pad = F.pad(actions_next, (0, 0, 0, max(0, L_bc - 1)))          # (B, T+L-1, A)
        act_unf = act_pad.unfold(dimension=1, size=T, step=1)               # (B, L, A, T)
        tgt_btla = act_unf.permute(0, 3, 1, 2).contiguous()                 # (B, T, L, A)

        mask_pad = F.pad(mask_next, (0, 0, 0, max(0, L_bc - 1)))            # (B, T+L-1, A)
        mask_unf = mask_pad.unfold(dimension=1, size=T, step=1)             # (B, L, A, T)
        dim_mask_btla = mask_unf.permute(0, 3, 1, 2).contiguous()           # (B, T, L, A)

        valid_pad = F.pad(valid_bt, (0, max(0, L_bc - 1)), value=False)     # (B, T+L-1)
        valid_unf = valid_pad.unfold(dimension=1, size=T, step=1)           # (B, L, T)
        pos_mask_btl = valid_unf.permute(0, 2, 1).contiguous()              # (B, T, L)

        # Combined per-element mask: valid timestep AND valid dim for this task.
        full_mask = pos_mask_btl.unsqueeze(-1).to(dim_mask_btla.dtype) * dim_mask_btla  # (B, T, L, A)

        # Policy forward with gradient isolation.
        pred_btla = policy_head(h_t_full.detach())                          # (B, T, L, A)

        diff_sq = (pred_btla.float() - tgt_btla.float()).pow(2)
        masked_sq = diff_sq * full_mask.float()
        denom = full_mask.float().sum().clamp_min(1.0)
        loss_bc = masked_sq.sum() / denom

    if rms_flow is not None:
        rms_flow.update(loss_flow)
        loss_flow_normed = rms_flow.normalize(loss_flow)
    else:
        loss_flow_normed = loss_flow

    if rms_rew is not None and float(loss_rew.item()) > 0.0:
        rms_rew.update(loss_rew)
        loss_rew_normed = rms_rew.normalize(loss_rew)
    else:
        loss_rew_normed = loss_rew

    if rms_bc is not None and float(loss_bc.item()) > 0.0:
        rms_bc.update(loss_bc)
        loss_bc_normed = rms_bc.normalize(loss_bc)
    else:
        loss_bc_normed = loss_bc

    loss = (
        loss_flow_normed
        + float(reward_weight) * loss_rew_normed
        + float(bc_weight) * loss_bc_normed
    )

    aux = {
        "flow_mse": flow_per.mean().detach(),
        "flow_per_sample": flow_per.mean(dim=1).detach(),  # (B_emp,) — for per-domain bucketing (unweighted)
        "bootstrap_mse": boot_mse.detach(),
        "loss_emp": loss_emp.detach(),
        "loss_self": loss_self.detach(),
        "sigma_mean": sigma_full.mean().detach(),
        "sigma_std": sigma_full.std().detach(),
        "loss_rew": loss_rew.detach(),
        "loss_bc": loss_bc.detach(),
        "loss_flow": loss_flow.detach(),
        "rms_flow": rms_flow.rms_val if rms_flow is not None else 1.0,
        "rms_rew": rms_rew.rms_val if rms_rew is not None else 1.0,
        "rms_bc": rms_bc.rms_val if rms_bc is not None else 1.0,
    }
    return loss, aux


def _is_pow2(n: int) -> bool:
    return (n > 0) and ((n & (n - 1)) == 0)


def make_tau_schedule(*, k_max: int, schedule: str, d: Optional[float] = None) -> Dict[str, Any]:
    """
    Returns a schedule dict:
      K = number of integration steps (also grid size)
      e = log2(K)  (step_idx)
      scale = k_max // K
      tau_idx[i] = discrete signal index at step i
      tau[i] = i/K
      dt = 1/K
    """
    assert _is_pow2(k_max), "k_max must be power of two"
    if schedule == "finest":
        K = k_max
    elif schedule == "shortcut":
        assert d is not None, "shortcut schedule requires --eval_d"
        inv = int(round(1.0 / float(d)))
        assert _is_pow2(inv), "eval_d must be 1/(power of two)"
        assert inv <= k_max, "eval_d must be >= 1/k_max"
        assert (k_max % inv) == 0, "k_max must be divisible by 1/eval_d"
        K = inv
    else:
        raise ValueError(f"unknown schedule: {schedule}")

    e = int(round(math.log2(K)))
    scale = k_max // K
    tau = [i / K for i in range(K)] + [1.0]
    tau_idx = [i * scale for i in range(K)] + [k_max]  # allow final clean index
    return dict(K=K, e=e, scale=scale, tau=tau, tau_idx=tau_idx, dt=1.0 / K, schedule=schedule, d=1.0 / K)


@torch.no_grad()
def sample_one_timestep_packed(
    dyn: Dynamics,
    *,
    past_packed: torch.Tensor,          # (B,t,n_spatial,d_spatial)
    k_max: int,
    sched: Dict[str, Any],
    actions: Optional[torch.Tensor] = None,     # (B,T,A) aligned to frames or None
    act_mask: Optional[torch.Tensor] = None,    # (B,T,A) or (A,) or None
    tau_ctx: float = 0.0,               # context corruption level
    lang_emb: Optional[torch.Tensor] = None,    # (B,lang_dim) or None
    use_amp: bool = True,               # match training bf16 autocast
    use_kv_cache: bool = False,         # cache time-attn K,V for context tokens
) -> torch.Tensor:
    """Generate next packed latent z_t given past length t.

    When `use_kv_cache=True` and `t > 0`, the time-attention K,V for the t
    context tokens is computed once in a prefill pass and reused across all K
    Euler denoising steps. Each denoising step then only runs the single new
    token through the transformer, attending to the cached context. This cuts
    per-step attention from O(t+1) to O(1) and gives roughly a 3x speedup at
    typical ctx_window=24 with K=4 denoising steps. Requires `Dynamics` to
    support the `return_kv_cache` / `kv_cache` kwargs.
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

    # start from noise at tau=0
    z = torch.randn((B, 1, n_spatial, d_spatial), device=device, dtype=dtype)

    emax = int(round(math.log2(k_max)))

    # Slightly corrupt past context tokens for robustness to autoregressive errors.
    # tau_ctx is the noise fraction: 0 = fully clean, 1 = fully noisy.
    # Signal index convention: 0 = fully noisy (sigma=0), k_max = fully clean (sigma=1),
    # so ctx_sig_idx = round((1 - tau_ctx) * k_max) to match the training convention.
    if tau_ctx > 0.0 and t > 0:
        z0_ctx = torch.randn_like(past_packed)
        past_input = ((1.0 - tau_ctx) * past_packed.float() + tau_ctx * z0_ctx.float()).to(dtype)
        ctx_sig_idx = min(int(round((1.0 - tau_ctx) * k_max)), k_max)
    else:
        past_input = past_packed
        ctx_sig_idx = k_max

    # broadcast (A,) -> (B,T,A) if needed (only if actions are present)
    if act_mask is not None and act_mask.dim() == 1:
        act_mask = act_mask.view(1, 1, -1)

    actions_in = None if actions is None else actions[:, : t + 1]
    actmask_in = None if act_mask is None else act_mask[:, : t + 1]

    # --- KV cache: prefill context tokens once ---
    kv_cache = None
    if use_kv_cache and t > 0:
        ctx_step_idxs = torch.full((B, t), emax, device=device, dtype=torch.long)
        ctx_signal_idxs = torch.full((B, t), ctx_sig_idx, device=device, dtype=torch.long)
        ctx_actions = None if actions_in is None else actions_in[:, :t]
        ctx_actmask = None if actmask_in is None else actmask_in[:, :t]

        with torch.autocast(device_type=device.type, enabled=(use_amp and device.type == "cuda"), dtype=torch.bfloat16):
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

    # Full-sequence path uses these (allocated once).
    step_idxs_full = torch.full((B, t + 1), emax, device=device, dtype=torch.long)
    step_idxs_full[:, -1] = e
    signal_idxs_full = torch.full((B, t + 1), ctx_sig_idx, device=device, dtype=torch.long)

    for i in range(K):
        tau_i = float(tau[i])
        sig_i = int(tau_idx[i])

        with torch.autocast(device_type=device.type, enabled=(use_amp and device.type == "cuda"), dtype=torch.bfloat16):
            if kv_cache is not None:
                # Decode: only the single new token (position t) attends to cached ctx.
                new_step_idxs = torch.full((B, 1), e, device=device, dtype=torch.long)
                new_signal_idxs = torch.full((B, 1), sig_i, device=device, dtype=torch.long)
                new_actions = None if actions_in is None else actions_in[:, -1:]
                new_actmask = None if actmask_in is None else actmask_in[:, -1:]

                x1_hat, _ = dyn(
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
                # Full-sequence (no-cache / t==0) path.
                signal_idxs_full[:, -1] = sig_i
                packed_seq = torch.cat([past_input, z], dim=1)  # (B,t+1,...)

                x1_hat_full, _ = dyn(
                    actions_in,
                    step_idxs_full,
                    signal_idxs_full,
                    packed_seq,
                    act_mask=actmask_in,
                    agent_tokens=None,
                    lang_emb=lang_emb,
                )
                x1_hat = x1_hat_full[:, -1:, :, :]  # (B,1,n_spatial,d_spatial)

        denom = max(1e-4, 1.0 - tau_i)
        b = (x1_hat.float() - z.float()) / denom
        z = (z.float() + b * dt).to(dtype)

    return z[:, 0]  # (B,n_spatial,d_spatial)


@torch.no_grad()
def sample_autoregressive_packed_sequence(
    dyn: Dynamics,
    *,
    z_gt_packed: torch.Tensor,                  # (B,T,n_spatial,d_spatial)
    ctx_length: int,
    horizon: int,
    k_max: int,
    sched: Dict[str, Any],
    actions: Optional[torch.Tensor] = None,     # (B,T,A) or None
    act_mask: Optional[torch.Tensor] = None,    # (B,T,A) or (A,) or None
    tau_ctx: float = 0.0,               # context corruption level
    lang_emb: Optional[torch.Tensor] = None,    # (B,lang_dim) or None
) -> torch.Tensor:
    B, T = z_gt_packed.shape[:2]
    L = min(T, ctx_length + horizon)
    ctx_length = min(ctx_length, L - 1)
    horizon = min(horizon, L - ctx_length)

    outs = [z_gt_packed[:, t] for t in range(ctx_length)]

    for t in range(ctx_length, ctx_length + horizon):
        past = torch.stack(outs, dim=1)  # (B,t,...)
        z_next = sample_one_timestep_packed(
            dyn,
            past_packed=past,
            k_max=k_max,
            sched=sched,
            actions=actions,
            act_mask=act_mask,
            tau_ctx=tau_ctx,
            lang_emb=lang_emb,
        )
        outs.append(z_next)

    return torch.stack(outs, dim=1)


@torch.no_grad()
def decode_packed_to_frames(
    decoder: Decoder,
    *,
    z_packed: torch.Tensor,     # (B,T',n_spatial,d_spatial)
    H: int, W: int, C: int, patch: int,
    packing_factor: int,
) -> torch.Tensor:
    z_btLd = unpack_spatial_to_bottleneck(z_packed, k=packing_factor)  # (B,T',L,D_b)
    patches_btnd = decoder(z_btLd)                                     # (B,T',Np,Dp) in [0,1]
    frames = temporal_unpatchify(patches_btnd, H, W, C, patch)         # (B,T',C,H,W) in [0,1]
    return frames.clamp(0, 1)


@torch.no_grad()
def log_dynamics_eval_wandb(
    *,
    gt: torch.Tensor,          # (B,T,C,H,W) float [0,1]
    pred: torch.Tensor,        # (B,T,C,H,W) float [0,1]
    ctx_length: int,
    step: int,
    tag: str,                  # wandb key prefix, e.g. "eval" or "val"
    max_items: int = 4,
    gap_px: int = 16,
):
    B, T, C, H, W = gt.shape
    Bv = min(B, max_items)

    def tile_time(x: torch.Tensor) -> torch.Tensor:
        x = x[:Bv]
        B_, T_, C_, H_, W_ = x.shape
        ctx = int(max(0, min(ctx_length, T_)))

        y = x.permute(0, 2, 3, 1, 4).contiguous().view(B_, C_, H_, T_ * W_)

        if gap_px > 0 and 0 < ctx < T_:
            split = ctx * W_
            left = y[..., :split]
            right = y[..., split:]
            gap = torch.zeros((B_, C_, H_, gap_px), device=y.device, dtype=y.dtype)
            y = torch.cat([left, gap, right], dim=-1)
        return y

    gt_t = tile_time(gt)
    pr_t = tile_time(pred)

    # Stack rows: GT / Pred
    panel = torch.cat([gt_t, pr_t], dim=2)   # (Bv,C,2H,TW+gap)
    big = torch.cat([panel[i] for i in range(Bv)], dim=1)  # (C,Bv*2H,TW+gap)

    big = (big.clamp(0, 1) * 255.0).to(torch.uint8)
    big_hwc = big.permute(1, 2, 0).cpu().numpy()

    wandb.log(
        {f"{tag}/viz": wandb.Image(big_hwc, caption=f"rows=GT/Pred | ctx={ctx_length} | T={T}")},
        step=step,
    )


@torch.no_grad()
def run_dynamics_eval(
    *,
    encoder: Encoder,
    decoder: Decoder,
    dyn: Dynamics,
    frames: torch.Tensor,            # (B,T,C,H,W) float [0,1]
    actions: Optional[torch.Tensor],    # (B,T,A) or None
    act_mask: Optional[torch.Tensor], # (A,) or None
    H: int, W: int, C: int, patch: int,
    packing_factor: int,
    k_max: int,
    ctx_length: int,
    horizon: int,
    sched: Dict[str, Any],
    max_items: int,
    step: int,
    tau_ctx: float = 0.0,
    lang_emb: Optional[torch.Tensor] = None,  # (B,lang_dim) or None
    tag: str = "eval",                         # wandb key prefix; use "val" for held-out set
):
    dyn_was_training = dyn.training
    dyn.eval()

    B, T = frames.shape[:2]
    T_eval = min(T, ctx_length + horizon)
    ctx_length = min(ctx_length, T_eval - 1)
    horizon = min(horizon, T_eval - ctx_length)

    frames_eval = frames[:, :T_eval]

    patches = temporal_patchify(frames_eval, patch)
    z_btLd, _ = encoder(patches)  # (B,T_eval,L,D_b)
    assert z_btLd.shape[2] % packing_factor == 0
    n_spatial = z_btLd.shape[2] // packing_factor
    z_gt_packed = pack_bottleneck_to_spatial(z_btLd, n_spatial=n_spatial, k=packing_factor)  # (B,T_eval,Sz,Dz)

    actions_eval = None if actions is None else actions[:, :T_eval]
    act_mask_eval = None if act_mask is None else act_mask[:, :T_eval] if act_mask.dim() == 3 else act_mask

    z_pred_packed = sample_autoregressive_packed_sequence(
        dyn,
        z_gt_packed=z_gt_packed,
        ctx_length=ctx_length,
        horizon=horizon,
        k_max=k_max,
        sched=sched,
        actions=actions_eval,
        act_mask=act_mask_eval,
        tau_ctx=tau_ctx,
        lang_emb=lang_emb,
    )

    pred_frames = decode_packed_to_frames(
        decoder,
        z_packed=z_pred_packed,
        H=H, W=W, C=C, patch=patch,
        packing_factor=packing_factor,
    )

    # floor baseline: repeat last context frame over horizon
    floor = frames_eval.clone()
    if horizon > 0:
        floor[:, ctx_length:ctx_length + horizon] = frames_eval[:, ctx_length - 1:ctx_length].expand(-1, horizon, -1, -1, -1)

    # metric on horizon only
    gt_h    = frames_eval[:, ctx_length:ctx_length + horizon]         # (B,Hz,C,H,W)
    pred_h  = pred_frames[:, ctx_length:ctx_length + horizon]
    floor_h = floor[:, ctx_length:ctx_length + horizon]

    mse_pred  = (pred_h.float()  - gt_h.float()).pow(2).mean()
    mse_floor = (floor_h.float() - gt_h.float()).pow(2).mean()

    psnr_pred  = 10.0 * torch.log10(1.0 / mse_pred.clamp_min(1e-12))
    psnr_floor = 10.0 * torch.log10(1.0 / mse_floor.clamp_min(1e-12))

    mse_ratio = mse_pred / mse_floor.clamp_min(1e-12)     # <1 is better
    psnr_gain = psnr_pred - psnr_floor                    # >0 is better

    # per-timestep horizon MSE: log first/mid/last
    # per_t: (Hz,)
    per_t_pred  = (pred_h.float()  - gt_h.float()).pow(2).mean(dim=(0,2,3,4))
    per_t_floor = (floor_h.float() - gt_h.float()).pow(2).mean(dim=(0,2,3,4))

    if horizon > 0:
        i0 = 0
        im = (horizon - 1) // 2
        i1 = horizon - 1

        wandb.log(
            {
                f"{tag}/mse_pred": float(mse_pred.item()),
                f"{tag}/mse_floor": float(mse_floor.item()),
                f"{tag}/mse_ratio_pred_over_floor": float(mse_ratio.item()),

                f"{tag}/psnr_pred": float(psnr_pred.item()),
                f"{tag}/psnr_floor": float(psnr_floor.item()),
                f"{tag}/psnr_gain_over_floor_db": float(psnr_gain.item()),

                # 1-indexed step labels in the horizon
                f"{tag}/mse_pred_t1": float(per_t_pred[i0].item()),
                f"{tag}/mse_pred_tmid": float(per_t_pred[im].item()),
                f"{tag}/mse_pred_tend": float(per_t_pred[i1].item()),

                f"{tag}/mse_floor_t1": float(per_t_floor[i0].item()),
                f"{tag}/mse_floor_tmid": float(per_t_floor[im].item()),
                f"{tag}/mse_floor_tend": float(per_t_floor[i1].item()),
            },
            step=step,
        )

    log_dynamics_eval_wandb(
        gt=frames_eval,
        pred=pred_frames,
        ctx_length=ctx_length,
        step=step,
        tag=tag,
        max_items=max_items,
    )

    if dyn_was_training:
        dyn.train()


@torch.no_grad()
def run_action_attribution_eval(
    *,
    encoder: Encoder,
    dyn: Dynamics,
    batches,                         # iterable of parsed (frames, actions, act_mask, lang_emb)
    patch: int,
    packing_factor: int,
    k_max: int,
    tau_ctx: float,
    step: int,
    tag: str,
    rng: torch.Generator,
    sched: Optional[Dict[str, Any]] = None,
):
    """Averages action-attribution ratios over many val batches.

    For each sequence, at each position t in [1, T-1], a teacher-forced
    one-step prediction is formed by running the shortcut sampler from pure
    noise with the ground-truth (packed) latents at positions 0..t-1 as
    context. The prediction is compared against z_gt[t] in latent space.

    Metric keys:
      {tag}/action_shuffle_ratio      : pool-level batch-permuted actions
      {tag}/action_timeshuffle_ratio  : per-sequence time-permuted actions
      {tag}/action_zero_ratio         : zero actions
    `pool-level` means the batch-shuffle permutation is drawn over the
    concatenation of all N val batches (not within each batch), which
    makes the intervention far more severe for multi-task mini-batches
    with only a few sequences per batch.

    The per-(seq,t) initial noise and tau_ctx corruption are drawn once
    from `rng` and shared across the four intervention calls, so the
    three ratios differ ONLY in the action tensor. That drops a lot of
    sampling variance from the ratio.
    """
    if sched is None:
        sched = make_tau_schedule(k_max=k_max, schedule="shortcut", d=1.0)

    dyn_was_training = dyn.training
    enc_was_training = encoder.training
    dyn.eval()
    encoder.eval()

    # Encode all batches; stash actions/masks/lang for the per-intervention calls.
    z_list, act_list, mask_list, lang_list = [], [], [], []
    for (frames, actions, act_mask, lang_emb) in batches:
        if actions is None:
            continue
        patches = temporal_patchify(frames, patch)
        z_btLd, _ = encoder(patches)
        assert z_btLd.shape[2] % packing_factor == 0
        n_spatial_local = z_btLd.shape[2] // packing_factor
        z_p = pack_bottleneck_to_spatial(z_btLd, n_spatial=n_spatial_local, k=packing_factor)
        z_list.append(z_p)
        act_list.append(actions)
        mask_list.append(act_mask)
        lang_list.append(lang_emb)

    if len(z_list) == 0:
        if dyn_was_training: dyn.train()
        if enc_was_training: encoder.train()
        return

    # Align B and T across batches (drop partial last batch if needed).
    B_common = min(z.shape[0] for z in z_list)
    T_common = min(z.shape[1] for z in z_list)
    z_list = [z[:B_common, :T_common] for z in z_list]
    act_list = [a[:B_common, :T_common] for a in act_list]
    mask_list = [
        (m[:B_common, :T_common] if (m is not None and m.dim() >= 2) else m)
        for m in mask_list
    ]

    device = z_list[0].device
    dtype  = z_list[0].dtype

    # Pool concatenation along batch dim: (N*B, T, S, D)
    z_pool = torch.cat(z_list, dim=0)
    a_pool = torch.cat(act_list, dim=0)
    if mask_list[0] is not None and mask_list[0].dim() >= 2:
        m_pool = torch.cat(mask_list, dim=0)
    else:
        m_pool = mask_list[0]
    if all(l is None for l in lang_list):
        l_pool = None
    else:
        ref = next(l for l in lang_list if l is not None)
        filled = [l if l is not None else torch.zeros_like(ref[:B_common]) for l in lang_list]
        l_pool = torch.cat([l[:B_common] for l in filled], dim=0)

    BP, TP = z_pool.shape[:2]
    if TP < 2:
        if dyn_was_training: dyn.train()
        if enc_was_training: encoder.train()
        return
    n_spatial = z_pool.shape[2]
    d_spatial = z_pool.shape[3]

    K      = int(sched["K"])
    tau_s  = sched["tau"]
    tau_i_s = sched["tau_idx"]
    dt     = float(sched["dt"])
    e_step = int(sched["e"])
    emax   = int(round(math.log2(k_max)))

    # Shared per-(seq,t) initial noise; same tensor reused for all interventions.
    noise_pool = torch.randn(
        (BP, TP - 1, n_spatial, d_spatial),
        generator=rng, device=rng.device, dtype=torch.float32,
    ).to(device=device, dtype=dtype)

    # tau_ctx-corrupted past context, matching the inference-time rollout.
    if tau_ctx > 0.0:
        ctx_noise = torch.randn(
            z_pool.shape, generator=rng, device=rng.device, dtype=torch.float32,
        ).to(device=device, dtype=dtype)
        z_past_input = ((1.0 - tau_ctx) * z_pool.float() + tau_ctx * ctx_noise.float()).to(dtype)
        ctx_sig_idx = min(int(round((1.0 - tau_ctx) * k_max)), k_max)
    else:
        z_past_input = z_pool
        ctx_sig_idx = k_max

    def _pred_mse(actions_in, mask_in):
        """Teacher-forced one-step MSE per (seq, t). Returns (BP, TP-1)."""
        per_t_mses = []
        for t in range(1, TP):
            past = z_past_input[:, :t]
            target_t = z_pool[:, t].float()

            z = noise_pool[:, t - 1:t].clone()  # (BP, 1, S, D)

            step_idxs_full = torch.full((BP, t + 1), emax, device=device, dtype=torch.long)
            step_idxs_full[:, -1] = e_step
            signal_idxs_full = torch.full((BP, t + 1), ctx_sig_idx, device=device, dtype=torch.long)

            act_in = actions_in[:, :t + 1] if actions_in is not None else None
            if mask_in is not None and mask_in.dim() >= 2:
                msk_in = mask_in[:, :t + 1]
            else:
                msk_in = mask_in

            for i in range(K):
                tau_i = float(tau_s[i])
                sig_i = int(tau_i_s[i])
                signal_idxs_full[:, -1] = sig_i
                packed_seq = torch.cat([past, z], dim=1)
                x1_hat_full, _ = dyn(
                    act_in, step_idxs_full, signal_idxs_full, packed_seq,
                    act_mask=msk_in, agent_tokens=None, lang_emb=l_pool,
                )
                x1_hat = x1_hat_full[:, -1:, :, :]
                denom = max(1e-4, 1.0 - tau_i)
                b = (x1_hat.float() - z.float()) / denom
                z = (z.float() + b * dt).to(dtype)

            z_pred_t = z[:, 0].float()
            diff_sq_t = (z_pred_t - target_t).pow(2).mean(dim=(1, 2))  # (BP,)
            per_t_mses.append(diff_sq_t)
        return torch.stack(per_t_mses, dim=1)  # (BP, TP-1)

    # Pool-level batch-shuffle permutation (over N*B sequences)
    perm_b = torch.randperm(BP, generator=rng, device=rng.device).to(device)
    # Per-sequence time-shuffle (independent permutation per seq, vectorized)
    perm_t_per_seq = torch.stack(
        [torch.randperm(TP, generator=rng, device=rng.device) for _ in range(BP)], dim=0
    ).to(device)
    a_tshuf = torch.gather(
        a_pool, 1,
        perm_t_per_seq.unsqueeze(-1).expand(-1, -1, a_pool.shape[-1]),
    )
    if m_pool is not None and m_pool.dim() >= 2:
        m_tshuf = torch.gather(
            m_pool, 1,
            perm_t_per_seq.unsqueeze(-1).expand(-1, -1, m_pool.shape[-1]),
        )
    else:
        m_tshuf = m_pool

    mse_real_bt  = _pred_mse(a_pool, m_pool)
    mse_bshuf_bt = _pred_mse(
        a_pool[perm_b],
        m_pool[perm_b] if (m_pool is not None and m_pool.dim() >= 2) else m_pool,
    )
    mse_tshuf_bt = _pred_mse(a_tshuf, m_tshuf)
    mse_zero_bt  = _pred_mse(torch.zeros_like(a_pool), m_pool)

    mse_real  = float(mse_real_bt.mean().item())
    mse_bshuf = float(mse_bshuf_bt.mean().item())
    mse_tshuf = float(mse_tshuf_bt.mean().item())
    mse_zero  = float(mse_zero_bt.mean().item())

    ratio_bshuf = mse_bshuf / max(mse_real, 1e-12)
    ratio_tshuf = mse_tshuf / max(mse_real, 1e-12)
    ratio_zero  = mse_zero  / max(mse_real, 1e-12)

    # Per-sequence ratio distribution (mean over t per seq). Headline ratios
    # above are pool-level (mean-of-MSEs); the per-seq stats below reveal
    # heterogeneity across sequences.
    real_seq_safe = mse_real_bt.mean(dim=1).clamp_min(1e-12)
    seq_ratios_b  = mse_bshuf_bt.mean(dim=1) / real_seq_safe
    seq_ratios_t  = mse_tshuf_bt.mean(dim=1) / real_seq_safe
    seq_ratios_z  = mse_zero_bt.mean(dim=1)  / real_seq_safe

    def _ratio_stats(ratios, key):
        out = {}
        if BP <= 1:
            out[f"{key}_std"] = 0.0
            return out
        out[f"{key}_std"] = float(ratios.std(unbiased=True).item())
        qs = torch.tensor([0.1, 0.5, 0.9], device=ratios.device, dtype=ratios.dtype)
        q = torch.quantile(ratios, qs)
        out[f"{key}_p10"] = float(q[0].item())
        out[f"{key}_p50"] = float(q[1].item())
        out[f"{key}_p90"] = float(q[2].item())
        return out

    log_payload = {
        f"{tag}/action_shuffle_ratio":      ratio_bshuf,
        f"{tag}/action_timeshuffle_ratio":  ratio_tshuf,
        f"{tag}/action_zero_ratio":         ratio_zero,
        f"{tag}/action_mse_real":           mse_real,
        f"{tag}/action_attr_pool_size":     BP,
        # One-step teacher-forced prediction MSE on val (latent space).
        f"{tag}/pred_mse":                  mse_real,
    }
    log_payload.update(_ratio_stats(seq_ratios_b, f"{tag}/action_shuffle_ratio"))
    log_payload.update(_ratio_stats(seq_ratios_t, f"{tag}/action_timeshuffle_ratio"))
    log_payload.update(_ratio_stats(seq_ratios_z, f"{tag}/action_zero_ratio"))

    wandb.log(log_payload, step=step)

    if dyn_was_training: dyn.train()
    if enc_was_training: encoder.train()


def parse_batch(batch, *, use_rewards: bool, device):
    """Mirror of the inline batch parsing in the train loop. Returns
    (frames, actions, act_mask, rewards, lang_emb), each on `device`.
    Kept as a helper so the validation block can reuse the exact same
    convention without duplicating logic."""
    obs_u8 = batch["obs"].to(device, non_blocking=True)         # (B,T+1,3,H,W) uint8
    act    = batch["act"].to(device, non_blocking=True)         # (B,T,A) float
    mask   = batch["act_mask"].to(device, non_blocking=True)    # (B,T,A) float
    rew_in = batch.get("rew", None)
    rew_in = None if rew_in is None else rew_in.to(device, non_blocking=True)
    lang_emb = batch["lang_emb"].to(device, non_blocking=True)  # (B,lang_dim)

    act = act.clamp(-1, 1) * mask

    frames = obs_u8[:, :-1].float() / 255.0
    actions = torch.zeros_like(act)
    actions[:, 1:] = act[:, :-1]
    act_mask = torch.zeros_like(mask)
    act_mask[:, 1:] = mask[:, :-1]

    rewards = rew_in if use_rewards else None

    return frames, actions, act_mask, rewards, lang_emb


def train(args):
    ddp, rank, world_size, local_rank = init_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    seed_everything(args.seed + rank)

    # Dataset and DataLoader
    from wm_dataset import WMDataset, collate_batch
    # Targeted-collection fine-tunes additionally load UNSEEN tasks; the dataset
    # filters to what actually has data on disk so runs without UNSEEN demos are
    # unaffected when none are present.
    full_task_list = list(TASK_SET) + list(UNSEEN_TASK_SET)
    tw_list = compute_task_weights(
        full_task_list, args.task_weighting,
        targeted_alpha=args.targeted_alpha,
    )
    task_weights = None if args.task_weighting == "valid_starts" else dict(zip(full_task_list, tw_list))
    dataset = WMDataset(
        data_dir=args.data_dirs,
        frames_dir=args.frame_dirs,
        seq_len=args.seq_len,
        img_size=224,
        action_dim=16,
        lang_dim=args.lang_dim,
        tasks_json=args.tasks_json,
        tasks=full_task_list,
        verbose=is_rank0(),
        cache_mb=args.cache_mb,
        ddp_partition=True,
        iid_sampling=True,
        samples_per_shard=args.samples_per_shard,
        task_weights=task_weights,
    )
    # Build per-rank lookup: local task_idx (emb_id) -> domain_idx.
    # Used to bucket per-sample losses by domain for logging.
    task_idx_to_domain_idx = torch.tensor(
        [DOMAINS.index(task_to_domain(t)) for t in dataset.tasks],
        dtype=torch.long, device=device,
    )

    # With ddp_partition + iid_sampling, tasks are already split across
    # ranks and __getitem__ ignores the index, so DistributedSampler is
    # unnecessary.  Plain shuffle keeps the DataLoader happy.
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        worker_init_fn=worker_init_fn,
        collate_fn=collate_batch,
    )

    # ---- validation data (rank 0 only) ----
    # Mirrors the train dataset construction but reads from a single held-out
    # directory. We only build it on rank 0 because the existing eval block
    # also runs on rank 0 only — other ranks idle at the next collective.
    val_loader = None
    val_iter = None
    if is_rank0() and args.val_every > 0 and args.val_frame_dir is not None:
        assert args.val_data_dir is not None, "--val_data_dir is required for validation"
        val_dataset = WMDataset(
            data_dir=args.val_data_dir,
            frames_dir=args.val_frame_dir,
            seq_len=args.seq_len,
            img_size=224,
            action_dim=16,
            lang_dim=args.lang_dim,
            tasks_json=args.tasks_json,
            tasks=TASK_SET,
            verbose=False,
            ddp_partition=False,
            iid_sampling=True,
            samples_per_shard=1,
            task_weights=task_weights,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.eval_batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True,
            worker_init_fn=worker_init_fn,
            collate_fn=collate_batch,
        )

        def _val_batch_iterator(loader):
            while True:
                for b in loader:
                    yield b
        val_iter = _val_batch_iterator(val_loader)

    # Load frozen tokenizer
    tok_override = {}
    if args.H is not None: tok_override["H"] = args.H
    if args.W is not None: tok_override["W"] = args.W
    if args.C is not None: tok_override["C"] = args.C
    if args.patch is not None: tok_override["patch"] = args.patch

    encoder, decoder, tok_args = load_frozen_tokenizer_from_pt_ckpt(
        args.tokenizer_ckpt, device=device, override=tok_override
    )

    H = int(tok_args.get("H", 224))
    W = int(tok_args.get("W", 224))
    C = int(tok_args.get("C", 3))
    patch = int(tok_args.get("patch", 4))
    n_latents = int(tok_args.get("n_latents", 16))
    d_bottleneck = int(tok_args.get("d_bottleneck", 32))

    assert H % patch == 0 and W % patch == 0
    assert n_latents % args.packing_factor == 0
    n_spatial = n_latents // args.packing_factor
    d_spatial = d_bottleneck * args.packing_factor

    # Build dynamics model
    dyn = Dynamics(
        d_model=args.d_model_dyn,
        d_bottleneck=d_bottleneck,
        d_spatial=d_spatial,
        n_spatial=n_spatial,
        n_register=args.n_register,
        n_agent=args.n_agent,
        n_heads=args.n_heads,
        depth=args.dyn_depth,
        k_max=args.k_max,
        dropout=args.dropout,
        mlp_ratio=args.mlp_ratio,
        time_every=args.time_every,
        lang_dim=args.lang_dim,
    ).to(device)

    # -----------------------
    # Reward head (optional)
    # -----------------------
    use_rewards = args.reward_weight > 0.0
    if use_rewards:
        if args.n_agent <= 0:
            raise RuntimeError("Reward modeling requires --n_agent > 0 (agent token features h_t).")
        rew_head = RewardHeadMTP(
            d_model=args.d_model_dyn,
            L=args.reward_L,
            num_bins=args.reward_num_bins,
            dropout=args.dropout,
            mlp_ratio=args.reward_mlp_ratio,
            log_low=args.reward_log_low,
            log_high=args.reward_log_high,
            pool_agent=args.reward_pool_agent,
        ).to(device)
    else:
        rew_head = None

    # -----------------------
    # BC (policy) head (optional, gradient-isolated)
    # -----------------------
    # BC gradients are isolated from the dynamics via h_t.detach() at the call
    # site.
    use_bc = args.bc_weight > 0.0
    if use_bc:
        if args.n_agent <= 0:
            raise RuntimeError("BC modeling requires --n_agent > 0 (agent token features h_t).")
        policy_head = PolicyHeadMTP(
            d_model=args.d_model_dyn,
            L=args.bc_L,
            act_dim_max=args.bc_act_dim,
            dropout=args.dropout,
            mlp_ratio=args.bc_mlp_ratio,
            pool_agent=args.bc_pool_agent,
        ).to(device)
    else:
        policy_head = None

    rms_flow = EmaRms().to(device)
    rms_rew  = EmaRms().to(device)
    rms_bc   = EmaRms().to(device)

    if is_rank0():
        print(dyn)
        dyn_param_count = sum(p.numel() for p in dyn.parameters() if p.requires_grad)
        rew_param_count = sum(p.numel() for p in rew_head.parameters() if p.requires_grad) if rew_head is not None else 0
        bc_param_count  = sum(p.numel() for p in policy_head.parameters() if p.requires_grad) if policy_head is not None else 0
        param_count = dyn_param_count + rew_param_count + bc_param_count
        if rew_head is not None:
            print(f"Learnable parameters (dynamics): {dyn_param_count:,}")
            print(f"Learnable parameters (reward head): {rew_param_count:,}")
        if policy_head is not None:
            print(f"Learnable parameters (policy/BC head, gradient-isolated): {bc_param_count:,}")
        print(f"Total learnable parameters: {param_count:,}")
        print(f"[tokenizer] H={H} W={W} C={C} patch={patch} n_lat={n_latents} d_b={d_bottleneck} packing={args.packing_factor}")

    if ddp:
        dyn = torch.nn.parallel.DistributedDataParallel(
            dyn, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False
        )
        if rew_head is not None:
            rew_head = torch.nn.parallel.DistributedDataParallel(
                rew_head, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False
            )
        if policy_head is not None:
            policy_head = torch.nn.parallel.DistributedDataParallel(
                policy_head, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False
            )

    if args.compile:
        dyn = torch.compile(dyn)
        if rew_head is not None:
            rew_head = torch.compile(rew_head)
        if policy_head is not None:
            policy_head = torch.compile(policy_head)

    # Optimizer. BC gradients stay isolated to policy_head params via h_t.detach()
    # at the call site (see dynamics_pretrain_loss); having them in the same
    # optimizer is fine and expected.
    params = list(dyn.parameters())
    if rew_head is not None:
        params += list(rew_head.parameters())
    if policy_head is not None:
        params += list(policy_head.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))
    use_amp = torch.cuda.is_available()

    # Initialize wandb
    if is_rank0():
        _dyn_for_count = dyn.module if hasattr(dyn, "module") else dyn
        dyn_param_count_total = sum(p.numel() for p in _dyn_for_count.parameters())
        dyn_param_count_trainable = sum(p.numel() for p in _dyn_for_count.parameters() if p.requires_grad)
        if rew_head is not None:
            _rew_for_count = rew_head.module if hasattr(rew_head, "module") else rew_head
            rew_param_count_trainable = sum(p.numel() for p in _rew_for_count.parameters() if p.requires_grad)
        else:
            rew_param_count_trainable = 0
        frames_per_opt_step = args.batch_size * world_size * max(1, args.grad_accum) * args.seq_len

        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            entity=args.wandb_entity,
            mode="online",
            config={
                **vars(args),
                "run/world_size": world_size,
                "run/effective_batch_seqs": args.batch_size * world_size * max(1, args.grad_accum),
                "run/frames_per_opt_step": frames_per_opt_step,
                "model/dyn_param_count_total": dyn_param_count_total,
                "model/dyn_param_count_trainable": dyn_param_count_trainable,
                "model/rew_param_count_trainable": rew_param_count_trainable,
                "model/total_param_count_trainable": dyn_param_count_trainable + rew_param_count_trainable,
                "model/n_spatial": n_spatial,
                "model/d_spatial": d_spatial,
                "model/n_latents": n_latents,
                "model/d_bottleneck": d_bottleneck,
                "tokenizer/H": H,
                "tokenizer/W": W,
                "tokenizer/C": C,
                "tokenizer/patch": patch,
            },
        )

    # Resume from checkpoint
    step = 0
    start_epoch = 0
    ckpt_dir = Path(args.ckpt_dir)
    if args.resume is not None:
        step, start_epoch = load_ckpt(Path(args.resume), dyn_model=dyn, rew_head=rew_head, policy_head=policy_head, opt=opt,
                                      rms_objects={"flow": rms_flow, "rew": rms_rew, "bc": rms_bc},
                                      strict=args.resume_strict)
        if is_rank0():
            print(f"[rank0] Resumed from {args.resume} (step={step}, epoch={start_epoch})")
    # Count LR warmup from the resume point, so fine-tuning resumes (which start
    # at a high step) still warm up. Fresh starts have resume_step=0 (warmup from
    # step 0).
    resume_step = step

    # Training loop
    dyn.train()
    if rew_head is not None:
        rew_head.train()
    t0 = time.time()
    grad_accum = max(1, int(args.grad_accum))

    # Per-domain flow-mse accumulator (flushed at each log_every cycle).
    domain_acc = PerDomainAccumulator(n_domains=len(DOMAINS), device=device)

    while step <= args.max_steps:
        for epoch in range(start_epoch, 10_000_000):

            for batch in loader:
                if step > args.max_steps:
                    break

                frames, actions, act_mask, rewards, lang_emb = parse_batch(
                    batch, use_rewards=use_rewards, device=device)

                # Frozen encoder -> packed spatial tokens z1
                with torch.no_grad():
                    patches = temporal_patchify(frames, patch)  # (B,T,Np,Dp)
                    z_btLd, _ = encoder(patches)                # (B,T,n_latents,d_b)
                    z1 = pack_bottleneck_to_spatial(z_btLd, n_spatial=n_spatial, k=args.packing_factor)  # (B,T,Sz,Dz)

                B = z1.shape[0]
                B_self = int(round(args.self_fraction * B))
                B_self = max(0, min(B - 1, B_self))
                do_step = ((step + 1) % grad_accum == 0)

                # emb_id is already computed by the dataset; materialize it once
                # up-front so it can feed the per-domain accumulator below.
                emb_id_batch = batch["emb_id"].to(device, non_blocking=True).long()  # (B,)

                sync_dyn = dyn.no_sync() if ddp and not do_step else nullcontext()
                sync_rew = rew_head.no_sync() if (rew_head is not None and ddp and not do_step) else nullcontext()
                sync_bc  = policy_head.no_sync() if (policy_head is not None and ddp and not do_step) else nullcontext()

                with sync_dyn, sync_rew, sync_bc:
                    with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                        loss, aux = dynamics_pretrain_loss(
                            dyn,
                            z1=z1,
                            actions=actions,
                            act_mask=act_mask,
                            k_max=args.k_max,
                            B_self=B_self,
                            step=step,
                            bootstrap_start=args.bootstrap_start,
                            lang_emb=lang_emb,
                            rew_head=rew_head,
                            rewards=rewards,
                            reward_weight=args.reward_weight,
                            policy_head=policy_head,
                            bc_weight=args.bc_weight,
                            rms_flow=rms_flow,
                            rms_rew=rms_rew,
                            rms_bc=rms_bc,
                        )

                    if ddp:
                        rms_flow.sync(world_size)
                        rms_rew.sync(world_size)
                        rms_bc.sync(world_size)

                    # Accumulate per-domain flow_mse for the empirical portion
                    # of the batch (first B_emp samples, aligned with flow_per_sample).
                    flow_ps = aux["flow_per_sample"]  # (B_emp,)
                    B_emp_local = int(flow_ps.shape[0])
                    domain_ids = task_idx_to_domain_idx[emb_id_batch[:B_emp_local]]
                    domain_acc.update(flow_ps, domain_ids)

                    if not torch.isfinite(loss):
                        raise RuntimeError(f"Non-finite loss at step {step}: loss={loss}")

                    loss_to_backprop = loss / grad_accum
                    loss_to_backprop.backward()

                grad_norm = 0.0
                grad_norm_dyn = 0.0
                grad_norm_rew = 0.0
                grad_norm_bc = 0.0
                if do_step:
                    _dyn = dyn.module if hasattr(dyn, "module") else dyn
                    clip_dyn = args.grad_clip_dyn if args.grad_clip_dyn > 0 else float('inf')
                    grad_norm_dyn = float(torch.nn.utils.clip_grad_norm_(
                        _dyn.parameters(), max_norm=clip_dyn,
                    ).item())
                    if rew_head is not None:
                        _rew = rew_head.module if hasattr(rew_head, "module") else rew_head
                        clip_rew = args.grad_clip_rew if args.grad_clip_rew > 0 else float('inf')
                        grad_norm_rew = float(torch.nn.utils.clip_grad_norm_(
                            _rew.parameters(), max_norm=clip_rew,
                        ).item())
                    if policy_head is not None:
                        _bc = policy_head.module if hasattr(policy_head, "module") else policy_head
                        clip_bc = args.grad_clip_bc if args.grad_clip_bc > 0 else float('inf')
                        grad_norm_bc = float(torch.nn.utils.clip_grad_norm_(
                            _bc.parameters(), max_norm=clip_bc,
                        ).item())
                    grad_norm = (grad_norm_dyn ** 2 + grad_norm_rew ** 2 + grad_norm_bc ** 2) ** 0.5

                    # LR warmup measured from the resume point (see resume_step).
                    steps_since_resume = step - resume_step
                    if args.warmup_steps > 0 and steps_since_resume < args.warmup_steps:
                        warmup_frac = (steps_since_resume + 1) / args.warmup_steps
                        for pg in opt.param_groups:
                            pg["lr"] = args.lr * warmup_frac

                    opt.step()
                    opt.zero_grad(set_to_none=True)

                # Evaluation / visualization
                if is_rank0() and args.eval_every > 0 and (step % args.eval_every == 0):

                    # Evaluate on a small slice of the current batch
                    B_eval = min(frames.shape[0], args.eval_batch_size)
                    frames_eval = frames[:B_eval]

                    actions_eval = actions[:B_eval]
                    act_mask_eval = act_mask[:B_eval]
                    lang_emb_eval = lang_emb[:B_eval]

                    sched = make_tau_schedule(k_max=args.k_max, schedule=args.eval_schedule, d=args.eval_d)

                    run_dynamics_eval(
                        encoder=encoder,
                        decoder=decoder,
                        dyn=(dyn.module if hasattr(dyn, "module") else dyn),
                        frames=frames_eval,
                        actions=actions_eval,
                        act_mask=act_mask_eval,
                        H=H, W=W, C=C, patch=patch,
                        packing_factor=args.packing_factor,
                        k_max=args.k_max,
                        ctx_length=args.eval_ctx,
                        horizon=args.eval_horizon,
                        sched=sched,
                        max_items=args.eval_max_items,
                        step=step,
                        tau_ctx=args.tau_ctx,
                        lang_emb=lang_emb_eval,
                        tag="eval",
                    )

                # Validation on a held-out dataset (rank 0 only)
                if val_iter is not None and (step % args.val_every == 0):
                    val_batch = next(val_iter)
                    val_frames, val_actions, val_act_mask, _val_rewards, val_lang_emb = parse_batch(
                        val_batch, use_rewards=use_rewards, device=device,
                    )

                    B_val = min(val_frames.shape[0], args.eval_batch_size)
                    val_frames    = val_frames[:B_val]
                    val_actions   = val_actions[:B_val]
                    val_act_mask  = val_act_mask[:B_val]
                    val_lang_emb  = val_lang_emb[:B_val]

                    sched_val = make_tau_schedule(k_max=args.k_max, schedule=args.eval_schedule, d=args.eval_d)

                    run_dynamics_eval(
                        encoder=encoder,
                        decoder=decoder,
                        dyn=(dyn.module if hasattr(dyn, "module") else dyn),
                        frames=val_frames,
                        actions=val_actions,
                        act_mask=val_act_mask,
                        H=H, W=W, C=C, patch=patch,
                        packing_factor=args.packing_factor,
                        k_max=args.k_max,
                        ctx_length=args.eval_ctx,
                        horizon=args.eval_horizon,
                        sched=sched_val,
                        max_items=args.eval_max_items,
                        step=step,
                        tau_ctx=args.tau_ctx,
                        lang_emb=val_lang_emb,
                        tag="val",
                    )

                    # Action-attribution metrics averaged over many val batches
                    # (batch-shuffle, time-shuffle, zero-action).
                    if args.action_attr_n_batches > 0:
                        attr_batches = [(val_frames, val_actions, val_act_mask, val_lang_emb)]
                        for _ in range(args.action_attr_n_batches - 1):
                            try:
                                b = next(val_iter)
                            except StopIteration:
                                break
                            f_i, a_i, m_i, _r_i, l_i = parse_batch(
                                b, use_rewards=use_rewards, device=device,
                            )
                            Bv = min(f_i.shape[0], args.eval_batch_size)
                            attr_batches.append((
                                f_i[:Bv], a_i[:Bv], m_i[:Bv], l_i[:Bv],
                            ))
                        _attr_rng = torch.Generator(device="cpu")
                        _attr_rng.manual_seed(12345)
                        run_action_attribution_eval(
                            encoder=encoder,
                            dyn=(dyn.module if hasattr(dyn, "module") else dyn),
                            batches=attr_batches,
                            patch=patch,
                            packing_factor=args.packing_factor,
                            k_max=args.k_max,
                            tau_ctx=args.tau_ctx,
                            step=step,
                            tag="val",
                            rng=_attr_rng,
                        )

                # Flush per-domain accumulators at log cadence (all ranks
                # must participate in the all_reduce; only rank 0 logs).
                domain_flow_means = None
                domain_flow_counts = None
                if step % args.log_every == 0:
                    domain_flow_means, domain_flow_counts = domain_acc.flush(ddp=ddp)

                # Logging
                if is_rank0() and (step % args.log_every == 0):

                    # Action shuffle loss ratio
                    _dyn_unwrapped = dyn.module if hasattr(dyn, "module") else dyn
                    with torch.no_grad():
                        loss_real, _ = dynamics_pretrain_loss(
                            _dyn_unwrapped,
                            z1=z1, actions=actions, act_mask=act_mask,
                            k_max=args.k_max, B_self=B_self, step=step,
                            bootstrap_start=args.bootstrap_start,
                            lang_emb=lang_emb,
                            rew_head=None, rewards=None, reward_weight=0.0,
                        )
                        perm = torch.randperm(actions.shape[0], device=actions.device)
                        loss_shuffled, _ = dynamics_pretrain_loss(
                            _dyn_unwrapped,
                            z1=z1, actions=actions[perm], act_mask=act_mask[perm] if act_mask is not None else None,
                            k_max=args.k_max, B_self=B_self, step=step,
                            bootstrap_start=args.bootstrap_start,
                            lang_emb=lang_emb,
                            rew_head=None, rewards=None, reward_weight=0.0,
                        )
                    action_shuffle_loss_ratio = loss_shuffled / loss_real.clamp_min(1e-8)

                    boot_over_flow = (
                        float(aux["bootstrap_mse"].item()) / max(float(aux["flow_mse"].item()), 1e-8)
                        if float(aux["flow_mse"].item()) > 0.0 else 0.0
                    )

                    # Weight norms
                    _dyn_log = dyn.module if hasattr(dyn, "module") else dyn
                    weight_norm_dyn = float(sum(p.float().norm().item() ** 2 for p in _dyn_log.parameters()) ** 0.5)
                    weight_norm_rew = 0.0
                    if rew_head is not None:
                        _rew_log = rew_head.module if hasattr(rew_head, "module") else rew_head
                        weight_norm_rew = float(sum(p.float().norm().item() ** 2 for p in _rew_log.parameters()) ** 0.5)
                    weight_norm_bc = 0.0
                    if policy_head is not None:
                        _bc_log = policy_head.module if hasattr(policy_head, "module") else policy_head
                        weight_norm_bc = float(sum(p.float().norm().item() ** 2 for p in _bc_log.parameters()) ** 0.5)

                    # Log to wandb
                    wandb.log(
                        {
                            "loss/total": float(loss.item()),
                            "loss/flow_total": float(aux["loss_flow"].item()),
                            "loss/flow_mse": float(aux["flow_mse"].item()),
                            "loss/bootstrap_mse": float(aux["bootstrap_mse"].item()),
                            "loss/bootstrap_over_flow": boot_over_flow,
                            "loss/loss_emp": float(aux["loss_emp"].item()),
                            "loss/loss_self": float(aux["loss_self"].item()),
                            "loss/reward_ce": float(aux["loss_rew"].item()),
                            "loss/bc_mse": float(aux["loss_bc"].item()),
                            "stats/action_shuffle_loss_ratio": float(action_shuffle_loss_ratio.item()),
                            "stats/sigma_mean": float(aux["sigma_mean"].item()),
                            "stats/sigma_std": float(aux["sigma_std"].item()),
                            "stats/B_self": float(B_self),
                            "stats/grad_norm": grad_norm,
                            "stats/grad_norm_dyn": grad_norm_dyn,
                            "stats/grad_norm_rew": grad_norm_rew,
                            "stats/grad_norm_bc": grad_norm_bc,
                            "stats/weight_norm_dyn": weight_norm_dyn,
                            "stats/weight_norm_rew": weight_norm_rew,
                            "stats/weight_norm_bc": weight_norm_bc,
                            "lr": float(opt.param_groups[0]["lr"]),
                            "time/hrs": (time.time() - t0) / 3600.0,
                            "rms/flow": aux["rms_flow"],
                            "rms/rew": aux["rms_rew"],
                            "rms/bc": aux["rms_bc"],
                        },
                        step=step,
                    )

                    # Per-domain flow_mse, aggregated over all ranks across the
                    # log_every window.
                    if domain_flow_means is not None:
                        dom_payload = {}
                        for di, dname in enumerate(DOMAINS):
                            m = float(domain_flow_means[di].item())
                            if not math.isnan(m):
                                dom_payload[f"domain/{dname}/flow_mse"] = m
                        if dom_payload:
                            wandb.log(dom_payload, step=step)

                    # Log to console
                    print(
                        f"step {step:07d} | loss={loss.item():.6f} "
                        f"| flow_mse={aux['flow_mse'].item():.6f} "
                        f"| boot_mse={aux['bootstrap_mse'].item():.6f} "
                        f"| rew_ce={aux['loss_rew'].item():.6f} "
                        f"| sigma={aux['sigma_mean'].item():.3f} | gnorm={grad_norm:.3f} | B_self={B_self}"
                    )

                # Checkpointing
                if is_rank0() and args.save_every > 0 and (step % args.save_every == 0) and step > 0 and do_step:
                    rms_state = {"flow": rms_flow.state_dict(), "rew": rms_rew.state_dict(), "bc": rms_bc.state_dict()}
                    ckpt_path = ckpt_dir / f"step_{step:07d}.pt"
                    save_ckpt(ckpt_path, step=step, epoch=epoch, dyn_model=dyn, rew_head=rew_head, policy_head=policy_head, opt=opt, args=args, rms_state=rms_state)
                    latest = ckpt_dir / "latest.pt"
                    save_ckpt(latest, step=step, epoch=epoch, dyn_model=dyn, rew_head=rew_head, policy_head=policy_head, opt=opt, args=args, rms_state=rms_state)
                    # log checkpoint as a wandb artifact (versioned, easy to retrieve from the UI)
                    artifact = wandb.Artifact(
                        name=f"{wandb.run.id}-dynamics",
                        type="model",
                        metadata={"step": step, "epoch": epoch},
                    )
                    artifact.add_file(str(ckpt_path), name=f"step_{step:07d}.pt")
                    wandb.log_artifact(artifact, aliases=["latest", f"step-{step}"])

                step += 1

            start_epoch = epoch + 1
            if step > args.max_steps:
                break  # exit the epoch loop too; otherwise the dataloader keeps spinning idle

    if ddp:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    p = argparse.ArgumentParser()

    # data (if using multiple datasets, make sure they align in order)
    p.add_argument("--data_dirs", type=str, nargs="+", default=[
        "./data/expert",
        "./data/mixed-large",
        "./data/mixed-small",
        "./data/zeros",
        "./data/collected",
    ])
    p.add_argument("--frame_dirs", type=str, nargs="+", default=[
        "./data/expert-shards",
        "./data/mixed-large-shards",
        "./data/mixed-small-shards",
        "./data/zeros-shards",
        "./data/collected-shards",
    ])
    p.add_argument("--tasks_json", type=str, default="../tasks.json")  # task metadata

    # validation data (held-out, single directory each)
    p.add_argument("--val_data_dir", type=str, default="./data/val",
                   help="optional single raw-data dir for validation")
    p.add_argument("--val_frame_dir", type=str, default="./data/val-shards",
                   help="optional single preprocessed-frames dir for validation")
    p.add_argument("--val_every", type=int, default=2_000,
                   help="run validation rollouts every N steps (0 disables)")

    p.add_argument("--seq_len", type=int, default=24)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--prefetch_factor", type=int, default=4,
                   help="batches prefetched per worker")
    p.add_argument("--cache_mb", type=int, default=13312,
                   help="per-worker LRU shard cache size in MB (WMDataset)")
    p.add_argument("--samples_per_shard", type=int, default=24,
                   help="sequences drawn from a shard before switching "
                        "(1 = pure iid, larger = less I/O)")
    p.add_argument("--task_weighting", type=str, default="valid_starts",
                   choices=["valid_starts", "uniform", "targeted"],
                   help="how to weight tasks during iid sampling. "
                        "'valid_starts' (default): P(task) ∝ #valid windows, "
                        "favors long-trajectory tasks. "
                        "'uniform': every task drawn with equal probability. "
                        "'targeted': split α uniformly over the targeted-collection "
                        "tasks (SEEN_TASK_SET ∪ UNSEEN_TASK_SET) and (1-α) uniformly "
                        "over the rest — use for targeted-collection fine-tunes.")
    p.add_argument("--targeted_alpha", type=float, default=0.5,
                   help="(--task_weighting=targeted only) total per-batch "
                        "probability mass allocated to the targeted-collection "
                        "tasks (split uniformly among them); the remaining 1-α is "
                        "spread uniformly over all other tasks.")

    # tokenizer restore
    p.add_argument("--tokenizer_ckpt", type=str, default="./logs/tokenizer_ckpts/latest.pt")
    p.add_argument("--H", type=int, default=None)
    p.add_argument("--W", type=int, default=None)
    p.add_argument("--C", type=int, default=None)
    p.add_argument("--patch", type=int, default=None)

    # dynamics arch
    p.add_argument("--d_model_dyn", type=int, default=1024)
    p.add_argument("--dyn_depth", type=int, default=16)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--mlp_ratio", type=float, default=4.0)
    p.add_argument("--time_every", type=int, default=2)

    p.add_argument("--packing_factor", type=int, default=2)
    p.add_argument("--n_register", type=int, default=4)
    p.add_argument("--n_agent", type=int, default=0)

    # shortcut / schedule
    p.add_argument("--k_max", type=int, default=64)
    p.add_argument("--bootstrap_start", type=int, default=10_000)
    p.add_argument("--self_fraction", type=float, default=0.25)

    # actions
    p.add_argument("--lang_dim", type=int, default=512)

    # rewards
    p.add_argument("--reward_weight", type=float, default=0)  # set >0 to enable
    p.add_argument("--reward_L", type=int, default=8)
    p.add_argument("--reward_num_bins", type=int, default=255)
    p.add_argument("--reward_log_low", type=float, default=-10.0)
    p.add_argument("--reward_log_high", type=float, default=10.0)
    p.add_argument("--reward_mlp_ratio", type=float, default=2.0)
    p.add_argument("--reward_pool_agent", type=str, default="attn", choices=["attn", "mean", "first"])

    # BC policy head (gradient-isolated from dynamics)
    p.add_argument("--bc_weight", type=float, default=0)  # set >0 to enable
    p.add_argument("--bc_L", type=int, default=8)
    p.add_argument("--bc_act_dim", type=int, default=16)
    p.add_argument("--bc_mlp_ratio", type=float, default=2.0)
    p.add_argument("--bc_pool_agent", type=str, default="attn", choices=["attn", "mean", "first"])

    # optim
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--max_steps", type=int, default=10_000_000)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--grad_clip_dyn", type=float, default=1.0)
    p.add_argument("--grad_clip_rew", type=float, default=1.0)
    p.add_argument("--grad_clip_bc", type=float, default=1.0)
    p.add_argument("--warmup_steps", type=int, default=1000)

    # eval / viz
    p.add_argument("--eval_every", type=int, default=1_000)
    p.add_argument("--eval_batch_size", type=int, default=4)
    p.add_argument("--eval_max_items", type=int, default=4)
    p.add_argument("--eval_ctx", type=int, default=8)
    p.add_argument("--eval_horizon", type=int, default=16)
    p.add_argument("--eval_schedule", type=str, default="shortcut", choices=["finest", "shortcut"])
    p.add_argument("--eval_d", type=float, default=0.25)
    p.add_argument("--tau_ctx", type=float, default=0.1)  # context corruption at inference
    p.add_argument("--action_attr_n_batches", type=int, default=32,
                   help="N val batches to average the action-attribution metrics over (0 disables).")

    # logging
    p.add_argument("--log_every", type=int, default=200)

    # wandb
    p.add_argument("--wandb_project", type=str, default="mmbench2-dynamics")
    p.add_argument("--wandb_run_name", type=str, default="default")
    p.add_argument("--wandb_entity", type=str, default=None)

    # ckpt
    p.add_argument("--ckpt_dir", type=str, default="./logs/dynamics_ckpts")
    p.add_argument("--save_every", type=int, default=5_000)
    p.add_argument("--resume", type=str, default="./logs/dynamics_ckpts/latest.pt")
    p.add_argument("--resume_strict", action="store_true", default=True,
                   help="Use strict=True when loading checkpoint (default). Use --no-resume_strict for finetuning with architecture changes (e.g. adding agent tokens).")
    p.add_argument("--no-resume_strict", dest="resume_strict", action="store_false")

    # misc
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--compile", action="store_true")

    args = p.parse_args()

    train(args)
