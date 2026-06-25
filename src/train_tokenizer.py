# train_tokenizer.py
"""Train the Dreamer 4 video tokenizer: a symmetric encoder/decoder Transformer
trained by masked auto-encoding of 224x224 RGB frames. Run from inside
``src/`` (flat imports), e.g. ``torchrun --nproc_per_node=8 train_tokenizer.py``.
"""
import os
import time
import random
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.amp import autocast
from torch.utils.data import DataLoader, DistributedSampler

import wandb

from task_set import TASK_SET, DOMAINS, UNSEEN_TASK_SET, task_to_domain, compute_task_weights
from sharded_frame_dataset import ShardedFrameDataset
from train_dynamics import PerDomainAccumulator
from model import (
    Encoder, Decoder, Tokenizer,
    temporal_patchify, temporal_unpatchify,
    recon_loss_from_mae, lpips_on_mae_recon,
    EmaRms,
)

try:
    import lpips
except ImportError:
    lpips = None

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


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
        dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(local_rank)
    return ddp, rank, world_size, local_rank


@torch.no_grad()
def log_tokenizer_viz_wandb(
    *,
    x_btchw: torch.Tensor,          # (B,T,C,H,W) float in [0,1]
    pred_btnd: torch.Tensor,        # (B,T,Np,Dp) float in [0,1]
    mae_mask_btNp1: torch.Tensor,   # (B,T,Np,1) bool True=masked
    patch: int,
    step: int,
    max_items: int = 8,
    max_T: int = 6,
    tag: str = "tokenizer/viz",
):
    B, T, C, H, W = x_btchw.shape
    Tv = min(T, max_T)
    Bv = min(B, max_items)

    # patchify target
    target_btnd = temporal_patchify(x_btchw[:, :Tv], patch)  # (B,Tv,Np,Dp)

    # panels (patch space)
    masked_input_btnd = torch.where(mae_mask_btNp1[:, :Tv], torch.zeros_like(target_btnd), target_btnd)
    recon_masked_btnd = torch.where(mae_mask_btNp1[:, :Tv], pred_btnd[:, :Tv], target_btnd)
    recon_full_btnd   = pred_btnd[:, :Tv]

    # to image space (B,T,C,H,W)
    target_img = temporal_unpatchify(target_btnd,       H, W, C, patch)
    masked_img = temporal_unpatchify(masked_input_btnd, H, W, C, patch)
    rmask_img  = temporal_unpatchify(recon_masked_btnd, H, W, C, patch)
    rfull_img  = temporal_unpatchify(recon_full_btnd,   H, W, C, patch)

    def tile_time(x: torch.Tensor) -> torch.Tensor:
        # (B,T,C,H,W) -> (B,C,H,T*W)
        x = x[:, :Tv]
        return x.permute(0, 2, 3, 1, 4).contiguous().view(x.shape[0], C, H, Tv * W)

    tgt = tile_time(target_img[:Bv])
    msk = tile_time(masked_img[:Bv])
    rm  = tile_time(rmask_img[:Bv])
    rf  = tile_time(rfull_img[:Bv])

    panel = torch.cat([tgt, msk, rm, rf], dim=2)  # (Bv,C,4H,Tv*W)
    big = torch.cat([panel[i] for i in range(Bv)], dim=1)  # (C,Bv*4H,Tv*W)

    big = (big.clamp(0, 1) * 255.0).to(torch.uint8)
    big_hwc = big.permute(1, 2, 0).cpu().numpy()

    wandb.log(
        {
            tag: wandb.Image(
                big_hwc,
                caption="rows=target/masked/recon_masked/recon_full",
            ),
            "tokenizer/masked_frac": float(mae_mask_btNp1[:, :Tv].float().mean().item()),
        },
        step=step,
    )


@torch.no_grad()
def log_tokenizer_val_viz_wandb(
    *,
    x_btchw: torch.Tensor,    # (B,T,C,H,W) float in [0,1]
    pred_btnd: torch.Tensor,  # (B,T,Np,Dp) float in [0,1]
    patch: int,
    step: int,
    max_items: int = 8,
    max_T: int = 6,
    tag: str = "val/viz",
):
    B, T, C, H, W = x_btchw.shape
    Tv = min(T, max_T)
    Bv = min(B, max_items)

    target_btnd = temporal_patchify(x_btchw[:, :Tv], patch)
    target_img  = temporal_unpatchify(target_btnd,         H, W, C, patch)
    recon_img   = temporal_unpatchify(pred_btnd[:, :Tv],   H, W, C, patch)

    def tile_time(x: torch.Tensor) -> torch.Tensor:
        x = x[:, :Tv]
        return x.permute(0, 2, 3, 1, 4).contiguous().view(x.shape[0], C, H, Tv * W)

    tgt = tile_time(target_img[:Bv])
    rec = tile_time(recon_img[:Bv])

    panel = torch.cat([tgt, rec], dim=2)                    # (Bv,C,2H,Tv*W)
    big = torch.cat([panel[i] for i in range(Bv)], dim=1)   # (C,Bv*2H,Tv*W)

    big = (big.clamp(0, 1) * 255.0).to(torch.uint8)
    big_hwc = big.permute(1, 2, 0).cpu().numpy()

    wandb.log(
        {tag: wandb.Image(big_hwc, caption="rows=target/recon (full reconstruction)")},
        step=step,
    )


@torch.no_grad()
def validate(
    *,
    model,
    val_loader,
    device,
    use_amp: bool,
    lpips_fn,
    args: argparse.Namespace,
    max_batches: int,
    task_idx_to_domain_idx: torch.Tensor = None,
    local_task_names: list = None,
    key_prefix: str = "val",
) -> dict:
    """Full-reconstruction validation (no MAE masking). Returns aggregated metrics.

    If `local_task_names` is provided (len = number of dataset-local tasks), also
    computes per-task PSNR. `key_prefix` namespaces the output keys (e.g. "val").
    """
    was_training = model.training
    model.eval()

    _model = model.module if hasattr(model, "module") else model
    mae = _model.encoder.mae
    saved_pmin, saved_pmax = mae.p_min, mae.p_max
    mae.p_min, mae.p_max = 0.0, 0.0  # disable masking for full reconstruction

    sum_mse = 0.0
    sum_lp = 0.0
    n_batches = 0
    first_x = None
    first_pred = None

    # Per-domain MSE sums (rank-local; val loader is not DDP-partitioned).
    n_domains = len(DOMAINS)
    dom_mse_sum = torch.zeros(n_domains, device=device, dtype=torch.float64)
    dom_count = torch.zeros(n_domains, device=device, dtype=torch.float64)

    # Optional per-task accumulators. Local task indices come straight from the
    # dataset (0..n_local_tasks-1) so we just bucket per task_idx_batch directly.
    track_per_task = local_task_names is not None
    n_local_tasks = len(local_task_names) if track_per_task else 0
    task_mse_sum = torch.zeros(n_local_tasks, device=device, dtype=torch.float64) if track_per_task else None
    task_count = torch.zeros(n_local_tasks, device=device, dtype=torch.float64) if track_per_task else None

    try:
        for batch in val_loader:
            if n_batches >= max_batches:
                break
            if isinstance(batch, dict):
                x = batch["frames"].to(device, non_blocking=True)
                task_idx_batch = batch["task_idx"].to(device, non_blocking=True).long()
            else:
                x = batch.to(device, non_blocking=True)
                task_idx_batch = None
            patches = temporal_patchify(x, args.patch)

            with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                pred, _, _ = model(patches)

            if first_x is None:
                first_x = x
                first_pred = pred.float()

            mse = (pred.float() - patches.float()).pow(2).mean()
            sum_mse += float(mse.item())

            if task_idx_batch is not None and task_idx_to_domain_idx is not None:
                mse_per = (pred.float() - patches.float()).pow(2).mean(dim=(1, 2, 3))  # (B,)
                domain_ids = task_idx_to_domain_idx[task_idx_batch]
                dom_mse_sum.index_add_(0, domain_ids, mse_per.to(torch.float64))
                dom_count.index_add_(0, domain_ids, torch.ones_like(mse_per, dtype=torch.float64))
                if track_per_task:
                    task_mse_sum.index_add_(0, task_idx_batch, mse_per.to(torch.float64))
                    task_count.index_add_(0, task_idx_batch, torch.ones_like(mse_per, dtype=torch.float64))

            if lpips_fn is not None and args.lpips_weight > 0.0:
                # Full reconstruction LPIPS: pass an all-True mask so `recon_masked == pred`.
                full_mask = torch.ones_like(patches[..., :1], dtype=torch.bool)
                lp = lpips_on_mae_recon(
                    lpips_fn, pred, patches, full_mask,
                    H=args.H, W=args.W, C=args.C, patch=args.patch,
                    subsample_frac=args.lpips_frac,
                )
                sum_lp += float(lp.item())

            n_batches += 1
    finally:
        mae.p_min, mae.p_max = saved_pmin, saved_pmax
        if was_training:
            model.train()

    if n_batches == 0:
        return {}, None, None

    avg_mse = sum_mse / n_batches
    avg_lp = sum_lp / n_batches
    psnr = 10.0 * np.log10(1.0 / max(avg_mse, 1e-10))
    out = {
        f"{key_prefix}/mse": avg_mse,
        f"{key_prefix}/psnr": float(psnr),
        f"{key_prefix}/n_batches": n_batches,
    }
    if lpips_fn is not None and args.lpips_weight > 0.0:
        out[f"{key_prefix}/lpips"] = avg_lp
    # Per-domain PSNR on full reconstruction (no MAE masking).
    for i, name in enumerate(DOMAINS):
        cnt = float(dom_count[i].item())
        if cnt <= 0:
            continue
        mse_i = float((dom_mse_sum[i] / dom_count[i]).item())
        if not np.isfinite(mse_i):
            continue
        out[f"{key_prefix}/domain/{name}/mse"] = mse_i
        out[f"{key_prefix}/domain/{name}/psnr"] = float(10.0 * np.log10(1.0 / max(mse_i, 1e-10)))
    # Per-task PSNR (only when local_task_names is provided — typically UNSEEN).
    if track_per_task:
        for i, name in enumerate(local_task_names):
            cnt = float(task_count[i].item())
            if cnt <= 0:
                continue
            mse_i = float((task_mse_sum[i] / task_count[i]).item())
            if not np.isfinite(mse_i):
                continue
            out[f"{key_prefix}/per_task/{name}/mse"] = mse_i
            out[f"{key_prefix}/per_task/{name}/psnr"] = float(10.0 * np.log10(1.0 / max(mse_i, 1e-10)))
    return out, first_x, first_pred


def save_ckpt(path: Path, *, step: int, epoch: int, model, opt, args: argparse.Namespace, rms_state: dict = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Unwrap DDP and torch.compile so the saved state_dict has plain keys
    # (no "module." or "_orig_mod." prefixes), making checkpoints portable
    # across compiled/uncompiled and DDP/non-DDP runs.
    target = model
    if hasattr(target, "module"):       # DDP wrapper
        target = target.module
    if hasattr(target, "_orig_mod"):    # torch.compile wrapper
        target = target._orig_mod
    obj = {
        "step": step,
        "epoch": epoch,
        "model": target.state_dict(),
        "opt": opt.state_dict(),
        "args": vars(args),
    }
    if rms_state is not None:
        obj["rms_state"] = rms_state
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def load_ckpt(path: Path, *, model, opt, rms_objects: dict = None) -> tuple[int, int]:
    ckpt = torch.load(path, map_location="cpu")
    state = ckpt["model"]
    # Unwrap DDP and torch.compile so we load into the raw nn.Module. The
    # checkpoint stores prefix-free keys (see save_ckpt), so loading into the
    # OptimizedModule wrapper would fail because it expects "_orig_mod." prefixes.
    target = model
    if hasattr(target, "module"):       # DDP wrapper
        target = target.module
    if hasattr(target, "_orig_mod"):    # torch.compile wrapper
        target = target._orig_mod
    target.load_state_dict(state, strict=True)
    opt.load_state_dict(ckpt["opt"])
    if rms_objects is not None:
        for k, v in rms_objects.items():
            if k in ckpt.get("rms_state", {}):
                v.load_state_dict(ckpt["rms_state"][k])
    return int(ckpt.get("step", 0)), int(ckpt.get("epoch", 0))


def train(args):
    assert torch.cuda.is_available()
    ddp, rank, world_size, local_rank = init_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    seed_everything(args.seed + rank)

    # ---- data ----
    # Also include UNSEEN tasks; ShardedFrameDataset filters to tasks with data
    # on disk, so runs without UNSEEN data are unaffected.
    full_task_list = list(TASK_SET) + list(UNSEEN_TASK_SET)
    tw_list = compute_task_weights(
        full_task_list, args.task_weighting,
        targeted_alpha=args.targeted_alpha,
    )
    task_weights = None if args.task_weighting == "valid_starts" else dict(zip(full_task_list, tw_list))
    dataset = ShardedFrameDataset(
        outdirs=args.data_dirs,
        tasks=full_task_list,
        seq_len=args.seq_len,
        iid_sampling=True,
        cache_size=args.shard_cache_size,
        samples_per_shard=args.samples_per_shard,
        ddp_partition=True,
        task_weights=task_weights,
        return_task_idx=True,
    )

    # Lookup for per-domain metrics: task_idx (into dataset.tasks) -> domain_idx.
    task_idx_to_domain_idx = torch.tensor(
        [DOMAINS.index(task_to_domain(t)) for t in dataset.tasks],
        dtype=torch.long, device=device,
    )

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True) if ddp else None

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        worker_init_fn=worker_init_fn,
    )

    # ---- validation data ----
    val_loader = None
    if args.val_data_dir is not None and args.val_every > 0:
        val_dataset = ShardedFrameDataset(
            outdirs=[args.val_data_dir],
            tasks=TASK_SET,
            seq_len=args.seq_len,
            iid_sampling=True,
            cache_size=args.shard_cache_size,
            samples_per_shard=args.val_samples_per_shard,
            # Don't partition val across ranks: val is small, and letting every
            # rank sample from the full pool maximizes the diversity of
            # sequences seen (especially the rank-0 first batch used for viz).
            ddp_partition=False,
            task_weights=task_weights,
            return_task_idx=True,
        )
        val_workers = max(1, args.num_workers // 2)
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,  # iid_sampling handles randomness
            num_workers=val_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=(val_workers > 0),
            prefetch_factor=args.prefetch_factor if val_workers > 0 else None,
            worker_init_fn=worker_init_fn,
        )

    # ---- UNSEEN validation data (optional) ----
    val_unseen_loader = None
    val_unseen_task_idx_to_domain_idx = None
    val_unseen_local_tasks = None
    if args.val_unseen_data_dir is not None and args.val_every > 0:
        val_unseen_dataset = ShardedFrameDataset(
            outdirs=[args.val_unseen_data_dir],
            tasks=list(UNSEEN_TASK_SET),
            seq_len=args.seq_len,
            iid_sampling=True,
            cache_size=args.shard_cache_size,
            samples_per_shard=args.val_samples_per_shard,
            ddp_partition=False,
            task_weights=None,  # uniform across the (≤10) UNSEEN tasks present on disk
            return_task_idx=True,
        )
        val_unseen_local_tasks = list(val_unseen_dataset.tasks)
        val_unseen_task_idx_to_domain_idx = torch.tensor(
            [DOMAINS.index(task_to_domain(t)) for t in val_unseen_local_tasks],
            dtype=torch.long, device=device,
        )
        val_unseen_workers = max(1, args.num_workers // 2)
        val_unseen_loader = DataLoader(
            val_unseen_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=val_unseen_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=(val_unseen_workers > 0),
            prefetch_factor=args.prefetch_factor if val_unseen_workers > 0 else None,
            worker_init_fn=worker_init_fn,
        )
        if is_rank0():
            print(f"[rank0] UNSEEN val loader: {len(val_unseen_local_tasks)} tasks from {args.val_unseen_data_dir}")

    # ---- model ----
    assert args.H % args.patch == 0 and args.W % args.patch == 0
    n_patches = (args.H // args.patch) * (args.W // args.patch)
    d_patch = args.patch * args.patch * args.C

    assert args.d_model % args.n_heads == 0, "d_model must be divisible by n_heads"

    enc = Encoder(
        patch_dim=d_patch,
        d_model=args.d_model,
        n_latents=args.n_latents,
        n_patches=n_patches,
        n_heads=args.n_heads,
        depth=args.depth,
        d_bottleneck=args.d_bottleneck,
        dropout=args.dropout,
        mlp_ratio=args.mlp_ratio,
        time_every=args.time_every,
        mae_p_min=args.mae_p_min,
        mae_p_max=args.mae_p_max,
    )
    dec = Decoder(
        d_bottleneck=args.d_bottleneck,
        d_model=args.d_model,
        n_heads=args.n_heads,
        depth=args.depth,
        n_latents=args.n_latents,
        n_patches=n_patches,
        d_patch=d_patch,
        dropout=args.dropout,
        mlp_ratio=args.mlp_ratio,
        time_every=args.time_every,
    )
    model = Tokenizer(enc, dec).to(device)

    rms_mse = EmaRms().to(device)
    rms_lp  = EmaRms().to(device)

    if is_rank0():
        print(model)
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Learnable parameters: {param_count:,}")

    if args.compile:
        model = torch.compile(model)

    if ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False
        )

    # ---- optim ----
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    use_amp = torch.cuda.is_available()

    # ---- lpips ----
    if args.lpips_weight > 0.0:
        assert lpips is not None, "pip install lpips"
        lpips_fn = lpips.LPIPS(net=args.lpips_net).to(device)
        lpips_fn.eval()
        lpips_fn.requires_grad_(False)
    else:
        lpips_fn = None

    # ---- wandb ----
    if is_rank0():
        _model_for_count = model.module if hasattr(model, "module") else model
        param_count_total = sum(p.numel() for p in _model_for_count.parameters())
        param_count_trainable = sum(p.numel() for p in _model_for_count.parameters() if p.requires_grad)
        param_count_enc = sum(p.numel() for p in _model_for_count.encoder.parameters() if p.requires_grad)
        param_count_dec = sum(p.numel() for p in _model_for_count.decoder.parameters() if p.requires_grad)
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
                "model/param_count_total": param_count_total,
                "model/param_count_trainable": param_count_trainable,
                "model/param_count_encoder": param_count_enc,
                "model/param_count_decoder": param_count_dec,
                "model/n_patches": n_patches,
                "model/d_patch": d_patch,
            },
        )

    # ---- resume ----
    step = 0
    start_epoch = 0
    ckpt_dir = Path(args.ckpt_dir)
    if args.resume is not None:
        step, start_epoch = load_ckpt(Path(args.resume), model=model, opt=opt,
                                      rms_objects={"mse": rms_mse, "lp": rms_lp})
        if is_rank0():
            print(f"[rank0] Resumed from {args.resume} (step={step}, epoch={start_epoch})")

    # ---- train ----
    model.train()
    t0 = time.time()
    grad_accum = max(1, int(args.grad_accum))

    # Per-domain MSE accumulator (flushed at each log_every boundary).
    domain_acc = PerDomainAccumulator(n_domains=len(DOMAINS), device=device)

    while step <= args.max_steps:
        for epoch in range(start_epoch, 10_000_000):
            if sampler is not None:
                sampler.set_epoch(epoch)

            for batch in loader:
                if step > args.max_steps:
                    break

                x = batch["frames"].to(device, non_blocking=True)  # (B,T,C,H,W)
                task_idx_batch = batch["task_idx"].to(device, non_blocking=True).long()
                patches = temporal_patchify(x, args.patch)

                if is_rank0() and step % args.log_every == 0:
                    with torch.no_grad():
                        _model = model.module if hasattr(model, "module") else model
                        z, _ = _model.encoder(patches)
                        zf = z.float()
                        wandb.log({
                            "debug/z_std": float(zf.std().item()),
                            "debug/z_mean": float(zf.mean().item()),
                            "debug/z_abs_max": float(zf.abs().max().item()),
                        }, step=step)

                with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                    pred, mae_mask, keep_prob = model(patches)

                # losses in fp32 (outside autocast).
                loss_mask = mae_mask
                mse = recon_loss_from_mae(pred, patches, loss_mask)
                rms_mse.update(mse)

                # Per-sample masked MSE for per-domain PSNR tracking.
                # Mirrors recon_loss_from_mae but sums per-sample instead of globally.
                with torch.no_grad():
                    diff_sq = (pred.float() - patches.float()).pow(2)   # (B,T,Np,Dp)
                    m = loss_mask.to(dtype=torch.float32)                # (B,T,Np,1)
                    sq_per = (diff_sq * m).sum(dim=(1, 2, 3))            # (B,)
                    denom_per = m.sum(dim=(1, 2, 3)) * diff_sq.shape[-1] # (B,)
                    mse_per_sample = sq_per / denom_per.clamp_min(1.0)   # (B,)
                    domain_ids = task_idx_to_domain_idx[task_idx_batch]
                    domain_acc.update(mse_per_sample, domain_ids)

                if lpips_fn is not None and args.lpips_weight > 0.0:
                    lp = lpips_on_mae_recon(
                        lpips_fn, pred, patches, loss_mask,
                        H=args.H, W=args.W, C=args.C, patch=args.patch,
                        subsample_frac=args.lpips_frac
                    )
                    rms_lp.update(lp)
                    loss = rms_mse.normalize(mse) + args.lpips_weight * rms_lp.normalize(lp)
                else:
                    lp = torch.zeros((), device=device)
                    loss = rms_mse.normalize(mse)

                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite loss at step {step}: loss={loss} mse={mse} lp={lp}")

                loss_to_backprop = loss / grad_accum

                loss_to_backprop.backward()

                do_step = ((step + 1) % grad_accum == 0)
                grad_norm = 0.0
                grad_norm_enc = 0.0
                grad_norm_dec = 0.0
                if do_step:
                    _model = model.module if hasattr(model, "module") else model
                    clip_enc = args.grad_clip_enc if args.grad_clip_enc > 0 else float('inf')
                    clip_dec = args.grad_clip_dec if args.grad_clip_dec > 0 else float('inf')
                    grad_norm_enc = float(torch.nn.utils.clip_grad_norm_(
                        _model.encoder.parameters(), max_norm=clip_enc,
                    ).item())
                    grad_norm_dec = float(torch.nn.utils.clip_grad_norm_(
                        _model.decoder.parameters(), max_norm=clip_dec,
                    ).item())
                    grad_norm = (grad_norm_enc ** 2 + grad_norm_dec ** 2) ** 0.5

                    # warmup
                    if args.warmup_steps > 0 and step < args.warmup_steps:
                        warmup_frac = (step + 1) / args.warmup_steps
                        for pg in opt.param_groups:
                            pg["lr"] = args.lr * warmup_frac

                    opt.step()
                    opt.zero_grad(set_to_none=True)

                # ---- logging ----
                # Flush the per-domain MSE accumulator on every rank at the log
                # boundary (all_reduce is collective). Only rank 0 logs to wandb.
                domain_mse_means = None
                if step % args.log_every == 0:
                    domain_mse_means, domain_mse_counts = domain_acc.flush(ddp=ddp)

                if is_rank0() and (step % args.log_every == 0):
                    psnr = 10.0 * torch.log10(1.0 / mse.clamp_min(1e-10))
                    _model = model.module if hasattr(model, "module") else model
                    weight_norm_enc = float(sum(p.float().norm().item() ** 2 for p in _model.encoder.parameters()) ** 0.5)
                    weight_norm_dec = float(sum(p.float().norm().item() ** 2 for p in _model.decoder.parameters()) ** 0.5)
                    loss_mse_normed = float(rms_mse.normalize(mse).item())
                    loss_lp_normed = float((args.lpips_weight * rms_lp.normalize(lp)).item()) if lpips_fn is not None else 0.0
                    log_dict = {
                        "loss/total": float(loss.item()),
                        "loss/mse": float(mse.item()),
                        "loss/lpips": float(lp.item()),
                        "loss/mse_normed": loss_mse_normed,
                        "loss/lpips_normed": loss_lp_normed,
                        "stats/psnr": float(psnr.item()),
                        "stats/keep_prob": float(keep_prob.mean().item()),
                        "stats/masked_frac": float(mae_mask.float().mean().item()),
                        "lr": float(opt.param_groups[0]["lr"]),
                        "stats/grad_norm": grad_norm,
                        "stats/grad_norm_enc": grad_norm_enc,
                        "stats/grad_norm_dec": grad_norm_dec,
                        "stats/weight_norm_enc": weight_norm_enc,
                        "stats/weight_norm_dec": weight_norm_dec,
                        "time/hrs": (time.time() - t0) / 3600.0,
                        "rms/mse": rms_mse.rms_val,
                        "rms/lp": rms_lp.rms_val,
                    }
                    if domain_mse_means is not None:
                        for i, name in enumerate(DOMAINS):
                            cnt = float(domain_mse_counts[i].item())
                            if cnt <= 0:
                                continue
                            mse_i = float(domain_mse_means[i].item())
                            if not np.isfinite(mse_i):
                                continue
                            psnr_i = 10.0 * np.log10(1.0 / max(mse_i, 1e-10))
                            log_dict[f"domain/{name}/mse"] = mse_i
                            log_dict[f"domain/{name}/psnr"] = float(psnr_i)
                    wandb.log(log_dict, step=step)

                if is_rank0() and (step % args.print_every == 0):
                    psnr = 10.0 * torch.log10(1.0 / mse.clamp_min(1e-10))
                    print(
                        f"step {step:07d} | loss={loss.item():.6f} "
                        f"| mse={mse.item():.6f} | lpips={lp.item():.4f} "
                        f"| psnr={psnr.item():.2f} | keep={keep_prob.mean().item():.3f} | gnorm={grad_norm:.3f}"
                    )

                # ---- viz ----
                if is_rank0() and args.viz_every > 0 and (step % args.viz_every == 0):
                    log_tokenizer_viz_wandb(
                        x_btchw=x,
                        pred_btnd=pred,
                        mae_mask_btNp1=mae_mask,
                        patch=args.patch,
                        step=step,
                        max_items=args.viz_max_items,
                        max_T=args.viz_max_T,
                    )

                # ---- validation ----
                if val_loader is not None and args.val_every > 0 and step > 0 and (step % args.val_every == 0):
                    val_metrics, val_x, val_pred = validate(
                        model=model,
                        val_loader=val_loader,
                        device=device,
                        use_amp=use_amp,
                        lpips_fn=lpips_fn,
                        args=args,
                        max_batches=args.val_batches,
                        task_idx_to_domain_idx=task_idx_to_domain_idx,
                        key_prefix="val",
                    )
                    if is_rank0() and val_metrics:
                        wandb.log(val_metrics, step=step)
                        print(
                            f"step {step:07d} | VAL mse={val_metrics['val/mse']:.6f} "
                            f"| psnr={val_metrics['val/psnr']:.2f}"
                            + (f" | lpips={val_metrics['val/lpips']:.4f}" if 'val/lpips' in val_metrics else "")
                        )
                        if val_x is not None and args.viz_every > 0:
                            log_tokenizer_val_viz_wandb(
                                x_btchw=val_x,
                                pred_btnd=val_pred,
                                patch=args.patch,
                                step=step,
                                max_items=args.viz_max_items,
                                max_T=args.viz_max_T,
                                tag="val/viz",
                            )

                    # ---- UNSEEN validation (optional) ----
                    if val_unseen_loader is not None:
                        val_un_metrics, val_un_x, val_un_pred = validate(
                            model=model,
                            val_loader=val_unseen_loader,
                            device=device,
                            use_amp=use_amp,
                            lpips_fn=lpips_fn,
                            args=args,
                            max_batches=args.val_batches,
                            task_idx_to_domain_idx=val_unseen_task_idx_to_domain_idx,
                            local_task_names=val_unseen_local_tasks,
                            key_prefix="val_unseen",
                        )
                        if is_rank0() and val_un_metrics:
                            wandb.log(val_un_metrics, step=step)
                            print(
                                f"step {step:07d} | VAL_UNSEEN mse={val_un_metrics['val_unseen/mse']:.6f} "
                                f"| psnr={val_un_metrics['val_unseen/psnr']:.2f}"
                                + (f" | lpips={val_un_metrics['val_unseen/lpips']:.4f}" if 'val_unseen/lpips' in val_un_metrics else "")
                            )
                            if val_un_x is not None and args.viz_every > 0:
                                log_tokenizer_val_viz_wandb(
                                    x_btchw=val_un_x,
                                    pred_btnd=val_un_pred,
                                    patch=args.patch,
                                    step=step,
                                    max_items=args.viz_max_items,
                                    max_T=args.viz_max_T,
                                    tag="val_unseen/viz",
                                )

                # ---- ckpt ----
                if is_rank0() and args.save_every > 0 and (step % args.save_every == 0) and step > 0 and do_step:
                    rms_state = {"mse": rms_mse.state_dict(), "lp": rms_lp.state_dict()}
                    ckpt_path = ckpt_dir / f"step_{step:07d}.pt"
                    save_ckpt(ckpt_path, step=step, epoch=epoch, model=model, opt=opt, args=args, rms_state=rms_state)
                    # also update a "latest" pointer
                    latest = ckpt_dir / "latest.pt"
                    save_ckpt(latest, step=step, epoch=epoch, model=model, opt=opt, args=args, rms_state=rms_state)
                    # log checkpoint as a wandb artifact (versioned, easy to retrieve from the UI)
                    artifact = wandb.Artifact(
                        name=f"{wandb.run.id}-tokenizer",
                        type="model",
                        metadata={"step": step, "epoch": epoch},
                    )
                    artifact.add_file(str(ckpt_path), name=f"step_{step:07d}.pt")
                    wandb.log_artifact(artifact, aliases=["latest", f"step-{step}"])

                step += 1

            start_epoch = epoch + 1
            if step > args.max_steps:
                break  # exit the epoch loop too; otherwise the outer `while` is never re-checked and the dataloader churns idle.

    if ddp:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    p = argparse.ArgumentParser()

    # data
    p.add_argument("--data_dirs", type=str, nargs="+", default=[
        "./data/expert-shards",
        "./data/mixed-large-shards",
        "./data/mixed-small-shards",
        "./data/zeros-shards",
    ])
    p.add_argument("--val_data_dir", type=str, default="./data/val-shards",
                   help="optional single directory of preprocessed shards used for validation")
    p.add_argument("--val_unseen_data_dir", type=str, default=None,
                   help="optional directory of preprocessed shards for UNSEEN-task validation. "
                        "Filtered to UNSEEN_TASK_SET. Logs a parallel `val_unseen/...` namespace "
                        "with per-task PSNR and a side-by-side viz panel under `val_unseen/viz`.")
    p.add_argument("--val_every", type=int, default=2_000,
                   help="run validation every N steps (0 disables)")

    p.add_argument("--val_batches", type=int, default=8,
                   help="number of batches per validation pass (per rank)")
    p.add_argument("--seq_len", type=int, default=24)
    p.add_argument("--num_workers", type=int, default=6)
    p.add_argument("--shard_cache_size", type=int, default=12,
                   help="number of shards held in each worker's LRU cache")
    p.add_argument("--prefetch_factor", type=int, default=4,
                   help="batches prefetched per worker")
    p.add_argument("--samples_per_shard", type=int, default=16,
                   help="number of sequences drawn from a loaded shard before "
                        "picking a new one (1 = pure iid, larger = much less I/O)")
    p.add_argument("--val_samples_per_shard", type=int, default=1,
                   help="samples_per_shard for the val loader; defaults to 1 "
                        "for maximum task diversity in val/viz (val is small "
                        "and runs infrequently, so I/O cost is bounded)")
    p.add_argument("--task_weighting", type=str, default="valid_starts",
                   choices=["valid_starts", "uniform", "targeted"],
                   help="Per-task sampling weighting. 'valid_starts' (default) "
                        "makes P(task) ∝ total valid_starts — skews toward "
                        "long-trajectory domains. 'uniform' "
                        "gives every task equal draw probability. 'targeted' "
                        "splits α uniformly over the targeted-collection tasks "
                        "(SEEN_TASK_SET ∪ UNSEEN_TASK_SET) and (1-α) uniformly "
                        "over the rest.")
    p.add_argument("--targeted_alpha", type=float, default=0.5,
                   help="(--task_weighting=targeted only) total per-batch "
                        "probability allocated to the targeted-collection tasks.")
    p.add_argument("--batch_size", type=int, default=12)

    # image / patching
    p.add_argument("--H", type=int, default=224)
    p.add_argument("--W", type=int, default=224)
    p.add_argument("--C", type=int, default=3)
    p.add_argument("--patch", type=int, default=14)

    # model
    p.add_argument("--d_model", type=int, default=512)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--depth", type=int, default=12)
    p.add_argument("--n_latents", type=int, default=64)
    p.add_argument("--d_bottleneck", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--mlp_ratio", type=float, default=4.0)
    p.add_argument("--time_every", type=int, default=1)
    p.add_argument("--mae_p_min", type=float, default=0.0)
    p.add_argument("--mae_p_max", type=float, default=0.9)

    # optim
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--max_steps", type=int, default=10_000_000)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--grad_clip_enc", type=float, default=1.0)
    p.add_argument("--grad_clip_dec", type=float, default=1.0)
    p.add_argument("--warmup_steps", type=int, default=0)

    # lpips
    p.add_argument("--lpips_weight", type=float, default=0.2)
    p.add_argument("--lpips_frac", type=float, default=0.5)
    p.add_argument("--lpips_net", type=str, default="alex", choices=["alex", "vgg", "squeeze"])

    # logging / viz
    p.add_argument("--log_every", type=int, default=200)
    p.add_argument("--print_every", type=int, default=200)
    p.add_argument("--viz_every", type=int, default=1_000)
    p.add_argument("--viz_max_items", type=int, default=4)
    p.add_argument("--viz_max_T", type=int, default=8)

    # wandb
    p.add_argument("--wandb_project", type=str, default="mmbench2-tokenizer")
    p.add_argument("--wandb_run_name", type=str, default="default")
    p.add_argument("--wandb_entity", type=str, default=None)

    # ckpt
    p.add_argument("--ckpt_dir", type=str, default="./logs/tokenizer_ckpts")
    p.add_argument("--save_every", type=int, default=10_000)
    p.add_argument("--resume", type=str, default=None)

    # misc
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--compile", action="store_true")

    train(p.parse_args())
