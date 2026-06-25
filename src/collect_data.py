# collect_data.py
"""
Data collection script for Dreamer 4 world model finetuning.

Two policy families:

1. **Cheap policies** (no WM, no GPU needed):
   - `zero`:   apply all-zero actions every step (passive observation baseline).
   - `random`: uniform [-1, 1] over the env's true action dims.

2. **WM-driven policies** (load tokenizer + dynamics + scorer):
   - `curiosity_u_r_norm`:   motion-normalized round-trip residual as MPC reward.

Output format:
  Writes PNG strips + demo .pt in the HuggingFace dataset layout. Run
  preprocess_dataset.py to convert to shards (matching the existing pipeline).

Usage examples:
  # Cheap policy on the SEEN task split (10 tasks, 5 episodes each):
  python collect_data.py --policy zero --task_set seen --n_episodes 5 \\
    --out_data_dir ./data/collected/zero

  # WM-driven curiosity collection on a single task:
  python collect_data.py --policy curiosity_u_r_norm --tasks walker-walk \\
    --tokenizer_ckpt ./logs/tokenizer_ckpts/latest.pt \\
    --dynamics_ckpt ./logs/dynamics_ckpts/latest.pt \\
    --n_episodes 50 --out_data_dir ./data/collected/round1
"""
import os
os.environ['MUJOCO_GL'] = os.getenv("MUJOCO_GL", 'egl')
os.environ['LAZY_LEGACY_OP'] = '0'
os.environ["TORCH_DISTRIBUTED_TIMEOUT"] = "1800"
os.environ['TORCHDYNAMO_INLINE_INBUILT_NN_MODULES'] = "1"
os.environ['TORCH_LOGS'] = "+recompiles"
import time
import argparse
import json
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import torch
from torch.amp import autocast

from model import (
    Encoder, Decoder, Dynamics,
    temporal_patchify, pack_bottleneck_to_spatial,
)
from train_dynamics import (
    load_frozen_tokenizer_from_pt_ckpt,
    make_tau_schedule,
    decode_packed_to_frames,
)
from interactive import load_dynamics_from_ckpt
from uncertainty import (
    URNormScorer,
    sample_predictions_for_actions,
)
from curiosity import curiosity_mpc_action
from env_wrapper import EnvCollector, save_raw_format
from task_set import SEEN_TASK_SET, UNSEEN_TASK_SET
from torchvision.utils import make_grid, save_image


# Policies that don't need a WM (no checkpoint loading, no GPU required).
CHEAP_POLICIES = {"zero", "random"}
# Policies that need encoder + decoder + dynamics + per-candidate scorer.
WM_POLICIES = {"curiosity_u_r_norm"}
ALL_POLICIES = CHEAP_POLICIES | WM_POLICIES

# Task-set presets selectable via --task_set. Falls back to the explicit
# --tasks list when the user passes one.
TASK_SET_PRESETS = {
    "seen":   list(SEEN_TASK_SET),
    "unseen": list(UNSEEN_TASK_SET),
    "both":   list(SEEN_TASK_SET) + list(UNSEEN_TASK_SET),
}


def _build_scorer(policy: str, *, encoder, decoder, patch, packing_factor, n_spatial, img_size):
    """Build the per-candidate URNormScorer for the WM-driven curiosity policy."""
    if policy == "curiosity_u_r_norm":
        return URNormScorer(
            encoder, decoder,
            patch=patch,
            packing_factor=packing_factor,
            n_spatial=n_spatial,
            H=img_size, W=img_size,
        )
    raise ValueError(f"Unknown policy: {policy}")


@torch.no_grad()
def encode_frame(
    encoder: Encoder,
    frame_u8: np.ndarray,
    *,
    patch: int,
    packing_factor: int,
    n_spatial: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Encode a single (3, H, W) uint8 frame to packed latent tokens.
    Returns: (1, 1, n_spatial, d_spatial) packed latent.
    """
    frame = torch.from_numpy(frame_u8).float().div(255.0)  # (3, H, W) in [0,1]
    frame = frame.unsqueeze(0).unsqueeze(0).to(device)      # (1, 1, 3, H, W)
    with autocast(device_type=device.type, dtype=torch.bfloat16):
        patches = temporal_patchify(frame, patch)             # (1, 1, Np, Dp)
        z, _ = encoder(patches)                               # (1, 1, n_latents, d_bottleneck)
    z_packed = pack_bottleneck_to_spatial(z, n_spatial=n_spatial, k=packing_factor)
    return z_packed.float()  # (1, 1, n_spatial, d_spatial)


@torch.no_grad()
def save_episode_png(
    ep_data: dict,
    *,
    decoder: Decoder,
    patch: int,
    packing_factor: int,
    img_size: int,
    out_path: str,
    device: torch.device,
    max_width: int = 897_792,
):
    """
    Save a two-row PNG for one episode: real env frames on top, WM predictions below.
    Columns are timesteps. WM row is blank if no predicted latents are available.
    """
    real_frames = torch.from_numpy(ep_data["frames"]).float()  # (N, 3, H, W) in [0, 255]
    N = real_frames.shape[0]

    wm_row = None
    if "wm_latents" in ep_data and ep_data["wm_latents"] is not None:
        # wm_latents: (N-1, n_spatial, d_spatial) — predictions for frames 1..N
        wm_packed = ep_data["wm_latents"].unsqueeze(0).to(device)  # (1, N-1, Sz, Dz)
        with autocast(device_type=device.type, dtype=torch.bfloat16):
            wm_frames = decode_packed_to_frames(
                decoder,
                z_packed=wm_packed,
                H=img_size, W=img_size, C=3,
                patch=patch,
                packing_factor=packing_factor,
            )  # (1, N-1, 3, H, W) in [0, 1]
        wm_frames = wm_frames[0].cpu()  # (N-1, 3, H, W)
        # Prepend blank frame to align with real (no prediction for frame 0)
        blank = torch.zeros(1, 3, img_size, img_size)
        wm_row = torch.cat([blank, wm_frames], dim=0)  # (N, 3, H, W)

    # Concatenate rows sequentially: real row then WM row
    # make_grid with nrow=N wraps after N frames, giving one row per episode
    rows = [real_frames / 255.0]
    if wm_row is not None:
        rows.append(wm_row)
    grid_input = torch.cat(rows, dim=0)  # (N*n_rows, 3, H, W) — real frames then WM frames

    grid = make_grid(grid_input, nrow=N, padding=0)  # (3, H*n_rows, W*N)
    total_width = grid.shape[-1]

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    if total_width <= max_width:
        save_image(grid, out_path)
    else:
        num_batches = (total_width + max_width - 1) // max_width
        stem, ext = out_path.rsplit(".", 1)
        for i in range(num_batches):
            start_col = i * max_width
            end_col = min((i + 1) * max_width, total_width)
            save_image(grid[:, :, start_col:end_col], f"{stem}-{i}.{ext}")

    print(f"[collect] Saved episode PNG -> {out_path}")


def collect_episode_simple(
    *,
    env: EnvCollector,
    policy: str,
    episode_id: int,
    max_steps: int,
    action_dim: int,
) -> dict:
    """Collect one episode under a no-WM action policy (`zero` or `random`).

    Returned dict matches the first 4 fields of ``collect_episode_with_uncertainty``
    (frames / actions / rewards / episode_id), so ``save_raw_format`` consumes
    both interchangeably. No uncertainties / wm_latents are produced.
    """
    if policy not in CHEAP_POLICIES:
        raise ValueError(
            f"collect_episode_simple supports {sorted(CHEAP_POLICIES)}, got {policy!r}"
        )

    obs_dict, _info = env.env.reset()
    obs = env._extract_rgb(obs_dict)
    real_dim = env._real_action_dim

    frames = [obs]
    actions_list = []
    rewards = [float('nan')]  # no transition produced frame 0

    for _step in range(max_steps):
        action = np.zeros(action_dim, dtype=np.float32)
        if policy == "random":
            action[:real_dim] = np.random.uniform(-1.0, 1.0, size=(real_dim,)).astype(np.float32)
        # zero policy: action stays all-zero by construction

        next_obs_dict, reward, terminated, truncated, _info = env.env.step(action[:real_dim])
        next_obs = env._extract_rgb(next_obs_dict)

        frames.append(next_obs)
        actions_list.append(action)
        rewards.append(float(reward))

        if terminated or truncated:
            break

    # Dummy final action (no next frame). NaN matches the WMDataset convention;
    # save_raw_format drops this slot during forward→incoming conversion.
    actions_list.append(np.full(action_dim, np.nan, dtype=np.float32))

    return {
        "frames":     np.stack(frames, axis=0),                # (N, 3, H, W) uint8
        "actions":    np.stack(actions_list, axis=0),           # (N, A) float32
        "rewards":    np.array(rewards, dtype=np.float32),      # (N,) float32
        "episode_id": episode_id,
    }


def collect_episode_with_uncertainty(
    *,
    env: EnvCollector,
    encoder: Encoder,
    dyn: Dynamics,
    sched: Dict[str, Any],
    k_max: int,
    policy: str,
    scorer,
    episode_id: int,
    max_steps: int,
    ctx_window: int,
    patch: int,
    packing_factor: int,
    n_spatial: int,
    action_dim: int,
    act_mask: Optional[torch.Tensor],
    lang_emb: Optional[torch.Tensor],
    tau_ctx: float,
    n_candidates: int,
    n_samples_unc: int,
    n_samples_mpc: int,
    n_elite: int,
    n_cem_iters: int,
    cem_init_std: float,
    cem_min_std: float,
    device: torch.device,
    save_wm_latents: bool = False,
    replan_every: int = 16,
    plan_horizon: Optional[int] = None,
    use_kv_cache: bool = False,
) -> dict:
    """
    Collect one episode, logging per-step scores from `scorer` (u_r_norm). For
    curiosity policies, CEM uses the same scorer as the planning reward.

    Returns dict with:
      "frames":       (N, 3, H, W) uint8
      "actions":      (N, action_dim) float32
      "rewards":      (N,) float32
      "episode_id":   int
      "uncertainties": list[float] — per-step scalar score under `scorer`
      "wm_latents":   (N-1, n_spatial, d_spatial) float32 — first-seed WM predictions (if save_wm_latents)
    """
    obs_dict, _ = env.env.reset()
    obs = env._extract_rgb(obs_dict)

    # Encode initial frame
    z0 = encode_frame(encoder, obs, patch=patch, packing_factor=packing_factor,
                      n_spatial=n_spatial, device=device)  # (1, 1, Sz, Dz)

    # History
    z_history = [z0[:, 0]]  # list of (1, Sz, Dz)
    frames = [obs]
    actions_list = []
    rewards = [float('nan')]  # no transition produced frame 0
    uncertainties = []
    # Per-step ground-truth divergence: ‖z_actual_next − mean(z_wm_predicted)‖_RMS.
    # NaN at index 0 (no prediction-vs-actual for the initial frame).
    prediction_errors = [float('nan')]
    wm_latents_list = []  # mean predicted latents per step
    plan_actions = []     # pre-planned actions for current chunk (curiosity)
    plan_offset = 0       # index into plan_actions
    replan_log = []       # replanning events with CEM stats

    for step in range(max_steps):
        # Build context window (last ctx_window frames)
        t_start = max(0, len(z_history) - ctx_window)
        z_ctx = torch.stack(z_history[t_start:], dim=1)  # (1, t, Sz, Dz)

        # Build action history for context
        if len(actions_list) > 0:
            acts_ctx = torch.stack(
                [torch.from_numpy(a).float().to(device) for a in actions_list[t_start:]],
                dim=0,
            ).unsqueeze(0)  # (1, t-1, A) — but we need (1, t, A) aligned to z_ctx
            # Prepend zero action for first frame in context
            zero_act = torch.zeros(1, 1, action_dim, device=device)
            if t_start == 0:
                acts_ctx = torch.cat([zero_act, acts_ctx], dim=1)  # (1, t, A)
            else:
                # When windowed, the first action in window is acts[t_start-1]
                first_act = torch.from_numpy(actions_list[max(0, t_start - 1)]).float().to(device)
                acts_ctx = torch.cat([first_act.unsqueeze(0).unsqueeze(0), acts_ctx], dim=1)
        else:
            acts_ctx = torch.zeros(1, 1, action_dim, device=device)

        # Select action
        if policy == "curiosity_u_r_norm":
            # Receding-horizon MPC: plan H actions, execute the first K (K<=H), then replan.
            if plan_offset >= len(plan_actions):
                H_plan = int(plan_horizon if plan_horizon is not None else replan_every)
                # Clip both the planning horizon and the execution stride to
                # what's left in the episode budget.
                H_plan = max(1, min(H_plan, max_steps - step))
                K_exec = max(1, min(int(replan_every), H_plan))
                t_plan = time.time()
                mpc_result = curiosity_mpc_action(
                    dyn,
                    past_packed=z_ctx,
                    past_actions=acts_ctx,
                    scorer=scorer,
                    k_max=k_max,
                    sched=sched,
                    act_mask=act_mask,
                    lang_emb=lang_emb,
                    tau_ctx=tau_ctx,
                    n_candidates=n_candidates,
                    horizon=H_plan,
                    n_samples=n_samples_mpc,
                    n_elite=n_elite,
                    n_cem_iters=n_cem_iters,
                    cem_init_std=cem_init_std,
                    cem_min_std=cem_min_std,
                    action_dim=action_dim,
                    device=device,
                    use_kv_cache=use_kv_cache,
                )
                if H_plan > 1 and "action_sequence" in mpc_result:
                    seq = mpc_result["action_sequence"]  # (H_plan, A)
                    # Receding-horizon: execute only the first K of the H plan.
                    plan_actions = [seq[h].cpu().numpy() for h in range(K_exec)]
                else:
                    plan_actions = [mpc_result["action"].cpu().numpy()]
                plan_offset = 0
                dt_plan = time.time() - t_plan
                replan_log.append({
                    "step": step,
                    "plan_horizon": H_plan,
                    "exec_K": K_exec,
                    "plan_unc": float(mpc_result["uncertainty"].item()),
                    "mean_cand_unc": float(mpc_result["mean_uncertainty"].item()),
                    "max_cand_unc": float(mpc_result["max_uncertainty"].item()),
                    "time_s": dt_plan,
                })
                print(f"    step {step}: planned H={H_plan}, exec K={K_exec}, "
                      f"plan_unc={replan_log[-1]['plan_unc']:.4f}, "
                      f"cem_gain={replan_log[-1]['max_cand_unc']/max(replan_log[-1]['mean_cand_unc'], 1e-8):.2f}x "
                      f"({dt_plan:.1f}s)")

            action = plan_actions[plan_offset]
            plan_offset += 1
        else:
            raise ValueError(f"Unknown policy: {policy}")

        # Per-step logging score (same signal as the planning reward) + WM prediction.
        # Uses a small sample count since this is purely for logging / post-hoc filtering.
        act_for_unc = torch.from_numpy(action).float().to(device).unsqueeze(0).unsqueeze(0)
        acts_full = torch.cat([acts_ctx, act_for_unc], dim=1)  # (1, t+1, A)
        predictions_KN = sample_predictions_for_actions(
            dyn,
            past_packed=z_ctx,
            candidate_actions=acts_full,
            k_max=k_max,
            sched=sched,
            act_mask=act_mask,
            tau_ctx=tau_ctx,
            lang_emb=lang_emb,
            n_samples=n_samples_unc,
            use_kv_cache=use_kv_cache,
        )  # (1, N, Sz, Dz)
        z_prev_K = z_ctx[0, -1].float().unsqueeze(0)  # (1, Sz, Dz)
        comps = scorer.score_components(predictions_KN, z_prev_K)            # dict of (1,) tensors
        uncertainties.append(float(comps["u_r_norm"].item()))
        if save_wm_latents:
            wm_latents_list.append(predictions_KN[0, 0].cpu())  # (Sz, Dz) — first seed

        # Step environment
        real_action = action[: env._real_action_dim]
        next_obs_dict, reward, terminated, truncated, info = env.env.step(real_action)
        next_obs = env._extract_rgb(next_obs_dict)
        done = terminated or truncated

        # Encode next frame
        z_next = encode_frame(encoder, next_obs, patch=patch, packing_factor=packing_factor,
                              n_spatial=n_spatial, device=device)

        # Actual next-z vs the WM's predicted next-z (mean over n_samples_unc denoising seeds).
        # No extra forward passes — both tensors are already in hand; RMS-norm over (Sz, Dz).
        z_pred_mean = predictions_KN[0].float().mean(dim=0)            # (Sz, Dz)
        z_actual    = z_next[0, 0].float()                              # (Sz, Dz)
        prediction_errors.append(float((z_pred_mean - z_actual).pow(2).mean().sqrt().item()))

        # Update history
        z_history.append(z_next[:, 0])
        frames.append(next_obs)
        actions_list.append(action if action.shape[0] == action_dim else env._pad_action(action))
        rewards.append(float(reward))

        if done:
            break

    # Dummy final action (no next frame to transition to). NaN matches the user's
    # dataset convention; save_raw_format drops this slot during forward→incoming
    # conversion, so it never reaches the model.
    actions_list.append(np.full(action_dim, np.nan, dtype=np.float32))

    N = len(frames)
    return {
        "frames": np.stack(frames, axis=0),                # (N, 3, H, W) uint8
        "actions": np.stack(actions_list, axis=0),          # (N, A) float32
        "rewards": np.array(rewards, dtype=np.float32),     # (N,) float32
        "episode_id": episode_id,
        "uncertainties": uncertainties,
        "prediction_errors": prediction_errors,             # (N,) RMS(z_actual - z_wm_pred); [0] = NaN
        "wm_latents": torch.stack(wm_latents_list, dim=0) if wm_latents_list else None,
        "replan_log": replan_log,
    }


def _load_wm(args, device):
    """Load tokenizer + dynamics + scheduler + scorer for WM-driven policies.

    Returns a dict bundling everything `collect_episode_with_uncertainty` and
    `save_episode_png` need. Only called when `args.policy ∈ WM_POLICIES`.
    """
    encoder, decoder, tok_args = load_frozen_tokenizer_from_pt_ckpt(
        args.tokenizer_ckpt, device=device,
    )
    # Modern tokenizer ckpts always record these; fail loudly rather than guessing.
    missing = [k for k in ("patch", "n_latents", "d_bottleneck") if k not in tok_args]
    if missing:
        raise KeyError(
            f"Tokenizer checkpoint is missing required keys {missing}; "
            f"only modern tokenizer ckpts (with explicit patch/n_latents/d_bottleneck) "
            f"are supported by collect_data.py."
        )
    patch = int(tok_args["patch"])
    n_latents = int(tok_args["n_latents"])
    d_bottleneck = int(tok_args["d_bottleneck"])
    n_spatial = n_latents // args.packing_factor

    dyn, _rew_head, _policy_head, dyn_meta = load_dynamics_from_ckpt(
        args.dynamics_ckpt,
        device=device,
        d_bottleneck=d_bottleneck,
        n_latents=n_latents,
        packing_factor=args.packing_factor,
    )
    k_max = dyn_meta["k_max"]
    if args.compile:
        dyn = torch.compile(dyn, mode="reduce-overhead")

    sched = make_tau_schedule(k_max=k_max, schedule=args.schedule, d=args.eval_d)
    scorer = _build_scorer(
        args.policy,
        encoder=encoder, decoder=decoder,
        patch=patch, packing_factor=args.packing_factor,
        n_spatial=n_spatial, img_size=args.img_size,
    )
    print(f"[collect] k_max={k_max}, schedule={args.schedule}, K={sched['K']} steps")
    print(f"[collect] patch={patch}, n_latents={n_latents}, d_bottleneck={d_bottleneck}, "
          f"n_spatial={n_spatial}, img_size={args.img_size}")
    print(f"[collect] scorer={type(scorer).__name__}")

    return dict(encoder=encoder, decoder=decoder, dyn=dyn, sched=sched, k_max=k_max,
                patch=patch, n_spatial=n_spatial, scorer=scorer)


def active_collection_round(args):
    """Run a collection sweep: load models if needed, then iterate over tasks."""
    needs_wm = args.policy in WM_POLICIES
    device = torch.device(f"cuda:{args.gpu}" if (torch.cuda.is_available() and needs_wm) else "cpu")

    # Global seed — overridden per-task below for reproducibility.
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print(f"[collect] policy={args.policy} ({'WM-driven' if needs_wm else 'no-WM'}), "
          f"tasks={args.tasks}, n_episodes={args.n_episodes}, seed={args.seed}")
    print(f"[collect] device={device}")

    wm = _load_wm(args, device) if needs_wm else None

    # task_meta only matters for WM policies (lang_emb + act_mask conditioning).
    task_meta = None
    if needs_wm and args.tasks_json and os.path.exists(args.tasks_json):
        with open(args.tasks_json, "r") as f:
            task_meta = json.load(f)

    for task_idx, task in enumerate(args.tasks):
        # Per-task reseed — guarantees that random/curiosity action noise and env
        # initial states are reproducible at (--seed, task) granularity.
        task_seed = args.seed + task_idx * 1000
        np.random.seed(task_seed)
        torch.manual_seed(task_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(task_seed)
        print(f"\n[collect] ====== task: {task} (seed={task_seed}) ======")
        run_task(task=task, args=args, wm=wm, task_meta=task_meta,
                 device=device, task_seed=task_seed)


def run_task(
    *,
    task: str,
    args,
    wm: Optional[dict],
    task_meta: Optional[dict],
    device: torch.device,
    task_seed: int,
):
    """Collect `args.n_episodes` episodes for a single task and save the shard.

    `wm` is a dict from `_load_wm` (encoder/decoder/dyn/sched/k_max/patch/n_spatial/scorer)
    when args.policy is WM-driven, or None for cheap policies.
    `task_meta` is the parsed tasks.json (only consulted for WM-driven policies).
    """
    # Per-task language embedding and action mask (only used by WM-driven policies).
    lang_emb = None
    act_mask = None
    if wm is not None and task_meta is not None and task in task_meta:
        meta = task_meta[task]
        if "text_embedding" in meta:
            lang_emb = torch.tensor(meta["text_embedding"], dtype=torch.float32,
                                    device=device).unsqueeze(0)
        if "action_dim" in meta:
            real_dim = int(meta["action_dim"])
            mask = torch.zeros(args.action_dim, device=device)
            mask[:real_dim] = 1.0
            act_mask = mask  # (A,)

    env = EnvCollector(task, img_size=args.img_size, action_dim=args.action_dim, seed=task_seed)

    all_episodes = []
    all_uncertainties = []      # populated only for WM-driven policies
    all_prediction_errors = []  # WM_pred-vs-actual_z divergence per env step (drops the leading NaN)

    for ep_idx in range(args.n_episodes):
        t0 = time.time()

        if args.policy in CHEAP_POLICIES:
            ep_data = collect_episode_simple(
                env=env, policy=args.policy,
                episode_id=ep_idx, max_steps=args.max_steps,
                action_dim=args.action_dim,
            )
        else:
            ep_data = collect_episode_with_uncertainty(
                env=env,
                encoder=wm["encoder"], dyn=wm["dyn"], sched=wm["sched"], k_max=wm["k_max"],
                policy=args.policy, scorer=wm["scorer"],
                episode_id=ep_idx, max_steps=args.max_steps, ctx_window=args.ctx_window,
                patch=wm["patch"], packing_factor=args.packing_factor, n_spatial=wm["n_spatial"],
                action_dim=args.action_dim, act_mask=act_mask, lang_emb=lang_emb,
                tau_ctx=args.tau_ctx,
                n_candidates=args.n_candidates,
                n_samples_unc=args.n_samples_unc, n_samples_mpc=args.n_samples_mpc,
                n_elite=args.n_elite, n_cem_iters=args.n_cem_iters,
                cem_init_std=args.cem_init_std, cem_min_std=args.cem_min_std,
                device=device,
                save_wm_latents=(args.save_vis_every > 0),
                replan_every=args.replan_every,
                plan_horizon=args.plan_horizon,
                use_kv_cache=args.use_kv_cache,
            )

        elapsed = time.time() - t0
        n_frames = ep_data["frames"].shape[0]
        total_reward = float(np.nansum(ep_data["rewards"]))
        real_dim = env._real_action_dim
        act_active = np.abs(ep_data["actions"][:-1, :real_dim])  # drop NaN dummy
        mean_act_mag = float(act_active.mean()) if act_active.size > 0 else 0.0
        act_sat_frac = float((act_active > 0.99).mean()) if act_active.size > 0 else 0.0

        ep_unc = ep_data.get("uncertainties") or []
        # prediction_errors carries a leading NaN (no prediction-vs-actual for
        # frame 0); drop it before aggregating across episodes.
        ep_perr_full = ep_data.get("prediction_errors") or []
        ep_perr = [v for v in ep_perr_full if not (isinstance(v, float) and np.isnan(v))]
        if ep_unc:
            mean_unc = float(np.mean(ep_unc))
            max_unc  = float(np.max(ep_unc))
            mean_perr = float(np.mean(ep_perr)) if ep_perr else float('nan')
            n_replans = len(ep_data.get("replan_log", []))
            print(f"  episode {ep_idx+1}/{args.n_episodes}: {n_frames} frames, "
                  f"mean_unc={mean_unc:.6f}, max_unc={max_unc:.6f}, "
                  f"mean_perr={mean_perr:.6f}, "
                  f"reward={total_reward:.2f}, |a|={mean_act_mag:.3f}, "
                  f"sat={act_sat_frac:.0%}, replans={n_replans}, time={elapsed:.1f}s")
            all_uncertainties.extend(ep_unc)
            all_prediction_errors.extend(ep_perr)
        else:
            print(f"  episode {ep_idx+1}/{args.n_episodes}: {n_frames} frames, "
                  f"reward={total_reward:.2f}, |a|={mean_act_mag:.3f}, "
                  f"sat={act_sat_frac:.0%}, time={elapsed:.1f}s")

        # WM-only visualization: real-vs-WM PNG side-by-side. Skipped for cheap
        # policies (no decoder, no wm_latents).
        if wm is not None and args.save_vis_every > 0 and ep_idx % args.save_vis_every == 0:
            mean_unc_for_name = float(np.mean(ep_unc)) if ep_unc else 0.0
            save_episode_png(
                ep_data,
                decoder=wm["decoder"],
                patch=wm["patch"],
                packing_factor=args.packing_factor,
                img_size=args.img_size,
                out_path=str(Path(args.out_data_dir) / f"{task}-ep{ep_idx:03d}-unc{mean_unc_for_name:.4f}.png"),
                device=device,
            )

        all_episodes.append(ep_data)

    if all_uncertainties:
        unc_arr = np.array(all_uncertainties)
        print(f"\n[collect] Summary for {task}:")
        print(f"  total episodes:  {len(all_episodes)}")
        print(f"  total frames:    {sum(ep['frames'].shape[0] for ep in all_episodes)}")
        print(f"  uncertainty mean: {unc_arr.mean():.6f}")
        print(f"  uncertainty std:  {unc_arr.std():.6f}")
        print(f"  uncertainty max:  {unc_arr.max():.6f}")
        print(f"  uncertainty p90:  {np.percentile(unc_arr, 90):.6f}")
        print(f"  uncertainty p99:  {np.percentile(unc_arr, 99):.6f}")
        if all_prediction_errors:
            perr_arr = np.array(all_prediction_errors)
            print(f"  prediction_error mean: {perr_arr.mean():.6f}  std: {perr_arr.std():.6f}  max: {perr_arr.max():.6f}")

    if args.uncertainty_threshold > 0:
        if not all_uncertainties:
            print(f"\n[collect] --uncertainty_threshold>0 ignored (no uncertainties for policy={args.policy!r})")
        else:
            filtered = [ep for ep in all_episodes
                        if (np.mean(ep["uncertainties"]) if ep.get("uncertainties") else 0.0)
                           >= args.uncertainty_threshold]
            print(f"\n[collect] Filtering: {len(filtered)}/{len(all_episodes)} episodes above "
                  f"threshold={args.uncertainty_threshold:.6f}")
            all_episodes = filtered

    if len(all_episodes) == 0:
        print("[collect] No episodes to save (all filtered out).")
        env.close()
        return

    save_result = save_raw_format(
        episodes=all_episodes,
        out_dir=args.out_data_dir,
        task=task,
        max_frames_per_png=args.png_max_frames,
    )

    rdim = env._real_action_dim
    stats_path = Path(args.out_data_dir) / f"{task}_collect_stats.json"

    def _ep_stats(ep):
        # Drop the trailing NaN dummy action when computing |a| / saturation.
        acts_active = np.abs(ep["actions"][:-1, :rdim]) if ep["actions"].shape[0] > 1 else np.zeros((0, rdim))
        s = {
            "episode_id":        ep["episode_id"],
            "n_frames":          int(ep["frames"].shape[0]),
            "total_reward":      float(np.nansum(ep["rewards"])),
            "mean_action_mag":   float(acts_active.mean()) if acts_active.size > 0 else 0.0,
            "action_saturation": float((acts_active > 0.99).mean()) if acts_active.size > 0 else 0.0,
            "reward_timeseries": ep["rewards"].tolist(),
        }
        if ep.get("uncertainties"):
            s["mean_unc"] = float(np.mean(ep["uncertainties"]))
            s["max_unc"]  = float(np.max(ep["uncertainties"]))
            s["uncertainty_timeseries"] = ep["uncertainties"]
            s["replan_log"] = ep.get("replan_log", [])
        if ep.get("prediction_errors"):
            ep_perr_full = ep["prediction_errors"]
            ep_perr = [v for v in ep_perr_full if not (isinstance(v, float) and np.isnan(v))]
            s["mean_prediction_error"] = float(np.mean(ep_perr)) if ep_perr else float('nan')
            s["max_prediction_error"]  = float(np.max(ep_perr))  if ep_perr else float('nan')
            # Carry the raw (NaN-leading) timeseries so post-hoc analysis can
            # align indices with frames / actions / rewards exactly.
            s["prediction_error_timeseries"] = ep_perr_full
        return s

    stats = {
        "task":         task,
        "policy":       args.policy,
        "n_episodes":   len(all_episodes),
        "total_frames": save_result["total_frames"],
        "episodes":     [_ep_stats(ep) for ep in all_episodes],
    }
    if all_uncertainties:
        stats["uncertainty_mean"] = float(np.mean(all_uncertainties))
        stats["uncertainty_std"]  = float(np.std(all_uncertainties))
        stats["uncertainty_max"]  = float(np.max(all_uncertainties))
    if all_prediction_errors:
        stats["prediction_error_mean"] = float(np.mean(all_prediction_errors))
        stats["prediction_error_std"]  = float(np.std(all_prediction_errors))
        stats["prediction_error_max"]  = float(np.max(all_prediction_errors))
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[collect] Saved collect stats to {stats_path}")

    env.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()

    # tasks — either pass --tasks explicitly or pick a preset via --task_set.
    p.add_argument("--tasks", type=str, nargs="+", default=None,
                   help="Explicit list of tasks. Overrides --task_set when provided.")
    p.add_argument("--task_set", type=str, default="seen",
                   choices=sorted(TASK_SET_PRESETS),
                   help="Task-set preset (used only if --tasks is not provided). "
                        "'seen'=10 SEEN_TASK_SET, 'unseen'=10 UNSEEN_TASK_SET, "
                        "'both'=20-task union.")

    # checkpoints — only loaded for WM-driven policies.
    p.add_argument("--tokenizer_ckpt", type=str,
                   default="./logs/tokenizer_ckpts/latest.pt",
                   help="Tokenizer checkpoint (only loaded for WM-driven policies).")
    p.add_argument("--dynamics_ckpt", type=str,
                   default="./logs/dynamics_ckpts/latest.pt",
                   help="Dynamics checkpoint (only loaded for WM-driven policies).")
    p.add_argument("--tasks_json", type=str, default="../tasks.json")

    # policy
    p.add_argument("--policy", type=str, default="zero",
                   choices=sorted(ALL_POLICIES),
                   help="Action policy. Cheap (no WM): 'zero' (all-zero actions), "
                        "'random' (uniform [-1,1] over the env's true action dims). "
                        "WM-driven: 'curiosity_u_r_norm' uses the motion-normalized "
                        "round-trip residual signal.")

    # env
    p.add_argument("--n_episodes", type=int, default=5)
    p.add_argument("--max_steps", type=int, default=500)
    p.add_argument("--img_size", type=int, default=224,
                   help="Image resolution passed to env / tokenizer (must match training).")

    # model
    p.add_argument("--packing_factor", type=int, default=2)
    p.add_argument("--action_dim", type=int, default=16)

    # inference schedule
    p.add_argument("--schedule", type=str, default="shortcut")
    p.add_argument("--eval_d", type=float, default=0.25)

    # context
    p.add_argument("--ctx_window", type=int, default=24)
    p.add_argument("--tau_ctx", type=float, default=0.1)

    # Curiosity MPC. Receding-horizon CEM: plan H actions, execute the first K (K <= H), then replan from the new state.
    p.add_argument("--n_candidates", type=int, default=128)
    p.add_argument("--plan_horizon", type=int, default=32,
                   help="CEM planning horizon H (imagined WM steps per plan).")
    p.add_argument("--replan_every", type=int, default=16,
                   help="Re-plan stride K (env steps executed per CEM plan). "
                        "K=plan_horizon recovers open-loop chunked CEM. K<H is "
                        "canonical receding-horizon MPC.")
    p.add_argument("--n_elite", type=int, default=32)
    p.add_argument("--n_cem_iters", type=int, default=3,
                   help="CEM iterations per plan.")
    p.add_argument("--cem_init_std", type=float, default=1.0,
                   help="Initial per-dim action std for CEM iter 0.")
    p.add_argument("--cem_min_std", type=float, default=0.05,
                   help="Minimum per-dim action std floor after elite update")
    p.add_argument("--use_kv_cache", action=argparse.BooleanOptionalAction, default=True,
                   help="Cache time-attn K,V for context tokens during CEM rollout. "
                        "Default on; pass --no-use_kv_cache to disable.")

    # uncertainty
    p.add_argument("--n_samples_unc", type=int, default=2,
                   help="N samples per env step for the per-step uncertainty log (a diagnostic, not a decision input).")
    p.add_argument("--n_samples_mpc", type=int, default=2, help="N samples per MPC candidate.")
    p.add_argument("--uncertainty_threshold", type=float, default=0.0,
                   help="Only keep episodes with mean uncertainty above this threshold (0 = keep all)")

    # output
    p.add_argument("--out_data_dir", type=str, required=True,
                   help="Output dir holding both PNG strips and demo .pt "
                        "(HuggingFace dataset layout).")
    p.add_argument("--png_max_frames", type=int, default=4008,
                   help="Max frames per PNG strip. "
                        "Default 4008 keeps PNG width <= 897792 px (PIL bomb-check cap).")

    # misc
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--seed", type=int, default=31,
                   help="Base RNG seed. Each task additionally reseeds at seed + idx*1000.")
    p.add_argument("--save_vis_every", type=int, default=0,
                   help="Write the real/WM PNG every Nth episode; 0 disables PNG saves. "
                        "PNGs require a WM (no-op for cheap policies).")
    p.add_argument("--compile", action="store_true", help="torch.compile the dynamics model")

    args = p.parse_args()
    if args.tasks is None:
        args.tasks = TASK_SET_PRESETS[args.task_set]
    active_collection_round(args)
