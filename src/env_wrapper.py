# env_wrapper.py
"""
Environment adapter for active data collection.

Wraps the project's unified environment interface (envs.make_env) for use
with the active collection pipeline. Also provides utilities for saving
collected episodes in raw HuggingFace dataset format.

Interface contract (after wrapping):
  - env.reset() -> (obs_dict, info)   where obs_dict['rgb'] is (3, H, W) uint8
  - env.step(action) -> (obs_dict, reward, terminated, truncated, info)
  - env.action_space.shape -> (action_dim,)
  - env.max_episode_steps -> int
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from envs import make_env as _make_env


# ---------------------------------------------------------------------------
# Minimal config object for envs.make_env
# ---------------------------------------------------------------------------

class _EnvCfg:
    """Minimal config object satisfying envs.make_env(cfg) requirements."""

    def __init__(self, task: str, img_size: int = 224, seed: int = 0):
        self.task = task
        self.obs = 'rgb'
        self.seed = seed
        self.child_env = True
        self.num_envs = 1
        self.save_video = False
        self.rank = 0
        self.render_size = img_size
        # These may be set by make_env after construction
        self.obs_shape = None
        self.action_dim = None
        self.episode_length = None

    def get(self, key, default=None):
        return getattr(self, key, default)


# ---------------------------------------------------------------------------
# Environment collector
# ---------------------------------------------------------------------------

class EnvCollector:
    """
    Collects episodes from a live environment and saves them in WMDataset shard format.
    """

    def __init__(self, task: str, img_size: int = 224, action_dim: int = 16, seed: int = 0,
                 render_size: int = 224):
        self.task = task
        self.img_size = img_size
        self.action_dim = action_dim

        # Render at render_size then downsample to img_size with bilinear if they differ,
        # matching preprocess_dataset.py. Defaults keep everything at 224 (no downsampling).
        cfg = _EnvCfg(task, img_size=render_size, seed=seed)
        self.env = _make_env(cfg)
        self._real_action_dim = cfg.action_dim
        self._max_episode_steps = cfg.episode_length

    def _extract_rgb(self, obs) -> np.ndarray:
        """
        Extract RGB frame from observation dict and downsample to img_size using bilinear
        interpolation, matching the preprocessing in preprocess_dataset.py exactly:
          float32 / 255 -> bilinear interpolate -> clamp -> uint8
        """
        if isinstance(obs, dict):
            frame = obs['rgb']
        else:
            frame = obs

        # Ensure (3, H, W)
        if frame.ndim == 3 and frame.shape[2] == 3:
            frame = np.transpose(frame, (2, 0, 1))

        if frame.dtype != np.uint8:
            if frame.max() <= 1.0:
                frame = (frame * 255).clip(0, 255).astype(np.uint8)
            else:
                frame = frame.clip(0, 255).astype(np.uint8)

        # Resize to img_size using bilinear if the rendered frame doesn't already match,
        # same as preprocess_dataset.py. Use the actual frame shape rather than the
        # configured render size, which can diverge from what the env actually returns.
        H_in, W_in = frame.shape[-2], frame.shape[-1]
        if H_in != self.img_size or W_in != self.img_size:
            t = torch.from_numpy(frame).unsqueeze(0).float()  # (1, 3, H, W)
            t = F.interpolate(t / 255.0, size=(self.img_size, self.img_size),
                              mode="bilinear", align_corners=False)
            frame = (t.squeeze(0).clamp(0.0, 1.0) * 255.0).to(torch.uint8).numpy()

        return frame

    def close(self):
        """Close the underlying environment to free resources."""
        if hasattr(self.env, 'close'):
            self.env.close()

    def _pad_action(self, action: np.ndarray) -> np.ndarray:
        """Zero-pad action to universal action_dim."""
        if action.shape[0] == self.action_dim:
            return action
        padded = np.zeros(self.action_dim, dtype=np.float32)
        padded[: action.shape[0]] = action
        return padded


# ---------------------------------------------------------------------------
# Raw HuggingFace-format saving (compatible with preprocess_dataset.py input)
# ---------------------------------------------------------------------------

def save_raw_format(
    episodes: list,
    out_dir: str,
    task: str,
    max_frames_per_png: int = 4008,
):
    """Save collected episodes in the raw HuggingFace dataset format.

    This produces the same on-disk layout as the MMBench2 dataset, so
    `preprocess_dataset.py` can convert it to shards on the training machine.
    Much smaller on disk than the post-shard format because PNG compresses
    static / low-motion frames very well.

    Creates:
      out_dir/{task}.pt        — demo file with episode/action/reward/terminated
      out_dir/{task}-{i}.png   — horizontal frame strips (224 x 224*N), with
                                  N <= max_frames_per_png to stay inside PIL's
                                  default decompression-bomb cap (178M pixels).

    Args:
      episodes: list of dicts with keys "frames" (N,3,H,W) uint8, "actions"
                (N,A) float32 forward convention, "rewards" (N,) float32,
                "episode_id" int.
      out_dir: single output directory (raw format keeps demo + PNGs together).
      max_frames_per_png: cap each PNG strip to N*224 columns. Default 4008
                          matches the existing HF dataset convention.

    Returns dict with total_frames, n_episodes, n_pngs, demo_path, out_dir.
    """
    from torchvision.io import write_png  # lazy: torchvision import is non-trivial

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Concatenate all episodes, converting forward → incoming action convention.
    all_frames = []
    all_actions = []
    all_rewards = []
    all_episodes = []
    for ep in episodes:
        N = ep["frames"].shape[0]
        eid = ep["episode_id"]
        # forward → incoming: shift actions right by one, NaN at index 0.
        act_fwd = ep["actions"]
        act_inc = np.full_like(act_fwd, np.nan)
        act_inc[1:] = act_fwd[:-1]
        all_frames.append(torch.from_numpy(ep["frames"]))
        all_actions.append(torch.from_numpy(act_inc))
        all_rewards.append(torch.from_numpy(ep["rewards"]))
        all_episodes.append(torch.full((N,), eid, dtype=torch.int64))

    all_frames    = torch.cat(all_frames, dim=0)    # (total, 3, H, W) uint8
    all_actions   = torch.cat(all_actions, dim=0)   # (total, A) float32, incoming
    all_rewards   = torch.cat(all_rewards, dim=0)   # (total,) float32
    all_episodes  = torch.cat(all_episodes, dim=0)  # (total,) int64
    total = all_frames.shape[0]

    H, W = int(all_frames.shape[-2]), int(all_frames.shape[-1])
    if (H, W) != (224, 224):
        raise ValueError(f"raw format expects 224x224 frames, got {H}x{W}")

    # Demo .pt — keys match what the HF dataset / WMDataset / preprocess pipeline
    # consume. `obs` (state vectors) is intentionally omitted because env-collected
    # data has no state-vector observations and WMDataset doesn't read it. We
    # include `terminated` (all-False) for full schema compatibility with the
    # downloaded HF dataset.
    demo_path = out_path / f"{task}.pt"
    torch.save({
        "episode":    all_episodes,
        "action":     all_actions,
        "reward":     all_rewards,
        "terminated": torch.zeros(total, dtype=torch.bool),
    }, demo_path)

    # PNG strips. Layout must round-trip through preprocess_dataset.py's read:
    #   read_image(...) -> (3, 224, 224*N)
    #   .view(3, 224, N, 224).permute(2, 0, 1, 3) -> (N, 3, 224, 224)
    # Inverse: (N,3,224,224).permute(1,2,0,3).contiguous().view(3,224,N*224)
    n_pngs = 0
    for i, start in enumerate(range(0, total, max_frames_per_png)):
        end = min(start + max_frames_per_png, total)
        chunk = all_frames[start:end]                              # (N_i, 3, 224, 224)
        N_i = int(chunk.shape[0])
        strip = chunk.permute(1, 2, 0, 3).contiguous().view(3, 224, N_i * 224)
        png_path = out_path / f"{task}-{i}.png"
        # write_png reads / writes uint8 directly — matches preprocess_dataset.py.
        write_png(strip, str(png_path))
        n_pngs += 1

    print(f"[save_raw_format] {task}: {total} frames in {n_pngs} PNG strip(s) -> {out_path}")
    print(f"[save_raw_format] {task}: demo -> {demo_path}")

    return {
        "total_frames": total,
        "n_episodes": len(episodes),
        "n_pngs": n_pngs,
        "demo_path": str(demo_path),
        "out_dir": str(out_path),
    }
