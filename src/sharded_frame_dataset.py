# sharded_frame_dataset.py
import json
import os
import bisect
import random
from collections import OrderedDict
from pathlib import Path
from typing import Sequence, List, Dict, Union, Mapping, Optional

import torch
import torch.distributed as dist
from torch.utils.data import Dataset


class ShardedFrameDataset(Dataset):
    """
    Samples contiguous sequences from preprocessed shards across multiple roots:

      root/<task>/<task>_index.json  with {"shard_name": num_frames, ...}
      root/<task>/*.pt               with {"frames": (N, 3, H, W) uint8}

    Returns: (T, 3, H, W) float32 in [0,1], where T = seq_len.

    If iid_sampling=True, ignores idx and samples a random sequence from this
    rank's slice of shards. Each worker holds a "current" shard for
    samples_per_shard draws before picking a new one — this amortizes the cost
    of loading a shard from disk over many sequences and is the main lever for
    avoiding I/O-bound training on large datasets.

    cache_size controls how many shards are kept in memory (LRU eviction) per
    worker process. With samples_per_shard > 1, a small cache (4-8) suffices
    since most accesses go to the current shard.

    ddp_partition controls whether iid_sampling shards are partitioned across
    DDP ranks. Set to False for rank-0-only validation loaders so rank 0 can
    still see the full validation set.
    """

    def __init__(
        self,
        outdirs: Union[str, Sequence[str]],
        tasks: Sequence[str] = (),
        seq_len: int = 16,
        iid_sampling: bool = True,
        cache_size: int = 8,
        samples_per_shard: int = 1,
        ddp_partition: bool = True,
        task_weights: Optional[Mapping[str, float]] = None,
        verbose: bool = True,
        return_task_idx: bool = False,
    ):
        super().__init__()
        assert outdirs is not None, "outdirs must be specified"

        if isinstance(outdirs, (str, Path)):
            self.outdirs = [str(outdirs)]
        else:
            self.outdirs = [str(p) for p in outdirs]

        self.tasks = list(tasks)
        self.seq_len = int(seq_len)
        self.iid_sampling = bool(iid_sampling)
        self._cache_size = max(1, int(cache_size))
        self.samples_per_shard = max(1, int(samples_per_shard))
        self.verbose = bool(verbose)
        self.return_task_idx = bool(return_task_idx)

        # Resolve optional per-task sampling weights. Mirrors WMDataset: a dict
        # {task_name: weight} sets relative task draw probabilities; tasks not
        # present get weight 0 (excluded from sampling). If None, fall back to
        # the legacy behavior where shard weight = num_starts, which makes
        # P(task) ∝ total valid_starts (short-trajectory domains get starved).
        if task_weights is not None:
            tw_list: List[float] = []
            missing: List[str] = []
            for t in self.tasks:
                if t in task_weights:
                    w = float(task_weights[t])
                    if w < 0:
                        raise ValueError(f"task_weights[{t!r}] is negative: {w}")
                    tw_list.append(w)
                else:
                    missing.append(t)
                    tw_list.append(0.0)
            if missing and self.verbose:
                print(f"[ShardedFrameDataset] Warning: {len(missing)} tasks have no "
                      f"task_weights entry and will be excluded from sampling "
                      f"(e.g. {missing[:5]})")
            self.task_weights: Optional[List[float]] = tw_list
        else:
            self.task_weights = None

        self.shards: List[Dict] = []
        self.cum_starts: List[int] = []
        total_starts = 0

        for root in self.outdirs:
            root = Path(root)
            for ti, task in enumerate(self.tasks):
                task_dir = root / task
                if not task_dir.exists():
                    continue

                index_path = task_dir / f"{task}_index.json"
                if index_path.exists():
                    # Fast path: read frame counts from the metadata index written by
                    # preprocess_dataset.py — no tensor data is loaded at init time.
                    with open(index_path) as f:
                        index = json.load(f)
                    for shard_name, num_frames in sorted(index.items()):
                        path = task_dir / shard_name
                        if not path.exists():
                            print(f"[ShardedFrameDataset] Shard {path} listed in index but missing, skipping")
                            continue
                        N = int(num_frames)
                        if N < self.seq_len:
                            print(f"[ShardedFrameDataset] Skipping {path} (N={N} < seq_len={self.seq_len})")
                            continue
                        num_starts = N - self.seq_len + 1
                        self.shards.append({"path": str(path), "num_frames": N, "num_starts": num_starts, "task_idx": ti})
                        total_starts += num_starts
                        self.cum_starts.append(total_starts)
                else:
                    # Slow fallback: load every shard to inspect its shape.
                    # Run preprocess_dataset.py to generate index files and avoid this.
                    print(f"[ShardedFrameDataset] No index for task={task} in {root}, scanning shards (slow)")
                    for fname in sorted(os.listdir(task_dir)):
                        if not fname.endswith(".pt"):
                            continue
                        path = task_dir / fname

                        try:
                            td = torch.load(path, map_location="cpu", weights_only=True)
                        except Exception as e:
                            print(f"[ShardedFrameDataset] Skipping shard {path} (load error): {e}")
                            continue

                        frames = td.get("frames", None)
                        if not isinstance(frames, torch.Tensor):
                            print(f"[ShardedFrameDataset] Skipping shard {path} (no 'frames' tensor)")
                            continue
                        if frames.ndim != 4 or frames.shape[1] != 3:
                            print(f"[ShardedFrameDataset] Skipping shard {path} (unexpected shape {frames.shape})")
                            continue

                        N = int(frames.shape[0])
                        if N < self.seq_len:
                            print(f"[ShardedFrameDataset] Skipping shard {path} (N={N} < seq_len={self.seq_len})")
                            continue

                        num_starts = N - self.seq_len + 1
                        self.shards.append({"path": str(path), "num_frames": N, "num_starts": num_starts, "task_idx": ti})
                        total_starts += num_starts
                        self.cum_starts.append(total_starts)

        self.total_starts = total_starts

        # ---- DDP-aware sharding for iid_sampling ----
        # When running under DDP with iid_sampling=True, partition shards across
        # ranks so each rank only samples from its own slice. This (a) makes the
        # "epoch" concept meaningful by removing cross-rank overlap, and (b)
        # shrinks each worker's shard pool by world_size, dramatically improving
        # per-worker LRU cache effectiveness.
        rank, world_size = 0, 1
        if (
            ddp_partition
            and self.iid_sampling
            and dist.is_available()
            and dist.is_initialized()
        ):
            rank = dist.get_rank()
            world_size = dist.get_world_size()

        if world_size > 1 and len(self.shards) >= world_size:
            # Strided (round-robin) partitioning rather than contiguous: shards
            # are appended in task order, so a contiguous slice would lock each
            # rank into a small set of tasks. Striding interleaves tasks across
            # ranks, so every rank (and therefore the rank-0 viz) sees a
            # task-diverse sample.
            self._iid_shard_indices = list(range(rank, len(self.shards), world_size))
        else:
            self._iid_shard_indices = list(range(len(self.shards)))

        # Precompute selection weights. Default: proportional to num_starts so
        # the marginal distribution over sequences stays uniform across shards
        # of varying length — but this makes P(task) ∝ total valid_starts.
        # When task_weights is set, keep the intra-task distribution ∝ num_starts
        # (for cache locality) but rescale per-task totals to match task_weights.
        # Shards of tasks with weight 0 (or tasks absent from the dict) are
        # dropped from the sampling pool entirely.
        if self._iid_shard_indices:
            if self.task_weights is None:
                weights = [float(self.shards[i]["num_starts"]) for i in self._iid_shard_indices]
            else:
                # Group this rank's shards by task, compute per-task scale.
                task_to_total_ns: Dict[int, float] = {}
                for i in self._iid_shard_indices:
                    ti = self.shards[i]["task_idx"]
                    task_to_total_ns[ti] = task_to_total_ns.get(ti, 0.0) + float(self.shards[i]["num_starts"])

                task_scale: Dict[int, float] = {}
                for ti, total_ns in task_to_total_ns.items():
                    w_task = float(self.task_weights[ti])
                    if w_task <= 0 or total_ns <= 0:
                        task_scale[ti] = 0.0
                    else:
                        task_scale[ti] = w_task / total_ns

                kept_indices: List[int] = []
                weights = []
                for i in self._iid_shard_indices:
                    ti = self.shards[i]["task_idx"]
                    s = task_scale.get(ti, 0.0)
                    if s <= 0:
                        continue
                    kept_indices.append(i)
                    weights.append(s * float(self.shards[i]["num_starts"]))
                self._iid_shard_indices = kept_indices

            total_w = float(sum(weights))
            if total_w > 0:
                self._iid_shard_weights = [w / total_w for w in weights]
            else:
                self._iid_shard_weights = []
        else:
            self._iid_shard_weights = []

        if self.total_starts == 0:
            print("[ShardedFrameDataset] WARNING: no usable sequences found in outdirs")
        else:
            extra = ""
            if world_size > 1 and self.iid_sampling and ddp_partition:
                extra = (
                    f", ddp_rank={rank}/{world_size}, "
                    f"local_shards={len(self._iid_shard_indices):,}"
                )
            if self.task_weights is not None:
                nz = sum(1 for w in self.task_weights if w > 0)
                extra += f", task_weights active ({nz}/{len(self.tasks)} tasks with w>0)"
            print(
                f"[ShardedFrameDataset] roots={len(self.outdirs)}, "
                f"shards={len(self.shards):,}, seq_starts={self.total_starts:,}, "
                f"samples_per_shard={self.samples_per_shard}{extra}"
            )

        # LRU shard cache: most-recently-used shards stay resident in memory.
        # With samples_per_shard > 1, the bulk of accesses hit the current
        # shard, so a small cache_size suffices.
        self._cache: OrderedDict[str, torch.Tensor] = OrderedDict()

        # Per-worker state for samples_per_shard. These attributes are inherited
        # by forked workers but mutated independently in each worker process,
        # so each worker maintains its own "currently active" shard.
        self._cur_shard_idx: Union[int, None] = None
        self._cur_shard_draws: int = 0

    def __len__(self) -> int:
        return self.total_starts

    def _load_shard(self, path: str) -> torch.Tensor:
        if path in self._cache:
            self._cache.move_to_end(path)  # mark as most-recently-used
            return self._cache[path]
        td = torch.load(path, map_location="cpu", weights_only=True)
        frames = td["frames"]
        if len(self._cache) >= self._cache_size:
            self._cache.popitem(last=False)  # evict least-recently-used
        self._cache[path] = frames
        return frames

    def _map_global_start_to_shard(self, global_start: int) -> tuple[int, int]:
        # global_start in [0, total_starts)
        shard_idx = bisect.bisect_right(self.cum_starts, global_start)
        prev_cum = 0 if shard_idx == 0 else self.cum_starts[shard_idx - 1]
        start_idx_in_shard = global_start - prev_cum
        return shard_idx, start_idx_in_shard

    def __getitem__(self, idx: int) -> torch.Tensor:
        if self.total_starts == 0:
            raise IndexError("Empty dataset")

        if self.iid_sampling:
            # Reuse the current shard for samples_per_shard draws before picking
            # a new one. This is the key locality optimization that lets large
            # datasets train without saturating disk bandwidth.
            if (
                self._cur_shard_idx is None
                or self._cur_shard_draws >= self.samples_per_shard
            ):
                self._cur_shard_idx = random.choices(
                    self._iid_shard_indices,
                    weights=self._iid_shard_weights,
                    k=1,
                )[0]
                self._cur_shard_draws = 0
            shard_idx = self._cur_shard_idx
            self._cur_shard_draws += 1

            meta = self.shards[shard_idx]
            start = random.randrange(meta["num_starts"])
        else:
            if idx < 0 or idx >= self.total_starts:
                raise IndexError(idx)
            shard_idx, start = self._map_global_start_to_shard(int(idx))
            meta = self.shards[shard_idx]

        frames = self._load_shard(meta["path"])  # (N, 3, H, W)

        end = start + self.seq_len
        seq_u8 = frames[start:end]  # (T, 3, H, W), guaranteed valid by construction
        seq = seq_u8.to(torch.float32) / 255.0
        if self.return_task_idx:
            return {"frames": seq, "task_idx": int(meta["task_idx"])}
        return seq
