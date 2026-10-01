# bench_loading.py
"""
Data-loading throughput of preprocessed shards at the training scripts' default loader
settings. Works on either shard format; run it once per format to compare, e.g.

    python tests/bench_loading.py --frame_dirs src/data/val-shards --data_dirs src/data/val --tasks ms-pick-cube
    python tests/bench_loading.py --frame_dirs src/data/val-shards-raw --data_dirs src/data/val --tasks ms-pick-cube

Reports the cost of opening a shard and reading one window, then samples/s for
ShardedFrameDataset (train_tokenizer.py defaults) and, if --data_dirs is given, WMDataset
(train_dynamics.py defaults). CPU only.

Raw shards are fully resident once cached, so with only a few shards on disk the raw
numbers are an upper bound. Pass --shard_cache_size 1 / --cache_mb 700 to force a reload
on every shard switch, which is what a full-size dataset does on most switches (and note
that this still reads from a warm page cache).
"""
import argparse
import contextlib
import io
import os
import random
import statistics
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from shard_io import list_shards, load_shard  # noqa: E402
from sharded_frame_dataset import ShardedFrameDataset  # noqa: E402
from wm_dataset import WMDataset, collate_batch  # noqa: E402


def worker_init_fn(worker_id):
    random.seed(torch.utils.data.get_worker_info().seed)


def micro(args):
    path = next((p for d in args.frame_dirs for t in args.tasks for p in list_shards(os.path.join(d, t))), None)
    if path is None:
        raise SystemExit(f"no shards for {args.tasks} under {args.frame_dirs}")
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        shard = load_shard(path)
        times.append(time.perf_counter() - t0)
    n = shard.shape[0]
    print(f"{os.path.basename(path)}: {n} frames, {os.path.getsize(path) / 1e6:.1f} MB "
          f"({os.path.getsize(path) / n / 1e3:.1f} KB/frame), open/load {min(times) * 1e3:.1f} ms")
    rng = random.Random(0)
    for T in sorted({args.seq_len, args.seq_len + 1}):
        if n <= T:
            continue
        times = []
        for _ in range(200):
            s = rng.randrange(n - T)
            t0 = time.perf_counter()
            shard[s:s + T]
            times.append(time.perf_counter() - t0)
        print(f"  random {T}-frame window: median {statistics.median(times) * 1e3:.2f} ms, "
              f"p95 {sorted(times)[189] * 1e3:.2f} ms")


def throughput(name, ds, batch_size, workers, collate, args):
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=workers, drop_last=True,
                        persistent_workers=workers > 0, prefetch_factor=args.prefetch_factor if workers > 0 else None,
                        worker_init_fn=worker_init_fn, collate_fn=collate)
    def batches():                      # small datasets have fewer batches per epoch than we time
        while True:
            yield from loader

    it = batches()
    for _ in range(args.warmup_batches):
        next(it)
    t0 = time.perf_counter()
    for _ in range(args.batches):
        next(it)
    dt = time.perf_counter() - t0
    print(f"{name}: {args.batches * batch_size / dt:.1f} samples/s "
          f"({dt / args.batches * 1e3:.0f} ms per batch of {batch_size}, {workers} workers)")


def main(args):
    torch.set_num_threads(4)
    micro(args)
    with contextlib.redirect_stdout(io.StringIO()):
        sf = ShardedFrameDataset(args.frame_dirs, tasks=args.tasks, seq_len=args.seq_len, iid_sampling=True,
                                 cache_size=args.shard_cache_size, samples_per_shard=args.tok_samples_per_shard,
                                 return_task_idx=True)
    throughput("ShardedFrameDataset (tokenizer loader)", sf, args.tok_batch_size, args.tok_num_workers, None, args)
    if args.data_dirs:
        wm = WMDataset(args.data_dirs, args.frame_dirs, seq_len=args.seq_len, img_size=224,
                       tasks_json=str(REPO / "tasks.json"), tasks=args.tasks, verbose=False, cache_mb=args.cache_mb,
                       iid_sampling=True, samples_per_shard=args.dyn_samples_per_shard)
        throughput("WMDataset (dynamics loader)", wm, args.dyn_batch_size, args.dyn_num_workers, collate_batch, args)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--frame_dirs", type=str, nargs="+", required=True, help="preprocessed shard directories")
    p.add_argument("--data_dirs", type=str, nargs="+", default=None,
                   help="raw-data directories paired with --frame_dirs (enables the WMDataset benchmark)")
    p.add_argument("--tasks", type=str, nargs="+", required=True)
    p.add_argument("--seq_len", type=int, default=24)
    p.add_argument("--prefetch_factor", type=int, default=4)
    p.add_argument("--warmup_batches", type=int, default=10)
    p.add_argument("--batches", type=int, default=60)
    # train_tokenizer.py defaults
    p.add_argument("--tok_batch_size", type=int, default=12)
    p.add_argument("--tok_num_workers", type=int, default=6)
    p.add_argument("--tok_samples_per_shard", type=int, default=16)
    p.add_argument("--shard_cache_size", type=int, default=12)
    # train_dynamics.py defaults
    p.add_argument("--dyn_batch_size", type=int, default=64)
    p.add_argument("--dyn_num_workers", type=int, default=4)
    p.add_argument("--dyn_samples_per_shard", type=int, default=24)
    p.add_argument("--cache_mb", type=int, default=13312)
    main(p.parse_args())
