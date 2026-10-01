# check_real_data.py
"""
End-to-end check of both shard formats on real MMBench2 data: preprocess the given
partitions/tasks with `--codec webp` and `--codec raw`, then verify that

  * both formats are bit-identical to the source PNG strips (decoded independently with PIL),
  * ShardedFrameDataset and WMDataset return identical samples for both formats, and
    WMDataset windows equal the source frames, and
  * multi-worker DataLoaders (fork and spawn) return the same batches as in-process reads.

It needs the raw files `<data_dir>/<partition>/<task>-*.png` and `<task>.pt`; a single
small task is enough. Shards are written to --work_dir (the raw format needs ~150 KB per
frame) and removed afterwards unless --keep is given. CPU only.

Usage (from the repo root):
    python tests/check_real_data.py --data_dir src/data --partitions val --tasks ms-pick-cube
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

REPO = Path(__file__).resolve().parents[1]
SRC = str(REPO / "src")
sys.path.insert(0, SRC)
os.environ["PYTHONPATH"] = SRC + os.pathsep + os.environ.get("PYTHONPATH", "")   # for spawned workers

from shard_io import list_shards, load_shard  # noqa: E402
from sharded_frame_dataset import ShardedFrameDataset  # noqa: E402
from wm_dataset import WMDataset, collate_batch  # noqa: E402

failures = []


def check(ok, what):
    print(f"  [{'ok' if ok else 'FAIL'}] {what}", flush=True)
    if not ok:
        failures.append(what)


def png_frames(part_dir, task):
    """Source frames (N, 3, 224, 224) uint8, decoded with PIL rather than torchvision."""
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    out, i = [], 0
    while os.path.exists(f"{part_dir}/{task}-{i}.png"):
        a = np.asarray(Image.open(f"{part_dir}/{task}-{i}.png").convert("RGB"))
        n = a.shape[1] // 224
        out.append(torch.from_numpy(a.reshape(224, n, 224, 3).transpose(1, 3, 0, 2).copy()))
        i += 1
    return torch.cat(out)


def same_batch(a, b):
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)
    return torch.equal(a, b)


def loader_matches(ds, ref, idxs, collate, ctx, batch_size=8):
    loader = DataLoader(Subset(ds, idxs), batch_size=batch_size, shuffle=False, num_workers=2,
                        collate_fn=collate, multiprocessing_context=ctx)
    stack = collate or torch.stack
    return all(same_batch(batch, stack([ref[i] for i in idxs[bi * batch_size:(bi + 1) * batch_size]]))
               for bi, batch in enumerate(loader))


def main(args):
    work = Path(args.work_dir or tempfile.mkdtemp(prefix="mmbench2-shard-check-"))
    print(f"work dir: {work}")
    try:
        for part in args.partitions:
            part_dir = os.path.join(args.data_dir, part)
            tasks = [t for t in args.tasks if os.path.exists(f"{part_dir}/{t}-0.png")]
            if not tasks:
                print(f"== {part}: none of {args.tasks} found under {part_dir}, skipping")
                continue
            out = {c: str(work / f"{part}-shards-{c}") for c in ("webp", "raw")}
            for codec in ("webp", "raw"):
                t0 = time.time()
                cmd = [sys.executable, "preprocess_dataset.py", "--filedir", os.path.abspath(part_dir), "--outdir", out[codec],
                       "--codec", codec, "--tasks", *tasks, "--num_workers", str(min(len(tasks), args.num_workers))]
                r = subprocess.run(cmd, cwd=SRC, capture_output=True, text=True)
                print(f"== {part}: preprocess --codec {codec}: exit {r.returncode}, {time.time() - t0:.1f}s")
                if r.returncode != 0:
                    print(r.stdout[-2000:], r.stderr[-2000:])
                    failures.append(f"{part}: preprocess --codec {codec}")
            for task in tasks:
                ref = png_frames(part_dir, task)
                wp, rp = list_shards(f"{out['webp']}/{task}"), list_shards(f"{out['raw']}/{task}")
                w = torch.cat([load_shard(p)[:] for p in wp])
                r = torch.cat([load_shard(p) for p in rp])
                n = ref.shape[0]
                sz = {k: sum(os.path.getsize(p) for p in v) / n / 1e3 for k, v in
                      (("png", glob.glob(f"{part_dir}/{task}-*.png")), ("webp", wp), ("raw", rp))}
                print(f"-- {part}/{task}: {n} frames, {len(wp)} shards | KB/frame: png {sz['png']:.1f}, "
                      f"webp {sz['webp']:.1f}, raw {sz['raw']:.1f} ({sz['raw'] / sz['webp']:.1f}x)")
                check(torch.equal(w, ref), "chunked shards == source PNGs")
                check(torch.equal(r, ref), "raw shards == source PNGs")
                del w, r

                # ShardedFrameDataset: sequential windows, both formats, in-process and through workers
                sk = dict(tasks=[task], seq_len=args.seq_len, iid_sampling=False, verbose=False)
                sw, sr = ShardedFrameDataset(out["webp"], **sk), ShardedFrameDataset(out["raw"], **sk)
                idxs = list(range(0, len(sw), max(1, len(sw) // args.max_items)))
                check(len(sw) == len(sr) and all(torch.equal(sw[i], sr[i]) for i in idxs),
                      f"ShardedFrameDataset chunked == raw ({len(idxs)} windows)")
                for ctx in ("fork", "spawn"):
                    check(loader_matches(sw, sr, idxs[:64], None, ctx), f"ShardedFrameDataset 2-worker loader ({ctx})")

                # WMDataset: windows must equal the source frames (single source, so starts index `ref` directly)
                wk = dict(seq_len=args.seq_len, img_size=224, tasks_json=str(REPO / "tasks.json"), tasks=[task], verbose=False)
                ww, wr = WMDataset(part_dir, out["webp"], **wk), WMDataset(part_dir, out["raw"], **wk)
                idxs = list(range(0, len(ww), max(1, len(ww) // args.max_items)))
                ok = len(ww) == len(wr)
                for i in idxs:
                    _, start = ww._lookup(i)
                    a, b = ww[i], wr[i]
                    ok = ok and torch.equal(a["obs"], ref[start:start + args.seq_len + 1]) and same_batch(a, b)
                check(ok, f"WMDataset chunked == raw == source windows ({len(idxs)} windows)")
                for ctx in ("fork", "spawn"):
                    check(loader_matches(ww, wr, idxs[:64], collate_batch, ctx), f"WMDataset 2-worker loader ({ctx})")
    finally:
        if not args.keep:
            shutil.rmtree(work, ignore_errors=True)
    if failures:
        raise SystemExit(f"{len(failures)} check(s) FAILED: {failures}")
    print("all checks passed")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir", type=str, default=str(REPO / "src" / "data"), help="dataset root holding the raw partitions")
    p.add_argument("--partitions", type=str, nargs="+", default=["val"])
    p.add_argument("--tasks", type=str, nargs="+", default=["ms-pick-cube"])
    p.add_argument("--seq_len", type=int, default=24)
    p.add_argument("--max_items", type=int, default=200, help="windows checked per dataset (evenly spaced)")
    p.add_argument("--num_workers", type=int, default=4, help="preprocessing workers")
    p.add_argument("--work_dir", type=str, default=None, help="where to write shards (default: a temp dir)")
    p.add_argument("--keep", action="store_true", help="keep the shards in --work_dir")
    main(p.parse_args())
