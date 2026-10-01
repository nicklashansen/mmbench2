# preprocess_dataset.py
import json
import multiprocessing as mp
from pathlib import Path

import torch
from torchvision.io import read_image
import torch.nn.functional as F

import argparse

from shard_io import CODECS, DEFAULT_CHUNK_FRAMES, check_codec, shard_filename, write_shard
from task_set import TASK_SET, UNSEEN_TASK_SET


def safe_save_frames(frames: torch.Tensor, out_path: Path, codec: str, chunk_frames: int) -> bool:
    """
    Save a shard via shard_io.write_shard (temp file + atomic rename).
    Returns True on success, False on failure.
    """
    try:
        write_shard(frames, out_path, codec=codec, chunk_frames=chunk_frames)
        print(f"  [OK] Saved shard with {frames.shape[0]} frames to {out_path}")
        return True
    except Exception as e:
        print(f"  [WARN] Failed saving shard {out_path}: {e}")
        return False


def process_task(args_tuple):
    task, filedir, outdir, target_size, shard_size, codec, chunk_frames = args_tuple
    task_out_dir = Path(outdir) / task
    index_path = task_out_dir / f"{task}_index.json"

    # skip if already done (index file is written last, so its presence means success)
    if index_path.exists():
        print(f"[{task}] already processed, skipping.")
        return 0

    task_out_dir.mkdir(parents=True, exist_ok=True)

    save_failures = 0   # any failed shard save leaves the task index unwritten (resumable)
    shard_frames = []   # list of (N_i, 3, target_size, target_size) uint8
    total_frames = 0    # running count of buffered frames
    shard_idx = 0
    shard_meta = {}     # shard filename -> num_frames, written as index at the end

    i = 0
    while True:
        png_path = Path(filedir) / f"{task}-{i}.png"
        if not png_path.exists():
            break

        print(f"[{task}] reading {png_path}")
        try:
            frames = read_image(str(png_path))  # (3, 224, 224 * num_frames), uint8
        except Exception as e:
            print(f"  [WARN] Skipping {png_path} (read error): {e}")
            i += 1
            continue

        C, H, W_total = frames.shape
        if H != 224 or W_total % 224 != 0:
            print(f"  [WARN] Skipping {png_path}, unexpected shape {frames.shape}")
            i += 1
            continue

        num_frames = W_total // 224
        if num_frames == 0:
            print(f"  [WARN] Skipping {png_path}, no frames detected")
            i += 1
            continue

        # Split horizontally: (num_frames, 3, 224, 224)
        frames = frames.view(C, 224, num_frames, 224)      # (3, 224, N, 224)
        frames = frames.permute(2, 0, 1, 3)                # (N, 3, 224, 224)

        # Downsample if needed; skip entirely when target matches source to avoid
        # any float round-trip artifacts in the saved frames.
        if target_size == 224:
            frames_u8 = frames.contiguous()
        else:
            frames_f = frames.to(torch.float32) / 255.0
            frames_f = F.interpolate(
                frames_f,
                size=(target_size, target_size),
                mode="bilinear",
                align_corners=False,
            )
            frames_u8 = (frames_f.clamp(0.0, 1.0) * 255.0).to(torch.uint8)

        shard_frames.append(frames_u8)
        total_frames += frames_u8.shape[0]

        # Flush complete shards; maintain running total to avoid re-summing the list
        while total_frames >= shard_size:
            concat = torch.cat(shard_frames, dim=0)
            to_save, remainder = concat[:shard_size], concat[shard_size:]
            shard_name = shard_filename(task, shard_idx, codec)
            out_path = task_out_dir / shard_name

            print(f"[{task}] saving shard {shard_idx} with {to_save.shape[0]} frames to {out_path}")
            ok = safe_save_frames(to_save, out_path, codec, chunk_frames)
            if ok:
                shard_meta[shard_name] = int(to_save.shape[0])
            else:
                save_failures += 1
                print(f"  [WARN] Continuing after failed save of {out_path} (check disk space/FS).")

            shard_frames = [remainder] if remainder.shape[0] > 0 else []
            total_frames = int(remainder.shape[0])
            shard_idx += 1

        i += 1

    # Flush remainder at the end
    if shard_frames:
        concat = torch.cat(shard_frames, dim=0)
        shard_name = shard_filename(task, shard_idx, codec)
        out_path = task_out_dir / shard_name
        print(f"[{task}] saving final shard {shard_idx} with {concat.shape[0]} frames to {out_path}")
        ok = safe_save_frames(concat, out_path, codec, chunk_frames)
        if ok:
            shard_meta[shard_name] = int(concat.shape[0])
        else:
            save_failures += 1

    # Write metadata index last — its presence signals that the task is complete.
    # If any shard save failed, do NOT write it: an index over partial data marks the
    # task done and silently truncates it (and skip-if-done reruns would never repair it).
    if save_failures > 0:
        print(f"[{task}] ERROR: {save_failures} shard saves failed; NOT writing index (rerun to resume)")
        return 1
    with open(index_path, "w") as f:
        json.dump(shard_meta, f, indent=2)
    print(f"[{task}] wrote index with {len(shard_meta)} shards to {index_path}")
    return 0


TASK_SET_PRESETS = {
    "trained": list(TASK_SET),                       # 200 training tasks
    "unseen":  list(UNSEEN_TASK_SET),                # 10 held-out tasks
    "all":     list(TASK_SET) + list(UNSEEN_TASK_SET),  # 210 total
}


def main(args):
    # Fail before reading any data: write_shard would otherwise reject every shard only
    # after each task's PNGs have been decoded.
    try:
        check_codec(args.codec, args.chunk_frames, args.target_size)
    except (ValueError, ImportError) as e:
        raise SystemExit(f"error: {e}")
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    if args.tasks is None:
        tasks = TASK_SET_PRESETS[args.task_set]
    else:
        tasks = list(args.tasks)
    task_args = [(task, args.filedir, args.outdir, args.target_size, args.shard_size,
                  args.codec, args.chunk_frames) for task in tasks]
    print(f"Processing {len(tasks)} tasks with {args.num_workers} parallel workers "
          f"(codec={args.codec}, chunk_frames={args.chunk_frames})")
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=args.num_workers) as pool:
        results = pool.map(process_task, task_args)
    n_failed = sum(1 for r in results if r)
    if n_failed:
        raise SystemExit(f"{n_failed}/{len(results)} tasks had failed shard saves; indexes withheld — rerun to resume.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--filedir", type=str, default="./data/val")
    p.add_argument("--outdir", type=str, default="./data/val-shards")
    p.add_argument("--target_size", type=int, default=224)
    p.add_argument("--shard_size", type=int, default=4096)
    p.add_argument("--codec", type=str, default="webp", choices=CODECS,
                   help="Shard format: 'webp' = chunked lossless WebP (.chunks, roughly 10-30x "
                        "smaller than raw, decoded per window at load time, slower to "
                        "preprocess); 'raw' = uint8 torch.save shards (.pt). Both are "
                        "bit-identical.")
    p.add_argument("--chunk_frames", type=int, default=DEFAULT_CHUNK_FRAMES,
                   help="Frames per encoded chunk for --codec webp. chunk_frames * target_size "
                        "must not exceed 16383 px (at most 73 at 224 px).")
    p.add_argument("--num_workers", type=int, default=16)
    p.add_argument("--tasks", type=str, nargs="+", default=None,
                   help="Explicit task list to preprocess. Overrides --task_set.")
    p.add_argument("--task_set", type=str, default="trained",
                   choices=sorted(TASK_SET_PRESETS),
                   help="Preset task list (used when --tasks is not given). "
                        "'trained'=TASK_SET (200), 'unseen'=UNSEEN_TASK_SET (10), "
                        "'all'=union (210).")
    main(p.parse_args())
