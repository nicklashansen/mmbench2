"""Download released world-model checkpoints from the Hugging Face Hub.

Pulls the tokenizer + dynamics checkpoint pair for one (or all) model variant(s)
from the public MMBench2 checkpoint repo. Each variant lives in its own folder
containing `tokenizer.pt` and `dynamics.pt` (weights-only).

Variants:
    base            pretrained world model
    coverage_aware  coverage-aware finetuned world model
    combined        finetuned with all targeted data-collection sources

Usage:
    python download_checkpoints.py                          # combined (default)
    python download_checkpoints.py --variant coverage_aware
    python download_checkpoints.py --variant all
    python download_checkpoints.py --variant base --local_dir ./checkpoints
"""
import argparse
import os

from huggingface_hub import snapshot_download

REPO_ID = "nicklashansen/mmbench2-models"
VARIANTS = ["base", "coverage_aware", "combined"]


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=VARIANTS + ["all"], default="combined",
                   help="which model variant to download (default: combined)")
    p.add_argument("--local_dir", type=str, default="./checkpoints",
                   help="destination directory (default: ./checkpoints)")
    args = p.parse_args()

    allow_patterns = None if args.variant == "all" else [f"{args.variant}/*"]
    allow_patterns = None if args.variant == "all" else [f"{args.variant}/*", "config.json"]
    try:
        snapshot_download(
            repo_id=REPO_ID,
            repo_type="model",
            local_dir=args.local_dir,
            allow_patterns=allow_patterns,
            token=os.environ.get("HF_TOKEN"),  # public repo; token optional
        )
    except Exception as e:
        import sys
        print(
            f"\nERROR: could not download from {REPO_ID}:\n  {type(e).__name__}: {e}\n\n"
            f"Check your network connection and that the repo id is correct. If the\n"
            f"repository requires authentication, run `hf auth login` or set HF_TOKEN.\n",
            file=sys.stderr,
        )
        sys.exit(1)

    wanted = VARIANTS if args.variant == "all" else [args.variant]
    print(f"\nDownloaded to {os.path.abspath(args.local_dir)}")
    for v in wanted:
        tok = os.path.join(args.local_dir, v, "tokenizer.pt")
        dyn = os.path.join(args.local_dir, v, "dynamics.pt")
        ok = os.path.exists(tok) and os.path.exists(dyn)
        print(f"  {v:14s} tokenizer={'OK' if os.path.exists(tok) else 'MISSING'}"
              f"  dynamics={'OK' if os.path.exists(dyn) else 'MISSING'}"
              + ("" if ok else "   <-- check repo id / network"))
