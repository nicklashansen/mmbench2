import os
from argparse import ArgumentParser

from huggingface_hub import snapshot_download


# Download the MMBench2 dataset from the Hugging Face Hub.
# Authenticate via the HF_TOKEN env var or a cached `hf auth login` if required.
DEFAULT_REPO_ID = "nicklashansen/mmbench2"


if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument("--local_dir", type=str, default="./data")
    p.add_argument("--repo_id", type=str, default=DEFAULT_REPO_ID,
                   help=f"HF dataset repo id (default: {DEFAULT_REPO_ID}).")
    p.add_argument("--subset", type=str, nargs="*", default=None,
                   help="optional subset(s) to download, e.g. 'val' or 'expert'")
    args = p.parse_args()

    allow_patterns = None
    if args.subset:
        allow_patterns = [f"{s.rstrip('/')}/*" for s in args.subset]

    snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        local_dir=args.local_dir,
        token=os.environ.get("HF_TOKEN"),
        allow_patterns=allow_patterns,
    )
