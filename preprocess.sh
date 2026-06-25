#!/usr/bin/env bash
# Preprocess raw MMBench2 partitions into the sharded format used for training.
# Each partition <name> is read from ./data/<name> and written to
# ./data/<name>-shards (paths are relative to src/, matching the README).
#
# Usage (run from the repo root):
#   bash preprocess.sh                # preprocess every partition under ./data
#   bash preprocess.sh expert val     # preprocess only the named partitions
#
# Set DATA_DIR to point at a different dataset root (default: ./data).
# For additional options (--target_size, --shard_size, --num_workers, ...),
# call src/preprocess_dataset.py directly.
set -euo pipefail
shopt -s nullglob

cd "$(dirname "$0")/src"
DATA_DIR="${DATA_DIR:-./data}"

# Resolve the list of partitions to preprocess.
if [ "$#" -gt 0 ]; then
    partitions=("$@")
else
    partitions=()
    for d in "$DATA_DIR"/*/; do
        name="$(basename "$d")"
        case "$name" in
            *-shards) continue ;;  # skip already-preprocessed outputs
        esac
        partitions+=("$name")
    done
fi

if [ "${#partitions[@]}" -eq 0 ]; then
    echo "No partitions found under $DATA_DIR. Download the dataset first (see README)."
    exit 1
fi

for name in "${partitions[@]}"; do
    src="$DATA_DIR/$name"
    out="$DATA_DIR/$name-shards"
    if [ ! -d "$src" ]; then
        echo "[skip] $src not found"
        continue
    fi
    echo "=== preprocessing '$name' -> '$name-shards' ==="
    python preprocess_dataset.py --filedir "$src" --outdir "$out"
done

echo "Preprocessing complete."
