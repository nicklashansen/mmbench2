#!/usr/bin/env bash
# Launch the interactive world-model web interface using downloaded checkpoints.
#
# Seeds rollouts from the live simulators, so no dataset is required —
# only the checkpoints (auto-downloaded from Hugging Face on first run) and a
# CUDA GPU. Open http://localhost:7860 once it starts.
#
# Usage:
#   ./run_interactive.sh                              # combined (default)
#   ./run_interactive.sh coverage_aware               # a different variant (base|coverage_aware|combined)
#   ./run_interactive.sh combined --task walker-run --port 7861   # extra args pass through to interactive.py
#
# Env vars:
#   CKPT_DIR   checkpoint directory (default: ./checkpoints)
set -euo pipefail
cd "$(dirname "$0")"                 # run from src/ (scripts use flat imports)

VARIANT="combined"
case "${1:-}" in
    base|coverage_aware|combined) VARIANT="$1"; shift ;;
esac

CKPT_DIR="${CKPT_DIR:-./checkpoints}"
TOK="$CKPT_DIR/$VARIANT/tokenizer.pt"
DYN="$CKPT_DIR/$VARIANT/dynamics.pt"

if [[ ! -f "$TOK" || ! -f "$DYN" ]]; then
    echo "Checkpoints for '$VARIANT' not found under $CKPT_DIR — downloading from Hugging Face..."
    python download_checkpoints.py --variant "$VARIANT" --local_dir "$CKPT_DIR"
fi

echo "Launching interactive interface with '$VARIANT' checkpoints (live-env seeding); open http://localhost:7860"
exec python interactive.py \
    --tokenizer_ckpt "$TOK" \
    --dynamics_ckpt "$DYN" \
    "$@"
