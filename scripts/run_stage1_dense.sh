#!/usr/bin/env bash
# Stage 1: custom dense loop equivalence (Gate A) + scheduler equivalence
set -e
cd "$(dirname "$0")/.."; export PYTHONPATH=.
IMG=$1; MSK=$2; PROMPT=$3
python tests/test_scheduler_equivalence.py --resolution 512 --steps 50
python -m samplers.dense_flux_fill --mode gate_a --image "$IMG" --mask "$MSK" \
  --prompt "$PROMPT" --out out/stage1 --seed 0 --steps 50
