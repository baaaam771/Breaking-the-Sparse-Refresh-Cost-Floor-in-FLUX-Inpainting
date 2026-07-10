#!/usr/bin/env bash
# DataLoader worker sweep (coarse -> 최적 근처를 다시 촘촘히)
set -e
cd "$(dirname "$0")/.."; export PYTHONPATH=.
MAN=${MAN:-data/coco_manifest.json}
for w in 0 2 4 8 12 16 20; do
  python -m tools.bench_workers --manifest $MAN --batch 4 --workers $w --steps 100 --pin-memory
done
echo "--- prefetch_factor sweep @ workers=8 ---"
for pf in 2 4 6 8; do
  python -m tools.bench_workers --manifest $MAN --batch 4 --workers 8 --prefetch-factor $pf --steps 100 --pin-memory
done
