#!/usr/bin/env bash
# Stage 6 (Fix 8): teacher dump -> router 학습 -> selector-only 평가 (mbfd_draft)
set -e
cd "$(dirname "$0")/.."; export PYTHONPATH=.
MAN=${MAN:-data/coco_manifest.json}
TEACH=${TEACH:-/mnt/HDD_12TB/bam_ki/flux_fill/router_teacher}
CKPT=${CKPT:-/mnt/HDD_12TB/bam_ki/flux_fill/router_ckpt}
OUT=${OUT:-out}; N=${N:-100}
PC=${PC:-}; PCARG=(); [ -n "$PC" ] && PCARG=(--prompt-cache "$PC")

# 전제조건: Stage 3의 reference와 baseline이 이미 있어야 함 (리뷰 2차 반영)
test -d "$OUT/dense_s50" || { echo "Missing $OUT/dense_s50: run Stage 3 first"; exit 1; }
test -d "$OUT/mbfd_c3_r03" || { echo "Missing $OUT/mbfd_c3_r03: run Stage 3 first"; exit 1; }

# 1) teacher trajectories (resumable; dense 50-step, 200 images)
python -m samplers.dump_router_teacher --manifest $MAN --out $TEACH \
  --steps 50 --limit 200 "${PCARG[@]}"

# 2) router 학습 (EMA + atomic rolling ckpt + resume)
python -m training.train_router --teacher $TEACH --out $CKPT \
  --steps 100000 --tau 1e-4 --resume

# 3) 평가: mbfd vs mbfd_draft, 동일 c/r (draft 호출 비용은 wall에 포함됨)
LAST=$(ls $CKPT/router_*.pt | tail -1)
python -m samplers.cached_flux_fill --manifest $MAN --out $OUT --method cache_sparse \
  --selector mbfd_draft --draft-ckpt "$LAST" --steps 50 --cache-period 3 --ratio 0.3 \
  --limit $N --tag mbfd_draft_c3_r03 "${PCARG[@]}"
python -m eval.region_metrics --run $OUT/mbfd_draft_c3_r03 --ref $OUT/dense_s50 \
  --manifest $MAN --out $OUT/mbfd_draft_c3_r03/metrics.json
python -m eval.assemble --runs $OUT/mbfd_c3_r03 $OUT/mbfd_draft_c3_r03 \
  --out $OUT/table_draft.md
