#!/usr/bin/env bash
# Stage 3-5: mapping 검증 -> heterogeneity(Q1, two-factor) -> PoC -> selector ablation
set -e
cd "$(dirname "$0")/.."; export PYTHONPATH=.
MAN=${MAN:-data/coco_manifest.json}; OUT=${OUT:-out}; N=${N:-100}
PC=${PC:-}; PCARG=(); [ -n "$PC" ] && PCARG=(--prompt-cache "$PC")

python -m eval.token_overlay --manifest $MAN --index 0 --out $OUT/overlay.png

# baselines 먼저 (heterogeneity Factor B(ii)가 dense sweep metrics를 사용)
python -m samplers.cached_flux_fill --manifest $MAN --out $OUT --method dense --steps 50 --limit $N --tag dense_s50 "${PCARG[@]}"
for S in 40 30 25 20; do
  python -m samplers.cached_flux_fill --manifest $MAN --out $OUT --method dense --steps $S --limit $N --tag dense_s$S "${PCARG[@]}"
done
for D in $OUT/dense_s40 $OUT/dense_s30 $OUT/dense_s25 $OUT/dense_s20; do
  python -m eval.region_metrics --run $D --ref $OUT/dense_s50 --manifest $MAN --out $D/metrics.json
done

# Q1: two-factor heterogeneity
python -m samplers.cached_flux_fill --manifest $MAN --out $OUT --method hetero \
  --steps 50 --limit 20 --tag hetero_s50 "${PCARG[@]}"
python -m eval.heterogeneity --run $OUT/hetero_s50 \
  --dense-reduced $OUT/dense_s20/metrics.json --out $OUT/hetero_report.json

for C in 2 3 5; do
  python -m samplers.cached_flux_fill --manifest $MAN --out $OUT --method reuse --steps 50 --cache-period $C --limit $N --tag reuse_c$C "${PCARG[@]}"
done

# Stage 4 PoC + Stage 5 ablation (c=3, r=0.3)
for SEL in mask mask_boundary mask_delta mask_frequency mbd mbfd random oracle; do
  python -m samplers.cached_flux_fill --manifest $MAN --out $OUT --method cache_sparse \
    --selector $SEL --steps 50 --cache-period 3 --ratio 0.3 --limit $N --tag ${SEL}_c3_r03 "${PCARG[@]}"
done
for D in $OUT/*_c3_r03 $OUT/reuse_c*; do
  python -m eval.region_metrics --run $D --ref $OUT/dense_s50 --manifest $MAN --out $D/metrics.json
done
python -m eval.assemble --runs $OUT/dense_s30 $OUT/dense_s20 $OUT/reuse_c3 \
  $OUT/mask_c3_r03 $OUT/random_c3_r03 $OUT/mbd_c3_r03 $OUT/mbfd_c3_r03 $OUT/oracle_c3_r03 \
  --out $OUT/table_main.md --csv $OUT/pareto_stage5.csv
