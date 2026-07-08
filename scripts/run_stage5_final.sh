#!/usr/bin/env bash
# Stage 7-8 (Fix 9): 3 seeds x (dense sweep / reuse / selector suite / block sweep)
# -> region metrics -> seed-aggregated main table + Pareto CSV + latency
set -e
cd "$(dirname "$0")/.."; export PYTHONPATH=.
MAN=${MAN:-data/coco_manifest.json}; OUT=${OUT:-out_final}; N=${N:-500}
PC=${PC:-}; PCARG=(); [ -n "$PC" ] && PCARG=(--prompt-cache "$PC")
SEEDS=${SEEDS:-"0 1000 2000"}          # latent_seed offsets

run() { python -m samplers.cached_flux_fill --manifest $MAN --limit $N "${PCARG[@]}" "$@"; }
met() { python -m eval.region_metrics --run "$1" --ref "$2" --manifest $MAN --out "$1/metrics.json"; }

ALL_RUNS=()
for S in $SEEDS; do
  O=$OUT/seed$S; REF=$O/dense_s50
  # dense reference + reduced-step baselines
  run --out $O --method dense --steps 50 --seed-offset $S --tag dense_s50
  for ST in 40 30 25 20 15; do run --out $O --method dense --steps $ST --seed-offset $S --tag dense_s$ST; done
  # r=0 anchored reuse
  run --out $O --method reuse --steps 50 --cache-period 3 --seed-offset $S --tag reuse_c3
  # selector suite (token-level)
  for SEL in mask mbd mbfd random oracle; do
    run --out $O --method cache_sparse --selector $SEL --steps 50 --cache-period 3 \
        --ratio 0.3 --seed-offset $S --tag ${SEL}_c3_r03
  done
  # Stage 7: structured blocks (true block Top-K; r_actual은 run.json에 기록됨)
  for B in 2 4; do
    run --out $O --method cache_sparse --selector mbfd --steps 50 --cache-period 3 \
        --ratio 0.3 --block $B --seed-offset $S --tag mbfd_c3_r03_b$B
  done
  for D in $O/dense_s40 $O/dense_s30 $O/dense_s25 $O/dense_s20 $O/dense_s15 \
           $O/reuse_c3 $O/*_c3_r03 $O/mbfd_c3_r03_b*; do
    met "$D" "$REF"
  done
  ALL_RUNS+=($O/dense_s40 $O/dense_s30 $O/dense_s25 $O/dense_s20 $O/dense_s15 $O/reuse_c3 $O/mask_c3_r03 $O/mbd_c3_r03 \
             $O/mbfd_c3_r03 $O/random_c3_r03 $O/oracle_c3_r03 \
             $O/mbfd_c3_r03_b2 $O/mbfd_c3_r03_b4)
done

# seed-aggregated table (mean±std) + Pareto CSV
python -m eval.assemble --runs "${ALL_RUNS[@]}" --out $OUT/table_main.md --csv $OUT/pareto.csv

# transformer-only latency profile (측정 latency vs 해석용 MAC ratio 분리 보고)
python -m eval.latency --resolution 512 --ratios 0.1 0.3 0.5 0.7 --out $OUT/latency_512.json
python -m eval.latency --resolution 1024 --ratios 0.1 0.3 0.5 --out $OUT/latency_1024.json
