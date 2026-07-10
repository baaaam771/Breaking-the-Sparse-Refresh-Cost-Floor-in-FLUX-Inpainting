#!/usr/bin/env bash
# Stage 3d (Lever B): anchor K/V cache — 품질(temb-staleness 비용) vs wall(절감) 측정
# 전제: Stage 3c와 동일 $OUT (1024², dense_s50 reference 재사용)
set -e
cd "$(dirname "$0")/.."; export PYTHONPATH=.
MAN=${MAN:-data/coco_manifest_1024.json}; OUT=${OUT:?set OUT}; N=${N:-100}
PC=${PC:-}; PCARG=(); [ -n "$PC" ] && PCARG=(--prompt-cache "$PC")
IMG=${IMG:-sample.png}; MSK=${MSK:-sample_mask.png}; PROMPT=${PROMPT:-"a photo"}
test -d "$OUT/dense_s50" || { echo "Missing $OUT/dense_s50: run Stage 3c first"; exit 1; }

# Gate: kv 경로도 anchor-step exact여야 함 (실모델)
python tests/test_cache_exactness.py --image "$IMG" --mask "$MSK" --prompt "$PROMPT" \
  --step-index 0 --ratios 0.3 --kv-cache

# latency: kv on/off
python -m eval.latency --resolution 1024 --ratios 0.15 0.3 --kv-cache \
  --out $OUT/latency_1024_kv.json

# 품질: Stage 3c의 세 운영점을 kv로 재실행 -> staleness 비용 격리
run() { python -m samplers.cached_flux_fill --manifest $MAN --out $OUT --limit $N \
        --steps 50 --method cache_sparse --selector mbd --dense-tail 4 --kv-cache \
        "${PCARG[@]}" "$@"; }
run --cache-period 3 --ratio 0.15 --tag mbd_c3_r015_t4_kv
run --cache-period 3 --ratio 0.3  --tag mbd_c3_r03_t4_kv
run --cache-period 2 --ratio 0.3  --tag mbd_c2_r03_t4_kv

for D in $OUT/mbd_c*_kv; do
  python -m eval.region_metrics --run $D --ref $OUT/dense_s50 --manifest $MAN --out $D/metrics.json
done
python -m eval.assemble --runs $OUT/mbd_c3_r015_t4 $OUT/mbd_c3_r015_t4_kv \
  $OUT/mbd_c3_r03_t4 $OUT/mbd_c3_r03_t4_kv $OUT/mbd_c2_r03_t4 $OUT/mbd_c2_r03_t4_kv \
  --out $OUT/table_kv.md --csv $OUT/pareto_kv.csv
