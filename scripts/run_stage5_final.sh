#!/usr/bin/env bash
# Stage 5 FINAL (1024², 3 seeds): frontier 승자 구성 + headline 운영점 selector/block ablation
# 예상: seed당 ~5.5h (N=100), 3 seeds ~17h — tmux 필수
set -e
cd "$(dirname "$0")/.."; export PYTHONPATH=.
MAN=${MAN:-data/coco_manifest_1024.json}; OUT=${OUT:?set OUT}; N=${N:-100}
PC=${PC:-}; PCARG=(); [ -n "$PC" ] && PCARG=(--prompt-cache "$PC")
SEEDS=${SEEDS:-"0 1000 2000"}

base() { python -m samplers.cached_flux_fill --manifest $MAN --limit $N --steps 50 \
         "${PCARG[@]}" "$@"; }

ALL=()
for S in $SEEDS; do
  O=$OUT/seed$S; REF=$O/dense_s50
  # dense frontier (s50 = reference)
  for ST in 50 40 30 25 20 15; do
    base --out $O --method dense --steps $ST --seed-offset $S --tag dense_s$ST
  done
  # r=0 reuse arms (저예산 frontier)
  base --out $O --method reuse --cache-period 2 --seed-offset $S --tag reuse_c2
  base --out $O --method reuse --cache-period 2 --dense-tail 4 --seed-offset $S --tag reuse_c2_t4
  base --out $O --method reuse --cache-period 3 --dense-tail 4 --seed-offset $S --tag reuse_c3_t4
  # refresh arms (중·고예산 frontier)
  sp() { base --out $O --method cache_sparse --dense-tail 4 --seed-offset $S "$@"; }
  sp --selector mbd --cache-period 2 --ratio 0.15 --dual-sparse --kv-cache --tag mbd_c2_r015_t4_dualkv
  sp --selector mbd --cache-period 2 --ratio 0.3  --dual-sparse --kv-cache --tag mbd_c2_r03_t4_dualkv
  sp --selector mbd --cache-period 2 --ratio 0.5  --dual-sparse --kv-cache --tag mbd_c2_r05_t4_dualkv
  sp --selector mbd --cache-period 3 --ratio 0.3  --dual-sparse --kv-cache --tag mbd_c3_r03_t4_dualkv
  sp --selector mbd --cache-period 2 --ratio 0.3  --kv-cache --tag mbd_c2_r03_t4_kv
  # headline(c2_r03_dualkv) selector ablation + Stage 7 block
  for SEL in mask mbfd random oracle; do
    sp --selector $SEL --cache-period 2 --ratio 0.3 --dual-sparse --kv-cache \
       --tag ${SEL}_c2_r03_t4_dualkv
  done
  for B in 2 4; do
    sp --selector mbd --cache-period 2 --ratio 0.3 --block $B --dual-sparse --kv-cache \
       --tag mbd_c2_r03_t4_dualkv_b$B
  done

  RUNS=($O/dense_s40 $O/dense_s30 $O/dense_s25 $O/dense_s20 $O/dense_s15 \
        $O/reuse_c2 $O/reuse_c2_t4 $O/reuse_c3_t4 \
        $O/mbd_c2_r015_t4_dualkv $O/mbd_c2_r03_t4_dualkv $O/mbd_c2_r05_t4_dualkv \
        $O/mbd_c3_r03_t4_dualkv $O/mbd_c2_r03_t4_kv \
        $O/mask_c2_r03_t4_dualkv $O/mbfd_c2_r03_t4_dualkv \
        $O/random_c2_r03_t4_dualkv $O/oracle_c2_r03_t4_dualkv \
        $O/mbd_c2_r03_t4_dualkv_b2 $O/mbd_c2_r03_t4_dualkv_b4)
  for D in "${RUNS[@]}"; do
    python -m eval.region_metrics --run $D --ref $REF --manifest $MAN --out $D/metrics.json
  done
  ALL+=("${RUNS[@]}")
done

python -m eval.assemble --runs "${ALL[@]}" --out $OUT/table_final.md --csv $OUT/pareto_final.csv

# transformer-only latency 공식 수치 (전 lever 조합)
python -m eval.latency --resolution 1024 --ratios 0.15 0.3 0.5 --out $OUT/latency_final_base.json
python -m eval.latency --resolution 1024 --ratios 0.15 0.3 0.5 --kv-cache --out $OUT/latency_final_kv.json
python -m eval.latency --resolution 1024 --ratios 0.15 0.3 0.5 --dual-sparse --kv-cache --out $OUT/latency_final_dualkv.json
python -m eval.latency --resolution 512  --ratios 0.15 0.3 0.5 --dual-sparse --kv-cache --out $OUT/latency_final_512_dualkv.json
