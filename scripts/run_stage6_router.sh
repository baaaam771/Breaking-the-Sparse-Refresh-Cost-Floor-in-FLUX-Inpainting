#!/usr/bin/env bash
# Stage 6 (headline 구성): learned router가 mbd–oracle gap(0.0051)을 얼마나 회수하나
#   teacher dump(1024, 200장, ~10.5GB) -> quantile-label 학습 -> headline 운영점 평가
# 판정: mbd_draft가 gap의 >=30% (mask-LPIPS <= 0.0285) 회수 시 본문 채택,
#       미달 시 bounded-negative로 appendix (양쪽 다 논문에 유효)
set -e
cd "$(dirname "$0")/.."; export PYTHONPATH=.
MAN=${MAN:-data/coco_manifest_1024.json}
TEACH=${TEACH:-/mnt/HDD_12TB/bam_ki/flux_fill/router_teacher_1024}
CKPT=${CKPT:-/mnt/HDD_12TB/bam_ki/flux_fill/router_ckpt_1024}
OUT=${OUT:?set OUT to the final dir}; N=${N:-100}
PC=${PC:-}; PCARG=(); [ -n "$PC" ] && PCARG=(--prompt-cache "$PC")
SEED_DIR=$OUT/seed0
test -d "$SEED_DIR/dense_s50" || { echo "Missing $SEED_DIR/dense_s50: run Stage 5 first"; exit 1; }
test -d "$SEED_DIR/mbd_c2_r03_t4_dualkv" || { echo "Missing headline run"; exit 1; }

# 1) teacher trajectories — manifest 뒷부분 200장 사용해 평가 100장과 겹침 방지
#    (manifest는 500장: 평가가 앞 100장을 쓰므로 dump는 뒤에서 200장)
python - << 'PY'
import json
m = json.load(open("data/coco_manifest_1024.json"))
m2 = dict(m); m2["items"] = m["items"][300:500]
json.dump(m2, open("data/coco_manifest_1024_teacher.json", "w"))
print("teacher manifest:", len(m2["items"]), "items (indices 300-499)")
PY
python -m samplers.dump_router_teacher --manifest data/coco_manifest_1024_teacher.json \
  --out $TEACH --steps 50 --limit 200 "${PCARG[@]}"

# 2) 학습 (quantile top-30% 라벨 = 선택 소비 방식과 정합; EMA+resume)
python -m training.train_router --teacher $TEACH --out $CKPT \
  --steps 60000 --bs 8 --label-mode quantile --quantile 0.7 \
  --cache-periods 2 3 --eval-every 2000 --save-every 5000 --resume

# 3) headline 운영점 평가: mbd_draft (+ 진단용 draft 기여 확인)
LAST=$(ls $CKPT/router_*.pt | tail -1); echo "using $LAST"
python -m samplers.cached_flux_fill --manifest $MAN --out $SEED_DIR --limit $N \
  --steps 50 --method cache_sparse --selector mbd_draft --draft-ckpt "$LAST" \
  --cache-period 2 --ratio 0.3 --dense-tail 4 --dual-sparse --kv-cache \
  "${PCARG[@]}" --tag mbd_draft_c2_r03_t4_dualkv
python -m eval.region_metrics --run $SEED_DIR/mbd_draft_c2_r03_t4_dualkv \
  --ref $SEED_DIR/dense_s50 --manifest $MAN \
  --out $SEED_DIR/mbd_draft_c2_r03_t4_dualkv/metrics.json
python -m eval.assemble --runs $SEED_DIR/mbd_c2_r03_t4_dualkv \
  $SEED_DIR/oracle_c2_r03_t4_dualkv $SEED_DIR/mbd_draft_c2_r03_t4_dualkv \
  --out $OUT/table_router.md
