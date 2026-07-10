#!/usr/bin/env bash
# Gate ladder (Fix 2): B0 -> B2. 실패 시 어떤 실험도 진행 금지 (DACE 규칙).
#   B0: 실제 single block official vs manual — 동일 shape이므로 합성 입력으로 충분
#   B1: full transformer official vs runner  -> run_stage1_dense.sh (--mode gate_a)
#   B2: fresh-cache sparse vs runner dense   — 실제 샘플 입력 + 상대오차 기준
#       (합성 randn 입력은 활성값 폭발로 shape 의존 bf16 GEMM 차이가 절대오차로
#        증폭됨 — tools/bisect_gate_b2 진단 결과. 상대오차가 올바른 판정 기준)
set -e
cd "$(dirname "$0")/.."; export PYTHONPATH=.
IMG=${1:?usage: run_stage2_cache.sh image.png mask.png "prompt"}
MSK=$2; PROMPT=$3

python tests/test_single_block_equivalence.py --resolution 512            # B0 bf16
python tests/test_single_block_equivalence.py --resolution 512 --fp32     # B0 fp32
# B2: t=1.0 (step 0)과 중간 step 둘 다, bf16 상대오차 + fp32 확인
python tests/test_cache_exactness.py --image "$IMG" --mask "$MSK" --prompt "$PROMPT" \
  --step-index 0  --ratios 0.1 0.3 0.7
python tests/test_cache_exactness.py --image "$IMG" --mask "$MSK" --prompt "$PROMPT" \
  --step-index 10 --ratios 0.3
python tests/test_cache_exactness.py --image "$IMG" --mask "$MSK" --prompt "$PROMPT" \
  --step-index 0  --ratios 0.3 --fp32
