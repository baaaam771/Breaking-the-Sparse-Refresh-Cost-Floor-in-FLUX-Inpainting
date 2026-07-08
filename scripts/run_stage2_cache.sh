#!/usr/bin/env bash
# Gate ladder (Fix 2): B0 -> B2. 실패 시 어떤 실험도 진행 금지 (DACE 규칙).
#   B0: 실제 single block official vs manual (원인 분리용)
#   B1: full transformer official vs runner  -> run_stage1_dense.sh (--mode gate_a)
#   B2: fresh-cache sparse vs runner dense
set -e
cd "$(dirname "$0")/.."; export PYTHONPATH=.
python tests/test_single_block_equivalence.py --resolution 512            # B0 bf16
python tests/test_single_block_equivalence.py --resolution 512 --fp32     # B0 fp32 (0 오차)
python tests/test_cache_exactness.py --resolution 512 --ratios 0.1 0.3 0.7   # B2 bf16
python tests/test_cache_exactness.py --resolution 512 --ratios 0.3 --fp32    # B2 fp32
