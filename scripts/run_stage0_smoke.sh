#!/usr/bin/env bash
# Stage 0: 공식 baseline 재현 + determinism 확인
set -e
cd "$(dirname "$0")/.."; export PYTHONPATH=.
IMG=${1:?usage: run_stage0_smoke.sh image.png mask.png "prompt"}
MSK=$2; PROMPT=$3; OUT=out/stage0

python -m samplers.dense_flux_fill --mode official --image "$IMG" --mask "$MSK" \
  --prompt "$PROMPT" --out $OUT/a --seed 0 --steps 50
python -m samplers.dense_flux_fill --mode official --image "$IMG" --mask "$MSK" \
  --prompt "$PROMPT" --out $OUT/b --seed 0 --steps 50
python - << 'PY'
from PIL import Image; import numpy as np
a=np.array(Image.open("out/stage0/a/output_official.png"))
b=np.array(Image.open("out/stage0/b/output_official.png"))
d=int(np.abs(a.astype(int)-b.astype(int)).max())
print("Stage0 determinism max|d| =", d); assert d==0
PY
