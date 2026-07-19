#!/usr/bin/env bash
# Stage 13-A (총평 2순위, 저비용): 추가 GPU에서 latency 재현.
# 목적: "특정 GPU의 kernel 선택에서만 나오는 speedup 아닌가"를 닫는다.
# 검증 대상 (절대값이 아니라):
#   (1) 방법 간 순서 유지 (dense > dual > dualkv 순 비용)
#   (2) 해상도 증가 시 speedup 증가
#   (3) analytic MAC 방향 일치 (realization은 하드웨어 의존으로 설명)
# 품질 재실행 불필요 — transformer-only latency만. 예상 소요 ~40분.
#
# 사용 (4090/A100/H100 등 아무 CUDA GPU):
#   OUT=/path/to/stage13_gpu2 bash scripts/run_stage13_gpu2.sh
set -e
cd "$(dirname "$0")/.."; export PYTHONPATH=.
OUT=${OUT:?예: .../stage13_gpu2}
mkdir -p "$OUT"
ITERS=${ITERS:-100}          # 총평 프로토콜: 20+ warmup / 100+ iters
RES_LIST=${RES_LIST:-"768 1024"}   # VRAM 여유 시 "768 1024 1536"

# ---- 환경 기록 (재현성: GPU/드라이버/torch/cuda/attention backend) ----
python - << 'PY' | tee "$OUT/env.json"
import json, torch, platform
info = dict(
    gpu=torch.cuda.get_device_name(0),
    capability=".".join(map(str, torch.cuda.get_device_capability(0))),
    vram_gb=round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1),
    torch=torch.__version__, cuda=torch.version.cuda,
    cudnn=torch.backends.cudnn.version(),
    sdpa_flash=torch.backends.cuda.flash_sdp_enabled(),
    sdpa_mem_efficient=torch.backends.cuda.mem_efficient_sdp_enabled(),
    python=platform.python_version(), compile_used=False, batch=1,
    dtype="bfloat16")
print(json.dumps(info, indent=1))
PY

# ---- FLUX transformer latency: dense/base/kv/dual/dualkv x r{.15,.3} ----
for RES in $RES_LIST; do
  test -f "$OUT/latency_flux_${RES}_base.json" || \
    python -m eval.latency --resolution $RES --ratios 0.15 0.3 \
      --iters $ITERS --out "$OUT/latency_flux_${RES}_base.json"
  test -f "$OUT/latency_flux_${RES}_kv.json" || \
    python -m eval.latency --resolution $RES --ratios 0.15 0.3 --kv-cache \
      --iters $ITERS --out "$OUT/latency_flux_${RES}_kv.json"
  test -f "$OUT/latency_flux_${RES}_dual.json" || \
    python -m eval.latency --resolution $RES --ratios 0.15 0.3 --dual-sparse \
      --iters $ITERS --out "$OUT/latency_flux_${RES}_dual.json"
  test -f "$OUT/latency_flux_${RES}_dualkv.json" || \
    python -m eval.latency --resolution $RES --ratios 0.15 0.3 --kv-cache \
      --dual-sparse --iters $ITERS --out "$OUT/latency_flux_${RES}_dualkv.json"
done

# ---- SD3.5 (모델이 이미 캐시돼 있거나 받을 수 있는 경우; 실패해도 계속) ----
python -m tools.sd3_latency --resolutions $RES_LIST --ratios 0.15 0.3 \
  --iters $ITERS --out "$OUT/sd3_latency_gpu2.md" || \
  echo "[skip] SD3 latency (모델 미확보) — FLUX 결과만으로도 충분"

# ---- 요약 표: dense 대비 speedup + 해상도 트렌드 ----
python - << 'PY'
import json, os
out = os.environ["OUT"]
rows = ["| res | variant | r | median ms | p10/p90 ms | speedup | MAC(analytic) |",
        "|---|---|---|---|---|---|---|"]
for res in os.environ.get("RES_LIST", "768 1024").split():
    dense_ms = None
    for var in ("base", "kv", "dual", "dualkv"):
        p = f"{out}/latency_flux_{res}_{var}.json"
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        if dense_ms is None and "dense" in d:
            dd = d["dense"]
            dense_ms = dd["median_ms"]
            rows.append(f"| {res} | dense | - | {dense_ms:.1f} | "
                        f"{dd.get('p10_ms', 0):.1f}/{dd.get('p90_ms', 0):.1f} "
                        f"| 1.00x | 1.000 |")
        for key, v in d.items():
            if not key.startswith("sparse_r"):
                continue
            r = key[len("sparse_r"):]
            sp = dense_ms / v["median_ms"] if dense_ms else 0
            rows.append(f"| {res} | {var} | {r} | {v['median_ms']:.1f} | "
                        f"{v.get('p10_ms', 0):.1f}/{v.get('p90_ms', 0):.1f} "
                        f"| {sp:.2f}x | {v.get('est_mac_ratio', 0):.3f} |")
with open(f"{out}/summary_gpu2.md", "w") as f:
    f.write("\n".join(rows) + "\n")
print("\n".join(rows))
PY
echo "done -> $OUT/summary_gpu2.md (env.json과 함께 전달)"
