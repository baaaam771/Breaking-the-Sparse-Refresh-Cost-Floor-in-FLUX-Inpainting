"""tools/floor_curve.py — Stage 15-A (신규 우선순위 1): refresh ratio를
r→0까지 내리는 실측 latency floor curve.

핵심 주장 "naive sparse는 r→0에서도 ~0.49가 남고, dual+KV는 ~0.24"를
analytic이 아닌 **실측 곡선**으로 직접 관찰한다.

구현 주의 (요구사항 반영):
- r→0은 k=1 (N=4096 중 1토큰, r=2.4e-4)로 실측 — 실제 sparse path가 전부
  실행됨: 텍스트 스트림, (naive/+dual에서) 남는 full K/V, (naive/+KV에서)
  dense dual, gather/scatter overhead. anchored reuse로 우회하지 않는다.
- r=1.0은 k=N (전 토큰 refresh) — dense 대비 sparse-path overhead 상한.
- 각 (lever, r)에서 warmup 후 iters회 반복, median/p10/p90.

  python -m tools.floor_curve --resolution 1024 --iters 50 \
      --out floor_curve.json
  python figures/make_fig_floor_curve.py --data floor_curve.json \
      --out fig_floor_curve.pdf
"""
import argparse
import json
import statistics
import time

import torch

from eval.latency import load_transformer_only  # 기존 로더 재사용
from models.flux_cache import FluxAnchorCache
from models.flux_sparse_transformer import FluxSparseRunner


LEVERS = {
    "naive":  dict(kv_cache=False, dual_sparse=False),
    "kv":     dict(kv_cache=True,  dual_sparse=False),
    "dual":   dict(kv_cache=False, dual_sparse=True),
    "dualkv": dict(kv_cache=True,  dual_sparse=True),
}


def _timeit(fn, iters, warmup=8):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    ts_ms = sorted(1e3 * t for t in ts)
    return dict(median_ms=statistics.median(ts_ms),
                p10_ms=ts_ms[max(int(0.1 * len(ts_ms)) - 1, 0)],
                p90_ms=ts_ms[int(0.9 * len(ts_ms)) - 1])


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resolution", type=int, default=1024)
    ap.add_argument("--text-len", type=int, default=512)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--ratios", type=float, nargs="+",
                    default=[0.0, 0.01, 0.025, 0.05, 0.1, 0.15, 0.3,
                             0.5, 1.0])
    ap.add_argument("--out", default="floor_curve.json")
    a = ap.parse_args()

    comps, x, pe, po, ts, gd, img_ids, txt_ids = load_transformer_only(
        a.resolution, a.text_len)
    runner = FluxSparseRunner(comps.transformer)
    dev = x.device
    N = x.shape[1]

    dense = _timeit(lambda: runner.dense_forward(
        x, pe, po, ts, gd, img_ids, txt_ids), a.iters)
    report = dict(resolution=a.resolution, N=N, iters=a.iters,
                  dense_ms=dense["median_ms"], dense=dense, curves={})
    print(f"dense: {dense['median_ms']:.1f} ms (N={N})")

    for lever, flags in LEVERS.items():
        cache = FluxAnchorCache()
        runner.dense_forward(x, pe, po, ts, gd, img_ids, txt_ids,
                             cache=cache, step_index=0,
                             record_kv=flags["kv_cache"],
                             record_dual=flags["dual_sparse"])
        pts = []
        for r in a.ratios:
            k = N if r >= 1.0 else max(1, int(r * N))   # r=0 -> k=1 (r→0 실측)
            hard = torch.sort(
                torch.randperm(N, device=dev)[:k]).values[None]
            t = _timeit(lambda: runner.sparse_forward(
                x, pe, po, ts, gd, img_ids, txt_ids, cache, hard,
                **flags), a.iters)
            pts.append(dict(r_requested=r, k=k, r_actual=k / N, **t,
                            ratio_vs_dense=t["median_ms"]
                            / dense["median_ms"]))
            print(f"[{lever}] r={r:<6} k={k:<5} "
                  f"{t['median_ms']:7.1f} ms  "
                  f"({t['median_ms'] / dense['median_ms']:.3f}x dense)")
        report["curves"][lever] = pts
        del cache
        torch.cuda.empty_cache()

    json.dump(report, open(a.out, "w"), indent=1)
    print("->", a.out)


if __name__ == "__main__":
    main()
