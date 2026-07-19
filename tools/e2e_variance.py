"""tools/e2e_variance.py — Stage 15-B: 실제 COCO 입력의 end-to-end latency
반복성. 기존 run.json들의 per-image wall_s 분포(각 arm 100 샘플)를
median / mean±std / p10–p90 / CV / peak VRAM으로 재집계한다 — 새 GPU 시간
불필요.

  python -m tools.e2e_variance \
      --runs $OUT/dense_s50 $OUT/reuse_c2_t4 $OUT/naive_c2_r03_t4 \
             $OUT/mbd_c2_r03_t4_dualkv $OUT/mbd_c2_r03_t4_kv \
      --out e2e_variance.md
"""
import argparse
import json
import os
import statistics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", default="e2e_variance.md")
    a = ap.parse_args()

    rows = ["| method | n | median(s) | mean±std | p10–p90 | CV | peak VRAM(GB) |",
            "|---|---|---|---|---|---|---|"]
    for run in a.runs:
        rj = os.path.join(run, "run.json")
        if not os.path.exists(rj):
            print(f"[skip] {run} (no run.json)")
            continue
        r = json.load(open(rj))
        ws = sorted(x["wall_s"] for x in r["rows"] if not x.get("warmup"))
        if len(ws) < 10:
            print(f"[skip] {run} (n={len(ws)} < 10)")
            continue
        vr = max((x.get("peak_vram_gb", 0.0) for x in r["rows"]), default=0.0)
        mean = statistics.mean(ws)
        std = statistics.stdev(ws)
        tag = os.path.basename(run.rstrip("/"))
        rows.append(
            f"| {tag} | {len(ws)} | {statistics.median(ws):.2f} "
            f"| {mean:.2f}±{std:.2f} "
            f"| {ws[max(int(0.1 * len(ws)) - 1, 0)]:.2f}–"
            f"{ws[int(0.9 * len(ws)) - 1]:.2f} "
            f"| {std / mean:.3f} | {vr:.1f} |")
        print(rows[-1])
    open(a.out, "w").write("\n".join(rows) + "\n")
    print("->", a.out)


if __name__ == "__main__":
    main()
