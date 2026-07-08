"""eval.heterogeneity — Q1 / Gate C: DACE의 two-factor deployment test 완전판 (Fix 4).

Factor A — spatial concentration (hetero run에서 직접 측정):
    top-30% token share, CV, in-mask/out-mask change ratio
Factor B — consequence:
    (i)  per-step E_rel = mean||v_t - v_{t-1}||^2 / mean||v_t||^2  (hetero run)
    (ii) step-reduction quality sensitivity  S_step = Q(dense-20) - Q(dense-50)
         (mask_lpips_to_ref 기준; --dense-full/--dense-reduced metrics.json 결합)

DACE Table 4 규칙: 변화가 '집중'되어 있고 '결과에 중요'할 때만 selective
correction이 이득. 둘 중 하나라도 약하면 r=0 anchored reuse가 정답인 regime.

    python -m eval.heterogeneity --run out/hetero_s50 \
        --dense-full out/dense_s50/metrics.json \
        --dense-reduced out/dense_s20/metrics.json \
        --out hetero_report.json
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--dense-full", default="",
                    help="metrics.json of the 50-step dense run (vs itself: skip)")
    ap.add_argument("--dense-reduced", default="",
                    help="metrics.json of a reduced-step dense run, e.g. dense_s20")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    run = json.load(open(Path(a.run) / "run.json"))
    per_step: dict[int, list[dict]] = {}
    for row in run["rows"]:
        for h in row.get("heterogeneity", []):
            per_step.setdefault(h["step"], []).append(h)

    steps = sorted(per_step)
    summary = []
    for s in steps:
        rows = per_step[s]
        io = [r["in_mask_mean"] / max(r["out_mask_mean"], 1e-12) for r in rows]
        summary.append({
            "step": s,
            "top30_share": statistics.mean(r["top30_share"] for r in rows),
            "cv": statistics.mean(r["cv"] for r in rows),
            "in_out_ratio": statistics.mean(io),
            "energy_ratio": statistics.mean(r["energy_ratio"] for r in rows
                                            if r.get("energy_ratio") is not None),
        })

    pooled = {
        # Factor A: spatial concentration
        "mean_top30_share": statistics.mean(x["top30_share"] for x in summary),
        "peak_top30_share": max(x["top30_share"] for x in summary),
        "mean_in_out_ratio": statistics.mean(x["in_out_ratio"] for x in summary),
        "peak_in_out_ratio": max(x["in_out_ratio"] for x in summary),
        # Factor B(i): change magnitude relative to prediction energy
        "mean_energy_ratio": statistics.mean(x["energy_ratio"] for x in summary),
        "n_images": len(run["rows"]),
    }

    # Factor B(ii): step-reduction sensitivity from the dense sweep
    step_sensitivity = None
    if a.dense_reduced:
        red = json.load(open(a.dense_reduced))["aggregate"]
        q_red = red.get("mask_lpips_to_ref")
        q_full = 0.0
        if a.dense_full:
            q_full = json.load(open(a.dense_full))["aggregate"].get("mask_lpips_to_ref", 0.0)
        if q_red is not None:
            step_sensitivity = q_red - q_full
            pooled["step_sensitivity_mask_lpips"] = step_sensitivity

    concentrated = pooled["mean_in_out_ratio"] > 2.0 and pooled["mean_top30_share"] > 0.5
    consequential = (step_sensitivity is None and pooled["mean_energy_ratio"] > 1e-3) or \
                    (step_sensitivity is not None and step_sensitivity > 0.02)
    if concentrated and consequential:
        verdict = ("Q1 SUPPORTED (both factors): change is concentrated in/near the "
                   "mask AND consequential -> selective refresh regime")
    elif concentrated:
        verdict = ("Q1 PARTIAL: concentrated but low consequence -> r=0 anchored "
                   "reuse likely sufficient (DACE lower-left cell)")
    elif consequential:
        verdict = ("Q1 PARTIAL: consequential but diffuse -> uniform step reduction "
                   "competitive; selective refresh needs the delta selector to earn it")
    else:
        verdict = "Q1 WEAK: neither factor holds -> aggressive reuse, avoid refresh cost"
    json.dump({"pooled": pooled, "per_step": summary, "verdict": verdict},
              open(a.out, "w"), indent=1)
    print(json.dumps(pooled, indent=1)); print(verdict)


if __name__ == "__main__":
    main()
