"""human_study.aggregate — 평가자 JSON들 -> win/tie/loss + bootstrap 95% CI.

숨김 left/right_method 키로 응답을 method 기준으로 복원하고,
질문별 x subset별로 집계. 평가자 간 다수결(이미지당 3인)과
개별 응답 단위 두 관점 모두 보고.

    python -m tools.human_study.aggregate --study study_ours_vs_dense30 \
        --ratings ratings_*.json --out human_study_report.md
"""
from __future__ import annotations

import argparse
import glob
import json
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path

QS = {"q0": "mask naturalness", "q1": "boundary blending", "q2": "overall preference"}


def _boot_ci(wins, losses, n=5000, seed=0):
    """win-rate = W/(W+L) (tie 제외)의 bootstrap 95% CI."""
    outcomes = [1] * wins + [0] * losses
    if not outcomes:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    ms = sorted(statistics.mean(rng.choices(outcomes, k=len(outcomes)))
                for _ in range(n))
    return ms[int(0.025 * n)], ms[int(0.975 * n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", required=True)
    ap.add_argument("--ratings", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    pairs = {p["id"]: p for p in
             json.load(open(Path(a.study) / "pairs.json"))["pairs"]}
    files = [f for pat in a.ratings for f in glob.glob(pat)]
    raters = [json.load(open(f)) for f in files]
    assert raters, "rating 파일 없음"
    label_a = json.load(open(Path(a.study) / "pairs.json"))["comparison"].split("_vs_")[0]

    # (pair, question) -> method-단위 응답들 ("A" win / "B" win / tie)
    votes = defaultdict(list)
    for r in raters:
        for pid_s, ansd in r["answers"].items():
            p = pairs[int(pid_s)]
            for q in QS:
                v = ansd.get(q)
                if v == "T":
                    votes[(int(pid_s), q)].append("T")
                elif v in ("L", "R"):
                    m = p["left_method"] if v == "L" else p["right_method"]
                    votes[(int(pid_s), q)].append("A" if m == label_a else "B")

    lines = [f"# Human study: {label_a} vs baseline "
             f"({len(raters)} raters, {len(pairs)} pairs)\n"]
    subsets = ["all", "random", "large_boxpoly"]
    for q, qname in QS.items():
        lines.append(f"\n## {qname}")
        lines.append("| subset | A win | tie | A loss | win-rate (95% CI) "
                     "| majority A-win/tie/loss |")
        lines.append("|---|---|---|---|---|---|")
        for sub in subsets:
            W = T = L = 0
            mw = mt = ml = 0
            for pid, p in pairs.items():
                if sub != "all" and p["subset"] != sub:
                    continue
                vs = votes.get((pid, q), [])
                W += vs.count("A"); T += vs.count("T"); L += vs.count("B")
                if vs:
                    c = Counter(vs)
                    top = c.most_common()
                    if len(top) > 1 and top[0][1] == top[1][1]:
                        mt += 1
                    elif top[0][0] == "A":
                        mw += 1
                    elif top[0][0] == "T":
                        mt += 1
                    else:
                        ml += 1
            lo, hi = _boot_ci(W, L)
            rate = W / max(W + L, 1)
            lines.append(f"| {sub} | {W} | {T} | {L} "
                         f"| {rate:.3f} [{lo:.3f}, {hi:.3f}] "
                         f"| {mw}/{mt}/{ml} |")

    # 평가자 간 일치도 (전 질문, tie 포함 3-way exact agreement)
    agree = tot = 0
    for key, vs in votes.items():
        if len(vs) >= 2:
            tot += 1
            if len(set(vs)) == 1:
                agree += 1
    lines.append(f"\nExact 3-way agreement (모든 평가자 동일 응답): "
                 f"{agree}/{tot} = {agree / max(tot, 1):.3f}")

    Path(a.out).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
