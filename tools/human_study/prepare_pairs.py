"""human_study.prepare_pairs — pairwise 비교 study 폴더 생성.

두 run의 composited(_pasted) 출력에서 비교쌍을 층화 샘플링해
자체 완결 study 폴더(index.html + pairs.json + 이미지)를 만든다.
좌우 위치는 pair별로 무작위 고정(seed) — 위치 편향 상쇄 + 평가자 간 동일 배치.
방법 라벨은 pairs.json의 숨김 필드에만 있고 화면에는 절대 노출되지 않는다.

    python -m tools.human_study.prepare_pairs \
      --run-a .../mbd_c2_r03_t4_dualkv --label-a ours \
      --run-b .../dense_s30           --label-b dense30 \
      --manifest data/coco_manifest_1024.json \
      --n 100 --subset-n 50 --seed 0 --out study_ours_vs_dense30
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def _overlay(image_path: str, mask_pt: Path, resolution: int) -> Image.Image:
    import sys
    sys.path.insert(0, ".")
    from data.dataset import load_image_rgb
    img = np.asarray(load_image_rgb(image_path, resolution)).astype(np.float32)
    m = torch.load(mask_pt).squeeze().numpy()
    if m.shape != img.shape[:2]:
        m = np.asarray(Image.fromarray((m * 255).astype(np.uint8))
                       .resize(img.shape[:2][::-1], Image.NEAREST)) / 255.0
    m = m[..., None]
    over = img * (1 - 0.45 * m) + np.array([255, 40, 40]) * 0.45 * m
    return Image.fromarray(over.astype(np.uint8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-a", required=True, help="method A (예: ours)")
    ap.add_argument("--run-b", required=True, help="method B (baseline)")
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--n", type=int, default=100, help="전체 무작위 pair 수")
    ap.add_argument("--subset-n", type=int, default=50,
                    help="large box/polygon(실패 유형) 추가 pair 수")
    ap.add_argument("--thumb", type=int, default=640, help="표시 해상도")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    rng = random.Random(a.seed)
    ra, rb, out = Path(a.run_a), Path(a.run_b), Path(a.out)
    (out / "img").mkdir(parents=True, exist_ok=True)

    items = {Path(it["sample_id"]).stem: it
             for it in json.load(open(a.manifest))["items"]}
    stems = sorted(s for s in items
                   if (ra / f"{s}_pasted.png").exists()
                   and (rb / f"{s}_pasted.png").exists()
                   and (ra / f"{s}_mask.pt").exists())
    assert stems, "공통 pasted 출력이 없음"

    hard = [s for s in stems
            if items[s]["bucket"] == "large"
            and items[s]["mask_type"] in ("box", "polygon")]
    rest = [s for s in stems if s not in set(hard)]
    rng.shuffle(hard); rng.shuffle(rest)
    chosen = [("random", s) for s in rest[:a.n]] + \
             [("large_boxpoly", s) for s in hard[:a.subset_n]]
    rng.shuffle(chosen)

    res = json.load(open(a.manifest))["resolution"]
    pairs = []
    for k, (subset, stem) in enumerate(chosen):
        it = items[stem]
        ref = _overlay(it["image"], ra / f"{stem}_mask.pt", res)
        T = a.thumb
        ref.resize((T, T), Image.LANCZOS).save(out / "img" / f"{k:04d}_ref.jpg",
                                               quality=90)
        ia = Image.open(ra / f"{stem}_pasted.png").resize((T, T), Image.LANCZOS)
        ib = Image.open(rb / f"{stem}_pasted.png").resize((T, T), Image.LANCZOS)
        flip = rng.random() < 0.5          # pair별 무작위 고정
        left, right = (ib, ia) if flip else (ia, ib)
        left.save(out / "img" / f"{k:04d}_L.jpg", quality=92)
        right.save(out / "img" / f"{k:04d}_R.jpg", quality=92)
        pairs.append({
            "id": k, "stem": stem, "subset": subset,
            "bucket": it["bucket"], "mask_type": it["mask_type"],
            # 숨김 정답 키: 화면 비노출, 집계 전용
            "left_method": a.label_b if flip else a.label_a,
            "right_method": a.label_a if flip else a.label_b,
        })

    json.dump({"comparison": f"{a.label_a}_vs_{a.label_b}",
               "n_pairs": len(pairs), "pairs": pairs},
              open(out / "pairs.json", "w"), indent=1)
    html_src = Path(__file__).parent / "index.html"
    shutil.copy(html_src, out / "index.html")
    print(f"study ready: {len(pairs)} pairs "
          f"({a.n} random + {min(a.subset_n, len(hard))} large-box/poly) "
          f"-> {out}/index.html")


if __name__ == "__main__":
    main()
