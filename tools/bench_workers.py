"""tools.bench_workers — DataLoader worker sweep on the REAL benchmark dataset.

Measures the actual per-sample CPU cost of this repo's pipeline: manifest read,
image load + resize, and the deterministic box/brush/polygon mask generation
(NOT a dummy mask — mask gen is part of the real cost).

해석 주의: 이 repo의 sampler는 batch=1로 sample당 50-step FLUX forward
(수십 초)를 돌리므로, 수십 ms의 데이터 로딩은 inference에서는 거의 가려짐.
이 벤치가 진짜 중요한 곳은 (a) router 학습, (b) manifest/teacher-dump 같은
CPU-측 대량 처리, (c) sampler의 --prefetch 켤지 판단.

    PYTHONPATH=. python -m tools.bench_workers --manifest data/coco_manifest.json \
        --workers 8 --steps 100 --pin-memory
"""
from __future__ import annotations

import argparse
import json
import time

import torch
from torch.utils.data import DataLoader

from data.dataset import FluxFillBenchmark


def _collate(items):
    return {
        "image": torch.stack([it["image"] for it in items]),
        "mask": torch.stack([it["mask"] for it in items]),
        "sample_id": [it["sample_id"] for it in items],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/coco_manifest.json")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--prefetch-factor", type=int, default=2)
    ap.add_argument("--pin-memory", action="store_true")
    a = ap.parse_args()

    ds = FluxFillBenchmark(a.manifest)
    kwargs = dict(batch_size=a.batch, shuffle=False, num_workers=a.workers,
                  pin_memory=a.pin_memory, drop_last=True, collate_fn=_collate)
    if a.workers > 0:
        kwargs.update(persistent_workers=True, prefetch_factor=a.prefetch_factor)
    dl = DataLoader(ds, **kwargs)

    it = iter(dl)

    def nxt():
        nonlocal it
        try:
            return next(it)
        except StopIteration:
            it = iter(dl)
            return next(it)

    for _ in range(5):                       # warm-up (worker spawn, page cache)
        nxt()

    t0 = time.perf_counter()
    n_img = 0
    for _ in range(a.steps):
        n_img += nxt()["image"].shape[0]
    dt = time.perf_counter() - t0
    print(json.dumps({
        "workers": a.workers, "batch": a.batch,
        "prefetch_factor": a.prefetch_factor if a.workers > 0 else None,
        "pin_memory": a.pin_memory,
        "images": n_img, "seconds": round(dt, 3),
        "img_per_sec": round(n_img / dt, 2),
        "ms_per_batch": round(dt / a.steps * 1000, 2),
    }))


if __name__ == "__main__":
    main()
