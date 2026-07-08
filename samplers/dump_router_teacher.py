"""samplers.dump_router_teacher — Stage 6 step 1: teacher dataset for the router.

Runs DENSE 50-step trajectories over the frozen manifest and stores, per sample,
one .pt shard with the full trajectory:

    latents  [S, N, 64]  packed z_t entering each step   (fp16)
    preds    [S, N, 64]  packed v_t at each step          (fp16)
    sigmas   [S]
    mask_tok [N]         token mask coverage
    token_hw (hp, wp)

Training (training/train_router.py) composes (step i, anchor a = i - i%c) pairs
on the fly for any cache period c and any tau — one dump serves every router
config. Size @512², 50 steps: 2·50·1024·64 fp16 ≈ 13 MB/sample.

    PYTHONPATH=. python -m samplers.dump_router_teacher --manifest data/coco_manifest.json \
        --out /mnt/HDD_12TB/bam_ki/flux_fill/router_teacher --steps 50 --limit 200
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from data.dataset import FluxFillBenchmark
from data.prompt_cache import load_cached
from models.flux_fill_loader import load_flux_fill
from models.flux_sparse_transformer import FluxSparseRunner
from samplers.dense_flux_fill import prepare_flux_fill_inputs, scheduler_step
from token_selectors.mask import mask_score


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--guidance", type=float, default=30.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--prompt-cache", default="")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    comps = load_flux_fill(keep_text_encoders=not a.prompt_cache)
    pipe, dev, dtype = comps.pipe, comps.device, comps.dtype
    runner = FluxSparseRunner(pipe.transformer)
    ds = FluxFillBenchmark(a.manifest)
    n = len(ds) if a.limit == 0 else min(a.limit, len(ds))

    index = []
    for i in range(n):
        s = ds[i]
        stem = Path(s["sample_id"]).stem
        shard_p = out / f"{stem}.pt"
        if shard_p.exists():                       # resumable
            index.append(shard_p.name)
            continue
        pe = po = None
        if a.prompt_cache:
            pe, po = load_cached(a.prompt_cache, s["prompt"], dev, dtype)
        state = prepare_flux_fill_inputs(
            pipe, s["image"], s["mask"], s["prompt"], s["latent_seed"],
            a.steps, a.guidance, dev, dtype, prompt_embeds=pe, pooled=po)
        grid = state.grid
        mask_tok = mask_score(s["mask"].unsqueeze(0).to(dev), grid)[0]

        lat_traj, v_traj, sig = [], [], []
        for j, t in enumerate(state.timesteps):
            lat_traj.append(state.latents[0].half().cpu())
            sig.append(float(pipe.scheduler.sigmas[j]))
            model_input = torch.cat([state.latents, state.cond], dim=2)
            timestep = t.expand(1).to(state.latents.dtype) / 1000
            v, _ = runner.dense_forward(model_input, state.prompt_embeds, state.pooled,
                                        timestep, state.guidance, state.img_ids,
                                        state.txt_ids)
            v_traj.append(v[0].half().cpu())
            state.latents = scheduler_step(pipe, v, t, state.latents)

        tmp = shard_p.with_suffix(".pt.tmp")       # atomic write
        torch.save({"latents": torch.stack(lat_traj), "preds": torch.stack(v_traj),
                    "sigmas": torch.tensor(sig), "mask_tok": mask_tok.cpu(),
                    "token_hw": grid.token_hw, "sample_id": s["sample_id"]}, tmp)
        tmp.rename(shard_p)
        index.append(shard_p.name)
        print(f"[{i+1}/{n}] {shard_p.name}")

    json.dump({"shards": index, "steps": a.steps},
              open(out / "index.json", "w"), indent=1)
    print(f"dumped {len(index)} trajectories -> {out}")


if __name__ == "__main__":
    main()
