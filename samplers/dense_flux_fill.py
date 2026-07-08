"""samplers.dense_flux_fill — Stage 0 (official baseline) + Stage 1 (custom dense loop).

Three entry points, matched to the plan:

  --mode official   run FluxFillPipeline untouched, save every reproducibility
                    artifact (plan Sec. 7: latents.pt, prompt_embeds.pt, ...).
  --mode custom     the decomposed loop of plan Sec. 8:
                        state = prepare_flux_fill_inputs(...)
                        for step: v = transformer_forward(state, t)
                                  state.latents = scheduler_step(...)
                        output = decode_latents(state)
                    with --forward {module|runner}:
                        module  = call the stock transformer (isolates loop math)
                        runner  = FluxSparseRunner.dense_forward (isolates our
                                  re-implemented block math — the path anchors
                                  and sparse steps will use)
  --mode gate_a     run official + custom(back-to-back, same seed) and report
                        latent max abs error / prediction max abs error /
                        decoded pixel error   (Gate A pass criteria).

Determinism: a fixed torch.Generator seeds the initial packed noise; identical
seeds must give identical official outputs twice (Stage-0 gate) before Gate A
is meaningful.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import torch

from models.flux_fill_loader import load_flux_fill
from models.flux_sparse_transformer import FluxSparseRunner, prepare_latent_image_ids
from utils.flow_math import calculate_shift
from utils.token_mapping import TokenGrid


# ----------------------------------------------------------------- state ------
@dataclass
class FluxFillState:
    latents: torch.Tensor                 # packed [B, N, 64]
    cond: torch.Tensor                    # packed mask+masked-image [B, N, 320]
    prompt_embeds: torch.Tensor
    pooled: torch.Tensor
    guidance: torch.Tensor | None
    img_ids: torch.Tensor
    txt_ids: torch.Tensor
    timesteps: torch.Tensor
    grid: TokenGrid
    generator: torch.Generator


@torch.no_grad()
def prepare_flux_fill_inputs(pipe, image, mask, prompt: str, seed: int,
                             num_steps: int, guidance_scale: float,
                             device: str, dtype,
                             prompt_embeds=None, pooled=None) -> FluxFillState:
    """Decomposed steps 1–7 of plan Sec. 8, delegating tensor math to the
    pipeline's own static helpers so packing/normalization stay byte-identical."""
    B = 1
    # normalize inputs: PIL or [C,H,W]/[1,H,W] torch in [0,1] -> 4D tensors
    if torch.is_tensor(image):
        image = image.unsqueeze(0) if image.dim() == 3 else image
        H, W = image.shape[-2:]
    else:
        W, H = image.size
    if torch.is_tensor(mask) and mask.dim() == 3:
        mask = mask.unsqueeze(0)
    grid = TokenGrid(H, W).validate()
    hp, wp = grid.token_hw

    # 1–2. image / mask preprocessing (identical processors)
    img_in = pipe.image_processor.preprocess(image, height=H, width=W).to(device, dtype)
    mask_in = pipe.mask_processor.preprocess(mask, height=H, width=W).to(device, dtype)
    masked = img_in * (1 - mask_in)

    # 6. prompt encoding (or the shared cache)
    if prompt_embeds is None:
        prompt_embeds, pooled, _ = pipe.encode_prompt(
            prompt=prompt, prompt_2=prompt, device=device)
    txt_ids = torch.zeros(prompt_embeds.shape[1], 3, device=device, dtype=dtype)

    # 3. initial packed noise (deterministic)
    generator = torch.Generator(device).manual_seed(seed)
    num_ch = pipe.vae.config.latent_channels
    hl, wl = grid.latent_hw
    noise = torch.randn(B, num_ch, hl, wl, generator=generator,
                        device=device, dtype=dtype)
    latents = pipe._pack_latents(noise, B, num_ch, hl, wl)
    img_ids = prepare_latent_image_ids(hp, wp, device, dtype)

    # 4–5. mask latent preparation + packing (pipeline helper)
    mask_packed, masked_image_latents = pipe.prepare_mask_latents(
        mask_in, masked, B, num_ch, 1, H, W, dtype, device, generator)
    cond = torch.cat([masked_image_latents, mask_packed], dim=-1)      # [B, N, 320]

    # 7. timesteps / sigmas with dynamic shift (identical schedule)
    sigmas = torch.linspace(1.0, 1.0 / num_steps, num_steps)
    mu = calculate_shift(
        hp * wp,
        pipe.scheduler.config.get("base_image_seq_len", 256),
        pipe.scheduler.config.get("max_image_seq_len", 4096),
        pipe.scheduler.config.get("base_shift", 0.5),
        pipe.scheduler.config.get("max_shift", 1.15),
    )
    pipe.scheduler.set_timesteps(sigmas=sigmas.tolist(), mu=mu, device=device)
    timesteps = pipe.scheduler.timesteps

    guidance = None
    if pipe.transformer.config.guidance_embeds:
        guidance = torch.full([B], guidance_scale, device=device, dtype=torch.float32)

    return FluxFillState(latents=latents, cond=cond, prompt_embeds=prompt_embeds,
                         pooled=pooled, guidance=guidance, img_ids=img_ids,
                         txt_ids=txt_ids, timesteps=timesteps, grid=grid,
                         generator=generator)


@torch.no_grad()
def transformer_forward(pipe, runner: FluxSparseRunner | None, state: FluxFillState,
                        t: torch.Tensor, use_runner: bool):
    """One dense prediction v_t. use_runner selects our decomposed forward."""
    model_input = torch.cat([state.latents, state.cond], dim=2)        # [B, N, 384]
    timestep = t.expand(state.latents.shape[0]).to(state.latents.dtype) / 1000
    if use_runner:
        v, _ = runner.dense_forward(
            model_input, state.prompt_embeds, state.pooled, timestep,
            state.guidance, state.img_ids, state.txt_ids)
        return v
    return pipe.transformer(
        hidden_states=model_input,
        timestep=timestep,
        guidance=state.guidance,
        pooled_projections=state.pooled,
        encoder_hidden_states=state.prompt_embeds,
        txt_ids=state.txt_ids,
        img_ids=state.img_ids,
        return_dict=False,
    )[0]


def scheduler_step(pipe, v: torch.Tensor, t: torch.Tensor, latents: torch.Tensor):
    return pipe.scheduler.step(v, t, latents, return_dict=False)[0]


@torch.no_grad()
def decode_latents(pipe, state: FluxFillState) -> torch.Tensor:
    lat = pipe._unpack_latents(state.latents, state.grid.height, state.grid.width,
                               pipe.vae_scale_factor)
    lat = lat / pipe.vae.config.scaling_factor + pipe.vae.config.shift_factor
    img = pipe.vae.decode(lat, return_dict=False)[0]
    return (img / 2 + 0.5).clamp(0, 1)


@torch.no_grad()
def run_custom_dense(pipe, state: FluxFillState, use_runner: bool,
                     save_predictions: bool = False):
    runner = FluxSparseRunner(pipe.transformer) if use_runner else None
    preds = []
    for t in state.timesteps:
        v = transformer_forward(pipe, runner, state, t, use_runner)
        if save_predictions:
            preds.append(v.detach().float().cpu())
        state.latents = scheduler_step(pipe, v, t, state.latents)
    return decode_latents(pipe, state), preds


# ------------------------------------------------------------------- main -----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["official", "custom", "gate_a"], required=True)
    ap.add_argument("--forward", choices=["module", "runner"], default="module")
    ap.add_argument("--image", required=True)
    ap.add_argument("--mask", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--guidance", type=float, default=30.0)
    ap.add_argument("--resolution", type=int, default=512)
    a = ap.parse_args()

    from PIL import Image
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    comps = load_flux_fill()
    pipe = comps.pipe
    image = Image.open(a.image).convert("RGB").resize((a.resolution, a.resolution))
    mask = Image.open(a.mask).convert("L").resize((a.resolution, a.resolution))

    if a.mode == "official":
        g = torch.Generator(comps.device).manual_seed(a.seed)
        res = pipe(prompt=a.prompt, image=image, mask_image=mask,
                   height=a.resolution, width=a.resolution,
                   num_inference_steps=a.steps, guidance_scale=a.guidance,
                   generator=g)
        res.images[0].save(out / "output_official.png")
        json.dump({"seed": a.seed, "steps": a.steps, "guidance": a.guidance},
                  open(out / "seed.json", "w"))
        return

    state = prepare_flux_fill_inputs(pipe, image, mask, a.prompt, a.seed,
                                     a.steps, a.guidance, comps.device, comps.dtype)
    if a.mode == "custom":
        img, _ = run_custom_dense(pipe, state, use_runner=(a.forward == "runner"))
        _save_img(img, out / f"output_custom_{a.forward}.png")
        return

    # ------------------------------------------------------------- gate A ---
    report = {}
    ref_state = prepare_flux_fill_inputs(pipe, image, mask, a.prompt, a.seed,
                                         a.steps, a.guidance, comps.device, comps.dtype)
    img_module, preds_m = run_custom_dense(pipe, ref_state, use_runner=False,
                                           save_predictions=True)
    state2 = prepare_flux_fill_inputs(pipe, image, mask, a.prompt, a.seed,
                                      a.steps, a.guidance, comps.device, comps.dtype)
    img_runner, preds_r = run_custom_dense(pipe, state2, use_runner=True,
                                           save_predictions=True)
    pred_err = max((pm - pr).abs().max().item() for pm, pr in zip(preds_m, preds_r))
    report["prediction_max_abs_error(module_vs_runner)"] = pred_err
    report["pixel_max_abs_error(module_vs_runner)"] = (img_module - img_runner).abs().max().item()

    g = torch.Generator(comps.device).manual_seed(a.seed)
    res = pipe(prompt=a.prompt, image=image, mask_image=mask,
               height=a.resolution, width=a.resolution,
               num_inference_steps=a.steps, guidance_scale=a.guidance,
               generator=g, output_type="pt")
    official = res.images[0].unsqueeze(0).to(img_module.device)
    report["pixel_max_abs_error(official_vs_custom)"] = (official - img_module).abs().max().item()
    _save_img(img_module, out / "gateA_custom_module.png")
    _save_img(img_runner, out / "gateA_custom_runner.png")
    _save_img(official, out / "gateA_official.png")
    json.dump(report, open(out / "gate_a_report.json", "w"), indent=1)
    print(json.dumps(report, indent=1))


def _save_img(t: torch.Tensor, path):
    from PIL import Image
    import numpy as np
    arr = (t[0].permute(1, 2, 0).float().cpu().numpy() * 255).round().astype("uint8")
    Image.fromarray(arr).save(path)


if __name__ == "__main__":
    main()
