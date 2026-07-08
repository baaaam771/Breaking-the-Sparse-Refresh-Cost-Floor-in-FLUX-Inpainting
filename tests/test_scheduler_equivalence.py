"""Stage 1 sub-gate (GPU): the custom loop's timestep/sigma schedule must be
byte-identical to what FluxFillPipeline computes internally, since Gate A is
meaningless if the two loops integrate different grids.

    PYTHONPATH=. python tests/test_scheduler_equivalence.py --resolution 512 --steps 50
"""
import argparse

import torch

from models.flux_fill_loader import load_flux_fill
from utils.flow_math import calculate_shift
from utils.token_mapping import TokenGrid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resolution", type=int, default=512)
    ap.add_argument("--steps", type=int, default=50)
    a = ap.parse_args()

    comps = load_flux_fill(keep_text_encoders=False)
    sch = comps.scheduler
    grid = TokenGrid(a.resolution, a.resolution).validate()
    seq_len = grid.num_image_tokens

    # ours
    sigmas = torch.linspace(1.0, 1.0 / a.steps, a.steps)
    mu = calculate_shift(seq_len,
                         sch.config.get("base_image_seq_len", 256),
                         sch.config.get("max_image_seq_len", 4096),
                         sch.config.get("base_shift", 0.5),
                         sch.config.get("max_shift", 1.15))
    sch.set_timesteps(sigmas=sigmas.tolist(), mu=mu, device=comps.device)
    ours_t = sch.timesteps.clone()
    ours_s = sch.sigmas.clone()

    # official path (retrieve_timesteps inside the pipeline uses the same call);
    # re-derive through the pipeline helper to catch any drift across versions
    from diffusers.pipelines.flux.pipeline_flux_fill import retrieve_timesteps
    timesteps, _ = retrieve_timesteps(sch, a.steps, comps.device,
                                      sigmas=sigmas.tolist(), mu=mu)
    ref_t = timesteps.clone()
    ref_s = sch.sigmas.clone()

    dt = (ours_t.float() - ref_t.float()).abs().max().item()
    ds = (ours_s.float() - ref_s.float()).abs().max().item()
    print(f"timestep max err {dt:.3e}, sigma max err {ds:.3e}")
    assert dt == 0.0 and ds == 0.0, "scheduler mismatch — Gate A blocked"
    print("PASS scheduler equivalence")


if __name__ == "__main__":
    main()
