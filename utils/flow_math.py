"""utils.flow_math — rectified-flow (FLUX) prediction-space helpers.

FLUX is a rectified-flow model under FlowMatchEulerDiscreteScheduler:

    z_t = (1 - sigma_t) * x0 + sigma_t * noise
    model output v_t ~ (noise - x0)          (velocity / flow prediction)

so the clean estimate is

    x0_hat = z_t - sigma_t * v_t                                  (Eq. F1)

NOT the DDPM  (z - sqrt(1-abar) eps)/sqrt(abar)  formula. Every module that
needs a clean estimate (frequency saliency source, x0 gates, metrics) must go
through this file so the parameterization lives in exactly one place.
"""
from __future__ import annotations

import torch


def clean_estimate(z_t: torch.Tensor, v_t: torch.Tensor, sigma_t) -> torch.Tensor:
    """x0_hat = z_t - sigma_t * v_t. Works on packed [B,N,C] or unpacked [B,C,H,W]."""
    if torch.is_tensor(sigma_t):
        sigma_t = sigma_t.to(z_t.dtype).to(z_t.device)
        while sigma_t.dim() < z_t.dim():
            sigma_t = sigma_t.unsqueeze(-1)
    return z_t - sigma_t * v_t


def sigma_for_step(scheduler, step_index: int) -> torch.Tensor:
    """Current sigma for a FlowMatchEulerDiscreteScheduler at a given step index."""
    return scheduler.sigmas[step_index]


def calculate_shift(
    image_seq_len: int,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
) -> float:
    """mu for dynamic timestep shifting — mirrors diffusers' FluxFillPipeline so the
    custom dense loop (Stage 1) reproduces the official schedule exactly."""
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return image_seq_len * m + b
