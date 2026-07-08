"""selectors.frequency — F_i: Haar high-frequency energy on the token grid.

Same single-level Haar DWT saliency as FreqSpec-Inpaint (Eq. 3 of WACV #415):

    E = mean_c (LH^2 + HL^2 + HH^2)

but computed over a *rectified-flow* clean estimate by default:

    x0_hat = z_t - sigma_t * v_anchor        (utils.flow_math.clean_estimate)

Frequency sources supported (plan Sec. 12), all in unpacked latent space
[B, C, Hl, Wl] and reduced to the (Hp, Wp) token grid:

    'anchor_x0'   clean estimate from the last anchor prediction  (default)
    'noisy'       current noisy latent z_t
    'masked'      the known-region (masked-image) latent
    'known'       encoder latent of the original image (upper-bound diagnostic)

One-Verifier's finding is that frequency's role is task/backbone dependent
(prior vs predictor vs inert); the Stage-5 ablation M+B+Δ vs M+B+F+Δ measures
exactly which role it plays on FLUX Fill.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from utils.token_mapping import TokenGrid, grid_to_token_scores

_SQ2 = 0.5  # normalized 2x2 Haar analysis weights (each filter has entries ±1/2)


def _haar_hf_energy(x: torch.Tensor) -> torch.Tensor:
    """Single-level Haar DWT high-frequency energy, channel-averaged.
    x: [B, C, H, W] with even H, W  ->  [B, 1, H/2, W/2]."""
    B, C, H, W = x.shape
    a = x[:, :, 0::2, 0::2]
    b = x[:, :, 0::2, 1::2]
    c = x[:, :, 1::2, 0::2]
    d = x[:, :, 1::2, 1::2]
    lh = (a - b + c - d) * _SQ2
    hl = (a + b - c - d) * _SQ2
    hh = (a - b - c + d) * _SQ2
    return (lh.pow(2) + hl.pow(2) + hh.pow(2)).mean(dim=1, keepdim=True)


def _minmax(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    B = x.shape[0]
    flat = x.view(B, -1)
    lo = flat.min(dim=1, keepdim=True).values
    hi = flat.max(dim=1, keepdim=True).values
    return ((flat - lo) / (hi - lo + eps)).view_as(x)


def frequency_score(latent: torch.Tensor, grid: TokenGrid) -> torch.Tensor:
    """Unpacked latent [B, C, Hl, Wl] -> F_i on the token grid, [B, N] in [0,1].

    The Haar transform halves the spatial resolution: (Hl, Wl) -> (Hl/2, Wl/2)
    which is exactly the (Hp, Wp) token grid, so one DWT level lands each
    energy value on its own packed token with no resampling.
    """
    hp, wp = grid.token_hw
    e = _haar_hf_energy(latent.float())
    if e.shape[-2:] != (hp, wp):  # defensive: e.g. latent already token-res
        e = F.adaptive_avg_pool2d(e, (hp, wp))
    return grid_to_token_scores(_minmax(e))
