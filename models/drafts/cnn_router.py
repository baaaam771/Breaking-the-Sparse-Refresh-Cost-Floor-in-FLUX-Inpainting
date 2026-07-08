"""models.drafts.cnn_router — smallest plug-in draft (Stage 6, role A: selector-only).

Input  (all on the (Hp, Wp) token grid, plan Sec. 13 draft candidate #3):
    packed noisy latent      64 ch
    packed mask              (token coverage) 1 ch
    cached target prediction 64 ch
    timestep                 sinusoidal, FiLM-injected
Output:
    one difficulty score per image token  ->  selectors.combo (eta term)

Trained offline against the router label of plan Sec. 14:
    y_i = 1[ ||v_T(t) - v_T(anchor)||^2 > tau ]      (BCE)  or a margin ranking loss.
The frozen FLUX target only generates the teacher dataset; it is never updated.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


def _timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10_000) * torch.arange(half, device=t.device) / half)
    ang = t.float()[:, None] * freqs[None] * 1000
    return torch.cat([ang.sin(), ang.cos()], dim=-1)


class ConvFiLMBlock(nn.Module):
    def __init__(self, ch: int, t_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.norm = nn.GroupNorm(8, ch)
        self.film = nn.Linear(t_dim, 2 * ch)

    def forward(self, x, temb):
        h = self.norm(x)
        scale, shift = self.film(temb)[:, :, None, None].chunk(2, dim=1)
        h = h * (1 + scale) + shift
        h = self.conv2(nn.functional.silu(self.conv1(nn.functional.silu(h))))
        return x + h


class CNNRouter(nn.Module):
    """~1M params. forward(packed_latent, mask_token, cached_pred, t) -> [B, N] logits."""

    def __init__(self, width: int = 96, depth: int = 4, t_dim: int = 128):
        super().__init__()
        in_ch = 64 + 1 + 64
        self.t_dim = t_dim
        self.inp = nn.Conv2d(in_ch, width, 1)
        self.blocks = nn.ModuleList(ConvFiLMBlock(width, t_dim) for _ in range(depth))
        self.out = nn.Conv2d(width, 1, 1)

    def forward(self, packed_latent, mask_token, cached_pred, t, token_hw):
        """packed_latent/cached_pred: [B, N, 64]; mask_token: [B, N]; t: [B]."""
        hp, wp = token_hw
        B, N, _ = packed_latent.shape

        def to_grid(x, c):
            return x.transpose(1, 2).reshape(B, c, hp, wp)

        x = torch.cat([
            to_grid(packed_latent.float(), 64),
            mask_token.float().view(B, 1, hp, wp),
            to_grid(cached_pred.float(), 64),
        ], dim=1)
        temb = _timestep_embedding(t, self.t_dim)
        h = self.inp(x)
        for blk in self.blocks:
            h = blk(h, temb)
        return self.out(h).flatten(1)                     # [B, N] logits


def router_bce_loss(logits: torch.Tensor, v_now: torch.Tensor, v_anchor: torch.Tensor,
                    tau: float) -> torch.Tensor:
    """Plan Sec. 14 router loss on offline teacher pairs."""
    y = ((v_now.float() - v_anchor.float()).pow(2).mean(-1) > tau).float()
    return nn.functional.binary_cross_entropy_with_logits(logits, y)
