"""utils.token_mapping — Stage 3: pixel mask -> FLUX packed image-token index.

Chain (FluxFillPipeline conventions):

    pixel grid (H x W)
      --VAE 8x-->            latent grid  (Hl x Wl),  Hl = H//8
      --2x2 packing-->        token grid  (Hp x Wp),  Hp = Hl//2 = H//16
      --row-major flatten-->  image token index i in [0, Hp*Wp)

The packed sequence fed to the transformer is [text tokens ; image tokens]
along dim=1 for attention, with image tokens in row-major (Hp, Wp) order via
`latents.view(B, C, Hp, 2, Wp, 2).permute(0,2,4,1,3,5).reshape(B, Hp*Wp, C*4)`.
This module owns every conversion so that selector scores computed on the
(Hp, Wp) grid index the packed sequence correctly, and provides the roundtrip
overlay demanded by the plan (mask -> token mask -> reconstructed pixel mask).

Pure torch; unit-tested on CPU (tests/test_token_mapping.py).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

VAE_SCALE = 8      # SDXL/FLUX VAE spatial compression
PACK = 2           # 2x2 latent packing


@dataclass(frozen=True)
class TokenGrid:
    height: int          # pixel H
    width: int           # pixel W

    @property
    def latent_hw(self):
        return self.height // VAE_SCALE, self.width // VAE_SCALE

    @property
    def token_hw(self):
        hl, wl = self.latent_hw
        return hl // PACK, wl // PACK

    @property
    def num_image_tokens(self):
        hp, wp = self.token_hw
        return hp * wp

    def validate(self):
        assert self.height % (VAE_SCALE * PACK) == 0 and self.width % (VAE_SCALE * PACK) == 0, (
            f"resolution {self.height}x{self.width} must be divisible by {VAE_SCALE * PACK} "
            "(VAE 8x then 2x2 packing)"
        )
        return self


# ---------------------------------------------------------------- packing ---
def pack_latents(lat: torch.Tensor) -> torch.Tensor:
    """[B, C, Hl, Wl] -> packed [B, Hp*Wp, C*4], row-major token order.
    Matches FluxFillPipeline._pack_latents."""
    B, C, Hl, Wl = lat.shape
    hp, wp = Hl // PACK, Wl // PACK
    x = lat.view(B, C, hp, PACK, wp, PACK)
    x = x.permute(0, 2, 4, 1, 3, 5)                # B, hp, wp, C, 2, 2
    return x.reshape(B, hp * wp, C * PACK * PACK)


def unpack_latents(tokens: torch.Tensor, grid: TokenGrid, channels: int) -> torch.Tensor:
    """packed [B, N, C*4] -> [B, C, Hl, Wl]. Matches FluxFillPipeline._unpack_latents."""
    B, N, D = tokens.shape
    hp, wp = grid.token_hw
    assert N == hp * wp and D == channels * PACK * PACK
    x = tokens.view(B, hp, wp, channels, PACK, PACK)
    x = x.permute(0, 3, 1, 4, 2, 5)                # B, C, hp, 2, wp, 2
    return x.reshape(B, channels, hp * PACK, wp * PACK)


# ------------------------------------------------------------------- masks ---
def pixel_mask_to_latent(mask_px: torch.Tensor, grid: TokenGrid) -> torch.Tensor:
    """[B,1,H,W] in {0,1} (1 = regenerate) -> latent-resolution coverage [B,1,Hl,Wl] in [0,1]."""
    hl, wl = grid.latent_hw
    return F.avg_pool2d(mask_px.float(), kernel_size=VAE_SCALE)[..., :hl, :wl]


def latent_mask_to_token(mask_lat: torch.Tensor, grid: TokenGrid) -> torch.Tensor:
    """[B,1,Hl,Wl] -> per-token coverage [B, N] in [0,1] (row-major token order)."""
    cov = F.avg_pool2d(mask_lat.float(), kernel_size=PACK)      # [B,1,Hp,Wp]
    return cov.flatten(2).squeeze(1)                            # [B, N]


def pixel_mask_to_token(mask_px: torch.Tensor, grid: TokenGrid) -> torch.Tensor:
    """Pixel mask -> per-token coverage in one hop (== avg over 16x16 pixel cell)."""
    return latent_mask_to_token(pixel_mask_to_latent(mask_px, grid), grid)


def token_scores_to_grid(scores: torch.Tensor, grid: TokenGrid) -> torch.Tensor:
    """[B, N] -> [B, 1, Hp, Wp] for wavelet / morphology ops on the token grid."""
    hp, wp = grid.token_hw
    return scores.view(scores.shape[0], 1, hp, wp)


def grid_to_token_scores(grid_map: torch.Tensor) -> torch.Tensor:
    """[B, 1, Hp, Wp] -> [B, N]."""
    return grid_map.flatten(2).squeeze(1)


def token_mask_to_pixel(token_mask: torch.Tensor, grid: TokenGrid) -> torch.Tensor:
    """[B, N] (binary or soft) -> pixel-resolution overlay [B,1,H,W] via nearest upsample.
    Used for the mandatory reconstructed-token-overlay figure (Stage 3)."""
    g = token_scores_to_grid(token_mask, grid)
    return F.interpolate(g, scale_factor=VAE_SCALE * PACK, mode="nearest")


# --------------------------------------------------------- sequence offsets ---
def image_token_positions(text_len: int, num_image_tokens: int, device=None) -> torch.Tensor:
    """Absolute positions of image tokens inside the joint [text; image] sequence
    used by the single-stream blocks."""
    return torch.arange(text_len, text_len + num_image_tokens, device=device)


def hard_easy_split(scores: torch.Tensor, ratio: float):
    """Top-`ratio` tokens by score. scores [B, N] -> (hard_idx [B,k], easy_idx [B,N-k]),
    both sorted ascending so gathers stay coalesced."""
    B, N = scores.shape
    k = max(1, int(round(ratio * N)))
    hard = torch.topk(scores, k, dim=1).indices
    hard, _ = torch.sort(hard, dim=1)
    mask = torch.ones(B, N, dtype=torch.bool, device=scores.device)
    mask.scatter_(1, hard, False)
    easy = mask.nonzero(as_tuple=False)[:, 1].view(B, N - k)
    return hard, easy


def blockify_scores(scores: torch.Tensor, grid: TokenGrid, block: int) -> torch.Tensor:
    """Block-mean smoothing of token scores (diagnostic only — for SELECTION use
    block_hard_easy_split, which does true block-level Top-K). block=1 is a no-op."""
    if block == 1:
        return scores
    g = token_scores_to_grid(scores, grid)
    hp, wp = grid.token_hw
    assert hp % block == 0 and wp % block == 0, "token grid must divide the block size"
    pooled = F.avg_pool2d(g, kernel_size=block)
    up = F.interpolate(pooled, scale_factor=block, mode="nearest")
    return grid_to_token_scores(up)


def block_hard_easy_split(scores: torch.Tensor, grid: TokenGrid,
                          ratio: float, block: int):
    """TRUE block-structured selection (Fix 3): reduce token scores to block
    means, Top-K over BLOCKS, expand winners to their token indices — so the
    hard set is always a union of whole (block x block) windows and
    k = kb * block^2 exactly.

    Returns (hard_idx [B, k], easy_idx [B, N-k], actual_ratio: float).
    With block == 1 this is plain token Top-K (actual_ratio == k/N)."""
    B, N = scores.shape
    if block == 1:
        hard, easy = hard_easy_split(scores, ratio)
        return hard, easy, hard.shape[1] / N

    hp, wp = grid.token_hw
    assert hp % block == 0 and wp % block == 0, "token grid must divide the block size"
    g = token_scores_to_grid(scores, grid)
    pooled = F.avg_pool2d(g, kernel_size=block).flatten(1)      # [B, nb]
    nb = pooled.shape[1]
    wb = wp // block
    kb = min(nb, max(1, int(round(ratio * N / (block * block)))))
    top_blocks = torch.topk(pooled, kb, dim=1).indices          # [B, kb]

    by, bx = top_blocks // wb, top_blocks % wb                  # block coords
    dy = torch.arange(block, device=scores.device)
    dx = torch.arange(block, device=scores.device)
    ty = by[:, :, None, None] * block + dy[None, None, :, None]  # [B,kb,b,1]
    tx = bx[:, :, None, None] * block + dx[None, None, None, :]  # [B,kb,1,b]
    hard = (ty * wp + tx).reshape(B, -1)                        # [B, kb*b*b]
    hard, _ = torch.sort(hard, dim=1)

    k = hard.shape[1]
    m = torch.ones(B, N, dtype=torch.bool, device=scores.device)
    m.scatter_(1, hard, False)
    easy = m.nonzero(as_tuple=False)[:, 1].view(B, N - k)
    return hard, easy, k / N
