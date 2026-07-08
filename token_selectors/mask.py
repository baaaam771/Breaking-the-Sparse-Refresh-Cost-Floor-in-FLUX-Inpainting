"""selectors.mask — M_i: per-token mask coverage (the strongest single prior in
FreqSpec-Inpaint; the Stage-4 PoC uses this alone)."""
import torch

from utils.token_mapping import TokenGrid, pixel_mask_to_token


def mask_score(mask_px: torch.Tensor, grid: TokenGrid) -> torch.Tensor:
    """[B,1,H,W] pixel mask (1 = regenerate) -> M_i in [0,1], shape [B, N]."""
    return pixel_mask_to_token(mask_px, grid)
