"""selectors.boundary — B_i = Dilate(M, k) - Erode(M, k) on the token grid.

The morphological gradient of the mask: a band around the seam, where
FreqSpec-Inpaint measured boundary errors to be the most visually severe
(bLPIPS) and where the WACV supplementary showed boundary terms give the
lowest seam error even when they cost a little accepted-x0 error.
"""
import torch
import torch.nn.functional as F

from utils.token_mapping import TokenGrid, token_scores_to_grid, grid_to_token_scores


def _binary(grid_map: torch.Tensor, thr: float = 0.5) -> torch.Tensor:
    return (grid_map > thr).float()


def boundary_score(mask_token: torch.Tensor, grid: TokenGrid, kernel: int = 3) -> torch.Tensor:
    """mask_token [B,N] coverage -> boundary band [B,N] in {0,1}.

    kernel is in *token* units (1 token = 16 px), so kernel=3 is a ±16 px band —
    the same order as the 32 px boundary ring used for bLPIPS in the WACV paper.
    """
    m = _binary(token_scores_to_grid(mask_token, grid))
    pad = kernel // 2
    dil = F.max_pool2d(m, kernel_size=kernel, stride=1, padding=pad)
    ero = -F.max_pool2d(-m, kernel_size=kernel, stride=1, padding=pad)
    return grid_to_token_scores(dil - ero).clamp(0, 1)
