"""selectors.delta — Δ_i = ||v_a - v_{a-}||² per packed token.

DACE's draft-free anchor/delta selector: the anchor-to-anchor change of the
target's own prediction, measured on states the cache stores anyway, so it is
free at sparse steps and requires no draft. In DACE this matched the oracle
within 1–2 FID; it is the primary deployable selector here.
"""
import torch


def delta_score(v_anchor: torch.Tensor, v_prev_anchor: torch.Tensor) -> torch.Tensor:
    """Packed predictions [B, N, C] from the two most recent anchors -> [B, N]."""
    return (v_anchor.float() - v_prev_anchor.float()).pow(2).mean(dim=-1)
