"""selectors.draft_disagreement — A^D_i = ||v_D(z_t) - v_T^cache||² per token.

Stage-6 optional plug-in draft in its safest role (selector-only): the draft
runs at the *current* step and its disagreement with the cached anchor target
prediction flags tokens whose true prediction has probably moved. One-Verifier
showed prediction agreement is the binding signal on transformer token grids;
this is its cheap, current-step approximation when the target is not run.
"""
import torch


def draft_disagreement_score(v_draft_t: torch.Tensor, v_target_anchor: torch.Tensor) -> torch.Tensor:
    """[B, N, C] draft prediction at t, [B, N, C] cached anchor target -> [B, N]."""
    return (v_draft_t.float() - v_target_anchor.float()).pow(2).mean(dim=-1)
