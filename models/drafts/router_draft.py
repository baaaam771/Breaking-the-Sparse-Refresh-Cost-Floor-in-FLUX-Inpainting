"""models.drafts.router_draft — inference wrapper: checkpoint -> per-token scores.

Role A (selector-only): the router consumes exactly what is free at a sparse
step — current packed z_t, token mask, cached anchor v_a, timestep — and emits
a difficulty score per image token, which combo_score rank-normalizes into the
eta term. Latency of this call IS part of the method and is inside the sampled
wall-clock (never subtracted, unlike the oracle's extra dense pass).
"""
from __future__ import annotations

from pathlib import Path

import torch

from models.drafts.cnn_router import CNNRouter
from models.flux_cache import FluxAnchorCache
from utils.token_mapping import TokenGrid


class RouterDraft:
    def __init__(self, model: CNNRouter, device):
        self.model = model.to(device).eval()
        self.device = device

    @classmethod
    def load(cls, ckpt_path: str, device, width: int = 96, depth: int = 4,
             use_ema: bool = True):
        ck = torch.load(Path(ckpt_path), map_location=device)
        model = CNNRouter(width, depth)
        model.load_state_dict(ck["ema" if use_ema and "ema" in ck else "model"])
        return cls(model, device)

    @torch.no_grad()
    def scores(self, latents: torch.Tensor, mask_tok: torch.Tensor,
               cache: FluxAnchorCache, t: torch.Tensor, grid: TokenGrid) -> torch.Tensor:
        """latents [B, N, 64] packed z_t -> [B, N] difficulty (sigmoid of logits)."""
        tt = (t if torch.is_tensor(t) else torch.tensor(t)).reshape(-1).float()
        logits = self.model(latents.float(), mask_tok.to(latents.device).float(),
                            cache.final_prediction.float(),
                            tt.to(latents.device) / 1000.0, grid.token_hw)
        return torch.sigmoid(logits)
