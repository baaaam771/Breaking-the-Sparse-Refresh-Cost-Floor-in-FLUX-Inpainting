"""eval.token_overlay — Stage 3 mandatory figure:
input | mask | latent mask | token mask | reconstructed token overlay.

Pure CPU; validates the mapping chain visually before any GPU experiment.

    python -m eval.token_overlay --manifest data/coco_manifest.json --index 0 --out overlay.png
"""
from __future__ import annotations

import argparse

import torch

from data.dataset import FluxFillBenchmark
from utils.token_mapping import (TokenGrid, pixel_mask_to_latent,
                                 pixel_mask_to_token, token_mask_to_pixel,
                                 token_scores_to_grid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import numpy as np
    from PIL import Image

    s = FluxFillBenchmark(a.manifest)[a.index]
    img, mask = s["image"], s["mask"].unsqueeze(0)          # [3,H,W], [1,1,H,W]
    H, W = img.shape[-2:]
    grid = TokenGrid(H, W).validate()

    lat = pixel_mask_to_latent(mask, grid)                  # [1,1,Hl,Wl]
    tok = pixel_mask_to_token(mask, grid)                   # [1,N]
    tok_bin = (tok > 0.5).float()
    rec = token_mask_to_pixel(tok_bin, grid)                # [1,1,H,W]

    def to_u8(x, size=(H, W)):
        x = torch.nn.functional.interpolate(x, size=size, mode="nearest")[0, 0]
        return (x.clamp(0, 1).numpy() * 255).astype("uint8")

    panels = [
        (img.permute(1, 2, 0).numpy() * 255).astype("uint8"),
        np.stack([to_u8(mask)] * 3, -1),
        np.stack([to_u8(lat)] * 3, -1),
        np.stack([to_u8(token_scores_to_grid(tok, grid))] * 3, -1),
        # overlay: red token mask over the image
        (0.6 * (img.permute(1, 2, 0).numpy() * 255)
         + 0.4 * np.stack([to_u8(rec), np.zeros((H, W)), np.zeros((H, W))], -1)
         ).astype("uint8"),
    ]
    canvas = np.concatenate(panels, axis=1)
    Image.fromarray(canvas).save(a.out)

    cov_px = mask.mean().item()
    cov_tok = tok_bin.mean().item()
    print(f"pixel coverage {cov_px:.3f} | binary token coverage {cov_tok:.3f} "
          f"| tokens {grid.num_image_tokens} | saved {a.out}")


if __name__ == "__main__":
    main()
