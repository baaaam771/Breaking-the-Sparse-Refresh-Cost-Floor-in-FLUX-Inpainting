"""Gate B0 (GPU, requires FLUX weights): OFFICIAL single-stream block forward
vs the manual reimplementation `_single_block_dense`, block-by-block, on the
real weights. This isolates the numeric-equivalence question from the cache
question — run it BEFORE Gate B2 (tests/test_cache_exactness.py) so a failure
there is attributable.

Gate ladder (Fix 2):
    B0  official block(x)      == _single_block_dense(block, x)     (this file)
    B1  official transformer   == FluxSparseRunner.dense_forward    (dense_flux_fill --mode gate_a)
    B2  fresh-cache sparse     == runner dense                      (test_cache_exactness.py)

    PYTHONPATH=. python tests/test_single_block_equivalence.py --resolution 512 [--fp32]
"""
import argparse

import torch

from models.flux_fill_loader import load_flux_fill
from models.flux_sparse_transformer import _single_block_dense, prepare_latent_image_ids
from utils.token_mapping import TokenGrid


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resolution", type=int, default=512)
    ap.add_argument("--text-len", type=int, default=512)
    ap.add_argument("--blocks", type=int, default=0,
                    help="0 = all 38 single blocks; n = first n blocks")
    ap.add_argument("--fp32", action="store_true")
    a = ap.parse_args()

    dtype = torch.float32 if a.fp32 else torch.bfloat16
    comps = load_flux_fill(dtype=dtype, keep_text_encoders=False)
    t, dev = comps.transformer, comps.device
    grid = TokenGrid(a.resolution, a.resolution).validate()
    hp, wp = grid.token_hw
    N, T = grid.num_image_tokens, a.text_len
    D = t.config.num_attention_heads * t.config.attention_head_dim

    torch.manual_seed(0)
    x = torch.randn(1, T + N, D, device=dev, dtype=dtype)
    temb = torch.randn(1, D, device=dev, dtype=dtype)
    ids = torch.cat([torch.zeros(T, 3, device=dev, dtype=dtype),
                     prepare_latent_image_ids(hp, wp, dev, dtype)], dim=0)
    rope = t.pos_embed(ids)
    cos, sin = rope if isinstance(rope, tuple) else (rope[0], rope[1])

    blocks = t.single_transformer_blocks
    n = len(blocks) if a.blocks == 0 else min(a.blocks, len(blocks))
    # fp32: SDPA backend/kernel choice can still introduce ~1e-6 reduction noise,
    # so "exact" means 1e-5, not literal 0 (리뷰 2차 반영)
    tol = 1e-5 if a.fp32 else 3e-2
    worst = 0.0
    ok = True
    for j in range(n):
        blk = blocks[j]
        # official diffusers path (v0.32.2 signature)
        official = blk(hidden_states=x, temb=temb, image_rotary_emb=(cos, sin))
        if isinstance(official, tuple):
            official = official[0]
        manual = _single_block_dense(blk, x, temb, cos, sin)
        err = (official.float() - manual.float()).abs().max().item()
        worst = max(worst, err)
        if err > tol:
            ok = False
            print(f"[Gate B0] block {j:2d}: max|d| = {err:.3e}  FAIL")
    print(f"[Gate B0] {n} blocks, worst max|d| = {worst:.3e}, tol = {tol}  "
          f"{'PASS' if ok else 'FAIL'}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
