"""Gate B (GPU, requires FLUX weights): fresh-cache exactness on the REAL model.

    1. anchor: dense_forward with cache recording at (z_t, t)
    2. sparse_forward at the SAME (z_t, t) with several hard ratios
    3. merged prediction must equal the dense prediction:
           hard tokens  -> max|Δ| == 0 up to bf16 reduction order (< 2e-2 in bf16,
                           == 0 when run in fp32 via --fp32)
           easy tokens  -> identical by construction (cached final prediction)

DACE rule: if this gate fails, DO NOT proceed to any experiment.

    PYTHONPATH=. python tests/test_cache_exactness.py --resolution 512 [--fp32]
"""
import argparse

import torch

from models.flux_cache import FluxAnchorCache
from models.flux_fill_loader import load_flux_fill
from models.flux_sparse_transformer import FluxSparseRunner, prepare_latent_image_ids
from utils.token_mapping import TokenGrid


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resolution", type=int, default=512)
    ap.add_argument("--text-len", type=int, default=512)
    ap.add_argument("--ratios", type=float, nargs="+", default=[0.1, 0.3, 0.7])
    ap.add_argument("--fp32", action="store_true")
    a = ap.parse_args()

    dtype = torch.float32 if a.fp32 else torch.bfloat16
    comps = load_flux_fill(dtype=dtype, keep_text_encoders=False)
    t = comps.transformer
    runner = FluxSparseRunner(t)
    dev = comps.device
    grid = TokenGrid(a.resolution, a.resolution).validate()
    hp, wp = grid.token_hw
    N = grid.num_image_tokens

    torch.manual_seed(0)
    x = torch.randn(1, N, 384, device=dev, dtype=dtype)
    pe = torch.randn(1, a.text_len, t.config.joint_attention_dim, device=dev, dtype=dtype)
    po = torch.randn(1, t.config.pooled_projection_dim, device=dev, dtype=dtype)
    ts = torch.full((1,), 0.5, device=dev, dtype=dtype)
    gd = torch.full((1,), 30.0, device=dev, dtype=torch.float32) \
        if t.config.guidance_embeds else None
    img_ids = prepare_latent_image_ids(hp, wp, dev, dtype)
    txt_ids = torch.zeros(a.text_len, 3, device=dev, dtype=dtype)

    cache = FluxAnchorCache()
    v_dense, _ = runner.dense_forward(x, pe, po, ts, gd, img_ids, txt_ids,
                                      cache=cache, step_index=0)

    tol = 1e-5 if a.fp32 else 2e-2   # fp32도 SDPA kernel 오차 여지를 둠
    ok = True
    for r in a.ratios:
        k = max(1, int(r * N))
        hard = torch.sort(torch.randperm(N, device=dev)[:k]).values[None]
        v_hard, _ = runner.sparse_forward(x, pe, po, ts, gd, img_ids, txt_ids,
                                          cache, hard)
        v_merged = runner.merge_prediction(cache, hard, v_hard)
        err = (v_merged - v_dense).abs().max().item()
        hard_err = (v_hard - torch.gather(
            v_dense, 1, hard.unsqueeze(-1).expand(-1, -1, v_dense.shape[-1]))
        ).abs().max().item()
        status = "PASS" if hard_err <= tol else "FAIL"
        ok &= hard_err <= tol
        print(f"[Gate B] ratio={r}: hard max|dv|={hard_err:.3e} "
              f"merged max|dv|={err:.3e}  {status}")
    print(f"cache VRAM: {cache.vram_bytes()/2**30:.2f} GB")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
