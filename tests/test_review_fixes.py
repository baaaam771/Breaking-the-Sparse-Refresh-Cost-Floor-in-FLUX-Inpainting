"""CPU tests for the review-fix batch (block selection, anchor x0, MAC
accounting, router training path)."""
import random

import torch

from models.drafts.cnn_router import CNNRouter, router_bce_loss
from models.flux_cache import FluxAnchorCache
from models.flux_sparse_transformer import estimate_transformer_macs
from token_selectors.combo import select_hard_tokens
from utils.token_mapping import TokenGrid, block_hard_easy_split


def test_true_block_selection():
    """Fix 3: hard set must be a union of whole (b x b) windows, k = kb*b^2."""
    grid = TokenGrid(512, 512).validate()               # 32x32 = 1024 tokens
    hp, wp = grid.token_hw
    torch.manual_seed(0)
    scores = torch.rand(2, grid.num_image_tokens)
    for b in (2, 4):
        hard, easy, r_act = block_hard_easy_split(scores, grid, 0.3, b)
        k = hard.shape[1]
        assert k % (b * b) == 0, f"b={b}: k={k} not a multiple of block area"
        assert abs(r_act - k / grid.num_image_tokens) < 1e-9
        # every selected token's whole window must be selected
        sel = torch.zeros(2, grid.num_image_tokens)
        sel.scatter_(1, hard, 1.0)
        g = sel.view(2, 1, hp, wp)
        pooled = torch.nn.functional.avg_pool2d(g, b)
        assert ((pooled == 0) | (pooled == 1)).all(), f"b={b}: partial window selected"
        # partition
        assert k + easy.shape[1] == grid.num_image_tokens
        # requested 0.3*1024 = 307.2 -> nearest whole-block count
        kb_expect = round(0.3 * 1024 / (b * b))
        assert k == kb_expect * b * b


def test_block1_matches_token_topk():
    grid = TokenGrid(128, 128).validate()
    s = torch.rand(1, grid.num_image_tokens)
    h1, e1, r1 = block_hard_easy_split(s, grid, 0.25, 1)
    h2, e2, r2 = select_hard_tokens(s, grid, 0.25, block=1)
    assert torch.equal(h1, h2) and abs(r1 - r2) < 1e-12


def test_anchor_clean_estimate_is_anchor_side():
    """Fix 1: x0_hat_a = z_a - sigma_a * v_a — anchor quantities only."""
    cache = FluxAnchorCache()
    z_a = torch.randn(1, 64, 64)
    v_a = torch.randn(1, 64, 64)
    sigma_a = torch.tensor(0.7)
    cache.final_prediction = v_a
    cache.set_anchor_context(z_a, sigma_a)
    assert torch.allclose(cache.anchor_clean_estimate, z_a - 0.7 * v_a, atol=1e-6)
    # later sparse-step quantities must NOT affect it
    frozen = cache.anchor_clean_estimate.clone()
    _ = z_a * 0  # (nothing in the API mutates it)
    assert torch.equal(cache.anchor_clean_estimate, frozen)
    # x0 recovery under the flow convention
    x0 = torch.randn(1, 64, 64); eps = torch.randn(1, 64, 64)
    s = 0.4
    cache2 = FluxAnchorCache()
    cache2.final_prediction = eps - x0
    cache2.set_anchor_context((1 - s) * x0 + s * eps, torch.tensor(s))
    assert torch.allclose(cache2.anchor_clean_estimate, x0, atol=1e-5)


def test_mac_estimator_sanity():
    """Fix 10: ratio in (0,1], monotonic in k, ->1 at k=N, and visibly larger
    than the naive Sq/S fraction (dense dual + full K/V floor)."""
    T, N, D = 512, 1024, 3072
    prev = 0.0
    for k in (16, 128, 307, 1024):
        m = estimate_transformer_macs(T, N, k, 19, 38, D)
        assert 0 < m["mac_ratio"] <= 1.0 + 1e-9
        assert m["mac_ratio"] > prev
        prev = m["mac_ratio"]
    full = estimate_transformer_macs(T, N, N, 19, 38, D)
    assert abs(full["mac_ratio"] - 1.0) < 1e-9
    r03 = estimate_transformer_macs(T, N, 307, 19, 38, D)
    naive = (T + 307) / (T + N)
    assert r03["mac_ratio"] > naive, "estimate must include the dense floor"


def test_router_training_path():
    """Fix 8: router forward/loss shapes and the label rule on a tiny grid."""
    torch.manual_seed(0)
    B, hp, wp = 2, 8, 8
    N = hp * wp
    model = CNNRouter(width=32, depth=2)
    lat = torch.randn(B, N, 64)
    mt = torch.rand(B, N)
    va = torch.randn(B, N, 64)
    vn = va.clone()
    vn[:, :10] += 1.0                                    # 10 changed tokens
    t = torch.rand(B)
    logits = model(lat, mt, va, t, (hp, wp))
    assert logits.shape == (B, N)
    loss = router_bce_loss(logits, vn, va, tau=1e-4)
    assert loss.ndim == 0 and torch.isfinite(loss)
    y = ((vn - va).pow(2).mean(-1) > 1e-4).float()
    assert y[:, :10].all() and not y[:, 10:].any()


def test_router_draft_wrapper_cpu():
    from models.drafts.router_draft import RouterDraft
    grid = TokenGrid(128, 128).validate()                # 8x8
    model = CNNRouter(width=32, depth=2)
    draft = RouterDraft(model, "cpu")
    cache = FluxAnchorCache()
    N = grid.num_image_tokens
    cache.final_prediction = torch.randn(1, N, 64)
    s = draft.scores(torch.randn(1, N, 64), torch.rand(1, N), cache,
                     torch.tensor([500.0]), grid)
    assert s.shape == (1, N) and (s >= 0).all() and (s <= 1).all()


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
