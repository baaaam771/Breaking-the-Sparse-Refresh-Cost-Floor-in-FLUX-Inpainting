"""Pre-GPU validation of the sparse execution MATH on a mock FLUX single-stream
stack (same submodule layout: norm/proj_mlp/act_mlp/attn(q,k,v,norm_q,norm_k)/
proj_out, RoPE, parallel MLP, gated residual).

Property under test = Gate B's core claim, checkable without the 12B model:

    with a FRESH cache (cache states == current dense states), the hard-query
    sparse block output equals the dense block output exactly for the query
    tokens, at any hard ratio, over a multi-block stack.

If this passes on the mock and the real model matches the mock's layout
(flux_fill_loader asserts it does), tests/test_cache_exactness.py on GPU is a
formality. Runs on CPU in seconds.
"""
import math

import torch
import torch.nn as nn

from models.flux_sparse_transformer import (_single_block_dense,
                                            _single_block_sparse)
from utils.token_mapping import hard_easy_split


class MockRMSNorm(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(d) + 0.5)

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.weight


class MockAdaLNZeroSingle(nn.Module):
    """Returns (normed, gate) like diffusers' AdaLayerNormZeroSingle."""
    def __init__(self, d):
        super().__init__()
        self.lin = nn.Linear(d, 3 * d)
        self.norm = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)

    def forward(self, x, emb):
        shift, scale, gate = self.lin(torch.nn.functional.silu(emb)).chunk(3, dim=-1)
        return self.norm(x) * (1 + scale[:, None]) + shift[:, None], gate


class MockAttn(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.heads = heads
        self.to_q = nn.Linear(d, d)
        self.to_k = nn.Linear(d, d)
        self.to_v = nn.Linear(d, d)
        self.norm_q = MockRMSNorm(d // heads)
        self.norm_k = MockRMSNorm(d // heads)


class MockSingleBlock(nn.Module):
    def __init__(self, d=64, heads=4, mlp_ratio=2):
        super().__init__()
        self.norm = MockAdaLNZeroSingle(d)
        self.proj_mlp = nn.Linear(d, d * mlp_ratio)
        self.act_mlp = nn.GELU(approximate="tanh")
        self.attn = MockAttn(d, heads)
        self.proj_out = nn.Linear(d + d * mlp_ratio, d)


def _rope_tables(S, head_dim):
    pos = torch.arange(S).float()
    freqs = torch.exp(-math.log(10_000) * torch.arange(head_dim // 2) / (head_dim // 2))
    ang = pos[:, None] * freqs[None]
    cos = torch.repeat_interleave(ang.cos(), 2, dim=-1)
    sin = torch.repeat_interleave(ang.sin(), 2, dim=-1)
    return cos, sin


def test_fresh_cache_exactness_multiblock():
    torch.manual_seed(0)
    B, T, N, D, L = 2, 5, 27, 64, 6
    S = T + N
    blocks = [MockSingleBlock(D) for _ in range(L)]
    temb = torch.randn(B, D)
    cos, sin = _rope_tables(S, D // 4)

    # ---- dense pass, recording per-block image-token inputs (== anchor) ----
    x = torch.randn(B, S, D, dtype=torch.float64)
    for b in blocks:
        b.double()
    temb, cos, sin = temb.double(), cos.double(), sin.double()
    cat = x.clone()
    cache = []
    dense_states = []
    for blk in blocks:
        cache.append(cat[:, T:].clone())
        cat = _single_block_dense(blk, cat, temb, cos, sin)
        dense_states.append(cat.clone())
    dense_final = cat

    # ---- sparse pass at the SAME state (fresh cache), various ratios -------
    for ratio in (0.1, 0.33, 0.7):
        scores = torch.rand(B, N)
        hard_idx, _ = hard_easy_split(scores, ratio)
        k = hard_idx.shape[1]
        q_text = x[:, :T]
        q_hard = torch.gather(x[:, T:], 1, hard_idx[..., None].expand(-1, -1, D))
        q_pos = torch.cat([torch.arange(T).expand(B, -1), hard_idx + T], dim=1)
        q_fresh = torch.cat([q_text, q_hard], dim=1)

        for j, blk in enumerate(blocks):
            ctx_img = cache[j].clone()
            ctx_img.scatter_(1, hard_idx[..., None].expand(-1, -1, D), q_fresh[:, T:])
            full_ctx = torch.cat([q_fresh[:, :T], ctx_img], dim=1)
            q_fresh = _single_block_sparse(blk, q_fresh, full_ctx, q_pos, temb, cos, sin)

        # hard-token outputs must equal dense outputs at those positions
        dense_hard = torch.gather(dense_final[:, T:], 1,
                                  hard_idx[..., None].expand(-1, -1, D))
        err_hard = (q_fresh[:, T:] - dense_hard).abs().max().item()
        err_text = (q_fresh[:, :T] - dense_final[:, :T]).abs().max().item()
        assert err_hard < 1e-9, f"ratio {ratio}: hard err {err_hard}"
        assert err_text < 1e-9, f"ratio {ratio}: text err {err_text}"
        print(f"fresh-cache exactness ratio={ratio}: hard {err_hard:.2e}, text {err_text:.2e}")


def test_stale_cache_changes_output():
    """Sanity: with a *stale* cache the sparse output must differ (otherwise the
    test above is vacuous)."""
    torch.manual_seed(1)
    B, T, N, D = 1, 4, 16, 64
    blk = MockSingleBlock(D).double()
    temb = torch.randn(B, D).double()
    cos, sin = (t.double() for t in _rope_tables(T + N, D // 4))
    x = torch.randn(B, T + N, D).double()
    dense = _single_block_dense(blk, x, temb, cos, sin)

    hard_idx = torch.tensor([[0, 3, 7]])
    q_pos = torch.cat([torch.arange(T).expand(B, -1), hard_idx + T], dim=1)
    q_fresh = torch.cat([x[:, :T],
                         torch.gather(x[:, T:], 1, hard_idx[..., None].expand(-1, -1, D))], 1)
    stale = x[:, T:] + 0.5 * torch.randn_like(x[:, T:])
    ctx_img = stale.clone()
    ctx_img.scatter_(1, hard_idx[..., None].expand(-1, -1, D), q_fresh[:, T:])
    out = _single_block_sparse(blk, q_fresh, torch.cat([q_fresh[:, :T], ctx_img], 1),
                               q_pos, temb, cos, sin)
    dense_hard = torch.gather(dense[:, T:], 1, hard_idx[..., None].expand(-1, -1, D))
    assert (out[:, T:] - dense_hard).abs().max() > 1e-3


if __name__ == "__main__":
    test_fresh_cache_exactness_multiblock()
    test_stale_cache_changes_output()
    print("PASS all mock sparse-math tests")
