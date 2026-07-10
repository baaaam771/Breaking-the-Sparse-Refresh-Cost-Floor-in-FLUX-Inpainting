"""models.flux_sparse_transformer — dense / anchor / sparse forward for FLUX Fill.

This is the execution layer the verifier papers deferred, ported from DACE
(ImageNet-64 DiT) to the FLUX.1 Fill [dev] architecture:

  * dual-stream 19 blocks : ALWAYS dense (plan Sec. 3 first-PoC safety rule)
  * single-stream 38 blocks:
        anchor step -> dense, recording each block's image-token INPUT states
        sparse step -> queries = [all text tokens ; hard image tokens] (fresh),
                       K/V ctx = [fresh text ; fresh hard ; anchor-cached easy]
                       — easy context is depth-correct and only time-stale,
                       never frozen across depth (DACE Sec. 4.1/4.2).

Three forward modes:
    dense_forward(...)                      Stage 1 (Gate A) & anchors
    sparse_forward(..., hard_idx)           Stage 4+ selective refresh
    (r = 0 needs no forward at all: the sampler reuses cache.final_prediction)

Exactness property (Gate B, tests/test_cache_exactness.py):
    with a fresh cache (anchor and sparse step at the same latent/timestep),
    sparse_forward's hard-token outputs equal dense outputs bit-for-bit modulo
    bf16 reduction order, at ANY hard ratio — because the scattered context is
    then literally the dense context.

Implementation notes:
  * We re-implement the single-stream block forward from its own submodules
    (norm / proj_mlp / act_mlp / attn.to_q|to_k|to_v|norm_q|norm_k / proj_out)
    for BOTH the dense and sparse paths, so the two paths share every numeric
    op and the exactness test is meaningful. Dual-stream blocks are invoked
    as-is. flux_fill_loader hard-asserts this module layout at load time.
  * Single-stream FLUX attention has no output projection: sdpa output is
    concatenated with the parallel MLP branch and passed through proj_out.
  * RoPE: diffusers FluxPosEmbed returns (cos, sin) of shape [S, head_dim];
    we gather rows for the query subset (per-batch) and keep full rows for K.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .flux_cache import FluxAnchorCache


# ------------------------------------------------------------------ helpers ---
def _rope_apply(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """diffusers Flux RoPE (use_real, unbind_dim=-1).
    x: [B, H, S, D]; cos/sin: [S, D] or [B, 1, S, D]."""
    if cos.dim() == 2:
        cos = cos[None, None]
        sin = sin[None, None]
    xf = x.float()
    x_pair = xf.reshape(*xf.shape[:-1], -1, 2)
    x_rot = torch.stack([-x_pair[..., 1], x_pair[..., 0]], dim=-1).reshape_as(xf)
    return (xf * cos.float() + x_rot * sin.float()).to(x.dtype)


def _heads(x: torch.Tensor, n_heads: int) -> torch.Tensor:
    B, S, D = x.shape
    return x.view(B, S, n_heads, D // n_heads).transpose(1, 2)          # [B,H,S,d]


def _unheads(x: torch.Tensor) -> torch.Tensor:
    B, H, S, d = x.shape
    return x.transpose(1, 2).reshape(B, S, H * d)


def _gather_tokens(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """x [B, S, D], idx [B, k] -> [B, k, D]."""
    return torch.gather(x, 1, idx.unsqueeze(-1).expand(-1, -1, x.shape[-1]))


def _scatter_tokens(base: torch.Tensor, idx: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """out-of-place scatter: base [B, S, D] with values [B, k, D] at idx [B, k]."""
    out = base.clone()
    out.scatter_(1, idx.unsqueeze(-1).expand(-1, -1, base.shape[-1]), values)
    return out


def prepare_latent_image_ids(hp: int, wp: int, device, dtype) -> torch.Tensor:
    """Row-major (Hp, Wp) positional ids — mirrors FluxFillPipeline."""
    ids = torch.zeros(hp, wp, 3, device=device, dtype=dtype)
    ids[..., 1] = torch.arange(hp, device=device, dtype=dtype)[:, None]
    ids[..., 2] = torch.arange(wp, device=device, dtype=dtype)[None, :]
    return ids.reshape(hp * wp, 3)


# --------------------------------------------------------------- single blk ---
def _single_block_dense(block, x: torch.Tensor, temb: torch.Tensor,
                        cos: torch.Tensor, sin: torch.Tensor,
                        return_kv_img_from: int | None = None):
    """Manual FluxSingleTransformerBlock forward on the full [text;image] seq.
    return_kv_img_from=T additionally returns the image-token (k, v) — k post-rope
    (rope depends on position only), v raw — for the anchor K/V cache (Lever B).
    No extra GEMMs: the same k/v used for attention are sliced."""
    residual = x
    normed, gate = block.norm(x, emb=temb)
    mlp_h = block.act_mlp(block.proj_mlp(normed))

    attn = block.attn
    q = _heads(attn.to_q(normed), attn.heads)
    k = _heads(attn.to_k(normed), attn.heads)
    v = _heads(attn.to_v(normed), attn.heads)
    if attn.norm_q is not None:
        q = attn.norm_q(q)
    if attn.norm_k is not None:
        k = attn.norm_k(k)
    q = _rope_apply(q, cos, sin)
    k = _rope_apply(k, cos, sin)
    o = F.scaled_dot_product_attention(q, k, v)
    o = _unheads(o).to(x.dtype)

    out = block.proj_out(torch.cat([o, mlp_h], dim=2))
    hidden = residual + gate.unsqueeze(1) * out
    if return_kv_img_from is not None:
        T = return_kv_img_from
        return hidden, k[:, :, T:], v[:, :, T:]
    return hidden


def _single_block_sparse_kv(
    block,
    q_fresh: torch.Tensor,        # [B, Sq, D]   fresh states: [text ; hard image]
    q_pos: torch.Tensor,          # [B, Sq]      absolute positions in the joint seq
    hard_idx: torch.Tensor,       # [B, k]       hard positions on the IMAGE grid
    T: int,
    temb: torch.Tensor,
    cos: torch.Tensor, sin: torch.Tensor,
    k_img_cache: torch.Tensor,    # [B, H, N, d] anchor image K (post-rope)
    v_img_cache: torch.Tensor,    # [B, H, N, d] anchor image V
) -> torch.Tensor:
    """Lever B: sparse block with the anchor K/V cache. Norm/K/V/MLP run only on
    the Sq fresh rows; easy image tokens contribute K/V frozen from the anchor
    (temb_a) — exact at the anchor step, a controlled temb-staleness
    approximation afterwards. Single-stream linear cost drops from
    D²(10·Sq + 2·S) to 12·Sq·D²."""
    residual = q_fresh
    q_norm, gate = block.norm(q_fresh, emb=temb)
    mlp_h = block.act_mlp(block.proj_mlp(q_norm))

    attn = block.attn
    q = _heads(attn.to_q(q_norm), attn.heads)
    k_new = _heads(attn.to_k(q_norm), attn.heads)
    v_new = _heads(attn.to_v(q_norm), attn.heads)
    if attn.norm_q is not None:
        q = attn.norm_q(q)
    if attn.norm_k is not None:
        k_new = attn.norm_k(k_new)
    cos_q = cos[q_pos].unsqueeze(1)
    sin_q = sin[q_pos].unsqueeze(1)
    q = _rope_apply(q, cos_q, sin_q)
    k_new = _rope_apply(k_new, cos_q, sin_q)          # text + hard rows, post-rope

    # scatter fresh hard K/V into the anchor image cache (dim 2 = token axis)
    idx = hard_idx[:, None, :, None].expand(-1, k_img_cache.shape[1], -1,
                                            k_img_cache.shape[-1])
    k_img = k_img_cache.scatter(2, idx, k_new[:, :, T:].to(k_img_cache.dtype))
    v_img = v_img_cache.scatter(2, idx, v_new[:, :, T:].to(v_img_cache.dtype))
    K = torch.cat([k_new[:, :, :T], k_img.to(k_new.dtype)], dim=2)
    V = torch.cat([v_new[:, :, :T], v_img.to(v_new.dtype)], dim=2)

    o = F.scaled_dot_product_attention(q, K, V)
    o = _unheads(o).to(q_fresh.dtype)
    out = block.proj_out(torch.cat([o, mlp_h], dim=2))
    return residual + gate.unsqueeze(1) * out


def _single_block_sparse(
    block,
    q_fresh: torch.Tensor,        # [B, Sq, D]   fresh states: [text ; hard image]
    ctx: torch.Tensor,            # [B, S,  D]   full ctx: fresh text/hard + cached easy
    q_pos: torch.Tensor,          # [B, Sq]      absolute positions of queries in seq
    temb: torch.Tensor,
    cos: torch.Tensor, sin: torch.Tensor,     # full-sequence rope tables [S, d]
) -> torch.Tensor:
    """Hard-query single-stream block: attention Sq x S, MLP on Sq only."""
    residual = q_fresh
    ctx_norm, gate = block.norm(ctx, emb=temb)              # per-token LN, batch-level mod
    q_norm = _gather_tokens(ctx_norm, q_pos)                # == norm(q_fresh): ctx holds fresh
    mlp_h = block.act_mlp(block.proj_mlp(q_norm))

    attn = block.attn
    q = _heads(attn.to_q(q_norm), attn.heads)
    k = _heads(attn.to_k(ctx_norm), attn.heads)
    v = _heads(attn.to_v(ctx_norm), attn.heads)
    if attn.norm_q is not None:
        q = attn.norm_q(q)
    if attn.norm_k is not None:
        k = attn.norm_k(k)
    cos_q = cos[q_pos].unsqueeze(1)                         # [B,1,Sq,d]
    sin_q = sin[q_pos].unsqueeze(1)
    q = _rope_apply(q, cos_q, sin_q)
    k = _rope_apply(k, cos, sin)
    o = F.scaled_dot_product_attention(q, k, v)
    o = _unheads(o).to(q_fresh.dtype)

    out = block.proj_out(torch.cat([o, mlp_h], dim=2))
    return residual + gate.unsqueeze(1) * out


# ------------------------------------------------------------------- runner ---
def estimate_transformer_macs(T: int, N: int, k: int, n_dual: int, n_single: int,
                              D: int, mlp_mult: int = 4,
                              kv_cached: bool = False) -> dict:
    """Analytic MAC estimate for one forward (Fix 10). Sparse execution does NOT
    make everything sparse; per single-stream block only Q / MLP / proj_out and
    the query-side attention shrink to Sq = T + k, while context norm and K/V
    projections stay full-S, and the 19 dual blocks stay fully dense. This
    function makes that explicit so 'ran 30% of tokens' is never conflated with
    '30% of transformer compute'. Returns MACs for dense and sparse plus ratio."""
    S = T + N
    Sq = T + k
    m = mlp_mult
    # dual block (both streams dense in every mode): qkv+out 4·D² per token per
    # stream, MLP 2·m·D² per token, joint attention 2·S²·D
    dual = n_dual * ((4 + 2 * m) * (N + T) * D * D + 2 * S * S * D)
    # single block dense: q/k/v 3·S·D², mlp m·S·D², proj_out (1+m)·S·D², attn 2·S²·D
    single_dense = n_single * ((3 + m + 1 + m) * S * D * D + 2 * S * S * D)
    # single block sparse: q Sq + k/v (2·S dense / 2·Sq with the anchor KV cache),
    # mlp m·Sq, proj_out (1+m)·Sq, attn 2·Sq·S·D
    kv_rows = 2 * Sq if kv_cached else 2 * S
    single_sparse = n_single * (
        ((1 + m + 1 + m) * Sq + kv_rows) * D * D + 2 * Sq * S * D)
    final_dense = N * D * 64
    final_sparse = k * D * 64
    dense = dual + single_dense + final_dense
    sparse = dual + single_sparse + final_sparse
    return {"dense_macs": dense, "sparse_macs": sparse,
            "mac_ratio": sparse / dense,
            "single_stream_share_dense": single_dense / dense}


@dataclass
class ForwardStats:
    mode: str                      # dense | anchor | sparse
    hard_ratio: float = 1.0
    single_attn_fraction: float = 1.0   # (Sq*S)/(S*S) averaged over single blocks
    single_linear_fraction: float = 1.0 # Sq/S (Q/MLP/proj_out side only; K/V stay full)
    est_transformer_mac_ratio: float = 1.0  # whole 57-block estimate incl. dense dual + full K/V


class FluxSparseRunner:
    """Owns the decomposed transformer forward. One instance per transformer."""

    def __init__(self, transformer):
        self.t = transformer

    # -------------------------------------------------------------- embeds ---
    def _embed(self, packed_model_input, prompt_embeds, pooled, timestep, guidance,
               img_ids, txt_ids):
        t = self.t
        x = t.x_embedder(packed_model_input)                       # [B, N, D]
        ctx = t.context_embedder(prompt_embeds)                    # [B, T, D]
        ts = timestep.to(x.dtype) * 1000
        if guidance is not None:
            g = guidance.to(x.dtype) * 1000
            temb = t.time_text_embed(ts, g, pooled)
        else:
            temb = t.time_text_embed(ts, pooled)
        ids = torch.cat([txt_ids, img_ids], dim=0)                 # [T+N, 3]
        rope = t.pos_embed(ids)
        cos, sin = (rope if isinstance(rope, tuple) else (rope[0], rope[1]))
        return x, ctx, temb, cos, sin

    def _dual_stream(self, x, ctx, temb, cos, sin):
        for block in self.t.transformer_blocks:
            out = block(hidden_states=x, encoder_hidden_states=ctx,
                        temb=temb, image_rotary_emb=(cos, sin))
            # diffusers returns (encoder_hidden_states, hidden_states)
            a, b = out
            if a.shape[1] == ctx.shape[1]:
                ctx, x = a, b
            else:
                x, ctx = a, b
        return x, ctx

    def _final(self, image_states, temb):
        h = self.t.norm_out(image_states, temb)
        return self.t.proj_out(h)                                   # [B, N, 64]

    # --------------------------------------------------------------- dense ---
    @torch.no_grad()
    def dense_forward(
        self,
        packed_model_input: torch.Tensor,      # [B, N, 384]
        prompt_embeds: torch.Tensor,
        pooled: torch.Tensor,
        timestep: torch.Tensor,                # [B] in [0,1] (sigma-style, pipeline units)
        guidance: torch.Tensor | None,
        img_ids: torch.Tensor,
        txt_ids: torch.Tensor,
        cache: FluxAnchorCache | None = None,  # pass to record an anchor
        step_index: int = -1,
        record_kv: bool = False,               # Lever B: also record image K/V per block
    ):
        x, ctx, temb, cos, sin = self._embed(
            packed_model_input, prompt_embeds, pooled, timestep, guidance, img_ids, txt_ids)
        x, ctx = self._dual_stream(x, ctx, temb, cos, sin)

        T = ctx.shape[1]
        cat = torch.cat([ctx, x], dim=1)
        if cache is not None:
            cache.begin_anchor(timestep, step_index)
            cache.entry_text_states = ctx.detach()
            cache.entry_image_states = x.detach()
        for block in self.t.single_transformer_blocks:
            if cache is not None:
                cache.record_single_input(cat[:, T:])
                if record_kv:
                    cat, k_img, v_img = _single_block_dense(
                        block, cat, temb, cos, sin, return_kv_img_from=T)
                    cache.record_single_kv(k_img, v_img)
                    continue
            cat = _single_block_dense(block, cat, temb, cos, sin)

        v = self._final(cat[:, T:], temb)
        if cache is not None:
            cache.finish_anchor(v, ctx, x)
            cache.image_token_positions = torch.arange(
                T, T + x.shape[1], device=x.device)
        stats = ForwardStats(mode="anchor" if cache is not None else "dense")
        return v, stats

    # -------------------------------------------------------------- sparse ---
    @torch.no_grad()
    def sparse_forward(
        self,
        packed_model_input: torch.Tensor,
        prompt_embeds: torch.Tensor,
        pooled: torch.Tensor,
        timestep: torch.Tensor,
        guidance: torch.Tensor | None,
        img_ids: torch.Tensor,
        txt_ids: torch.Tensor,
        cache: FluxAnchorCache,
        hard_idx: torch.Tensor,                # [B, k] image-token indices
        kv_cache: bool = False,                # Lever B: anchor K/V for easy tokens
    ):
        """Selective refresh. Returns (v_hard [B,k,64], stats).
        The sampler merges: v = scatter(cache.final_prediction, hard_idx, v_hard)."""
        assert not cache.is_empty(), "sparse_forward requires a recorded anchor"
        x, ctx, temb, cos, sin = self._embed(
            packed_model_input, prompt_embeds, pooled, timestep, guidance, img_ids, txt_ids)

        # dual stream stays dense (safety rule) — full image token set
        x, ctx = self._dual_stream(x, ctx, temb, cos, sin)

        B, N, D = x.shape
        T = ctx.shape[1]
        k = hard_idx.shape[1]
        S = T + N
        dev = x.device

        # fresh queries: [text ; hard image]
        q_text = ctx
        q_hard = _gather_tokens(x, hard_idx)
        text_pos = torch.arange(T, device=dev).unsqueeze(0).expand(B, -1)
        hard_pos = hard_idx + T
        q_pos = torch.cat([text_pos, hard_pos], dim=1)              # [B, T+k]

        q_fresh = torch.cat([q_text, q_hard], dim=1)
        attn_frac, lin_frac = 0.0, 0.0
        if kv_cache:
            assert cache.single_block_kv, \
                "kv_cache=True requires dense_forward(..., record_kv=True) at the anchor"
        for j, block in enumerate(self.t.single_transformer_blocks):
            if kv_cache:
                k_img, v_img = cache.single_block_kv[j]
                q_fresh = _single_block_sparse_kv(block, q_fresh, q_pos, hard_idx, T,
                                                  temb, cos, sin, k_img, v_img)
                lin_frac += (T + k) / S              # K/V도 Sq행으로 축소
            else:
                cached_img = cache.single_block_inputs[j]           # [B, N, D] anchor depth-j
                ctx_img = _scatter_tokens(cached_img.to(x.dtype), hard_idx, q_fresh[:, T:])
                full_ctx = torch.cat([q_fresh[:, :T], ctx_img], dim=1)
                q_fresh = _single_block_sparse(block, q_fresh, full_ctx, q_pos, temb, cos, sin)
                lin_frac += (T + k) / S
            attn_frac += (T + k) / S
        n_single = len(self.t.single_transformer_blocks)

        v_hard = self._final(q_fresh[:, T:], temb)                  # [B, k, 64]
        macs = estimate_transformer_macs(
            T, N, k, len(self.t.transformer_blocks), n_single, x.shape[-1],
            kv_cached=kv_cache)
        stats = ForwardStats(
            mode="sparse",
            hard_ratio=k / N,
            single_attn_fraction=attn_frac / n_single,
            single_linear_fraction=lin_frac / n_single,
            est_transformer_mac_ratio=macs["mac_ratio"],
        )
        return v_hard, stats

    # ---------------------------------------------------------- accounting ---
    @staticmethod
    def merge_prediction(cache: FluxAnchorCache, hard_idx: torch.Tensor,
                         v_hard: torch.Tensor) -> torch.Tensor:
        """v_i = fresh for hard, anchor-cached for easy (plan Sec. 1 merge rule)."""
        return _scatter_tokens(cache.final_prediction.to(v_hard.dtype), hard_idx, v_hard)
