"""Stage 3 gate: pack/unpack roundtrip, mask->token->pixel consistency,
row-major ordering, structured block selection. Pure CPU — runs anywhere."""
import torch

from utils.token_mapping import (TokenGrid, blockify_scores, grid_to_token_scores,
                                 hard_easy_split, latent_mask_to_token, pack_latents,
                                 pixel_mask_to_latent, pixel_mask_to_token,
                                 token_mask_to_pixel, token_scores_to_grid,
                                 unpack_latents)


def test_pack_unpack_roundtrip():
    grid = TokenGrid(512, 768).validate()
    hl, wl = grid.latent_hw
    lat = torch.randn(2, 16, hl, wl)
    packed = pack_latents(lat)
    assert packed.shape == (2, grid.num_image_tokens, 64)
    rec = unpack_latents(packed, grid, channels=16)
    assert torch.equal(rec, lat)


def test_row_major_token_order():
    """Token i of a (Hp, Wp) grid must be row-major: i = y*Wp + x."""
    grid = TokenGrid(128, 256).validate()          # Hp=8, Wp=16
    hl, wl = grid.latent_hw
    lat = torch.zeros(1, 1, hl, wl)
    lat[0, 0, 2:4, 6:8] = 7.0                      # exactly token (y=1, x=3)
    packed = pack_latents(lat)
    nz = packed.abs().sum(-1).nonzero()[:, 1].tolist()
    hp, wp = grid.token_hw
    assert nz == [1 * wp + 3]


def test_mask_to_token_coverage():
    grid = TokenGrid(256, 256).validate()          # 16x16 tokens
    m = torch.zeros(1, 1, 256, 256)
    m[..., :16, :16] = 1                           # exactly one full token cell
    tok = pixel_mask_to_token(m, grid)
    assert tok.shape == (1, 256)
    assert torch.isclose(tok[0, 0], torch.tensor(1.0))
    assert tok[0, 1:].abs().max() == 0

    # half-covered token
    m2 = torch.zeros(1, 1, 256, 256)
    m2[..., :8, :16] = 1
    tok2 = pixel_mask_to_token(m2, grid)
    assert torch.isclose(tok2[0, 0], torch.tensor(0.5))


def test_token_overlay_roundtrip():
    """binary pixel mask aligned to 16px cells -> token -> pixel must be identity."""
    grid = TokenGrid(256, 256).validate()
    m = torch.zeros(1, 1, 256, 256)
    m[..., 32:96, 64:160] = 1                      # 16px-aligned block
    tok = (pixel_mask_to_token(m, grid) > 0.5).float()
    rec = token_mask_to_pixel(tok, grid)
    assert torch.equal(rec, m)


def test_hard_easy_split_partition():
    scores = torch.rand(3, 100)
    hard, easy = hard_easy_split(scores, 0.3)
    assert hard.shape == (3, 30) and easy.shape == (3, 70)
    for b in range(3):
        union = torch.cat([hard[b], easy[b]]).sort().values
        assert torch.equal(union, torch.arange(100))
        # hard really is top-k
        thr = scores[b][hard[b]].min()
        assert (scores[b][easy[b]] <= thr + 1e-6).all()


def test_blockify_contiguity():
    grid = TokenGrid(256, 256).validate()          # 16x16 tokens
    scores = torch.rand(1, 256)
    b2 = blockify_scores(scores, grid, 2)
    g = token_scores_to_grid(b2, grid)
    # every 2x2 window is constant
    assert torch.equal(g[..., 0::2, 0::2], g[..., 1::2, 1::2])
    hard, _ = hard_easy_split(b2, 0.25)
    hard_grid = torch.zeros(256)
    hard_grid[hard[0]] = 1
    hg = hard_grid.view(16, 16)
    assert torch.equal(hg[0::2, 0::2], hg[1::2, 1::2])  # selections are 2x2 blocks


def test_latent_hop_equivalence():
    grid = TokenGrid(512, 512).validate()
    m = (torch.rand(1, 1, 512, 512) > 0.7).float()
    direct = pixel_mask_to_token(m, grid)
    two_hop = latent_mask_to_token(pixel_mask_to_latent(m, grid), grid)
    assert torch.allclose(direct, two_hop, atol=1e-6)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
