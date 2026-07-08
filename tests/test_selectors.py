"""CPU unit tests for the selector stack + mask generation determinism."""
import torch

from data.masks import BUCKETS, MaskSpec, make_mask
from token_selectors.boundary import boundary_score
from token_selectors.combo import PRESETS, combo_score, rank_norm, select_hard_tokens
from token_selectors.delta import delta_score
from token_selectors.frequency import frequency_score
from token_selectors.mask import mask_score
from utils.flow_math import clean_estimate
from utils.token_mapping import TokenGrid


def test_mask_determinism_and_buckets():
    for bucket, (lo, hi) in BUCKETS.items():
        for mt in ("box", "brush", "polygon"):
            spec = MaskSpec("img_001.jpg", mt, bucket, seed=0)
            m1, m2 = make_mask(256, 256, spec), make_mask(256, 256, spec)
            assert torch.equal(m1, m2), f"{mt}/{bucket} not deterministic"
            cov = m1.mean().item()
            assert 0.5 * lo <= cov <= 1.6 * hi, f"{mt}/{bucket} cov {cov:.3f} way off"


def test_mask_seed_is_stable():
    """stable_seed must not depend on process hash salt and must separate specs."""
    from data.masks import stable_seed
    assert stable_seed("img.jpg", "brush", "medium", 0) == \
           stable_seed("img.jpg", "brush", "medium", 0)
    assert stable_seed("img.jpg", "brush", "medium", 0) != \
           stable_seed("img.jpg", "brush", "medium", 1)
    assert stable_seed("img.jpg", "brush", "medium", 0) != \
           stable_seed("img.jpg", "box", "medium", 0)


def test_mask_determinism_across_processes():
    """The REAL guarantee: two fresh Python processes (different hash salts by
    construction) must produce byte-identical masks for the same spec."""
    import subprocess
    import sys
    code = (
        "import hashlib, torch; from data.masks import MaskSpec, make_mask; "
        "m = make_mask(256, 256, MaskSpec('img_007.jpg', 'brush', 'large', 3)); "
        "print(hashlib.sha256(m.numpy().tobytes()).hexdigest())"
    )
    outs = []
    for salt in ("0", "12345"):
        env = dict(__import__("os").environ, PYTHONHASHSEED=salt, PYTHONPATH=".")
        r = subprocess.run([sys.executable, "-c", code], capture_output=True,
                           text=True, env=env, check=True)
        outs.append(r.stdout.strip())
    assert outs[0] == outs[1], f"masks differ across processes: {outs}"


def test_rank_norm_bounds_and_order():
    x = torch.tensor([[3.0, 1.0, 2.0, 10.0]])
    r = rank_norm(x)
    assert r.min() == 0 and r.max() == 1
    assert torch.equal(r.argsort(), x.argsort())


def test_boundary_is_a_band():
    grid = TokenGrid(256, 256).validate()
    m = torch.zeros(1, 1, 256, 256)
    m[..., 64:192, 64:192] = 1
    mt = mask_score(m, grid)
    b = boundary_score(mt, grid, kernel=3)
    inside = mt.view(16, 16)[6:10, 6:10]           # deep interior tokens
    bg = b.view(16, 16)
    assert bg[6:10, 6:10].sum() == 0               # interior excluded
    assert bg[3:5, 4:12].sum() > 0                 # seam included
    assert b.max() <= 1 and b.min() >= 0


def test_frequency_targets_texture():
    grid = TokenGrid(256, 256).validate()          # latent 32x32, tokens 16x16
    lat = torch.zeros(1, 16, 32, 32)
    lat[..., 16:, 16:] = torch.randn(1, 16, 16, 16)    # HF texture in one quadrant
    f = frequency_score(lat, grid).view(16, 16)
    assert f[8:, 8:].mean() > 5 * f[:8, :8].mean() + 1e-6


def test_delta_and_combo_shapes():
    B, N, C = 2, 64, 16
    va, vb = torch.randn(B, N, C), torch.randn(B, N, C)
    d = delta_score(va, vb)
    assert d.shape == (B, N) and (d >= 0).all()
    grid = TokenGrid(128, 128).validate()          # 8x8 = 64 tokens
    m = (torch.rand(B, 1, 128, 128) > 0.6).float()
    mt = mask_score(m, grid)
    bt = boundary_score(mt, grid)
    s = combo_score(PRESETS["mbfd"], mask=mt, boundary=bt,
                    frequency=torch.rand(B, N), delta=d)
    assert s.shape == (B, N)
    hard, easy, r_act = select_hard_tokens(s, grid, 0.25)
    assert abs(r_act - 0.25) < 1e-6
    assert hard.shape[1] == 16


def test_mask_dominates_when_alone():
    """PRESETS['mask'] must pick exactly the mask tokens when coverage is binary."""
    grid = TokenGrid(128, 128).validate()
    m = torch.zeros(1, 1, 128, 128)
    m[..., :64, :64] = 1                            # 16 of 64 tokens fully covered
    mt = mask_score(m, grid)
    s = combo_score(PRESETS["mask"], mask=mt, boundary=None)
    hard, _, _ = select_hard_tokens(s, grid, 16 / 64)
    picked = torch.zeros(64)
    picked[hard[0]] = 1
    assert torch.equal(picked, (mt[0] > 0.5).float())


def test_clean_estimate_flow_convention():
    """x0 = z - sigma*v must invert z = (1-s)x0 + s*eps with v = eps - x0."""
    x0 = torch.randn(1, 8, 16)
    eps = torch.randn(1, 8, 16)
    s = torch.tensor(0.37)
    z = (1 - s) * x0 + s * eps
    v = eps - x0
    rec = clean_estimate(z, v, s)
    assert torch.allclose(rec, x0, atol=1e-6)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
