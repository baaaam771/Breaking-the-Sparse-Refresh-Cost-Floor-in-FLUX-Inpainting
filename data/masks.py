"""data.masks — deterministic inpainting mask generation.

Requirements from the plan (Sec. 5):
  * mask types: box, free-form brush, irregular polygon
  * coverage buckets:  small < 15%,  15% <= medium <= 35%,  large > 35%
  * the SAME image·mask·prompt·seed tuple must be shared by every method
    -> masks are a pure function of (sample_id, mask_seed, type, bucket)

Pure torch, CPU-friendly, unit-testable.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import torch

BUCKETS = {"small": (0.02, 0.15), "medium": (0.15, 0.35), "large": (0.35, 0.60)}
MASK_TYPES = ("box", "brush", "polygon")


def stable_seed(*parts: object) -> int:
    """Cross-process deterministic seed. Python's builtin hash() is salted per
    process (PYTHONHASHSEED) for str inputs, so it must NEVER seed benchmark
    randomness — different stages / sessions would silently get different
    masks. SHA-256 of the joined parts is stable everywhere."""
    text = "||".join(map(str, parts))
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") & 0x7FFFFFFF


def _gen(seed: int) -> torch.Generator:
    g = torch.Generator("cpu")
    g.manual_seed(seed)
    return g


def _coverage(m: torch.Tensor) -> float:
    return m.float().mean().item()


def _stamp_disk(m: torch.Tensor, y: int, x: int, radius: int):
    """Set a filled disk to 1 touching only its bounding box (O(r²), not O(HW))."""
    H, W = m.shape[-2:]
    y0, y1 = max(0, y - radius), min(H, y + radius + 1)
    x0, x1 = max(0, x - radius), min(W, x + radius + 1)
    ly = torch.arange(y0, y1).unsqueeze(1)
    lx = torch.arange(x0, x1).unsqueeze(0)
    disk = (ly - y).square() + (lx - x).square() <= radius * radius
    patch = m[0, y0:y1, x0:x1]
    patch[disk] = 1.0


def box_mask(H: int, W: int, target_cov: float, g: torch.Generator) -> torch.Tensor:
    m = torch.zeros(1, H, W)
    ar = 0.5 + torch.rand(1, generator=g).item()            # aspect in [0.5, 1.5]
    h = min(H, max(8, int(round(math.sqrt(target_cov * H * W * ar)))))
    w = min(W, max(8, int(round(target_cov * H * W / h))))
    top = torch.randint(0, max(1, H - h + 1), (1,), generator=g).item()
    left = torch.randint(0, max(1, W - w + 1), (1,), generator=g).item()
    m[:, top:top + h, left:left + w] = 1.0
    return m


def brush_mask(H: int, W: int, target_cov: float, g: torch.Generator) -> torch.Tensor:
    """Random-walk thick strokes; strokes are added until coverage enters range.
    Disks are stamped via their bounding box only (O(r²) per point, not O(HW))."""
    m = torch.zeros(1, H, W)
    radius = max(6, int(0.04 * min(H, W)))
    for _ in range(64):
        y = torch.randint(radius, H - radius, (1,), generator=g).item()
        x = torch.randint(radius, W - radius, (1,), generator=g).item()
        ang = torch.rand(1, generator=g).item() * 2 * math.pi
        for _ in range(torch.randint(8, 24, (1,), generator=g).item()):
            _stamp_disk(m, y, x, radius)
            ang += (torch.rand(1, generator=g).item() - 0.5) * 1.2
            step = radius * (1.0 + torch.rand(1, generator=g).item())
            y = int(min(max(y + step * math.sin(ang), radius), H - radius - 1))
            x = int(min(max(x + step * math.cos(ang), radius), W - radius - 1))
        if _coverage(m) >= target_cov:
            break
    return m


def polygon_mask(H: int, W: int, target_cov: float, g: torch.Generator) -> torch.Tensor:
    """Irregular star-convex polygon around a random center, rasterized by angle test."""
    cy = (0.25 + 0.5 * torch.rand(1, generator=g).item()) * H
    cx = (0.25 + 0.5 * torch.rand(1, generator=g).item()) * W
    base_r = math.sqrt(target_cov * H * W / math.pi)
    n_vert = torch.randint(5, 12, (1,), generator=g).item()
    radii = base_r * (0.6 + 0.8 * torch.rand(n_vert, generator=g))            # jagged
    angles = torch.linspace(0, 2 * math.pi, n_vert + 1)[:-1]
    yy, xx = torch.meshgrid(torch.arange(H).float(), torch.arange(W).float(), indexing="ij")
    theta = torch.atan2(yy - cy, xx - cx) % (2 * math.pi)
    dist = torch.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    # piecewise-linear interpolation of the radius over angle
    idx = (theta / (2 * math.pi) * n_vert)
    lo = idx.floor().long() % n_vert
    hi = (lo + 1) % n_vert
    frac = idx - idx.floor()
    r_at = radii[lo] * (1 - frac) + radii[hi] * frac
    return (dist <= r_at).float().unsqueeze(0)


_FN = {"box": box_mask, "brush": brush_mask, "polygon": polygon_mask}


@dataclass(frozen=True)
class MaskSpec:
    sample_id: str
    mask_type: str      # box | brush | polygon
    bucket: str         # small | medium | large
    seed: int = 0


def make_mask(H: int, W: int, spec: MaskSpec) -> torch.Tensor:
    """Deterministic [1, H, W] mask in {0,1}, 1 = regenerate."""
    assert spec.mask_type in MASK_TYPES and spec.bucket in BUCKETS
    lo, hi = BUCKETS[spec.bucket]
    g = _gen(stable_seed(spec.sample_id, spec.mask_type, spec.bucket, spec.seed))
    target = lo + (hi - lo) * torch.rand(1, generator=g).item()
    m = _FN[spec.mask_type](H, W, target, g)
    return (m > 0.5).float()


def spec_for_index(sample_id: str, index: int, seed: int = 0) -> MaskSpec:
    """Balanced round-robin over 3 types x 3 buckets, deterministic in index."""
    t = MASK_TYPES[index % 3]
    b = list(BUCKETS)[(index // 3) % 3]
    return MaskSpec(sample_id=sample_id, mask_type=t, bucket=b, seed=seed)
