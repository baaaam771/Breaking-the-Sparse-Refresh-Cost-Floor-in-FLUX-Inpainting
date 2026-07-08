"""eval.region_metrics — plan Sec. 18 quality metrics (Fixes 5, 6 적용판).

Metric naming (Fix 5): everything measured against the 50-step dense output is
suffixed `_to_ref` — it is a FINAL-OUTPUT divergence, not a per-timestep
trajectory metric. (A true LPIPS_t would compare per-step x0_hat sequences.)

Output separation (Fix 6): the sampler saves both
    {stem}.png          raw model output
    {stem}_pasted.png   M*x_model + (1-M)*x_input  (composited)
because FLUX Fill is mask-conditioned but does not mathematically pin known
pixels. We report raw AND pasted variants, plus known-region preservation
against the ORIGINAL INPUT (the question that actually matters), which needs
--manifest to reload inputs.

    python -m eval.region_metrics --run out/mbfd_r03 --ref out/dense_s50 \\
        --manifest data/coco_manifest.json --out out/mbfd_r03/metrics.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F


def _load_png(path) -> torch.Tensor:
    from PIL import Image
    import numpy as np
    return torch.from_numpy(np.array(Image.open(path).convert("RGB"))).permute(2, 0, 1)[None].float() / 255.0


def boundary_ring(mask: torch.Tensor, px: int = 32) -> torch.Tensor:
    """[1,1,H,W] mask -> 32px ring around the seam (dilate - erode at pixel res)."""
    k = 2 * px + 1
    dil = F.max_pool2d(mask, k, stride=1, padding=px)
    ero = -F.max_pool2d(-mask, k, stride=1, padding=px)
    return (dil - ero).clamp(0, 1)


class Metrics:
    def __init__(self, device="cuda"):
        import lpips
        # spatial=True returns a per-pixel distance map -> exact region averaging
        self.lpips = lpips.LPIPS(net="vgg", spatial=True).to(device).eval()
        self.device = device

    @torch.no_grad()
    def region_lpips(self, a, b, region):
        """Mean LPIPS distance over `region` ([1,1,H,W] in {0,1}); pass ones for global."""
        dmap = self.lpips(a * 2 - 1, b * 2 - 1)            # [1,1,h,w]
        r = F.interpolate(region, size=dmap.shape[-2:], mode="nearest")
        return ((dmap * r).sum() / r.sum().clamp_min(1)).item()

    @staticmethod
    def psnr(a, b, region=None):
        if region is not None:
            diff = ((a - b) ** 2 * region).sum() / (region.sum() * a.shape[1]).clamp_min(1)
        else:
            diff = ((a - b) ** 2).mean()
        return (10 * torch.log10(1.0 / diff.clamp_min(1e-12))).item()

    @staticmethod
    @torch.no_grad()
    def region_ssim(a, b, region):
        """Mean of the per-pixel SSIM map over `region` (known-region SSIM).
        Uses torchmetrics' full similarity image so the restriction is exact."""
        from torchmetrics.functional.image import structural_similarity_index_measure
        _, smap = structural_similarity_index_measure(
            a, b, data_range=1.0, return_full_image=True)
        r = F.interpolate(region, size=smap.shape[-2:], mode="nearest")
        return ((smap.mean(1, keepdim=True) * r).sum() / r.sum().clamp_min(1)).item()


def _inputs_by_stem(manifest: str) -> dict:
    from data.dataset import FluxFillBenchmark
    ds = FluxFillBenchmark(manifest)
    return {Path(ds[i]["sample_id"]).stem: i for i in range(len(ds))}, ds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="method output dir (pngs + *_mask.pt)")
    ap.add_argument("--ref", required=True, help="dense 50-step reference dir")
    ap.add_argument("--manifest", default="",
                    help="reload ORIGINAL inputs for known-region preservation metrics")
    ap.add_argument("--gt", default="", help="optional ground-truth image dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    M = Metrics(a.device)
    stem_map = ds = None
    if a.manifest:
        stem_map, ds = _inputs_by_stem(a.manifest)

    rows = []
    run_dir, ref_dir = Path(a.run), Path(a.ref)
    for png in sorted(run_dir.glob("*.png")):
        if png.stem.endswith("_pasted"):
            continue
        ref_png = ref_dir / png.name
        mask_pt = run_dir / f"{png.stem}_mask.pt"
        if not ref_png.exists() or not mask_pt.exists():
            continue
        x = _load_png(png).to(a.device)
        r = _load_png(ref_png).to(a.device)
        mask = torch.load(mask_pt).to(a.device)[None]          # [1,1,H,W]
        ring = boundary_ring(mask)
        known = 1 - mask

        row = {
            "sample_id": png.stem,
            # Fix 5: *_to_ref = divergence from the 50-step dense FINAL output
            "lpips_to_ref": M.region_lpips(x, r, torch.ones_like(mask)),
            "mask_lpips_to_ref": M.region_lpips(x, r, mask),
            "boundary_lpips_to_ref": M.region_lpips(x, r, ring),
            "known_psnr_to_ref": M.psnr(x, r, known),
            "known_ssim_to_ref": M.region_ssim(x, r, known),
        }
        pasted_p = run_dir / f"{png.stem}_pasted.png"
        if pasted_p.exists():
            xp = _load_png(pasted_p).to(a.device)
            ref_pasted = ref_dir / f"{png.stem}_pasted.png"
            rp = _load_png(ref_pasted).to(a.device) if ref_pasted.exists() else r
            row["mask_lpips_pasted_to_ref"] = M.region_lpips(xp, rp, mask)
            row["boundary_lpips_pasted_to_ref"] = M.region_lpips(xp, rp, ring)  # seam quality
        if ds is not None and png.stem in stem_map:
            inp = ds[stem_map[png.stem]]["image"][None].to(a.device)
            # Fix 6 core question: how well does the RAW output preserve the input?
            row["known_psnr_to_input"] = M.psnr(x, inp, known)
            row["known_ssim_to_input"] = M.region_ssim(x, inp, known)
            row["mask_lpips_to_input"] = M.region_lpips(x, inp, mask)  # diagnostic
        if a.gt:
            gt = _load_png(Path(a.gt) / png.name).to(a.device)
            row["mask_lpips_to_gt"] = M.region_lpips(x, gt, mask)
            row["known_psnr_to_gt"] = M.psnr(x, gt, known)
        rows.append(row)

    keys = sorted({k for r in rows for k in r if k != "sample_id"})
    agg = {k: sum(r[k] for r in rows if k in r) / max(sum(k in r for r in rows), 1)
           for k in keys} if rows else {}
    json.dump({"n": len(rows), "aggregate": agg, "rows": rows},
              open(a.out, "w"), indent=1)
    print(json.dumps({"n": len(rows), "aggregate": agg}, indent=1))


if __name__ == "__main__":
    main()
