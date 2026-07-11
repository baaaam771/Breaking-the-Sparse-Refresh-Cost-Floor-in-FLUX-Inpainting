"""training.train_router — Stage 6 step 2: train the CNN router on teacher dumps.

Pairs are composed on the fly from full trajectories: for a shard and step i,
anchor a = i - (i % c) with c sampled from --cache-periods, giving the router
the exact input distribution it sees at inference (z_t, mask, cached v_a, t).
Label per token (plan Sec. 14):  y_i = 1[ ||v_i(t) - v_i(a)||^2 > tau ].

Checkpoints are atomic (tmp+rename), rolling (last 3 kept), resumable
(--resume auto-picks the latest), and carry model/EMA/optimizer/step —
the same discipline as the ImageNet-64 target.pt retraining.

    PYTHONPATH=. python -m training.train_router \
        --teacher /mnt/HDD_12TB/bam_ki/flux_fill/router_teacher \
        --out /mnt/HDD_12TB/bam_ki/flux_fill/router_ckpt \
        --steps 100000 --tau 1e-4
"""
from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

import torch

import torch.nn.functional as _F

from models.drafts.cnn_router import CNNRouter, router_bce_loss


def quantile_labels(v_now, v_anchor, q: float):
    """y_i = 1[change_i > per-sample quantile q]. Selector는 top-k RANKING으로
    소비되므로 절대 임계(tau)보다 분위 라벨이 목적과 정합적 (FLUX v 스케일에
    대한 tau 재보정도 불필요)."""
    d = (v_now - v_anchor).pow(2).mean(-1)               # [B, N]
    thr = torch.quantile(d, q, dim=1, keepdim=True)
    return (d > thr).float()


def router_quantile_loss(logits, v_now, v_anchor, q: float):
    return _F.binary_cross_entropy_with_logits(
        logits, quantile_labels(v_now, v_anchor, q))


class TeacherPairs:
    """Lazy pair sampler over trajectory shards (holdout split by shard hash)."""

    def __init__(self, root: str, cache_periods, split: str, holdout_frac=0.1):
        idx = json.load(open(Path(root) / "index.json"))
        shards = sorted(idx["shards"])
        cut = max(1, int(len(shards) * holdout_frac))
        self.shards = shards[cut:] if split == "train" else shards[:cut]
        assert self.shards, f"no shards for split={split}"
        self.root = Path(root)
        self.steps = idx["steps"]
        self.cache_periods = list(cache_periods)
        self._cache: dict[str, dict] = {}

    def _shard(self, name):
        if name not in self._cache:
            if len(self._cache) > 8:               # bounded RAM
                self._cache.pop(next(iter(self._cache)))
            self._cache[name] = torch.load(self.root / name, map_location="cpu")
        return self._cache[name]

    def sample(self, rng: random.Random):
        sh = self._shard(rng.choice(self.shards))
        c = rng.choice(self.cache_periods)
        i = rng.randrange(1, self.steps)
        if i % c == 0:                             # anchor steps have no sparse pair
            i = min(i + 1, self.steps - 1)
        a = i - (i % c)
        return {
            "latent": sh["latents"][i].float(),    # [N, 64] z_t
            "v_now": sh["preds"][i].float(),
            "v_anchor": sh["preds"][a].float(),
            "mask_tok": sh["mask_tok"].float(),
            "sigma": sh["sigmas"][i].float(),
            "token_hw": sh["token_hw"],
        }

    def batch(self, bs, rng, device):
        items = [self.sample(rng) for _ in range(bs)]
        hw = items[0]["token_hw"]
        stack = lambda k: torch.stack([it[k] for it in items]).to(device)
        return (stack("latent"), stack("mask_tok"), stack("v_anchor"),
                stack("sigma"), stack("v_now"), tuple(hw))


def _save_ckpt(path: Path, step, model, ema, opt, keep=3):
    path.mkdir(parents=True, exist_ok=True)
    p = path / f"router_{step:07d}.pt"
    tmp = p.with_suffix(".pt.tmp")
    torch.save({"step": step, "model": model.state_dict(),
                "ema": ema.state_dict(), "opt": opt.state_dict()}, tmp)
    tmp.rename(p)                                   # atomic
    for old in sorted(path.glob("router_*.pt"))[:-keep]:
        old.unlink()
    return p


@torch.no_grad()
def _auroc(model, pairs, rng, device, tau, n_batches=20, bs=8,
           label_mode="tau", q=0.7):
    """Threshold-free ranking quality of the router on held-out shards."""
    model.eval()
    scores, labels = [], []
    for _ in range(n_batches):
        lat, mt, va, sig, vn, hw = pairs.batch(bs, rng, device)
        logits = model(lat, mt, va, sig, hw)
        y = (quantile_labels(vn, va, q) if label_mode == "quantile"
             else ((vn - va).pow(2).mean(-1) > tau).float())
        scores.append(logits.flatten().cpu())
        labels.append(y.flatten().cpu())
    s, y = torch.cat(scores), torch.cat(labels)
    if y.min() == y.max():
        return float("nan")
    order = s.argsort()
    ranks = torch.empty_like(order, dtype=torch.float)
    ranks[order] = torch.arange(len(s), dtype=torch.float)
    n_pos, n_neg = y.sum(), (1 - y).sum()
    return ((ranks[y == 1].sum() - n_pos * (n_pos - 1) / 2) / (n_pos * n_neg)).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=100_000)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--tau", type=float, default=1e-4)
    ap.add_argument("--label-mode", choices=["quantile", "tau"], default="quantile")
    ap.add_argument("--quantile", type=float, default=0.7)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--cache-periods", type=int, nargs="+", default=[2, 3, 4, 5])
    ap.add_argument("--width", type=int, default=96)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--save-every", type=int, default=5000)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    dev = a.device
    train = TeacherPairs(a.teacher, a.cache_periods, "train")
    val = TeacherPairs(a.teacher, a.cache_periods, "val")
    rng, vrng = random.Random(0), random.Random(1)

    model = CNNRouter(a.width, a.depth).to(dev)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)

    start = 0
    out = Path(a.out)
    if a.resume:
        cks = sorted(out.glob("router_*.pt"))
        if cks:
            ck = torch.load(cks[-1], map_location=dev)
            model.load_state_dict(ck["model"]); ema.load_state_dict(ck["ema"])
            opt.load_state_dict(ck["opt"]); start = ck["step"]
            print(f"resumed {cks[-1].name} @ step {start}")

    for step in range(start, a.steps):
        model.train()
        lat, mt, va, sig, vn, hw = train.batch(a.bs, rng, dev)
        logits = model(lat, mt, va, sig, hw)
        loss = (router_quantile_loss(logits, vn, va, a.quantile)
                if a.label_mode == "quantile"
                else router_bce_loss(logits, vn, va, a.tau))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), a.grad_clip)
        opt.step()
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.lerp_(pm, 1 - a.ema)
            for be, bm in zip(ema.buffers(), model.buffers()):
                be.copy_(bm)

        if (step + 1) % a.eval_every == 0:
            auc = _auroc(ema, val, vrng, dev, a.tau,
                         label_mode=a.label_mode, q=a.quantile)
            print(f"step {step+1}: loss {loss.item():.4f}  val AUROC(EMA) {auc:.4f}")
        if (step + 1) % a.save_every == 0 or step + 1 == a.steps:
            p = _save_ckpt(out, step + 1, model, ema, opt)
            print(f"saved {p.name}")


if __name__ == "__main__":
    main()
