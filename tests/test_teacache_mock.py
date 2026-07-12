"""faithful TeaCache 이식 검증 (mock transformer, GPU 불필요).
공식 정책과의 일치: (a) cnt==0/last 강제 dense, (b) 누적<thresh면 skip,
(c) skip = x_embed + prev_residual -> final head만, (d) calc가 residual 갱신,
(e) prev_mod는 매 step 갱신."""
import torch
import torch.nn as nn
from models.flux_sparse_transformer import FluxSparseRunner


class _Norm1(nn.Module):
    def forward(self, x, emb=None):
        return x * 2.0 + emb.mean(), None   # deterministic modulated input


class _DualBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = _Norm1()


class _MockT(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer_blocks = nn.ModuleList([_DualBlock()])
        self.single_transformer_blocks = nn.ModuleList([])


def _mk_runner():
    r = FluxSparseRunner.__new__(FluxSparseRunner)
    r.t = _MockT()
    B, N, T, D = 1, 8, 4, 16
    x0 = torch.randn(B, N, D)
    ctx0 = torch.randn(B, T, D)
    temb = torch.ones(B, D)
    r._embed = lambda *a, **k: (x0.clone(), ctx0.clone(), temb, None, None)
    # dense 경로: dual/single 대신 고정 변환 (x -> x + 3)
    r._dual_stream = lambda x, ctx, temb, cos, sin: (x + 3.0, ctx)
    r._final = lambda h, temb: h * 10.0
    return r, x0


def test_forced_and_skip_and_residual():
    r, x0 = _mk_runner()
    tc = {"cnt": 0, "num_steps": 4, "rel_l1_thresh": 1e9,   # 임계 매우 큼 -> 중간 step 전부 skip
          "accumulated": 0.0, "prev_mod": None, "prev_residual": None}
    args = (None, None, None, torch.tensor([0.5]), None, None, None)

    v0, calc0 = r.teacache_forward(*args, tc)             # step 0: 강제 dense
    assert calc0 and torch.allclose(v0, (x0 + 3.0) * 10.0)
    assert torch.allclose(tc["prev_residual"], torch.full_like(x0, 3.0))

    v1, calc1 = r.teacache_forward(*args, tc)             # step 1: skip (누적<thresh)
    assert not calc1
    assert torch.allclose(v1, (x0 + 3.0) * 10.0)          # x_embed + residual -> head

    v2, calc2 = r.teacache_forward(*args, tc)             # step 2: skip
    assert not calc2

    v3, calc3 = r.teacache_forward(*args, tc)             # step 3 == num_steps-1: 강제 dense
    assert calc3
    assert tc["cnt"] == 0                                  # wrap-around 리셋
    print("PASS forced-dense at first/last, skip = residual+head only")


def test_accumulation_triggers_dense():
    r, x0 = _mk_runner()
    # modulated input이 매 step 동일 -> rel=0 -> rescale(0)=coeffs[-1]=0.264...
    # thresh=0.4면 step1 누적 0.264 -> skip, step2 누적 0.528 -> dense
    tc = {"cnt": 0, "num_steps": 10, "rel_l1_thresh": 0.4,
          "accumulated": 0.0, "prev_mod": None, "prev_residual": None}
    args = (None, None, None, torch.tensor([0.5]), None, None, None)
    calcs = [r.teacache_forward(*args, tc)[1] for _ in range(4)]
    assert calcs == [True, False, True, False], calcs
    print("PASS accumulated rel-L1 triggers periodic dense (poly1d bias=0.264/step)")


if __name__ == "__main__":
    test_forced_and_skip_and_residual()
    test_accumulation_triggers_dense()
