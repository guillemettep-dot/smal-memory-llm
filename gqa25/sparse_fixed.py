"""Fixed random sparse linear layers: every output unit is wired to exactly `fan_in` inputs, the
positions drawn once at construction (seeded) and kept for life -- they live in the state_dict as a
buffer, so a checkpoint carries its own wiring.

Storage is COMPACT (only the fan_in weights per output are parameters, so the optimizer states are
sized to the real parameter count); compute is DENSE: the compact weights are scattered into a
zero matrix each forward and the product runs as one tensor-core GEMM. With fan_in 150 of 2,676
inputs the GEMM does 18x the necessary flops, but at these widths that is a few ms per micro-batch,
whereas a per-output gather has no data reuse and is memory-bound (~240 GB/micro-batch here).
"""
import torch, torch.nn as nn


def _wiring(groups, n_out, n_in, fan_in, seed):
    g = torch.Generator().manual_seed(seed)
    idx = torch.empty(groups, n_out, fan_in, dtype=torch.long)
    for a in range(groups):
        for o in range(n_out):
            idx[a, o] = torch.randperm(n_in, generator=g)[:fan_in]
    return idx


class FixedSparse(nn.Module):
    """G independent sparse maps (n_in -> n_out), e.g. one per head. x:(..., G, n_in) -> (..., G, n_out)."""
    def __init__(self, groups, n_in, n_out, fan_in, seed):
        super().__init__()
        assert 0 < fan_in <= n_in
        self.G, self.n_in, self.n_out, self.fan_in = groups, n_in, n_out, fan_in
        self.register_buffer("idx", _wiring(groups, n_out, n_in, fan_in, seed))          # (G, n_out, fan_in)
        bound = 1.0 / fan_in ** 0.5
        self.w = nn.Parameter(torch.empty(groups, n_out, fan_in).uniform_(-bound, bound))

    def dense(self, dtype=None):
        """(G, n_in, n_out): compact weights scattered to their fixed positions (differentiable)."""
        w = self.w if dtype is None else self.w.to(dtype)
        d = torch.zeros(self.G, self.n_out, self.n_in, device=w.device, dtype=w.dtype)
        d = d.scatter(2, self.idx, w)                    # out-of-place: autograd gathers the grad back
        return d.transpose(1, 2)

    def forward(self, x):
        return torch.einsum("...gi,gio->...go", x, self.dense(x.dtype))

    def extra_repr(self):
        return "groups=%d, in=%d, out=%d, fan_in=%d (params %d, dense would be %d)" % (
            self.G, self.n_in, self.n_out, self.fan_in, self.w.numel(), self.G * self.n_in * self.n_out)


if __name__ == "__main__":
    torch.manual_seed(0)
    m = FixedSparse(groups=3, n_in=64, n_out=20, fan_in=7, seed=123).cuda()
    # 1. exactly fan_in DISTINCT connections per output, inside range
    assert m.idx.shape == (3, 20, 7) and int(m.idx.max()) < 64 and int(m.idx.min()) >= 0
    assert all(len(set(m.idx[a, o].tolist())) == 7 for a in range(3) for o in range(20))
    # 2. wiring is deterministic (same seed -> same idx), stored in the state_dict, differs per group
    m2 = FixedSparse(3, 64, 20, 7, seed=123); assert torch.equal(m.idx.cpu(), m2.idx)
    assert "idx" in m.state_dict() and not torch.equal(m.idx[0], m.idx[1])
    # 3. forward == the explicit per-output gather-and-dot
    x = torch.randn(5, 3, 64, device="cuda")
    y = m(x)
    ref = torch.stack([torch.stack([(x[:, a, m.idx[a, o]] * m.w[a, o]).sum(-1) for o in range(20)], -1) for a in range(3)], 1)
    assert y.shape == (5, 3, 20) and torch.allclose(y, ref, atol=1e-5), (y - ref).abs().max()
    # 4. gradient reaches every compact weight and only its own wired input positions carry grad
    y.sum().backward(); assert m.w.grad.shape == m.w.shape and torch.isfinite(m.w.grad).all()
    xg = torch.zeros(5, 3, 64, device="cuda", requires_grad=True); m(xg).sum().backward()
    wired = torch.zeros(3, 64, device="cuda", dtype=torch.bool); [wired[a].index_fill_(0, m.idx[a].flatten(), True) for a in range(3)]
    assert (xg.grad[0].abs() > 0).cpu().eq(wired.cpu()).all() or (xg.grad[0][~wired].abs().max() == 0), "grad leaked to unwired inputs"
    # 5. the real sizes: params vs dense
    big = FixedSparse(97, 2676, 500, 150, seed=1)
    print(big); print(FixedSparse(97, 500, 128, 150, seed=2)); print(FixedSparse(1, 6208, 2048, 150, seed=3))
    print("SPARSE_OK")
