# Block-sparse fixed-random layer with fused GELU, for the compressor's wide first stage.
#   wiring : each 32-wide OUTPUT block of each group reads NB random 32-wide INPUT blocks, drawn once from a seed,
#            distinct within the output block, never moved (a buffer in the state_dict).
#   input  : NOT materialised. Three sources per (token, group): the raw state (a view of the scan emission,
#            K_S columns), the probe reads (K_R) and the shared ask (K_A, identical for every group); blocks are
#            addressed by id: [0, K_S/32) state, then reads, then ask. GELU is applied on load.
#   fwd    : out(N,G,NOUT) fp32 = sum_j gelu(X_blk[idx]) @ W_blk           (program = token tile x group x out-block)
#   bwd dX : per INPUT block, the out-blocks that read it (inverse map, padded) -> dX = sum dH @ W^T, times gelu'
#            written straight into the three source gradients (state grad has the emission's layout)
#   bwd dW : per (group, out-block, slot): sum over tokens of gelu(X)^T @ dH
import torch, triton, triton.language as tl
BLK = 32
import os as _os
FUSED_BWD = _os.environ.get("BS_FUSED_BWD") is not None   # opt-in: one input-stationary backward kernel vs the two-kernel version
def make_wiring(groups, n_in, n_out, nb, seed):
    assert n_in % BLK == 0 and n_out % BLK == 0
    nib, nob = n_in // BLK, n_out // BLK
    nb = min(nb, nib)                                            # a small input cannot supply nb distinct blocks
    gen = torch.Generator().manual_seed(seed)
    idx = torch.stack([torch.stack([torch.randperm(nib, generator=gen)[:nb] for _ in range(nob)]) for _ in range(groups)])
    return idx.to(torch.int32)                                   # (G, NOB, NB)
def inverse_map(idx, n_in):
    """(G, NIB, MAXC) int32 of packed (ob*NB + j) reading each input block, -1 padded."""
    G, NOB, NB = idx.shape; nib = n_in // BLK
    lists = [[[] for _ in range(nib)] for _ in range(G)]
    for g in range(G):
        for ob in range(NOB):
            for j in range(NB): lists[g][int(idx[g, ob, j])].append(ob * NB + j)
    maxc = max(len(l) for gl in lists for l in gl)
    inv = torch.full((G, nib, maxc + 1), -1, dtype=torch.int32)   # last column = count
    for g in range(G):
        for ib in range(nib):
            for c, v in enumerate(lists[g][ib]): inv[g, ib, c] = v
            inv[g, ib, maxc] = len(lists[g][ib])
    return inv
def expand_dense(w, idx, n_in):
    G, nob, nb = idx.shape; dense = w.new_zeros(G, n_in, nob * BLK)
    for g in range(G):
        for ob in range(nob):
            for j in range(nb):
                ib = int(idx[g, ob, j]); dense[g, ib * BLK:(ib + 1) * BLK, ob * BLK:(ob + 1) * BLK] += w[g, ob, j]
    return dense

@triton.jit
def _gelu(x):
    return 0.5 * x * (1.0 + tl.erf(x * 0.7071067811865476))
@triton.jit
def _gelu_grad(x):
    return 0.5 * (1.0 + tl.erf(x * 0.7071067811865476)) + x * 0.3989422804014327 * tl.exp(-0.5 * x * x)
@triton.jit
def _load_blk(s_ptr, r_ptr, a_ptr, rows, g, ib, G, KS, KR, KA, NBS, NBR, rm):
    """(BM, 32) pre-activation block ib of group g for token rows, from the source the id falls in."""
    c = tl.arange(0, 32)
    n64 = rows.to(tl.int64)
    is_s = ib < NBS
    is_r = (ib >= NBS) & (ib < NBS + NBR)
    is_a = ib >= NBS + NBR
    xs = tl.load(s_ptr + ((n64 * G + g) * KS + ib * 32)[:, None] + c[None, :], mask=rm[:, None] & is_s, other=0.0)
    xr = tl.load(r_ptr + ((n64 * G + g) * KR + (ib - NBS) * 32)[:, None] + c[None, :], mask=rm[:, None] & is_r, other=0.0)
    xa = tl.load(a_ptr + n64[:, None] * KA + (ib - NBS - NBR) * 32 + c[None, :], mask=rm[:, None] & is_a, other=0.0)
    return xs.to(tl.float32) + xr.to(tl.float32) + xa.to(tl.float32)

@triton.jit
def _bs_fwd(s_ptr, r_ptr, a_ptr, w_ptr, idx_ptr, out_ptr, N, G, GS, KS, KR, KA, NBS, NBR, NOB,
            NB: tl.constexpr, BM: tl.constexpr, GELU: tl.constexpr):
    pid_m = tl.program_id(0); g = tl.program_id(1); ob = tl.program_id(2)
    rows = pid_m * BM + tl.arange(0, BM); rm = rows < N
    acc = tl.zeros((BM, 32), dtype=tl.float32)
    for j in tl.static_range(NB):
        ib = tl.load(idx_ptr + (g * NOB + ob) * NB + j).to(tl.int64)
        x = _load_blk(s_ptr, r_ptr, a_ptr, rows, g, ib, G, KS, KR, KA, NBS, NBR, rm)
        if GELU:
            x = _gelu(x)
        wt = tl.load(w_ptr + (((g * NOB + ob) * NB + j) * 32 + tl.arange(0, 32)[:, None]) * 32 + tl.arange(0, 32)[None, :]).to(tl.bfloat16)
        acc += tl.dot(x.to(tl.bfloat16), wt)
    oo = (rows.to(tl.int64) * GS + g) * (NOB * 32) + ob * 32          # GS >= G: the buffer may hold extra groups written by the caller
    tl.store(out_ptr + oo[:, None] + tl.arange(0, 32)[None, :], acc, mask=rm[:, None])

@triton.jit
def _bs_bwd_dx(s_ptr, r_ptr, a_ptr, w_ptr, inv_ptr, dh_ptr, ds_ptr, dr_ptr, da_ptr, N, G, GS, KS, KR, KA, NBS, NBR, NOB, NIB,
               NB: tl.constexpr, MAXC: tl.constexpr, BM: tl.constexpr, GELU: tl.constexpr):
    pid_m = tl.program_id(0); g = tl.program_id(1); ib = tl.program_id(2).to(tl.int64)
    rows = pid_m * BM + tl.arange(0, BM); rm = rows < N
    n64 = rows.to(tl.int64); c = tl.arange(0, 32)
    acc = tl.zeros((BM, 32), dtype=tl.float32)
    cnt = tl.load(inv_ptr + (g * NIB + ib) * MAXC + (MAXC - 1))
    for k in range(0, cnt):
        code = tl.load(inv_ptr + (g * NIB + ib) * MAXC + k)
        ob = code // NB; j = code % NB
        dh = tl.load(dh_ptr + ((n64 * GS + g) * (NOB * 32) + ob * 32)[:, None] + c[None, :], mask=rm[:, None], other=0.0)
        wt = tl.load(w_ptr + (((g * NOB + ob) * NB + j) * 32 + tl.arange(0, 32)[:, None]) * 32 + tl.arange(0, 32)[None, :]).to(tl.bfloat16)
        acc += tl.dot(dh.to(tl.bfloat16), tl.trans(wt))
    if GELU:
        xpre = _load_blk(s_ptr, r_ptr, a_ptr, rows, g, ib, G, KS, KR, KA, NBS, NBR, rm)
        dx = acc * _gelu_grad(xpre)
    else:
        dx = acc
    is_s = ib < NBS; is_r = (ib >= NBS) & (ib < NBS + NBR); is_a = ib >= NBS + NBR
    tl.store(ds_ptr + ((n64 * G + g) * KS + ib * 32)[:, None] + c[None, :], dx.to(ds_ptr.dtype.element_ty), mask=rm[:, None] & is_s)
    tl.store(dr_ptr + ((n64 * G + g) * KR + (ib - NBS) * 32)[:, None] + c[None, :], dx.to(tl.bfloat16), mask=rm[:, None] & is_r)
    tl.store(da_ptr + ((n64 * G + g) * KA + (ib - NBS - NBR) * 32)[:, None] + c[None, :], dx, mask=rm[:, None] & is_a)

@triton.jit
def _bs_bwd_dw(s_ptr, r_ptr, a_ptr, idx_ptr, dh_ptr, dw_ptr, N, G, GS, KS, KR, KA, NBS, NBR, NOB,
               NB: tl.constexpr, BM: tl.constexpr, GELU: tl.constexpr):
    g = tl.program_id(0); ob = tl.program_id(1); j = tl.program_id(2)
    ib = tl.load(idx_ptr + (g * NOB + ob) * NB + j).to(tl.int64)
    c = tl.arange(0, 32)
    acc = tl.zeros((32, 32), dtype=tl.float32)
    for m0 in range(0, N, BM):
        rows = m0 + tl.arange(0, BM); rm = rows < N; n64 = rows.to(tl.int64)
        x = _load_blk(s_ptr, r_ptr, a_ptr, rows, g, ib, G, KS, KR, KA, NBS, NBR, rm)
        if GELU:
            x = _gelu(x)
        dh = tl.load(dh_ptr + ((n64 * GS + g) * (NOB * 32) + ob * 32)[:, None] + c[None, :], mask=rm[:, None], other=0.0)
        acc += tl.dot(tl.trans(x.to(tl.bfloat16)), dh.to(tl.bfloat16))
    tl.store(dw_ptr + (((g * NOB + ob) * NB + j) * 32 + c[:, None]) * 32 + c[None, :], acc)


@triton.jit
def _bs_bwd_fused(s_ptr, r_ptr, a_ptr, w_ptr, inv_ptr, dh_ptr, ds_ptr, dr_ptr, da_ptr, dw_ptr, N, G, GS, KS, KR, KA, NBS, NBR, NOB, NIB, INVS,
                  NB: tl.constexpr, MAXC: tl.constexpr, BM: tl.constexpr, NT: tl.constexpr, GELU: tl.constexpr):
    # INVS = row stride of the inverse map (its last entry is the count); MAXC = power-of-two >= INVS-1 accumulator slots
    chunk = tl.program_id(0); g = tl.program_id(1); ib = tl.program_id(2).to(tl.int64)
    c = tl.arange(0, 32)
    cnt = tl.load(inv_ptr + (g * NIB + ib) * INVS + (INVS - 1))
    is_s = ib < NBS; is_r = (ib >= NBS) & (ib < NBS + NBR); is_a = ib >= NBS + NBR
    accw = tl.zeros((MAXC, 32, 32), dtype=tl.float32)
    for ti in range(0, NT):
        rows = (chunk * NT + ti) * BM + tl.arange(0, BM); rm = rows < N; n64 = rows.to(tl.int64)
        xpre = _load_blk(s_ptr, r_ptr, a_ptr, rows, g, ib, G, KS, KR, KA, NBS, NBR, rm)
        if GELU:
            xg = _gelu(xpre)
        else:
            xg = xpre
        xb = xg.to(tl.bfloat16)
        accx = tl.zeros((BM, 32), dtype=tl.float32)
        for k in tl.static_range(MAXC):
            code = tl.load(inv_ptr + (g * NIB + ib) * INVS + tl.minimum(k, INVS - 2))
            ok = k < cnt
            ob = tl.maximum(code, 0) // NB; j = tl.maximum(code, 0) % NB
            dh = tl.load(dh_ptr + ((n64 * GS + g) * (NOB * 32) + ob * 32)[:, None] + c[None, :], mask=rm[:, None] & ok, other=0.0).to(tl.bfloat16)
            wt = tl.load(w_ptr + (((g * NOB + ob) * NB + j) * 32 + tl.arange(0, 32)[:, None]) * 32 + tl.arange(0, 32)[None, :]).to(tl.bfloat16)
            accx += tl.dot(dh, tl.trans(wt))
            dwk = tl.dot(tl.trans(xb), dh)                                          # (32, 32), zero when masked
            accw += tl.where((tl.arange(0, MAXC) == k)[:, None, None], dwk[None, :, :], 0.0)
        if GELU:
            dx = accx * _gelu_grad(xpre)
        else:
            dx = accx
        tl.store(ds_ptr + ((n64 * G + g) * KS + ib * 32)[:, None] + c[None, :], dx.to(ds_ptr.dtype.element_ty), mask=rm[:, None] & is_s)
        tl.store(dr_ptr + ((n64 * G + g) * KR + (ib - NBS) * 32)[:, None] + c[None, :], dx.to(tl.bfloat16), mask=rm[:, None] & is_r)
        tl.store(da_ptr + ((n64 * G + g) * KA + (ib - NBS - NBR) * 32)[:, None] + c[None, :], dx, mask=rm[:, None] & is_a)
    for k in tl.static_range(MAXC):
        code = tl.load(inv_ptr + (g * NIB + ib) * INVS + tl.minimum(k, INVS - 2))
        ok = k < cnt
        ob = tl.maximum(code, 0) // NB; j = tl.maximum(code, 0) % NB
        wk = tl.sum(tl.where((tl.arange(0, MAXC) == k)[:, None, None], accw, 0.0), axis=0)
        tl.atomic_add(dw_ptr + (((g * NOB + ob) * NB + j) * 32 + c[:, None]) * 32 + c[None, :], wk, mask=(c[:, None] >= 0) & ok)

class _BSFunnel(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xs, xr, xa, w, idx, inv, gelu, out_groups):
        # xs (N,G,KS) bf16 pre-activation state (a view of the emission), xr (N,G,KR) bf16 or None, xa (N,KA) bf16 shared or None,
        # w (G,NOB,NB,32,32) bf16. Block ids: [0,KS/32) state, then reads, then ask. gelu: apply GELU on load (and its derivative in backward).
        N, G, KS = xs.shape; NOB, NB = idx.shape[1], idx.shape[2]
        ctx.has_r, ctx.has_a, ctx.gelu = xr is not None, xa is not None, bool(gelu)
        xr = xr.contiguous() if xr is not None else xs.new_zeros(1); xa = xa.contiguous() if xa is not None else xs.new_zeros(1)
        KR = xr.shape[-1] if ctx.has_r else 0; KA = xa.shape[-1] if ctx.has_a else 0
        xs, w = xs.contiguous(), w.contiguous()
        GS = out_groups or G                              # the caller may append groups of its own into [:, G:] in place
        out = torch.empty(N, GS, NOB * 32, device=xs.device, dtype=torch.float32)
        BM = 64
        _bs_fwd[(triton.cdiv(N, BM), G, NOB)](xs, xr, xa, w, idx, out, N, G, GS, KS, KR, KA, KS // 32, KR // 32, NOB, NB=NB, BM=BM, GELU=ctx.gelu, num_warps=4)
        ctx.save_for_backward(xs, xr, xa, w, idx, inv); ctx.KR, ctx.KA, ctx.GS = KR, KA, GS
        return out
    @staticmethod
    def backward(ctx, dh):
        xs, xr, xa, w, idx, inv = ctx.saved_tensors
        N, G, KS = xs.shape; KR, KA = ctx.KR, ctx.KA; GS = ctx.GS; NOB, NB = idx.shape[1], idx.shape[2]; NIB = (KS + KR + KA) // 32
        dh = dh.contiguous()                              # (N, GS, NOB*32) fp32; the kernels cast tiles in registers
        ds = torch.empty_like(xs); dr = torch.empty_like(xr) if ctx.has_r else xs.new_zeros(1)
        da_part = torch.empty(N, G, KA, device=xs.device, dtype=torch.float32) if ctx.has_a else torch.zeros(1, device=xs.device)
        BM = 64
        if FUSED_BWD:
            dhb = dh.to(torch.bfloat16)
            dw = torch.zeros(G, NOB, NB, 32, 32, device=xs.device, dtype=torch.float32)
            NT = 32                                                    # tokens per program = BM*NT; few chunks -> few atomics on dW
            INVS = inv.shape[-1]; MAXC = 1 << max(1, (INVS - 2)).bit_length() if (INVS - 1) & (INVS - 2) else INVS - 1   # next pow2 >= INVS-1
            _bs_bwd_fused[(triton.cdiv(N, BM * NT), G, NIB)](xs, xr, xa, w, inv, dhb, ds, dr, da_part, dw, N, G, GS, KS, KR, KA, KS // 32, KR // 32, NOB, NIB, INVS,
                                                              NB=NB, MAXC=MAXC, BM=BM, NT=NT, GELU=ctx.gelu, num_warps=4)
        else:
            _bs_bwd_dx[(triton.cdiv(N, BM), G, NIB)](xs, xr, xa, w, inv, dh, ds, dr, da_part, N, G, GS, KS, KR, KA, KS // 32, KR // 32, NOB, NIB,
                                                     NB=NB, MAXC=inv.shape[-1], BM=BM, GELU=ctx.gelu, num_warps=4)
            dw = torch.empty(G, NOB, NB, 32, 32, device=xs.device, dtype=torch.float32)
            _bs_bwd_dw[(G, NOB, NB)](xs, xr, xa, idx, dh, dw, N, G, GS, KS, KR, KA, KS // 32, KR // 32, NOB, NB=NB, BM=BM, GELU=ctx.gelu, num_warps=4)
        return ds, (dr if ctx.has_r else None), (da_part.sum(1).to(xa.dtype) if ctx.has_a else None), dw.to(w.dtype), None, None, None, None

def bs_funnel(xs, xr, xa, w, idx, inv, gelu=True, out_groups=None):
    return _BSFunnel.apply(xs, xr, xa, w, idx, inv, gelu, out_groups)

if __name__ == "__main__":
    import time, torch.nn.functional as F
    torch.manual_seed(0)
    G, KS, KR, KA, NOUT, NB = 96, 2048, 64, 512, 512, 5
    idx = make_wiring(G, KS + KR + KA, NOUT, NB, 20260905); inv = inverse_map(idx, KS + KR + KA)
    print("wiring: %d input blocks, %d out blocks x %d; an input block is read by at most %d out blocks" % ((KS + KR + KA) // 32, NOUT // 32, NB, inv.shape[-1] - 1))
    idx, inv = idx.cuda(), inv.cuda()
    w = (torch.randn(G, NOUT // 32, NB, 32, 32, device="cuda") * 0.02).bfloat16().requires_grad_()
    def ref(xs, xr, xa):    # dense reference in fp32 with the same wiring
        dense = expand_dense(w.detach().float(), idx.cpu(), KS + KR + KA).cuda()
        X = torch.cat([xs.float(), xr.float(), xa.float().unsqueeze(1).expand(-1, G, -1)], -1)
        return torch.einsum('ngk,gko->ngo', F.gelu(X), dense)
    N = 192
    xs = (torch.randn(N, G, KS, device="cuda") * 0.7).bfloat16().requires_grad_(); xr = torch.randn(N, G, KR, device="cuda").bfloat16().requires_grad_(); xa = torch.randn(N, KA, device="cuda").bfloat16().requires_grad_()
    out = bs_funnel(xs, xr, xa, w, idx, inv); r = ref(xs, xr, xa)
    rel = lambda a, b: ((a.float() - b.float()).abs().max() / (b.float().abs().max() + 1e-12)).item()
    print("fwd vs fp32 dense reference: rel diff %.2e" % rel(out, r))
    gW = torch.randn_like(out)
    g1 = torch.autograd.grad((out * gW).sum(), (xs, xr, xa, w))
    # reference grads: same wiring, fp32 math, bf16-rounded inputs
    xs2, xr2, xa2 = [t.detach().float().requires_grad_() for t in (xs, xr, xa)]; w2 = w.detach().float().requires_grad_()
    dense = expand_dense(w2, idx.cpu(), KS + KR + KA)
    X = torch.cat([xs2, xr2, xa2.unsqueeze(1).expand(-1, G, -1)], -1); r2 = torch.einsum('ngk,gko->ngo', F.gelu(X), dense)
    g2 = torch.autograd.grad((r2 * gW).sum(), (xs2, xr2, xa2, w2))
    print("grads vs reference: " + ", ".join("%s %.1e" % (n, rel(a, b)) for n, a, b in zip(("state", "reads", "ask", "W"), g1, g2)))
    assert rel(out, r) < 1e-2 and all(rel(a, b) < 3e-2 for a, b in zip(g1, g2))
    # timing at the real size
    N = 16384
    xs = (torch.randn(N, G, KS, device="cuda") * 0.7).bfloat16().requires_grad_(); xr = torch.randn(N, G, KR, device="cuda").bfloat16().requires_grad_(); xa = torch.randn(N, KA, device="cuda").bfloat16().requires_grad_()
    def fb():
        o = bs_funnel(xs, xr, xa, w, idx, inv); torch.autograd.grad(o, (xs, xr, xa, w), grad_outputs=o)
    for _ in range(2): fb()
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(5): fb()
    torch.cuda.synchronize(); tb = (time.time() - t0) / 5 * 1e3
    with torch.no_grad():
        torch.cuda.synchronize(); t0 = time.time()
        for _ in range(5): bs_funnel(xs, xr, xa, w, idx, inv)
        torch.cuda.synchronize(); tf = (time.time() - t0) / 5 * 1e3
    print("block-sparse stage 1 at B16 x T1024 (N=16384, 96 groups, 2624 -> 512, 160 inputs/unit): fwd %.1f ms | fwd+bwd %.1f ms  [dense expanded path measured ~57 ms GEMM + ~40 ms memory ops]" % (tf, tb))
    print("BS_OK")
