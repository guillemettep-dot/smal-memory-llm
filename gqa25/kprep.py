# Fused "write-path prep" for the kslot mixer: from the three GEMM outputs of a layer -- key features f = phi(x), write
# pre-activation z (from wvb), decay pre-activation gw (from wg) -- produce in ONE kernel what ~30 small kernels produced:
#     k    = normalize( M . normalize(f) )                       (lane mix M: (NG, La, La) from the lane gates)
#     beta = M . clamp(1.2 sigmoid(z + bias_h) - 0.1, 0, 1)
#     g    = M . ( -softplus(gw) [* (beta_raw > 0) if GTIE] )
#     qq_r = normalize( M . normalize(f @ Wq[h, r]) )            r = 1..R read queries per head
# and ONE backward kernel for df, dz, dgw, dM, dWq. fp32 inside; bf16 in/out where the model uses bf16.
import torch, triton, triton.language as tl, os
PREC = tl.constexpr("ieee" if os.environ.get("KP_IEEE") else "tf32")
EPS = tl.constexpr(1e-12)

@triton.jit
def _rownorm(x):                       # (BM, D) -> x / max(||x||, eps), and the norm
    n = tl.maximum(tl.sqrt(tl.sum(x * x, 1)), EPS)
    return x / n[:, None], n

@triton.jit
def _softplus(x):
    return tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(tl.minimum(x, 20.0))))

@triton.jit
def _sigmoid(x):
    return 1.0 / (1.0 + tl.exp(-x))

@triton.jit
def _prep_fwd(f_ptr, z_ptr, gw_ptr, bias_ptr, m_ptr, wq_ptr, k_ptr, b_ptr, g_ptr, q_ptr, N, H,
              LA: tl.constexpr, DK: tl.constexpr, DV: tl.constexpr, R: tl.constexpr, BM: tl.constexpr, GTIE: tl.constexpr):
    pid = tl.program_id(0); ng = tl.program_id(1)
    rows = pid * BM + tl.arange(0, BM); rm = rows < N; n64 = rows.to(tl.int64)
    d = tl.arange(0, DK); c = tl.arange(0, DV); la = tl.arange(0, LA)
    # ---- raw per-lane quantities, stacked on a lane axis
    K0 = tl.zeros((LA, BM, DK), dtype=tl.float32); Braw = tl.zeros((LA, BM, DV), dtype=tl.float32); Graw = tl.zeros((LA, BM, DV), dtype=tl.float32)
    for l in tl.static_range(LA):
        h = ng * LA + l
        fo = (n64 * H + h) * DK
        f = tl.load(f_ptr + fo[:, None] + d[None, :], mask=rm[:, None], other=0.0).to(tl.float32)
        k0, _ = _rownorm(f)
        vo = n64 * (H * DV) + h * DV
        z = tl.load(z_ptr + vo[:, None] + c[None, :], mask=rm[:, None], other=0.0).to(tl.float32) + tl.load(bias_ptr + h)
        gw = tl.load(gw_ptr + vo[:, None] + c[None, :], mask=rm[:, None], other=0.0).to(tl.float32)
        b = tl.minimum(tl.maximum(1.2 * _sigmoid(z) - 0.1, 0.0), 1.0)
        g = -_softplus(gw)
        if GTIE:
            g = g * (b > 0.0).to(tl.float32)
        sel = (la == l).to(tl.float32)
        K0 += sel[:, None, None] * k0[None, :, :]; Braw += sel[:, None, None] * b[None, :, :]; Graw += sel[:, None, None] * g[None, :, :]
    # ---- lane mix and outputs
    for m in tl.static_range(LA):
        h = ng * LA + m
        Mrow = tl.load(m_ptr + (ng * LA + m) * LA + la)                                   # M[ng, m, :]
        k1 = tl.sum(Mrow[:, None, None] * K0, 0); k, _ = _rownorm(k1)
        beta = tl.sum(Mrow[:, None, None] * Braw, 0); g = tl.sum(Mrow[:, None, None] * Graw, 0)
        fo = (n64 * H + h) * DK; vo = n64 * (H * DV) + h * DV
        tl.store(k_ptr + fo[:, None] + d[None, :], k.to(k_ptr.dtype.element_ty), mask=rm[:, None])
        tl.store(b_ptr + vo[:, None] + c[None, :], beta.to(b_ptr.dtype.element_ty), mask=rm[:, None])
        tl.store(g_ptr + vo[:, None] + c[None, :], g, mask=rm[:, None])
    # ---- read queries: per r, per lane q0 = f @ Wq[h, r]; normalize; mix; normalize
    for r in tl.static_range(R):
        Q1 = tl.zeros((LA, BM, DK), dtype=tl.float32)
        for l in tl.static_range(LA):
            h = ng * LA + l
            fo = (n64 * H + h) * DK
            f = tl.load(f_ptr + fo[:, None] + d[None, :], mask=rm[:, None], other=0.0)
            W = tl.load(wq_ptr + ((h * R + r) * DK + d[:, None]) * DK + d[None, :])           # (DK, DK)
            q0 = tl.dot(f.to(W.dtype), W, input_precision=PREC)
            q1, _ = _rownorm(q0)
            sel = (la == l).to(tl.float32)
            Q1 += sel[:, None, None] * q1[None, :, :]
        for m in tl.static_range(LA):
            h = ng * LA + m
            Mrow = tl.load(m_ptr + (ng * LA + m) * LA + la)
            q2 = tl.sum(Mrow[:, None, None] * Q1, 0); qq, _ = _rownorm(q2)
            qo = ((n64 * H + h) * R + r) * DK
            tl.store(q_ptr + qo[:, None] + d[None, :], qq.to(q_ptr.dtype.element_ty), mask=rm[:, None])

@triton.jit
def _prep_bwd(f_ptr, z_ptr, gw_ptr, bias_ptr, m_ptr, wq_ptr, dk_ptr, db_ptr, dg_ptr, dq_ptr,
              df_ptr, dz_ptr, dgw_ptr, dm_ptr, dwq_ptr, N, H, NG,
              LA: tl.constexpr, DK: tl.constexpr, DV: tl.constexpr, R: tl.constexpr, BM: tl.constexpr, GTIE: tl.constexpr):
    pid = tl.program_id(0); ng = tl.program_id(1)
    rows = pid * BM + tl.arange(0, BM); rm = rows < N; n64 = rows.to(tl.int64)
    d = tl.arange(0, DK); c = tl.arange(0, DV); la = tl.arange(0, LA)
    rmf = rm.to(tl.float32)
    # recompute raw quantities and norms
    K0 = tl.zeros((LA, BM, DK), dtype=tl.float32); Nf = tl.zeros((LA, BM), dtype=tl.float32)
    Braw = tl.zeros((LA, BM, DV), dtype=tl.float32); Graw = tl.zeros((LA, BM, DV), dtype=tl.float32)
    Sg = tl.zeros((LA, BM, DV), dtype=tl.float32); Sw = tl.zeros((LA, BM, DV), dtype=tl.float32)   # sigmoid(z+bias), sigmoid(gw)
    for l in tl.static_range(LA):
        h = ng * LA + l
        fo = (n64 * H + h) * DK; vo = n64 * (H * DV) + h * DV
        f = tl.load(f_ptr + fo[:, None] + d[None, :], mask=rm[:, None], other=0.0).to(tl.float32)
        k0, nf = _rownorm(f)
        z = tl.load(z_ptr + vo[:, None] + c[None, :], mask=rm[:, None], other=0.0).to(tl.float32) + tl.load(bias_ptr + h)
        gw = tl.load(gw_ptr + vo[:, None] + c[None, :], mask=rm[:, None], other=0.0).to(tl.float32)
        s = _sigmoid(z); b = tl.minimum(tl.maximum(1.2 * s - 0.1, 0.0), 1.0); g = -_softplus(gw)
        if GTIE:
            g = g * (b > 0.0).to(tl.float32)
        sel = (la == l).to(tl.float32)
        K0 += sel[:, None, None] * k0[None, :, :]; Nf += sel[:, None] * nf[None, :]
        Braw += sel[:, None, None] * b[None, :, :]; Graw += sel[:, None, None] * g[None, :, :]
        Sg += sel[:, None, None] * s[None, :, :]; Sw += sel[:, None, None] * _sigmoid(gw)[None, :, :]
    dK0 = tl.zeros((LA, BM, DK), dtype=tl.float32); dBraw = tl.zeros((LA, BM, DV), dtype=tl.float32); dGraw = tl.zeros((LA, BM, DV), dtype=tl.float32)
    dM = tl.zeros((LA, LA), dtype=tl.float32)
    # ---- k / beta / g mixed outputs: backward through normalize (k) and the lane mix
    for m in tl.static_range(LA):
        h = ng * LA + m
        Mrow = tl.load(m_ptr + (ng * LA + m) * LA + la)
        k1 = tl.sum(Mrow[:, None, None] * K0, 0); k, n1 = _rownorm(k1)
        fo = (n64 * H + h) * DK; vo = n64 * (H * DV) + h * DV
        dk = tl.load(dk_ptr + fo[:, None] + d[None, :], mask=rm[:, None], other=0.0).to(tl.float32)
        dk1 = (dk - k * tl.sum(k * dk, 1)[:, None]) / n1[:, None]
        dbm = tl.load(db_ptr + vo[:, None] + c[None, :], mask=rm[:, None], other=0.0).to(tl.float32)
        dgm = tl.load(dg_ptr + vo[:, None] + c[None, :], mask=rm[:, None], other=0.0).to(tl.float32)
        # dM[m, l] += <dk1, K0_l> + <dbm, Braw_l> + <dgm, Graw_l>
        contrib = tl.sum(tl.sum(dk1[None, :, :] * K0, 2), 1) + tl.sum(tl.sum(dbm[None, :, :] * Braw, 2), 1) + tl.sum(tl.sum(dgm[None, :, :] * Graw, 2), 1)   # (LA,)
        dM += ((la == m).to(tl.float32))[:, None] * contrib[None, :]
        dK0 += Mrow[:, None, None] * dk1[None, :, :]; dBraw += Mrow[:, None, None] * dbm[None, :, :]; dGraw += Mrow[:, None, None] * dgm[None, :, :]
    # ---- read queries backward (recompute per r), accumulating into df directly (per lane) and dWq partials
    dF = tl.zeros((LA, BM, DK), dtype=tl.float32)
    for r in tl.static_range(R):
        Q1 = tl.zeros((LA, BM, DK), dtype=tl.float32); N0 = tl.zeros((LA, BM), dtype=tl.float32)
        for l in tl.static_range(LA):
            h = ng * LA + l
            fo = (n64 * H + h) * DK
            f = tl.load(f_ptr + fo[:, None] + d[None, :], mask=rm[:, None], other=0.0)
            W = tl.load(wq_ptr + ((h * R + r) * DK + d[:, None]) * DK + d[None, :])
            q0 = tl.dot(f.to(W.dtype), W, input_precision=PREC); q1, n0 = _rownorm(q0)
            sel = (la == l).to(tl.float32)
            Q1 += sel[:, None, None] * q1[None, :, :]; N0 += sel[:, None] * n0[None, :]
        dQ1 = tl.zeros((LA, BM, DK), dtype=tl.float32)
        for m in tl.static_range(LA):
            h = ng * LA + m
            Mrow = tl.load(m_ptr + (ng * LA + m) * LA + la)
            q2 = tl.sum(Mrow[:, None, None] * Q1, 0); qq, n2 = _rownorm(q2)
            qo = ((n64 * H + h) * R + r) * DK
            dqq = tl.load(dq_ptr + qo[:, None] + d[None, :], mask=rm[:, None], other=0.0).to(tl.float32)
            dq2 = (dqq - qq * tl.sum(qq * dqq, 1)[:, None]) / n2[:, None]
            contrib = tl.sum(tl.sum(dq2[None, :, :] * Q1, 2), 1)
            dM += ((la == m).to(tl.float32))[:, None] * contrib[None, :]
            dQ1 += Mrow[:, None, None] * dq2[None, :, :]
        for l in tl.static_range(LA):
            h = ng * LA + l
            sel = (la == l).to(tl.float32)
            q1 = tl.sum(sel[:, None, None] * Q1, 0); dq1 = tl.sum(sel[:, None, None] * dQ1, 0); n0 = tl.sum(sel[:, None] * N0, 0)
            dq0 = (dq1 - q1 * tl.sum(q1 * dq1, 1)[:, None]) / n0[:, None]
            fo = (n64 * H + h) * DK
            f = tl.load(f_ptr + fo[:, None] + d[None, :], mask=rm[:, None], other=0.0)
            W = tl.load(wq_ptr + ((h * R + r) * DK + d[:, None]) * DK + d[None, :])
            dfl = tl.dot(dq0.to(W.dtype), tl.trans(W), input_precision=PREC)                  # (BM, DK)
            dF += sel[:, None, None] * dfl[None, :, :]
            dW = tl.dot(tl.trans(f.to(W.dtype)), (dq0 * rmf[:, None]).to(W.dtype), input_precision=PREC)   # (DK, DK)
            tl.store(dwq_ptr + (((pid * H + h) * R + r) * DK + d[:, None]) * DK + d[None, :], dW)
    # ---- through the raw per-lane nonlinearities to the inputs
    for l in tl.static_range(LA):
        h = ng * LA + l
        sel = (la == l).to(tl.float32)
        k0 = tl.sum(sel[:, None, None] * K0, 0); dk0 = tl.sum(sel[:, None, None] * dK0, 0); nf = tl.sum(sel[:, None] * Nf, 0)
        df = (dk0 - k0 * tl.sum(k0 * dk0, 1)[:, None]) / nf[:, None] + tl.sum(sel[:, None, None] * dF, 0)
        b = tl.sum(sel[:, None, None] * Braw, 0); s = tl.sum(sel[:, None, None] * Sg, 0); sw = tl.sum(sel[:, None, None] * Sw, 0)
        db = tl.sum(sel[:, None, None] * dBraw, 0); dg = tl.sum(sel[:, None, None] * dGraw, 0)
        pre = 1.2 * s - 0.1
        inside = ((pre >= 0.0) & (pre <= 1.0)).to(tl.float32)                                # clamp gradient, inclusive like torch
        dz = db * 1.2 * s * (1.0 - s) * inside
        if GTIE:
            dgw = dg * (b > 0.0).to(tl.float32) * (-sw)
        else:
            dgw = dg * (-sw)
        fo = (n64 * H + h) * DK; vo = n64 * (H * DV) + h * DV
        tl.store(df_ptr + fo[:, None] + d[None, :], df.to(df_ptr.dtype.element_ty), mask=rm[:, None])
        tl.store(dz_ptr + vo[:, None] + c[None, :], dz.to(dz_ptr.dtype.element_ty), mask=rm[:, None])
        tl.store(dgw_ptr + vo[:, None] + c[None, :], dgw.to(dgw_ptr.dtype.element_ty), mask=rm[:, None])
    tl.store(dm_ptr + ((pid * NG + ng) * LA + la[:, None]) * LA + la[None, :], dM)

class _Prep(torch.autograd.Function):
    @staticmethod
    def forward(ctx, f, z, gw, bias, M, wq, La, gtie):
        B, T, H, DK = f.shape; DV = z.shape[-1] // H; R = wq.shape[1]; N = B * T; NG = H // La
        f, z, gw, bias, M, wq = f.contiguous(), z.contiguous(), gw.contiguous(), bias.contiguous().float(), M.contiguous().float(), wq.contiguous()
        k = torch.empty(B, T, H, DK, device=f.device, dtype=f.dtype); beta = torch.empty(B, T, H * DV, device=f.device, dtype=z.dtype)
        g = torch.empty(B, T, H * DV, device=f.device, dtype=torch.float32); qq = torch.empty(B, T, H, R, DK, device=f.device, dtype=f.dtype)
        BM = 32
        _prep_fwd[(triton.cdiv(N, BM), NG)](f, z, gw, bias, M, wq, k, beta, g, qq, N, H, LA=La, DK=DK, DV=DV, R=R, BM=BM, GTIE=gtie, num_warps=4)
        ctx.save_for_backward(f, z, gw, bias, M, wq); ctx.meta = (La, gtie, BM)
        return k, beta, g, qq
    @staticmethod
    def backward(ctx, dk, db, dg, dq):
        f, z, gw, bias, M, wq = ctx.saved_tensors; La, gtie, BM = ctx.meta
        B, T, H, DK = f.shape; DV = z.shape[-1] // H; R = wq.shape[1]; N = B * T; NG = H // La; nb = triton.cdiv(N, BM)
        dk, db, dg, dq = dk.contiguous(), db.contiguous(), dg.contiguous(), dq.contiguous()
        df = torch.empty_like(f); dz = torch.empty_like(z); dgw = torch.empty_like(gw)
        dm_part = torch.empty(nb, NG, La, La, device=f.device, dtype=torch.float32); dwq_part = torch.empty(nb, H, R, DK, DK, device=f.device, dtype=torch.float32)
        _prep_bwd[(nb, NG)](f, z, gw, bias, M, wq, dk, db, dg, dq, df, dz, dgw, dm_part, dwq_part, N, H, NG, LA=La, DK=DK, DV=DV, R=R, BM=BM, GTIE=gtie, num_warps=4)
        dbias = dz.float().view(N, H, DV).sum((0, 2))
        return df, dz, dgw, dbias.to(bias.dtype), dm_part.sum(0), dwq_part.sum(0).to(wq.dtype), None, None

def prep_fused(f, z, gw, bias, M, wq, La=4, gtie=False):
    return _Prep.apply(f, z, gw, bias, M, wq, La, bool(gtie))

# ---- reference: the model's chain (memlm_m5 slot_gates + _lmix + normalize + _read_queries), given the lane matrix M
def lmix_ref(t, M, La):
    B, T, H = t.shape[:3]; NG = H // La; sh = t.shape
    tg = t.reshape(B, T, NG, La, -1)
    return torch.einsum('btglx,gml->btgmx', tg.float(), M.float()).to(t.dtype).reshape(sh)
def prep_reference(f, z, gw, bias, M, wq, La=4, gtie=False):
    import torch.nn.functional as F
    B, T, H, DK = f.shape; DV = z.shape[-1] // H; R = wq.shape[1]
    zz = z.view(B, T, H, DV).float() + bias.view(1, 1, H, 1)
    beta = (1.2 * torch.sigmoid(zz) - 0.1).clamp(0.0, 1.0)
    g = -F.softplus(gw.float()).view(B, T, H, DV)
    if gtie: g = g * (beta > 0).float()
    k = F.normalize(F.normalize(f, dim=-1).float(), dim=-1)
    k = F.normalize(lmix_ref(k, M, La), dim=-1)
    beta = lmix_ref(beta, M, La); g = lmix_ref(g, M, La)
    qq = F.normalize(torch.einsum('bthd,hrde->bthre', f.float(), wq.float()), dim=-1)
    qq = F.normalize(lmix_ref(qq, M, La), dim=-1)
    return k.to(f.dtype), beta.reshape(B, T, -1).to(z.dtype), g.reshape(B, T, -1), qq.to(f.dtype)
