"""CPU fallbacks for the three Triton kernels the m8 model calls at inference (scan/emission, fused reads,
block-sparse funnel). Pure torch, run on any device; `install()` patches them into kcol / memlm_m5.
Semantics copied from the kernels' own references (test_kcol.py ref, kcol.reads torch branches,
bsparse.__main__ dense reference); verified against the kernels by test_cpu_shim.py."""
import torch, torch.nn.functional as F
import kcol, bsparse, ktriton

def _scan(k, g, beta, v, L, S0=None):
    """fp32 column-gated delta-rule recurrence for all heads: returns (emit (B,T,VT,DK) fp32, final states list per head (B,DK,dv_g))."""
    B, T, G, DK = k.shape
    VT, dvs = L["VT"], L["dv"]
    emit = torch.zeros(B, T, VT, DK, dtype=torch.float32, device=k.device)
    kf, gf, bf_, vf = k.float(), g.float(), beta.float(), v.float()
    finals = []
    if len(set(dvs)) == 1:                                   # uniform payload: all heads in one batched loop
        dv = dvs[0]
        gv, bv, vv = gf.view(B, T, G, dv), bf_.view(B, T, G, dv), vf.view(B, T, G, dv)
        S = torch.stack(S0, 1) if S0 is not None else torch.zeros(B, G, DK, dv, dtype=torch.float32, device=k.device)
        for t in range(T):
            Sp = S * gv[:, t].exp().unsqueeze(2)
            u = bv[:, t] * (vv[:, t] - torch.einsum('bgd,bgdv->bgv', kf[:, t], Sp))
            S = Sp + kf[:, t].unsqueeze(-1) * u.unsqueeze(2)
            emit[:, t] = S.transpose(2, 3).reshape(B, VT, DK)
        finals = [S[:, gi] for gi in range(G)]
    else:                                                    # ragged: per head, as the reference does
        for gi in range(G):
            a0, b0 = L["vsl"][gi]
            S = S0[gi].clone() if S0 is not None else torch.zeros(B, DK, b0 - a0, dtype=torch.float32, device=k.device)
            for t in range(T):
                Sp = S * gf[:, t, a0:b0].exp()[:, None, :]
                u = (vf[:, t, a0:b0] - torch.einsum('bd,bdv->bv', kf[:, t, gi], Sp)) * bf_[:, t, a0:b0]
                S = Sp + kf[:, t, gi][:, :, None] * u[:, None, :]
                emit[:, t, a0:b0] = S.transpose(1, 2)
            finals.append(S)
    return emit, finals

def kslot_cpu(k, g, beta, v, q, L):
    """kcol.kslot for R == 0 (the production mode: reads happen on the emission). emit rows = payload columns,
    head-major (test_kcol.ref); fp32 recurrence, returned in k.dtype like the kernel's bf16 emission."""
    B, T = k.shape[0], k.shape[1]
    assert L["R"] == 0, "cpu_shim: in-kernel reads (R > 0) are not used by the model"
    emit, _ = _scan(k, g, beta, v, L)
    out = torch.empty(B, T, L["OT"], dtype=k.dtype, device=k.device)
    return out, emit.reshape(B, T, L["ST"]).to(k.dtype)

def kslot_s_cpu(k, g, beta, v, q, L, s0=None):
    """kcol.kslot_s: initial state in and fp32 final state out, both PROGRAM-indexed (B*NP, dk, CB): program p holds
    head pg2g[p], payload columns pg2c[p] .. +CB (zero-padded past the head's dv)."""
    B, T, G, DK = k.shape
    assert L["R"] == 0, "cpu_shim: in-kernel reads (R > 0) are not used by the model"
    NP, CB, dvs = L["NP"], L["CB"], L["dv"]
    pg2g, pg2c = L["pg2g"].tolist(), L["pg2c"].tolist()
    S0 = None
    if s0 is not None:
        s0 = s0.float().view(B, NP, DK, CB)
        S0 = [torch.zeros(B, DK, dvs[gi], dtype=torch.float32, device=k.device) for gi in range(G)]
        for p in range(NP):
            gi, c0 = pg2g[p], pg2c[p]; w = min(CB, dvs[gi] - c0)
            S0[gi][:, :, c0:c0 + w] = s0[:, p, :, :w]
    emit, finals = _scan(k, g, beta, v, L, S0)
    sfin = torch.zeros(B, NP, DK, CB, dtype=torch.float32, device=k.device)
    for p in range(NP):
        gi, c0 = pg2g[p], pg2c[p]; w = min(CB, dvs[gi] - c0)
        sfin[:, p, :, :w] = finals[gi][:, :, c0:c0 + w]
    out = torch.empty(B, T, L["OT"], dtype=k.dtype, device=k.device)
    return out, emit.reshape(B, T, L["ST"]).to(k.dtype), sfin.reshape(B * NP, DK, CB)

def reads_kernel_cpu(emit, q, L):
    """kcol.reads_kernel contract via the torch branches of kcol.reads."""
    B, T = emit.shape[0], emit.shape[1]
    dk, G, VT = L["dk"], L["G"], L["VT"]
    if q.dim() == 4:                                         # shared queries (B,T,R,dk)
        R = q.shape[2]
        o = kcol._mm(emit, q.reshape(B * T, R, dk), L).reshape(B, T, R, VT)
        per = [o[..., a:b] for (a, b) in L["vsl"]]
        packed = o.reshape(B, T, R * VT)
    else:                                                    # per-head queries (B,T,G,R,dk), uniform payload
        R = q.shape[3]; dv = L["dv"][0]
        S = emit.view(B * T * G, dv, dk)
        o = torch.bmm(S, q.reshape(B * T * G, R, dk).transpose(1, 2).to(S.dtype))
        packed = o.transpose(1, 2).reshape(B, T, G * R * dv)
        pv = packed.view(B, T, G, R, dv)
        per = [pv[:, :, g] for g in range(G)]
    return packed, per

_dense_cache = {}
def bs_funnel_cpu(xs, xr, xa, w, idx, inv, gelu=True, out_groups=None):
    """bsparse.bs_funnel via the expanded dense matrices (bsparse.__main__ reference). The dense expansion is
    cached per wiring (idx buffer): at inference the weights behind a wiring never change."""
    N, G, KS = xs.shape; NOB = idx.shape[1]
    n_in = KS + (xr.shape[-1] if xr is not None else 0) + (xa.shape[-1] if xa is not None else 0)
    key = (idx.data_ptr(), n_in, tuple(w.shape))
    dense = _dense_cache.get(key)
    if dense is None:
        dense = bsparse.expand_dense(w.detach().float(), idx.cpu(), n_in).to(xs.device)
        _dense_cache[key] = dense
    parts = [xs.float()]
    if xr is not None: parts.append(xr.float())
    if xa is not None: parts.append(xa.float().unsqueeze(1).expand(-1, G, -1))
    X = torch.cat(parts, -1)
    if gelu: X = F.gelu(X)
    out = torch.einsum('ngk,gko->ngo', X, dense)
    GS = out_groups or G
    if GS > G: out = torch.cat([out, out.new_zeros(N, GS - G, NOB * 32)], 1)
    return out


def fused_delta_stream_cpu(q, k, v, g, beta, scale=None, store_state=False, initial_state=None, state_dtype=torch.float32):
    """ktriton._fwd: per head scalar gates. S = a*S + beta * k (x) (v - a*k^T S); o = scale * q^T S (read AFTER the write).
    q,k:(B,T,H,dk) v:(B,T,H,dv) g,beta:(B,T,H). Returns (o in q.dtype, S stream (B,T,H,dk,dv) or None)."""
    B, T, H, dk = q.shape; dv = v.shape[-1]
    if scale is None: scale = dk ** -0.5
    S = initial_state.float().clone() if initial_state is not None else torch.zeros(B, H, dk, dv, dtype=torch.float32, device=q.device)
    qf, kf, vf, gf, bf_ = q.float(), k.float(), v.float(), g.float(), beta.float()
    o = torch.empty(B, T, H, dv, dtype=torch.float32, device=q.device)
    St = torch.empty(B, T, H, dk, dv, dtype=state_dtype, device=q.device) if store_state else None
    for t in range(T):
        a = gf[:, t].exp()[:, :, None, None]                                   # (B,H,1,1)
        kS = torch.einsum('bhd,bhdv->bhv', kf[:, t], S)
        S = a * S + bf_[:, t][:, :, None, None] * (kf[:, t].unsqueeze(-1) * (vf[:, t] - a[:, :, :, 0] * kS).unsqueeze(2))
        o[:, t] = scale * torch.einsum('bhd,bhdv->bhv', qf[:, t], S)
        if store_state: St[:, t] = S.to(state_dtype)
    return o.to(q.dtype), St

def delta_stream_cpu(q, k, v, g, beta, scale=None, initial_state=None):
    return fused_delta_stream_cpu(q, k, v, g, beta, scale=scale, store_state=True, initial_state=initial_state)

def _no2(*a, **k):
    raise NotImplementedError("cpu_shim: delta_stream2 has no CPU fallback (not used by m8)")

def gist_funnel2_bs_cpu(oS, rd, aske, gq, ShF, qr, bsw, NH, La, dk, dv, dt):
    """memlm_m5._gist_funnel2_bs with its bf16 casts removed (the CPU has no bf16 units; the fp32 model would otherwise
    mix dtypes at the hub einsum). Same operations in fp32."""
    w1, idx1, inv1, hub, w2, idx2, inv2, w3, awp = bsw
    B, Tb = oS.shape[0], oS.shape[1]; N = B * Tb
    aw = aske.shape[-1]; W = La * dk * dv; RW = La * 2 * dv
    xs = oS.reshape(N, NH, W).float()
    xr = (gq.float() * rd.float()).reshape(N, NH, RW)
    xa = aske.reshape(N, aw).float()
    if awp > aw: xa = F.pad(xa, (0, awp - aw))
    h = bs_funnel_cpu(xs, xr, xa, w1, idx1, inv1, gelu=True, out_groups=NH + 1)
    Sm = ShF.view(B, Tb, 1, dk, dv).float()
    hrd = torch.einsum('btqk,bthkv->bthqv', qr.float(), Sm).reshape(N, 2 * dv)
    xh = torch.cat([F.gelu(ShF.reshape(N, dk * dv).float()), F.gelu(gq.float() * hrd), F.gelu(xa)], -1)
    h[:, NH] = torch.mm(xh, hub.float())
    h2 = bs_funnel_cpu(h, None, None, w2, idx2, inv2, gelu=True)
    h3 = torch.bmm(F.gelu(h2).transpose(0, 1), w3.float())
    out = h3.transpose(0, 1).reshape(B, Tb, -1)
    return out.to(dt) if isinstance(dt, torch.dtype) else out

def bmm_f32_cpu(X, W):
    return torch.bmm(X.float(), W.float())

def install(M):
    """Patch the kernels out of kcol and the model module M (memlm_m5)."""
    kcol.kslot, kcol.kslot_s, kcol.reads_kernel = kslot_cpu, kslot_s_cpu, reads_kernel_cpu
    bsparse.bs_funnel = bs_funnel_cpu
    ktriton.delta_stream, ktriton.fused_delta_stream = delta_stream_cpu, fused_delta_stream_cpu
    ktriton.delta_stream2, ktriton.fused_delta_stream2 = _no2, _no2
    for name, fn in (("kslot", kslot_cpu), ("kslot_s", kslot_s_cpu), ("reads_kernel", reads_kernel_cpu),
                     ("bs_funnel", bs_funnel_cpu), ("_bmm_f32", bmm_f32_cpu), ("_gist_funnel2_bs", gist_funnel2_bs_cpu)):
        if hasattr(M, name): setattr(M, name, fn)
    return M
