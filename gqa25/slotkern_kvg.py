"""Key-matched slot scan with OUTPUT STRIDE G > 1, Triton -- forward reuses slotkern_kvt._kvfwd (which
already stores the PRE-write state every G steps); this file adds the backward that G=1 never needed.

WHY A STRIDE.  memlm_sh shares ONE memory across every layer above the writer.  At G=1 the scan emits
the full state at every token and every reader re-reads that whole copy per token: measured 8x the
read traffic of the per-layer model, and the shared model ran 1.6x SLOWER end to end despite a 4.5x
cheaper scan.  At G=16 a reader at token t reads the copy taken at the start of its 16-token block;
the <=15 tokens written since then are all inside the 16-token exact window, so nothing becomes
unreachable, while read traffic, emission memory and the backward's saved state all drop 16x.

THE BACKWARD.  G=1 could load the stored state of every step.  With only one stored state per segment,
each segment (taken in reverse) is REPLAYED forward from its stored start into per-program scratch,
then walked backward with exactly the adjoints of slotkern_kvt._kvbwd (see its docstring).  The only
difference: an emitted copy -- and so an incoming dkbuf/dvbuf gradient -- exists at the FIRST step of
each segment only.  Validated against slotkern_kv.ref_scan_kv(G=...) in float64 (test_kvg.py).
"""
import torch
import triton
import triton.language as tl

from slotkern_kvt import _kvfwd, _p2


def _nw_g(NP, DKP, DVP):
    """Measured on this GPU (RTX 3060) with the real kernels, 192 x (64+64): 16 warps beat 8 on both
    passes (fwd 14.4 vs 16.5 ms, bwd 20.3 vs 20.8 ms per head at T=1024).  Smaller tiles keep 8."""
    per = NP * (DKP + DVP)
    return 16 if per >= 32768 else (8 if per > 8192 else 4)


def _nw_gb(NP, DKP, DVP):
    """Backward warps.  The per-head timing above was measured with ONE program on the GPU; in the
    model B*H = 24 programs run at once on 28 SMs and the replay scratch contends for bandwidth.
    Measured at that concurrency (B=2, T=1024, 12 heads, 192 x 64+64, stride 16), forward / backward:
    nw 4: 64.7 / 104.1 ms, nw 8: 14.8 / 54.9, nw 16: 13.6 / 71.5, nw 32: 17.9 / 97.0.  So the forward
    keeps 16 and the backward takes 8.  Every nw 1..16 matches an fp64 reference to ~2e-5 at T=1024."""
    per = NP * (DKP + DVP)
    return 8 if per >= 32768 else _nw_g(NP, DKP, DVP)


@triton.jit
def _kvbwd_g(k_ptr, q_ptr, v_ptr, g_ptr, fr_ptr, temp_ptr,
             kbuf_ptr, vbuf_ptr, obuf_ptr, dkbuf_ptr, dvbuf_ptr,
             sk_ptr, sv_ptr, so_ptr,
             dk_ptr, dq_ptr, dv_ptr, dg_ptr, dfr_ptr, dtemp_ptr,
             T, H, G, NB, EPS,
             N: tl.constexpr, DK: tl.constexpr, DV: tl.constexpr,
             NP: tl.constexpr, DKP: tl.constexpr, DVP: tl.constexpr,
             HARD: tl.constexpr, HARDG: tl.constexpr, DETACH: tl.constexpr, OCCB: tl.constexpr = False,
             FRT: tl.constexpr = False):
    # FRT (sh_frtok): fr is PER TOKEN (B, T, H) and so is its gradient, stored per step instead of summed.
    pid = tl.program_id(0)
    b = pid // H
    h = pid % H
    on = tl.arange(0, NP)
    ok_ = tl.arange(0, DKP)
    oe = tl.arange(0, DVP)
    nm = on < N
    km = ok_ < DK
    em = oe < DV
    nk = nm[:, None] & km[None, :]
    ne = nm[:, None] & em[None, :]
    fr = tl.load(fr_ptr)
    temp = tl.load(temp_ptr)
    dK = tl.zeros((NP, DKP), dtype=tl.float32)
    dV = tl.zeros((NP, DVP), dtype=tl.float32)
    do = tl.zeros((NP,), dtype=tl.float32)
    dfr = 0.0
    dtemp = 0.0
    kbase = k_ptr + (b * T * H + h) * DK
    qbase = q_ptr + (b * T * H + h) * DK
    vbase = v_ptr + (b * T * H + h) * DV
    gbase = g_ptr + (b * T * H + h)
    sbase = pid * G                                   # this program's scratch rows
    for sb in range(0, NB):
        tb = NB - 1 - sb
        t0 = tb * G
        t1 = tl.minimum(t0 + G, T)
        so = ((b * NB + tb) * H + h)
        K = tl.load(kbuf_ptr + so * N * DK + on[:, None] * DK + ok_[None, :], mask=nk, other=0.0)
        V = tl.load(vbuf_ptr + so * N * DV + on[:, None] * DV + oe[None, :], mask=ne, other=0.0)
        o = tl.load(obuf_ptr + so * N + on, mask=nm, other=0.0)
        # ---- replay the segment forward, keeping each step's PRE-write state ----
        for t in range(t0, t1):
            j = t - t0
            tl.store(sk_ptr + (sbase + j) * NP * DKP + on[:, None] * DKP + ok_[None, :], K)
            tl.store(sv_ptr + (sbase + j) * NP * DVP + on[:, None] * DVP + oe[None, :], V)
            tl.store(so_ptr + (sbase + j) * NP + on, o)
            kt = tl.load(kbase + t * H * DK + ok_, mask=km, other=0.0)
            qt = tl.load(qbase + t * H * DK + ok_, mask=km, other=0.0)
            vv = tl.load(vbase + t * H * DV + oe, mask=em, other=0.0)
            graw = tl.load(gbase + t * H)
            if FRT:
                fr = tl.load(fr_ptr + b * T * H + t * H + h)
            g = (graw > 0.5).to(tl.float32) if HARDG else graw
            Kn2 = tl.sum(K * K, axis=1)
            kt2 = tl.sum(qt * qt, axis=0)
            kinv = 1.0 / tl.maximum(tl.sqrt(Kn2), 1e-6)
            qinv = 1.0 / tl.maximum(tl.sqrt(kt2), 1e-6)
            s = temp * tl.sum((K * kinv[:, None]) * (qt * qinv)[None, :], axis=1)
            if OCCB:
                z = s - fr * (o >= EPS).to(tl.float32)
            else:
                z = s - fr * o
            empty = (o < EPS) & nm
            cs = tl.cumsum(empty.to(tl.int32), axis=0)
            okrow = nm & ((empty == 0) | (cs == 1))
            z = tl.where(okrow, z, -1e30)
            if HARD:
                am = tl.argmax(z, axis=0)
                a = (on == am).to(tl.float32)
            else:
                mz = tl.max(z, axis=0)
                ez = tl.where(okrow, tl.exp(z - mz), 0.0)
                a = ez / tl.sum(ez, axis=0)
            w = g * a
            omw = 1.0 - w
            K = omw[:, None] * K + w[:, None] * kt[None, :]
            V = omw[:, None] * V + w[:, None] * vv[None, :]
            o = omw * o + w
        # every thread must see the whole scratch the block just wrote before reading it back
        tl.debug_barrier()
        # ---- walk the segment backward ----
        for ti in range(t0, t1):
            t = t1 - 1 - (ti - t0)
            j = t - t0
            K = tl.load(sk_ptr + (sbase + j) * NP * DKP + on[:, None] * DKP + ok_[None, :])
            V = tl.load(sv_ptr + (sbase + j) * NP * DVP + on[:, None] * DVP + oe[None, :])
            o = tl.load(so_ptr + (sbase + j) * NP + on)
            kt = tl.load(kbase + t * H * DK + ok_, mask=km, other=0.0)
            qt = tl.load(qbase + t * H * DK + ok_, mask=km, other=0.0)
            vv = tl.load(vbase + t * H * DV + oe, mask=em, other=0.0)
            graw = tl.load(gbase + t * H)
            if FRT:
                fr = tl.load(fr_ptr + b * T * H + t * H + h)
            g = (graw > 0.5).to(tl.float32) if HARDG else graw
            Kn2 = tl.sum(K * K, axis=1)
            kt2 = tl.sum(qt * qt, axis=0)
            knorm = tl.sqrt(Kn2)
            qnorm = tl.sqrt(kt2)
            kinv = 1.0 / tl.maximum(knorm, 1e-6)
            qinv = 1.0 / tl.maximum(qnorm, 1e-6)
            Kn = K * kinv[:, None]
            qn = qt * qinv
            s = temp * tl.sum(Kn * qn[None, :], axis=1)
            if OCCB:
                occ = (o >= EPS).to(tl.float32)
            else:
                occ = o
            z = s - fr * occ
            empty = (o < EPS) & nm
            cs = tl.cumsum(empty.to(tl.int32), axis=0)
            okrow = nm & ((empty == 0) | (cs == 1))
            z = tl.where(okrow, z, -1e30)
            mz = tl.max(z, axis=0)
            ez = tl.where(okrow, tl.exp(z - mz), 0.0)
            asoft = ez / tl.sum(ez, axis=0)
            if HARD:
                am = tl.argmax(z, axis=0)
                ah = (on == am).to(tl.float32)
            else:
                ah = asoft
            w = g * ah
            omw = 1.0 - w
            # the emitted copy (and so its incoming gradient) exists only at the FIRST step of a segment
            first = t == t0
            dkb = tl.load(dkbuf_ptr + so * N * DK + on[:, None] * DK + ok_[None, :], mask=nk & first, other=0.0)
            dvb = tl.load(dvbuf_ptr + so * N * DV + on[:, None] * DV + oe[None, :], mask=ne & first, other=0.0)
            dw = tl.sum((kt[None, :] - K) * dK, axis=1) + tl.sum((vv[None, :] - V) * dV, axis=1) + (1.0 - o) * do
            dw = tl.where(nm, dw, 0.0)
            dkt = tl.sum(w[:, None] * dK, axis=0)
            dvv = tl.sum(w[:, None] * dV, axis=0)
            dg = tl.sum(ah * dw, axis=0)
            da = g * dw
            sa = tl.sum(asoft * da, axis=0)
            dz = asoft * (da - sa)
            dz = tl.where(okrow, dz, 0.0)
            if FRT:
                tl.store(dfr_ptr + b * T * H + t * H + h, -tl.sum(occ * dz, axis=0))
            else:
                dfr += -tl.sum(occ * dz, axis=0)
            dtemp += tl.sum(dz * s, axis=0) / temp
            dKn = dz[:, None] * temp * qn[None, :]
            dqn = tl.sum(dz[:, None] * temp * Kn, axis=0)
            prj = tl.sum(dKn * Kn, axis=1)
            dKm_full = kinv[:, None] * (dKn - Kn * prj[:, None])
            cl = (knorm < 1e-6)
            dK_match = tl.zeros((NP, DKP), dtype=tl.float32) if DETACH else tl.where(cl[:, None], 0.0, dKm_full)
            prjq = tl.sum(dqn * qn, axis=0)
            dkt_match = tl.where(qnorm < 1e-6, qinv * dqn, qinv * (dqn - qn * prjq))
            dK = omw[:, None] * dK + dkb + dK_match
            dV = omw[:, None] * dV + dvb
            if OCCB:
                do = omw * do                 # the indicator is a step: no d z / d o
            else:
                do = omw * do - fr * dz
            tl.store(dk_ptr + (b * T * H + h) * DK + t * H * DK + ok_, dkt, mask=km)
            tl.store(dq_ptr + (b * T * H + h) * DK + t * H * DK + ok_, dkt_match, mask=km)
            tl.store(dv_ptr + (b * T * H + h) * DV + t * H * DV + oe, dvv, mask=em)
            tl.store(dg_ptr + (b * T * H + h) + t * H, dg)
        # the next segment's replay overwrites this scratch: finish reading it first
        tl.debug_barrier()
    if not FRT:
        tl.atomic_add(dfr_ptr, dfr)
    tl.atomic_add(dtemp_ptr, dtemp)


class _KVScanG(torch.autograd.Function):
    @staticmethod
    def forward(ctx, k, q, v, gate, fr, temp, N, G, hard, hard_gate, eps, detach_addr, nw, nwb, occ_bin=False):
        B, T, H, DK = k.shape
        DV = v.shape[-1]
        NB = (T + G - 1) // G
        f = lambda x: x.contiguous().to(torch.float32)
        k, q, v, gate = f(k), f(q), f(v), f(gate)
        frt = fr.dim() == 3                        # sh_frtok: one fr per token, shaped like the gate
        if frt:
            assert fr.shape == gate.shape, (fr.shape, gate.shape)
            fr = f(fr)
        else:
            fr = fr.to(torch.float32).reshape(1)
        temp = torch.as_tensor(temp, dtype=torch.float32, device=k.device).reshape(1)
        kbuf = torch.empty(B, NB, H, N, DK, device=k.device, dtype=torch.float32)
        vbuf = torch.empty(B, NB, H, N, DV, device=k.device, dtype=torch.float32)
        obuf = torch.empty(B, NB, H, N, device=k.device, dtype=torch.float32)
        _kvfwd[(B * H,)](k, q, v, gate, fr, temp, kbuf, vbuf, obuf, T, H, G, NB, eps,
                         N=N, DK=DK, DV=DV, NP=_p2(N), DKP=_p2(DK), DVP=_p2(DV),
                         HARD=bool(hard), HARDG=bool(hard_gate), OCCB=bool(occ_bin), FRT=frt, num_warps=nw)
        ctx.save_for_backward(k, q, v, gate, fr, temp, kbuf, vbuf, obuf)
        ctx.meta = (N, int(G), bool(hard), bool(hard_gate), float(eps), bool(detach_addr), int(nw), int(nwb), bool(occ_bin), frt)
        return kbuf, vbuf

    @staticmethod
    def backward(ctx, dkbuf, dvbuf):
        k, q, v, gate, fr, temp, kbuf, vbuf, obuf = ctx.saved_tensors
        N, G, hard, hard_gate, eps, detach_addr, nw, nwb, occ_bin, frt = ctx.meta
        B, T, H, DK = k.shape
        DV = v.shape[-1]
        NB = kbuf.shape[1]
        NP, DKP, DVP = _p2(N), _p2(DK), _p2(DV)
        dev = k.device
        sk = torch.empty(B * H * G, NP, DKP, device=dev, dtype=torch.float32)
        sv = torch.empty(B * H * G, NP, DVP, device=dev, dtype=torch.float32)
        so = torch.empty(B * H * G, NP, device=dev, dtype=torch.float32)
        dk = torch.zeros_like(k); dq = torch.zeros_like(q); dv = torch.zeros_like(v); dg = torch.zeros_like(gate)
        dfr = torch.zeros_like(fr) if frt else torch.zeros(1, device=dev, dtype=torch.float32)
        dtemp = torch.zeros(1, device=dev, dtype=torch.float32)
        dkbuf = torch.zeros_like(kbuf) if dkbuf is None else dkbuf.contiguous().float()
        dvbuf = torch.zeros_like(vbuf) if dvbuf is None else dvbuf.contiguous().float()
        _kvbwd_g[(B * H,)](k, q, v, gate, fr, temp, kbuf, vbuf, obuf, dkbuf, dvbuf, sk, sv, so,
                           dk, dq, dv, dg, dfr, dtemp, T, H, G, NB, eps,
                           N=N, DK=DK, DV=DV, NP=NP, DKP=DKP, DVP=DVP,
                           HARD=hard, HARDG=hard_gate, DETACH=detach_addr, OCCB=occ_bin, FRT=frt, num_warps=nwb)
        return dk, dq, dv, dg, dfr, dtemp.reshape(()), None, None, None, None, None, None, None, None, None


def kv_scan_g(k, v, gate, fr, N, temp, G=16, hard=False, hard_gate=False, eps=1e-4, detach_addr=False, q=None, nw=None, nwb=None, occ_bin=False):
    """Triton key-matched scan, output stride G (any G >= 1), with autograd.
    Returns (kbuf, vbuf), each (B, ceil(T/G), H, N, D*): the PRE-write state at t = 0, G, 2G, ..."""
    DK, DV = k.shape[-1], v.shape[-1]
    if nwb is None:                     # an explicit nw with no nwb sets BOTH passes
        nwb = _nw_gb(_p2(N), _p2(DK), _p2(DV)) if nw is None else nw
    if nw is None:
        nw = _nw_g(_p2(N), _p2(DK), _p2(DV))
    return _KVScanG.apply(k, k if q is None else q, v, gate, fr, temp, N, G, hard, hard_gate, eps, detach_addr, nw, nwb, bool(occ_bin))
