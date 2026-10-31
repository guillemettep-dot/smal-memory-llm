"""Key-matched slot scan, HARD address only, with an UNDO-LOG backward.  Same contract as
slotkern_kvg.kv_scan_g(hard=True): returns (kbuf, vbuf), the PRE-write state every G steps.

WHY.  kv_scan_g's backward replays each G-step segment from its stored snapshot and writes the WHOLE
state (N x (DK + DV) floats) to scratch at every step, then reads it back walking backward: ~6.4 GB of
traffic per backward at sh21's shape, which is why the backward costs 4-5x the forward.

THE UNDO LOG.  With a hard address a step changes exactly ONE row: every other row is computed as
1.0 * K + 0.0 * kt, which is K bit for bit.  So the forward saves, per (b, t, h), the chosen row and that
row's PRE-write key, value and occupancy (DK + DV + 1 floats, the size of the inputs), plus the final
state.  The backward starts from the final state and, at each step t taken in reverse, puts the saved
row back: that IS the pre-write state of step t, exactly, with no replay and no scratch.  The adjoint
code that follows is kv_scan_g's, line for line, so the gradients are expected to be bitwise equal to it.

The forward's arithmetic is _kvfwd's, line for line (same snapshots); the extra saves are masked stores of
the chosen row straight from the state.  (A first version read the row with a masked sum over rows: exact,
but a cross-warp reduction plus a temporary tile per step -- forward 13 -> 42 ms at sh21's shape, measured.)"""
import torch
import triton
import triton.language as tl

from slotkern_kvt import _p2
from slotkern_kvg import _nw_g, _nw_gb, kv_scan_g


@triton.jit
def _kvfwd_u(k_ptr, q_ptr, v_ptr, g_ptr, fr_ptr, temp_ptr, kbuf_ptr, vbuf_ptr, obuf_ptr,
             am_ptr, ko_ptr, vo_ptr, oo_ptr, kf_ptr, vf_ptr, of_ptr,
             T, H, G, NB, EPS,
             N: tl.constexpr, DK: tl.constexpr, DV: tl.constexpr,
             NP: tl.constexpr, DKP: tl.constexpr, DVP: tl.constexpr,
             HARDG: tl.constexpr, OCCB: tl.constexpr, FRT: tl.constexpr):
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
    K = tl.zeros((NP, DKP), dtype=tl.float32)
    V = tl.zeros((NP, DVP), dtype=tl.float32)
    o = tl.zeros((NP,), dtype=tl.float32)
    kbase = k_ptr + (b * T * H + h) * DK
    qbase = q_ptr + (b * T * H + h) * DK
    vbase = v_ptr + (b * T * H + h) * DV
    gbase = g_ptr + (b * T * H + h)
    for t in range(0, T):
        if t % G == 0:
            tb = t // G
            tl.store(kbuf_ptr + ((b * NB + tb) * H + h) * N * DK + on[:, None] * DK + ok_[None, :],
                     K.to(kbuf_ptr.dtype.element_ty), mask=nk)
            tl.store(vbuf_ptr + ((b * NB + tb) * H + h) * N * DV + on[:, None] * DV + oe[None, :],
                     V.to(vbuf_ptr.dtype.element_ty), mask=ne)
            tl.store(obuf_ptr + ((b * NB + tb) * H + h) * N + on, o, mask=nm)
        kt = tl.load(kbase + t * H * DK + ok_, mask=km, other=0.0)
        qt = tl.load(qbase + t * H * DK + ok_, mask=km, other=0.0)
        vv = tl.load(vbase + t * H * DV + oe, mask=em, other=0.0)
        g = tl.load(gbase + t * H)
        if FRT:
            fr = tl.load(fr_ptr + b * T * H + t * H + h)
        if HARDG:
            g = (g > 0.5).to(tl.float32)
        Kn2 = tl.sum(K * K, axis=1)
        kt2 = tl.sum(qt * qt, axis=0)
        kinv = 1.0 / tl.maximum(tl.sqrt(Kn2), 1e-6)
        qinv = 1.0 / tl.maximum(tl.sqrt(kt2), 1e-6)
        Kn = K * kinv[:, None]
        qn = qt * qinv
        s = temp * tl.sum(Kn * qn[None, :], axis=1)
        if OCCB:
            z = s - fr * (o >= EPS).to(tl.float32)
        else:
            z = s - fr * o
        empty = (o < EPS) & nm
        cs = tl.cumsum(empty.to(tl.int32), axis=0)
        okrow = nm & ((empty == 0) | (cs == 1))
        z = tl.where(okrow, z, -1e30)
        am = tl.argmax(z, axis=0)
        sel = on == am
        a = sel.to(tl.float32)
        # ---- the undo log: the chosen row BEFORE this write ----
        ix = b * T * H + t * H + h
        tl.store(am_ptr + ix, am.to(tl.int32))
        # a MASKED STORE of the whole tile: every row points at the same DK addresses and only the chosen
        # row is unmasked, so the threads that hold it write it -- no cross-warp reduction, no temporary tile
        tl.store(ko_ptr + ix * DK + on[:, None] * 0 + ok_[None, :], K, mask=sel[:, None] & km[None, :])
        tl.store(vo_ptr + ix * DV + on[:, None] * 0 + oe[None, :], V, mask=sel[:, None] & em[None, :])
        tl.store(oo_ptr + ix + on * 0, o, mask=sel)
        w = g * a
        omw = 1.0 - w
        K = omw[:, None] * K + w[:, None] * kt[None, :]
        V = omw[:, None] * V + w[:, None] * vv[None, :]
        o = omw * o + w
    fo = (b * H + h)
    tl.store(kf_ptr + fo * NP * DKP + on[:, None] * DKP + ok_[None, :], K)
    tl.store(vf_ptr + fo * NP * DVP + on[:, None] * DVP + oe[None, :], V)
    tl.store(of_ptr + fo * NP + on, o)


@triton.jit
def _kvbwd_u(k_ptr, q_ptr, v_ptr, g_ptr, fr_ptr, temp_ptr,
             dkbuf_ptr, dvbuf_ptr,
             am_ptr, ko_ptr, vo_ptr, oo_ptr, kf_ptr, vf_ptr, of_ptr,
             dk_ptr, dq_ptr, dv_ptr, dg_ptr, dfr_ptr, dtemp_ptr,
             T, H, G, NB, EPS,
             N: tl.constexpr, DK: tl.constexpr, DV: tl.constexpr,
             NP: tl.constexpr, DKP: tl.constexpr, DVP: tl.constexpr,
             HARDG: tl.constexpr, DETACH: tl.constexpr, OCCB: tl.constexpr, FRT: tl.constexpr):
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
    fo = (b * H + h)
    K = tl.load(kf_ptr + fo * NP * DKP + on[:, None] * DKP + ok_[None, :])
    V = tl.load(vf_ptr + fo * NP * DVP + on[:, None] * DVP + oe[None, :])
    o = tl.load(of_ptr + fo * NP + on)
    dK = tl.zeros((NP, DKP), dtype=tl.float32)
    dV = tl.zeros((NP, DVP), dtype=tl.float32)
    do = tl.zeros((NP,), dtype=tl.float32)
    dfr = 0.0
    dtemp = 0.0
    kbase = k_ptr + (b * T * H + h) * DK
    qbase = q_ptr + (b * T * H + h) * DK
    vbase = v_ptr + (b * T * H + h) * DV
    gbase = g_ptr + (b * T * H + h)
    for ti in range(0, T):
        t = T - 1 - ti
        # ---- undo step t's write: the saved row goes back, giving the PRE-write state of step t ----
        ix = b * T * H + t * H + h
        am = tl.load(am_ptr + ix)
        sel = on == am
        kold = tl.load(ko_ptr + ix * DK + ok_, mask=km, other=0.0)
        vold = tl.load(vo_ptr + ix * DV + oe, mask=em, other=0.0)
        oold = tl.load(oo_ptr + ix)
        K = tl.where(sel[:, None], kold[None, :], K)
        V = tl.where(sel[:, None], vold[None, :], V)
        o = tl.where(sel, oold, o)
        # ---- from here: kv_scan_g's backward step, line for line ----
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
        ah = sel.to(tl.float32)                 # the forward's own argmax, from the log
        w = g * ah
        omw = 1.0 - w
        # an emitted copy (and so an incoming gradient) exists only where t is a multiple of G
        first = (t % G) == 0
        so = ((b * NB + t // G) * H + h)
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
            do = omw * do
        else:
            do = omw * do - fr * dz
        tl.store(dk_ptr + (b * T * H + h) * DK + t * H * DK + ok_, dkt, mask=km)
        tl.store(dq_ptr + (b * T * H + h) * DK + t * H * DK + ok_, dkt_match, mask=km)
        tl.store(dv_ptr + (b * T * H + h) * DV + t * H * DV + oe, dvv, mask=em)
        tl.store(dg_ptr + (b * T * H + h) + t * H, dg)
    if not FRT:
        tl.atomic_add(dfr_ptr, dfr)
    tl.atomic_add(dtemp_ptr, dtemp)


class _KVScanU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, k, q, v, gate, fr, temp, N, G, hard_gate, eps, detach_addr, nw, nwb, occ_bin):
        B, T, H, DK = k.shape
        DV = v.shape[-1]
        NB = (T + G - 1) // G
        NP, DKP, DVP = _p2(N), _p2(DK), _p2(DV)
        f = lambda x: x.contiguous().to(torch.float32)
        k, q, v, gate = f(k), f(q), f(v), f(gate)
        frt = fr.dim() == 3
        if frt:
            assert fr.shape == gate.shape, (fr.shape, gate.shape)
            fr = f(fr)
        else:
            fr = fr.to(torch.float32).reshape(1)
        temp = torch.as_tensor(temp, dtype=torch.float32, device=k.device).reshape(1)
        dev = k.device
        kbuf = torch.empty(B, NB, H, N, DK, device=dev, dtype=torch.float32)
        vbuf = torch.empty(B, NB, H, N, DV, device=dev, dtype=torch.float32)
        obuf = torch.empty(B, NB, H, N, device=dev, dtype=torch.float32)
        am = torch.empty(B, T, H, device=dev, dtype=torch.int32)
        ko = torch.empty(B, T, H, DK, device=dev, dtype=torch.float32)
        vo = torch.empty(B, T, H, DV, device=dev, dtype=torch.float32)
        oo = torch.empty(B, T, H, device=dev, dtype=torch.float32)
        kf = torch.empty(B, H, NP, DKP, device=dev, dtype=torch.float32)
        vf = torch.empty(B, H, NP, DVP, device=dev, dtype=torch.float32)
        of = torch.empty(B, H, NP, device=dev, dtype=torch.float32)
        _kvfwd_u[(B * H,)](k, q, v, gate, fr, temp, kbuf, vbuf, obuf, am, ko, vo, oo, kf, vf, of,
                           T, H, G, NB, eps, N=N, DK=DK, DV=DV, NP=NP, DKP=DKP, DVP=DVP,
                           HARDG=bool(hard_gate), OCCB=bool(occ_bin), FRT=frt, num_warps=nw)
        ctx.save_for_backward(k, q, v, gate, fr, temp, am, ko, vo, oo, kf, vf, of)
        ctx.meta = (N, int(G), NB, bool(hard_gate), float(eps), bool(detach_addr), int(nwb), bool(occ_bin), frt)
        return kbuf, vbuf

    @staticmethod
    def backward(ctx, dkbuf, dvbuf):
        k, q, v, gate, fr, temp, am, ko, vo, oo, kf, vf, of = ctx.saved_tensors
        N, G, NB, hard_gate, eps, detach_addr, nwb, occ_bin, frt = ctx.meta
        B, T, H, DK = k.shape
        DV = v.shape[-1]
        NP, DKP, DVP = _p2(N), _p2(DK), _p2(DV)
        dev = k.device
        dk = torch.zeros_like(k); dq = torch.zeros_like(q); dv = torch.zeros_like(v); dg = torch.zeros_like(gate)
        dfr = torch.zeros_like(fr) if frt else torch.zeros(1, device=dev, dtype=torch.float32)
        dtemp = torch.zeros(1, device=dev, dtype=torch.float32)
        kshape = (B, NB, H, N, DK); vshape = (B, NB, H, N, DV)
        dkbuf = torch.zeros(kshape, device=dev) if dkbuf is None else dkbuf.contiguous().float()
        dvbuf = torch.zeros(vshape, device=dev) if dvbuf is None else dvbuf.contiguous().float()
        _kvbwd_u[(B * H,)](k, q, v, gate, fr, temp, dkbuf, dvbuf, am, ko, vo, oo, kf, vf, of,
                           dk, dq, dv, dg, dfr, dtemp, T, H, G, NB, eps,
                           N=N, DK=DK, DV=DV, NP=NP, DKP=DKP, DVP=DVP,
                           HARDG=hard_gate, DETACH=detach_addr, OCCB=occ_bin, FRT=frt, num_warps=nwb)
        return dk, dq, dv, dg, dfr, dtemp.reshape(()), None, None, None, None, None, None, None, None


def kv_scan_u(k, v, gate, fr, N, temp, G=16, hard=False, hard_gate=False, eps=1e-4, detach_addr=False, q=None,
              nw=None, nwb=None, occ_bin=False):
    """Drop-in for kv_scan_g.  hard=False has no undo log (a soft write changes every row): it falls back."""
    if not hard:
        return kv_scan_g(k, v, gate, fr, N, temp, G=G, hard=False, hard_gate=hard_gate, eps=eps,
                         detach_addr=detach_addr, q=q, nw=nw, nwb=nwb, occ_bin=occ_bin)
    DK, DV = k.shape[-1], v.shape[-1]
    if nwb is None:
        nwb = _nw_gb(_p2(N), _p2(DK), _p2(DV)) if nw is None else nw
    if nw is None:
        nw = _nw_g(_p2(N), _p2(DK), _p2(DV))
    return _KVScanU.apply(k, k if q is None else q, v, gate, fr, temp, N, G, hard_gate, eps, detach_addr,
                          nw, nwb, bool(occ_bin))
