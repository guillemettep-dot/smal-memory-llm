"""KEY-MATCHED slot scan, Triton.  Same maths as slotkern_kv.ref_scan_kv, which stays the reference
this is validated against -- the python loop is correct but builds ~30k autograd nodes per forward
and runs the GPU at 31%.

Per (b,h), state K(N,DK), V(N,DV), o(N), all zero-initialised:
    kbuf_t, vbuf_t = K, V                       (pre-write, every G steps)
    s   = temp * <K_n/|K_n|, k_t/|k_t|>         cosine match against the STORED keys
    z   = s - fr*o,  masked so only occupied rows and the FIRST free row compete
    a   = softmax(z)   [or onehot(argmax z) if HARD]
    w   = g_t * a      [g_t binarised if HARDG]
    K <- (1-w)K + w k_t      V <- (1-w)V + w v_t      o <- (1-w)o + w

The empties-as-one mask matters for correctness, not just tidiness: every empty row scores identically
(cosine against a zero key is 0 for all of them), so without it a soft write splits one value across
the whole free pool and leaves every empty row holding the same direction.
"""
import torch, triton, triton.language as tl

NW_FWD = 1


def _nw(NP, DKP, DVP):
    """Warps per program, from the tile a program must hold in registers.

    One program owns NP*(DKP+DVP) floats spread over 32*num_warps threads. Past ~128 floats per
    thread the compiler spills to local memory and the scan collapses: measured on the 2-head shape
    (8,192 floats) num_warps=1 spills 394 B forward and 1,064 B backward and runs 7.4x slower than
    num_warps=4, which spills nothing. 8 warps was slower again, so 4 is the cap.

    The bound is deliberately set at 2,048 so every shape run before this change -- the 8-head
    control at 1,536 and the 24-row arm at 2,048 -- keeps num_warps=1 exactly as it had."""
    per = NP * (DKP + DVP)
    if per <= 2048:
        return 1
    if per <= 4096:
        return 2
    if per <= 8192:
        return 4
    return 8
NEG = -1e30


@triton.jit
def _kvfwd(k_ptr, q_ptr, v_ptr, g_ptr, fr_ptr, temp_ptr, kbuf_ptr, vbuf_ptr, obuf_ptr,
           T, H, G, NB, EPS,
           N: tl.constexpr, DK: tl.constexpr, DV: tl.constexpr,
           NP: tl.constexpr, DKP: tl.constexpr, DVP: tl.constexpr,
           HARD: tl.constexpr, HARDG: tl.constexpr, OCCB: tl.constexpr = False, FRT: tl.constexpr = False):
    # FRT (sh_frtok): fr_ptr holds one free-row preference PER TOKEN, laid out (B, T, H) like the gate.
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
        kt = tl.load(kbase + t * H * DK + ok_, mask=km, other=0.0)          # STORED
        qt = tl.load(qbase + t * H * DK + ok_, mask=km, other=0.0)          # used to SEARCH
        vv = tl.load(vbase + t * H * DV + oe, mask=em, other=0.0)
        g = tl.load(gbase + t * H)
        if FRT:
            fr = tl.load(fr_ptr + b * T * H + t * H + h)
        if HARDG:
            g = (g > 0.5).to(tl.float32)
        # cosine against the stored keys; clamp the NORM (not its square) exactly as the reference does
        # NORMALISE FIRST, THEN DOT -- the same order as the reference.  Dotting first and scaling
        # after is algebraically identical but not numerically, and inside this recurrence the softmax
        # amplifies the difference every step: measured 3.7e-3 drift by T=512 in soft mode, while hard
        # mode (an argmax, insensitive to tiny perturbations) matched at 1e-7 and hid the problem.
        Kn2 = tl.sum(K * K, axis=1)
        kt2 = tl.sum(qt * qt, axis=0)
        kinv = 1.0 / tl.maximum(tl.sqrt(Kn2), 1e-6)
        qinv = 1.0 / tl.maximum(tl.sqrt(kt2), 1e-6)
        Kn = K * kinv[:, None]
        qn = qt * qinv
        s = temp * tl.sum(Kn * qn[None, :], axis=1)
        if OCCB:                         # penalty on 'ever written' ([o >= EPS]), not on o itself
            z = s - fr * (o >= EPS).to(tl.float32)
        else:
            z = s - fr * o
        # only occupied rows and the FIRST free row compete
        empty = (o < EPS) & nm
        cs = tl.cumsum(empty.to(tl.int32), axis=0)
        okrow = nm & ((empty == 0) | (cs == 1))
        z = tl.where(okrow, z, -1e30)   # literal: a @jit kernel cannot read a module global
        if HARD:
            am = tl.argmax(z, axis=0)
            a = (on == am).to(tl.float32)
        else:
            mz = tl.max(z, axis=0)
            ez = tl.exp(z - mz)
            ez = tl.where(okrow, ez, 0.0)
            a = ez / tl.sum(ez, axis=0)
        w = g * a
        omw = 1.0 - w
        K = omw[:, None] * K + w[:, None] * kt[None, :]
        V = omw[:, None] * V + w[:, None] * vv[None, :]
        o = omw * o + w


def _p2(x):
    n = 1
    while n < x:
        n *= 2
    return n


def kv_scan_fwd(k, v, gate, fr, N, temp, G=1, hard=False, hard_gate=False, eps=1e-4, q=None):
    """Forward only (no autograd) -- for validation against ref_scan_kv."""
    B, T, H, DK = k.shape
    DV = v.shape[-1]
    NB = (T + G - 1) // G
    f = lambda x: x.contiguous().to(torch.float32)
    k, v, gate = f(k), f(v), f(gate)
    q = k if q is None else f(q)
    fr = fr.to(torch.float32).reshape(1)
    temp = torch.as_tensor(temp, dtype=torch.float32, device=k.device).reshape(1)
    kbuf = torch.empty(B, NB, H, N, DK, device=k.device, dtype=torch.float32)
    vbuf = torch.empty(B, NB, H, N, DV, device=k.device, dtype=torch.float32)
    obuf = torch.empty(B, NB, H, N, device=k.device, dtype=torch.float32)
    _kvfwd[(B * H,)](k, q, v, gate, fr, temp, kbuf, vbuf, obuf,
                     T, H, G, NB, eps,
                     N=N, DK=DK, DV=DV, NP=_p2(N), DKP=_p2(DK), DVP=_p2(DV),
                     HARD=bool(hard), HARDG=bool(hard_gate), num_warps=_nw(_p2(N), _p2(DK), _p2(DV)))
    return kbuf, vbuf, obuf


@triton.jit
def _kvbwd(k_ptr, q_ptr, v_ptr, g_ptr, fr_ptr, temp_ptr,
           kbuf_ptr, vbuf_ptr, obuf_ptr, dkbuf_ptr, dvbuf_ptr,
           dk_ptr, dq_ptr, dv_ptr, dg_ptr, dfr_ptr, dtemp_ptr,
           T, H, EPS,
           N: tl.constexpr, DK: tl.constexpr, DV: tl.constexpr,
           NP: tl.constexpr, DKP: tl.constexpr, DVP: tl.constexpr,
           HARD: tl.constexpr, HARDG: tl.constexpr, DETACH: tl.constexpr):
    """Reverse pass, G=1 only: the forward stored K, V and o for EVERY step, so each step's a and w
    are recomputed from stored state instead of checkpoint-and-replay.

    Adjoints, with dK'/dV'/do' the POST-write state adjoints carried back from t+1:
        dw   = sum_d (k_t - K) dK' + sum_d (v_t - V) dV' + (1-o) do'
        dk_t+= sum_n w dK'            dv_t = sum_n w dV'      dg = sum_n a_h dw
        dz   = a_s (g dw - a_s . g dw)                    [masked rows: 0]
        dfr -= sum_n o dz             dtemp += sum_n dz s / temp
        through the cosine (y = x/|x|, so dx = (dy - y<y,dy>)/|x|, and NO projection term where the
        norm hit its 1e-6 clamp, because there the derivative of the clamp is zero):
        dK  += kinv (dKn - Kn <Kn,dKn>)   dk_t += qinv (dqn - qn <qn,dqn>)
        carry: dK = (1-w) dK' + dkbuf_t + dK_match     dV = (1-w) dV' + dvbuf_t
               do = (1-w) do' - fr dz
    HARD/HARDG are straight-through in the forward, so w uses the hard value while dz uses the SOFT
    softmax and dg passes through unchanged."""
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
    for ti in range(0, T):
        t = T - 1 - ti
        so = ((b * T + t) * H + h)
        K = tl.load(kbuf_ptr + so * N * DK + on[:, None] * DK + ok_[None, :], mask=nk, other=0.0)
        V = tl.load(vbuf_ptr + so * N * DV + on[:, None] * DV + oe[None, :], mask=ne, other=0.0)
        o = tl.load(obuf_ptr + so * N + on, mask=nm, other=0.0)
        kt = tl.load(kbase + t * H * DK + ok_, mask=km, other=0.0)      # STORED
        qt = tl.load(qbase + t * H * DK + ok_, mask=km, other=0.0)      # used to SEARCH
        vv = tl.load(vbase + t * H * DV + oe, mask=em, other=0.0)
        graw = tl.load(gbase + t * H)
        g = (graw > 0.5).to(tl.float32) if HARDG else graw
        # --- recompute the forward at this step ---
        Kn2 = tl.sum(K * K, axis=1)
        kt2 = tl.sum(qt * qt, axis=0)
        knorm = tl.sqrt(Kn2)
        qnorm = tl.sqrt(kt2)
        kinv = 1.0 / tl.maximum(knorm, 1e-6)
        qinv = 1.0 / tl.maximum(qnorm, 1e-6)
        Kn = K * kinv[:, None]
        qn = qt * qinv
        s = temp * tl.sum(Kn * qn[None, :], axis=1)
        z = s - fr * o
        empty = (o < EPS) & nm
        cs = tl.cumsum(empty.to(tl.int32), axis=0)
        okrow = nm & ((empty == 0) | (cs == 1))
        z = tl.where(okrow, z, -1e30)
        mz = tl.max(z, axis=0)
        ez = tl.exp(z - mz)
        ez = tl.where(okrow, ez, 0.0)
        asoft = ez / tl.sum(ez, axis=0)
        if HARD:
            am = tl.argmax(z, axis=0)
            ah = (on == am).to(tl.float32)
        else:
            ah = asoft
        w = g * ah
        omw = 1.0 - w
        # --- adjoints ---
        dkb = tl.load(dkbuf_ptr + so * N * DK + on[:, None] * DK + ok_[None, :], mask=nk, other=0.0)
        dvb = tl.load(dvbuf_ptr + so * N * DV + on[:, None] * DV + oe[None, :], mask=ne, other=0.0)
        dw = tl.sum((kt[None, :] - K) * dK, axis=1) + tl.sum((vv[None, :] - V) * dV, axis=1) + (1.0 - o) * do
        dw = tl.where(nm, dw, 0.0)
        dkt = tl.sum(w[:, None] * dK, axis=0)
        dvv = tl.sum(w[:, None] * dV, axis=0)
        dg = tl.sum(ah * dw, axis=0)
        da = g * dw
        sa = tl.sum(asoft * da, axis=0)
        dz = asoft * (da - sa)
        dz = tl.where(okrow, dz, 0.0)
        dfr += -tl.sum(o * dz, axis=0)
        dtemp += tl.sum(dz * s, axis=0) / temp
        dKn = dz[:, None] * temp * qn[None, :]
        dqn = tl.sum(dz[:, None] * temp * Kn, axis=0)
        prj = tl.sum(dKn * Kn, axis=1)
        dKm_full = kinv[:, None] * (dKn - Kn * prj[:, None])
        # ZERONAN: no gradient path into a never-written (exactly zero) row -- see slotkern_kv.py.
        cl = (knorm < 1e-6)
        # DETACH: the address treated the buffer as a constant, so no gradient returns through it.
        # Forward is untouched -- this term exists only in the backward.
        dK_match = tl.zeros((NP, DKP), dtype=tl.float32) if DETACH else tl.where(cl[:, None], 0.0, dKm_full)
        prjq = tl.sum(dqn * qn, axis=0)
        dkt_match = tl.where(qnorm < 1e-6, qinv * dqn, qinv * (dqn - qn * prjq))
        # --- carry to the pre-write state at t ---
        dK = omw[:, None] * dK + dkb + dK_match
        dV = omw[:, None] * dV + dvb
        do = omw * do - fr * dz
        # the store path (sum_n w dK') belongs to the KEY; the match path belongs to the QUERY.
        # When q is k they are the same tensor and autograd sums them, reproducing the old total.
        tl.store(dk_ptr + (b * T * H + h) * DK + t * H * DK + ok_, dkt, mask=km)
        tl.store(dq_ptr + (b * T * H + h) * DK + t * H * DK + ok_, dkt_match, mask=km)
        tl.store(dv_ptr + (b * T * H + h) * DV + t * H * DV + oe, dvv, mask=em)
        tl.store(dg_ptr + (b * T * H + h) + t * H, dg)
    tl.atomic_add(dfr_ptr, dfr)
    tl.atomic_add(dtemp_ptr, dtemp)


class _KVScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, k, q, v, gate, fr, temp, N, hard, hard_gate, eps, detach_addr):
        kbuf, vbuf, obuf = kv_scan_fwd(k, v, gate, fr, N, temp, G=1, hard=hard, hard_gate=hard_gate, eps=eps, q=q)
        ctx.save_for_backward(k.contiguous().float(), q.contiguous().float(), v.contiguous().float(), gate.contiguous().float(),
                              fr.float().reshape(1), torch.as_tensor(temp, dtype=torch.float32,
                              device=k.device).reshape(1), kbuf, vbuf, obuf)
        ctx.meta = (N, bool(hard), bool(hard_gate), float(eps), bool(detach_addr))
        return kbuf, vbuf

    @staticmethod
    def backward(ctx, dkbuf, dvbuf):
        k, q, v, gate, fr, temp, kbuf, vbuf, obuf = ctx.saved_tensors
        N, hard, hard_gate, eps, detach_addr = ctx.meta
        B, T, H, DK = k.shape
        DV = v.shape[-1]
        dk = torch.zeros_like(k); dq = torch.zeros_like(q); dv = torch.zeros_like(v); dg = torch.zeros_like(gate)
        dfr = torch.zeros(1, device=k.device, dtype=torch.float32)
        dtemp = torch.zeros(1, device=k.device, dtype=torch.float32)
        _kvbwd[(B * H,)](k, q, v, gate, fr, temp, kbuf, vbuf, obuf,
                         dkbuf.contiguous().float(), dvbuf.contiguous().float(),
                         dk, dq, dv, dg, dfr, dtemp,
                         T, H, eps,
                         N=N, DK=DK, DV=DV, NP=_p2(N), DKP=_p2(DK), DVP=_p2(DV),
                         HARD=hard, HARDG=hard_gate, DETACH=detach_addr, num_warps=_nw(_p2(N), _p2(DK), _p2(DV)))
        return dk, dq, dv, dg, dfr, dtemp.reshape(()), None, None, None, None, None


def kv_scan(k, v, gate, fr, N, temp, hard=False, hard_gate=False, eps=1e-4, detach_addr=False, q=None, occ_bin=False):
    """Triton key-matched scan with autograd. G=1 only (which is what the model uses)."""
    if occ_bin:
        raise NotImplementedError("occ_bin is implemented in slotkern_kvg.kv_scan_g only (the G=1 backward was not patched)")
    return _KVScan.apply(k, k if q is None else q, v, gate, fr, temp, N, hard, hard_gate, eps, detach_addr)
