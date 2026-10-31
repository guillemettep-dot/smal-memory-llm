"""Triton fused-recurrent gated delta rule, emitting the per-token state stream.

One program per (b,h). S=[dk,dv] stays resident in registers across a real loop over
T (T passed non-constexpr, so Triton emits a loop, not 2048 unrolled copies). This is
the primitive a per-token bus needs: o_t AND S_t for every t, in ONE kernel launch,
instead of a Python loop firing a dozen ops per token.

Matches _delta_ref exactly:
    a=exp(g); kS=k.S; S = a*S + b*k (x) (v - a*kS); o = dk^-0.5 * q.S   (read after write)
"""
import torch, triton, triton.language as tl

@triton.jit
def _fwd(q_ptr, k_ptr, v_ptr, g_ptr, b_ptr, o_ptr, s_ptr, s0_ptr,
         B, T, H, scale, HAS_S0: tl.constexpr,
         DK: tl.constexpr, DV: tl.constexpr,
         PK: tl.constexpr, PV: tl.constexpr, STORE_S: tl.constexpr):
    pid = tl.program_id(0)               # pid = b*H + h
    b = pid // H
    h = pid % H
    offs_k = tl.arange(0, PK)            # PK/PV = next pow2 >= DK/DV
    offs_v = tl.arange(0, PV)
    mk = offs_k < DK
    mv = offs_v < DV
    S = tl.zeros((PK, PV), dtype=tl.float32)
    if HAS_S0:
        s0_off = (b * H + h) * DK * DV
        S = tl.load(s0_ptr + s0_off + offs_k[:, None] * DV + offs_v[None, :],
                    mask=mk[:, None] & mv[None, :], other=0.0).to(tl.float32)

    # base offsets into (B,T,H,D) tensors for token 0
    qkv_base = (b * T * H + h) * DK      # advances by H*DK per token
    v_base = (b * T * H + h) * DV
    g_base = (b * T * H + h)             # (B,T,H) -> advances by H per token

    for t in range(0, T):
        pk = k_ptr + qkv_base + t * H * DK + offs_k
        pq = q_ptr + qkv_base + t * H * DK + offs_k
        pv = v_ptr + v_base + t * H * DV + offs_v
        k = tl.load(pk, mask=mk, other=0.0).to(tl.float32)
        q = tl.load(pq, mask=mk, other=0.0).to(tl.float32)
        v = tl.load(pv, mask=mv, other=0.0).to(tl.float32)
        g = tl.load(g_ptr + g_base + t * H).to(tl.float32)
        beta = tl.load(b_ptr + g_base + t * H).to(tl.float32)
        a = tl.exp(g)
        # kS[v] = sum_k k[k] * S[k,v]   (uses OLD S)
        kS = tl.sum(k[:, None] * S, axis=0)                 # [DV]
        # S = a*S + beta * k (x) (v - a*kS)
        S = a * S + beta * (k[:, None] * (v[None, :] - a * kS[None, :]))
        # o[v] = scale * sum_k q[k]*S[k,v]   (read AFTER write)
        o = scale * tl.sum(q[:, None] * S, axis=0)          # [DV]
        tl.store(o_ptr + v_base + t * H * DV + offs_v, o.to(o_ptr.dtype.element_ty),
                 mask=mv)
        if STORE_S:
            # s layout (B,T,H,DK,DV) contiguous
            s_off = ((b * T + t) * H + h) * DK * DV
            tl.store(s_ptr + s_off + offs_k[:, None] * DV + offs_v[None, :],
                     S.to(s_ptr.dtype.element_ty),
                     mask=mk[:, None] & mv[None, :])


def fused_delta_stream(q, k, v, g, beta, scale=None, store_state=False,
                       initial_state=None, state_dtype=torch.float32):
    """q,k:(B,T,H,dk)  v:(B,T,H,dv)  g,beta:(B,T,H). Returns (o, S_stream|None)."""
    B, T, H, dk = q.shape
    dv = v.shape[-1]
    if scale is None:
        scale = dk ** -0.5
    q, k, v, g, beta = [x.contiguous() for x in (q, k, v, g, beta)]
    o = torch.empty(B, T, H, dv, device=q.device, dtype=q.dtype)
    if store_state:
        S = torch.empty(B, T, H, dk, dv, device=q.device, dtype=state_dtype)
    else:
        S = torch.empty(1, device=q.device, dtype=torch.float32)
    s0 = initial_state.contiguous().float() if initial_state is not None else S
    _fwd[(B * H,)](q, k, v, g, beta, o, S, s0, B, T, H, scale,
                   HAS_S0=initial_state is not None,
                   DK=dk, DV=dv,
                   PK=triton.next_power_of_2(dk), PV=triton.next_power_of_2(dv),
                   STORE_S=store_state)
    return o, (S if store_state else None)


@triton.jit
def _bwd(q_ptr, k_ptr, v_ptr, g_ptr, b_ptr, s_ptr, s0_ptr,
         do_ptr, ds_ptr, dsT_ptr,
         dq_ptr, dk_ptr, dv_ptr, dg_ptr, db_ptr, ds0_ptr,
         B, T, H, scale,
         HAS_S0: tl.constexpr, HAS_DS: tl.constexpr, HAS_DST: tl.constexpr,
         DK: tl.constexpr, DV: tl.constexpr,
         PK: tl.constexpr, PV: tl.constexpr):
    # Reverse scan. Forward per token t:
    #   kS   = k . S_{t-1}
    #   u    = v - a*kS                      (a = exp(g))
    #   S_t  = a*S_{t-1} + beta * k (x) u
    #   o_t  = scale * q . S_t
    # Adjoint, given dO_t, per-token external dS_t (bus reads), and dS_T:
    #   dS_t   += scale * q (x) dO_t                     (o reads S_t)
    #   du     = beta * (k . dS_t)          [DV]
    #   dk     = beta * (dS_t . u) - a * beta * (S_{t-1} . du_kS ... ) -- via dkS below
    #   dv     = du
    #   dkS    = -a * du                    [DV]
    #   dbeta  = sum(k (x) u * dS_t)
    #   da     = sum(S_{t-1} * dS_t) + kS . dkS ; dg = a*da
    #   dS_{t-1} = a*dS_t - a*beta* k (x) (k . dS_t)  + k (x) dkS
    #            = a*(dS_t - beta * k (x) (k.dS_t)) + k (x) dkS
    pid = tl.program_id(0)
    b = pid // H
    h = pid % H
    offs_k = tl.arange(0, PK)
    offs_v = tl.arange(0, PV)
    mk_ = offs_k < DK
    mv_ = offs_v < DV
    m2 = mk_[:, None] & mv_[None, :]

    dS = tl.zeros((PK, PV), dtype=tl.float32)
    if HAS_DST:
        off = (b * H + h) * DK * DV
        dS = tl.load(dsT_ptr + off + offs_k[:, None] * DV + offs_v[None, :],
                     mask=m2, other=0.0).to(tl.float32)

    qkv_base = (b * T * H + h) * DK
    v_base = (b * T * H + h) * DV
    g_base = (b * T * H + h)

    for i in range(0, T):
        t = T - 1 - i
        pk = k_ptr + qkv_base + t * H * DK + offs_k
        pq = q_ptr + qkv_base + t * H * DK + offs_k
        pv = v_ptr + v_base + t * H * DV + offs_v
        k = tl.load(pk, mask=mk_, other=0.0).to(tl.float32)
        q = tl.load(pq, mask=mk_, other=0.0).to(tl.float32)
        v = tl.load(pv, mask=mv_, other=0.0).to(tl.float32)
        g = tl.load(g_ptr + g_base + t * H).to(tl.float32)
        beta = tl.load(b_ptr + g_base + t * H).to(tl.float32)
        a = tl.exp(g)
        # S_{t-1}: from stream at t-1, or s0/zeros at t=0
        if t > 0:
            sp = ((b * T + (t - 1)) * H + h) * DK * DV
            Sp = tl.load(s_ptr + sp + offs_k[:, None] * DV + offs_v[None, :],
                         mask=m2, other=0.0).to(tl.float32)
        else:
            Sp = tl.zeros((PK, PV), dtype=tl.float32)
            if HAS_S0:
                s0o = (b * H + h) * DK * DV
                Sp = tl.load(s0_ptr + s0o + offs_k[:, None] * DV + offs_v[None, :],
                             mask=m2, other=0.0).to(tl.float32)
        kS = tl.sum(k[:, None] * Sp, axis=0)            # [PV]
        u = v - a * kS
        # S_t (for dq): recompute
        St = a * Sp + beta * (k[:, None] * u[None, :])
        do = tl.load(do_ptr + v_base + t * H * DV + offs_v,
                     mask=mv_, other=0.0).to(tl.float32)
        # dq from o_t = scale * q . S_t
        dq = scale * tl.sum(St * do[None, :], axis=1)
        tl.store(dq_ptr + qkv_base + t * H * DK + offs_k,
                 dq.to(dq_ptr.dtype.element_ty), mask=mk_)
        # accumulate into dS_t
        dS += scale * (q[:, None] * do[None, :])
        if HAS_DS:
            so = ((b * T + t) * H + h) * DK * DV
            dS += tl.load(ds_ptr + so + offs_k[:, None] * DV + offs_v[None, :],
                          mask=m2, other=0.0).to(tl.float32)
        kdS = tl.sum(k[:, None] * dS, axis=0)           # [PV]  (k . dS_t)
        du = beta * kdS
        dkS = -a * du
        dv_ = du
        dk_ = beta * tl.sum(dS * u[None, :], axis=1) + tl.sum(Sp * dkS[None, :], axis=1)
        dbeta = tl.sum((k[:, None] * u[None, :]) * dS)
        da = tl.sum(Sp * dS) - tl.sum(kS * du)
        dg = a * da
        tl.store(dv_ptr + v_base + t * H * DV + offs_v,
                 dv_.to(dv_ptr.dtype.element_ty), mask=mv_)
        tl.store(dk_ptr + qkv_base + t * H * DK + offs_k,
                 dk_.to(dk_ptr.dtype.element_ty), mask=mk_)
        tl.store(dg_ptr + g_base + t * H, dg.to(dg_ptr.dtype.element_ty))
        tl.store(db_ptr + g_base + t * H, dbeta.to(db_ptr.dtype.element_ty))
        # dS_{t-1}
        dS = a * (dS - beta * (k[:, None] * kdS[None, :]))
    ds0o = (b * H + h) * DK * DV
    tl.store(ds0_ptr + ds0o + offs_k[:, None] * DV + offs_v[None, :],
             dS.to(ds0_ptr.dtype.element_ty), mask=m2)


class _DeltaStream(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, g, beta, scale, initial_state):
        o, S = fused_delta_stream(q, k, v, g, beta, scale=scale,
                                  store_state=True, initial_state=initial_state)
        ctx.save_for_backward(q, k, v, g, beta, S,
                              initial_state if initial_state is not None
                              else torch.empty(0, device=q.device))
        ctx.scale = scale if scale is not None else q.shape[-1] ** -0.5
        return o, S

    @staticmethod
    def backward(ctx, do, dS_stream):
        q, k, v, g, beta, S, s0 = ctx.saved_tensors
        B, T, H, dk = q.shape
        dv = v.shape[-1]
        has_s0 = s0.numel() > 0
        dq = torch.empty_like(q, dtype=torch.float32)
        dk_ = torch.empty_like(k, dtype=torch.float32)
        dv_ = torch.empty_like(v, dtype=torch.float32)
        dg = torch.empty_like(g, dtype=torch.float32)
        db = torch.empty_like(beta, dtype=torch.float32)
        ds0 = torch.empty(B, H, dk, dv, device=q.device, dtype=torch.float32)
        has_ds = dS_stream is not None and bool((dS_stream != 0).any())
        dummy = torch.empty(1, device=q.device, dtype=torch.float32)
        _bwd[(B * H,)](q, k, v, g, beta, S,
                       s0 if has_s0 else dummy,
                       do.contiguous(),
                       dS_stream.contiguous() if has_ds else dummy,
                       dummy,
                       dq, dk_, dv_, dg, db, ds0,
                       B, T, H, ctx.scale,
                       HAS_S0=has_s0, HAS_DS=has_ds, HAS_DST=False,
                       DK=dk, DV=dv,
                       PK=triton.next_power_of_2(dk), PV=triton.next_power_of_2(dv))
        return (dq.to(q.dtype), dk_.to(k.dtype), dv_.to(v.dtype),
                dg.to(g.dtype), db.to(beta.dtype), None,
                ds0.to(s0.dtype) if has_s0 else None)


def delta_stream(q, k, v, g, beta, scale=None, initial_state=None):
    """Differentiable fused scan returning (o, per-token state stream).
    Gradients flow through o AND through any use of the state stream."""
    return _DeltaStream.apply(q, k, v, g, beta, scale, initial_state)


@triton.jit
def _fwd2(q_ptr, k_ptr, v_ptr, g_ptr, b_ptr, o_ptr, s_ptr, s0_ptr,
          B, T, H, scale, HAS_S0: tl.constexpr,
          DK: tl.constexpr, DV: tl.constexpr,
          PK: tl.constexpr, BV: tl.constexpr, STORE_S: tl.constexpr):
    pid = tl.program_id(0)
    vb = tl.program_id(1)
    b = pid // H
    h = pid % H
    offs_k = tl.arange(0, PK)
    offs_v = vb * BV + tl.arange(0, BV)
    mk = offs_k < DK
    mv = offs_v < DV
    S = tl.zeros((PK, BV), dtype=tl.float32)
    if HAS_S0:
        s0_off = (b * H + h) * DK * DV
        S = tl.load(s0_ptr + s0_off + offs_k[:, None] * DV + offs_v[None, :],
                    mask=mk[:, None] & mv[None, :], other=0.0).to(tl.float32)
    qkv_base = (b * T * H + h) * DK
    v_base = (b * T * H + h) * DV
    g_base = (b * T * H + h)
    for t in range(0, T):
        k = tl.load(k_ptr + qkv_base + t * H * DK + offs_k, mask=mk, other=0.0).to(tl.float32)
        q = tl.load(q_ptr + qkv_base + t * H * DK + offs_k, mask=mk, other=0.0).to(tl.float32)
        v = tl.load(v_ptr + v_base + t * H * DV + offs_v, mask=mv, other=0.0).to(tl.float32)
        g = tl.load(g_ptr + g_base + t * H).to(tl.float32)
        beta = tl.load(b_ptr + g_base + t * H).to(tl.float32)
        a = tl.exp(g)
        kS = tl.sum(k[:, None] * S, axis=0)
        S = a * S + beta * (k[:, None] * (v[None, :] - a * kS[None, :]))
        o = scale * tl.sum(q[:, None] * S, axis=0)
        tl.store(o_ptr + v_base + t * H * DV + offs_v, o.to(o_ptr.dtype.element_ty), mask=mv)
        if STORE_S:
            s_off = ((b * T + t) * H + h) * DK * DV
            tl.store(s_ptr + s_off + offs_k[:, None] * DV + offs_v[None, :],
                     S.to(s_ptr.dtype.element_ty), mask=mk[:, None] & mv[None, :])


@triton.jit
def _bwd2(q_ptr, k_ptr, v_ptr, g_ptr, b_ptr, s_ptr, s0_ptr,
          do_ptr, ds_ptr,
          dq_ptr, dk_ptr, dv_ptr, dg_ptr, db_ptr, ds0_ptr,
          B, T, H, scale,
          HAS_S0: tl.constexpr, HAS_DS: tl.constexpr,
          DK: tl.constexpr, DV: tl.constexpr,
          PK: tl.constexpr, BV: tl.constexpr):
    pid = tl.program_id(0)
    vb = tl.program_id(1)
    b = pid // H
    h = pid % H
    offs_k = tl.arange(0, PK)
    offs_v = vb * BV + tl.arange(0, BV)
    mk_ = offs_k < DK
    mv_ = offs_v < DV
    m2 = mk_[:, None] & mv_[None, :]
    dS = tl.zeros((PK, BV), dtype=tl.float32)
    qkv_base = (b * T * H + h) * DK
    v_base = (b * T * H + h) * DV
    g_base = (b * T * H + h)
    for i in range(0, T):
        t = T - 1 - i
        k = tl.load(k_ptr + qkv_base + t * H * DK + offs_k, mask=mk_, other=0.0).to(tl.float32)
        q = tl.load(q_ptr + qkv_base + t * H * DK + offs_k, mask=mk_, other=0.0).to(tl.float32)
        v = tl.load(v_ptr + v_base + t * H * DV + offs_v, mask=mv_, other=0.0).to(tl.float32)
        g = tl.load(g_ptr + g_base + t * H).to(tl.float32)
        beta = tl.load(b_ptr + g_base + t * H).to(tl.float32)
        a = tl.exp(g)
        if t > 0:
            sp = ((b * T + (t - 1)) * H + h) * DK * DV
            Sp = tl.load(s_ptr + sp + offs_k[:, None] * DV + offs_v[None, :],
                         mask=m2, other=0.0).to(tl.float32)
        else:
            Sp = tl.zeros((PK, BV), dtype=tl.float32)
            if HAS_S0:
                s0o = (b * H + h) * DK * DV
                Sp = tl.load(s0_ptr + s0o + offs_k[:, None] * DV + offs_v[None, :],
                             mask=m2, other=0.0).to(tl.float32)
        kS = tl.sum(k[:, None] * Sp, axis=0)
        u = v - a * kS
        St = a * Sp + beta * (k[:, None] * u[None, :])
        do = tl.load(do_ptr + v_base + t * H * DV + offs_v, mask=mv_, other=0.0).to(tl.float32)
        dq = scale * tl.sum(St * do[None, :], axis=1)
        tl.atomic_add(dq_ptr + qkv_base + t * H * DK + offs_k, dq, mask=mk_)
        dS += scale * (q[:, None] * do[None, :])
        if HAS_DS:
            so = ((b * T + t) * H + h) * DK * DV
            dS += tl.load(ds_ptr + so + offs_k[:, None] * DV + offs_v[None, :],
                          mask=m2, other=0.0).to(tl.float32)
        kdS = tl.sum(k[:, None] * dS, axis=0)
        du = beta * kdS
        dkS = -a * du
        dk_ = beta * tl.sum(dS * u[None, :], axis=1) + tl.sum(Sp * dkS[None, :], axis=1)
        dbeta = tl.sum((k[:, None] * u[None, :]) * dS)
        da = tl.sum(Sp * dS) - tl.sum(kS * du)
        tl.store(dv_ptr + v_base + t * H * DV + offs_v, du, mask=mv_)
        tl.atomic_add(dk_ptr + qkv_base + t * H * DK + offs_k, dk_, mask=mk_)
        tl.atomic_add(dg_ptr + g_base + t * H, a * da)
        tl.atomic_add(db_ptr + g_base + t * H, dbeta)
        dS = a * (dS - beta * (k[:, None] * kdS[None, :]))
    ds0o = (b * H + h) * DK * DV
    tl.store(ds0_ptr + ds0o + offs_k[:, None] * DV + offs_v[None, :], dS, mask=m2)


def _pick_bv(dv):
    for bv in (16, 8):
        if dv % bv == 0:
            return bv
    return dv


def fused_delta_stream2(q, k, v, g, beta, scale=None, store_state=False,
                        initial_state=None):
    B, T, H, dk = q.shape
    dv = v.shape[-1]
    if scale is None:
        scale = dk ** -0.5
    q, k, v, g, beta = [x.contiguous() for x in (q, k, v, g, beta)]
    o = torch.empty(B, T, H, dv, device=q.device, dtype=q.dtype)
    S = (torch.empty(B, T, H, dk, dv, device=q.device, dtype=torch.float32)
         if store_state else torch.empty(1, device=q.device, dtype=torch.float32))
    s0 = initial_state.contiguous().float() if initial_state is not None else S
    BV = _pick_bv(dv)
    _fwd2[(B * H, (dv + BV - 1) // BV)](
        q, k, v, g, beta, o, S, s0, B, T, H, scale,
        HAS_S0=initial_state is not None, DK=dk, DV=dv,
        PK=triton.next_power_of_2(dk), BV=BV, STORE_S=store_state)
    return o, (S if store_state else None)


class _DeltaStream2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, g, beta, scale, initial_state):
        o, S = fused_delta_stream2(q, k, v, g, beta, scale=scale,
                                   store_state=True, initial_state=initial_state)
        ctx.save_for_backward(q, k, v, g, beta, S,
                              initial_state if initial_state is not None
                              else torch.empty(0, device=q.device))
        ctx.scale = scale if scale is not None else q.shape[-1] ** -0.5
        return o, S

    @staticmethod
    def backward(ctx, do, dS_stream):
        q, k, v, g, beta, S, s0 = ctx.saved_tensors
        B, T, H, dk = q.shape
        dv = v.shape[-1]
        has_s0 = s0.numel() > 0
        f32 = dict(device=q.device, dtype=torch.float32)
        dq = torch.zeros(q.shape, **f32)
        dk_ = torch.zeros(k.shape, **f32)
        dv_ = torch.empty(v.shape, **f32)
        dg = torch.zeros(g.shape, **f32)
        db = torch.zeros(beta.shape, **f32)
        ds0 = torch.empty(B, H, dk, dv, **f32)
        has_ds = dS_stream is not None and bool((dS_stream != 0).any())
        dummy = torch.empty(1, **f32)
        BV = _pick_bv(dv)
        _bwd2[(B * H, (dv + BV - 1) // BV)](
            q, k, v, g, beta, S,
            s0 if has_s0 else dummy,
            do.contiguous(),
            dS_stream.contiguous() if has_ds else dummy,
            dq, dk_, dv_, dg, db, ds0,
            B, T, H, ctx.scale,
            HAS_S0=has_s0, HAS_DS=has_ds, DK=dk, DV=dv,
            PK=triton.next_power_of_2(dk), BV=BV)
        return (dq.to(q.dtype), dk_.to(k.dtype), dv_.to(v.dtype),
                dg.to(g.dtype), db.to(beta.dtype), None,
                ds0.to(s0.dtype) if has_s0 else None)


def delta_stream2(q, k, v, g, beta, scale=None, initial_state=None):
    """Column-parallel differentiable fused scan -> (o, per-token state stream)."""
    return _DeltaStream2.apply(q, k, v, g, beta, scale, initial_state)
