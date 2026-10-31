"""kslot: delta-rule memory scan with PER-SLOT decay and write strength, ragged payloads,
R read queries per head, and per-token state emission -- one kernel for the main read
path and the compressor's shadow path.

Why it exists (measured on m4/m300, 2026-09-03): a dk-address store cannot survive 1,024
writes landing on top of it; replaying real writes at 1% frequency lifted single-cabinet
retrieval 10.9% -> 42.6% at 64 tokens. Writing rarely needs a gate that reaches exactly
zero PER SLOT, with the unwritten slots not decaying either -- neither is expressible in
the library kernel (per-head scalars), hence this one.

Recurrence, per head, per token, with a = exp(g) and beta both dk-vectors:
    S' = diag(a) S_{t-1}
    u  = v - k^T S'                       (residual at the address, read from the decayed state)
    S_t = S' + (beta * k) (x) u          (row r receives beta_r k_r u)
    o_r = q_r^T S_t                       (R reads)
With uniform rows this is exactly kpair2/kpair3's rule. The read at k still converges to v
with effective strength sum_r beta_r k_r^2, so it is still a delta rule.

Parallelism: the recurrence is independent per PAYLOAD COLUMN, so a program owns
(batch, head, block of CB columns). Heads of any payload width share one compiled kernel
through a column mask; values, reads and the emission use packed ragged layouts with
per-head offsets (see slot_layout).

Backward: S_{t-1} is read from the emission (no replay); the dk-vector adjoints of a head are
written per column block to a partial buffer and summed with one index_add (no atomics:
deterministic, and no memory fences inside the sequential chain).
g (log-decay) is passed in fp32: bf16 cannot represent a = 0.9999, a 7k-token half-life.
"""
import torch, triton, triton.language as tl


@triton.jit
def _fwd(k_ptr, g_ptr, b_ptr, v_ptr, q_ptr, out_ptr, emit_ptr, s0_ptr, sfin_ptr,
         pg2g_ptr, pg2c_ptr, dvt_ptr, voff_ptr, ooff_ptr,
         T, G, NP, VT, ST, OT,
         DK: tl.constexpr, PK: tl.constexpr, CB: tl.constexpr, R: tl.constexpr, QSHARED: tl.constexpr, HAS_S0: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    b = pid // NP
    p = pid % NP
    gi = tl.load(pg2g_ptr + p).to(tl.int64)
    j0 = tl.load(pg2c_ptr + p).to(tl.int64)
    dvg = tl.load(dvt_ptr + gi).to(tl.int64)
    vo = tl.load(voff_ptr + gi).to(tl.int64)
    oo = tl.load(ooff_ptr + gi).to(tl.int64)
    ok = tl.arange(0, PK)                  # key lanes padded to a power of two; km masks the real DK
    km = ok < DK
    oc = j0 + tl.arange(0, CB)
    cm = oc < dvg
    sb0 = pid * DK * CB
    st_off = ok[:, None] * CB + tl.arange(0, CB)[None, :]
    if HAS_S0:
        S = tl.load(s0_ptr + sb0 + st_off, mask=km[:, None], other=0.0)          # continue from the previous chunk's fp32 state
    else:
        S = tl.zeros((PK, CB), dtype=tl.float32)
    kb = (b * T * G + gi) * DK
    if QSHARED:
        qb = (b * T * R) * DK          # q:(B,T,R,dk) shared by every head
        qs = R * DK
    else:
        qb = ((b * T * G + gi) * R) * DK
        qs = G * R * DK
    vb = b * T * VT + vo
    sb = b * T * ST
    ob = b * T * OT + oo
    # prefetch token 0
    k_n = tl.load(k_ptr + kb + ok, mask=km, other=0.0)
    g_n = tl.load(g_ptr + vb + oc, mask=cm, other=0.0)
    b_n = tl.load(b_ptr + vb + oc, mask=cm, other=0.0)
    v_n = tl.load(v_ptr + vb + oc, mask=cm, other=0.0)
    for t in range(0, T):
        k = k_n.to(tl.float32)
        gg = g_n.to(tl.float32)
        be = b_n.to(tl.float32)
        vv = v_n.to(tl.float32)
        tn = tl.minimum(t + 1, T - 1)
        k_n = tl.load(k_ptr + kb + tn * G * DK + ok, mask=km, other=0.0)
        g_n = tl.load(g_ptr + vb + tn * VT + oc, mask=cm, other=0.0)
        b_n = tl.load(b_ptr + vb + tn * VT + oc, mask=cm, other=0.0)
        v_n = tl.load(v_ptr + vb + tn * VT + oc, mask=cm, other=0.0)
        a = tl.exp(gg)
        Sp = a[None, :] * S                      # decay per PAYLOAD column
        kS = tl.sum(k[:, None] * Sp, axis=0)
        u = (vv - kS) * be                       # per-column write strength on the RESIDUAL
        S = Sp + k[:, None] * u[None, :]         # write direction is exactly k
        for r in tl.static_range(R):
            qv = tl.load(q_ptr + qb + t * qs + r * DK + ok, mask=km, other=0.0).to(tl.float32)
            o = tl.sum(qv[:, None] * S, axis=0)
            tl.store(out_ptr + ob + t * OT + r * dvg + oc, o.to(out_ptr.dtype.element_ty), mask=cm)
        tl.store(emit_ptr + sb + t * ST + (vo + oc)[None, :] * DK + ok[:, None],
                 S.to(emit_ptr.dtype.element_ty), mask=cm[None, :] & km[:, None])
    tl.store(sfin_ptr + sb0 + st_off, S, mask=km[:, None])                  # final state, fp32, for the next chunk


@triton.jit
def _bwd(k_ptr, g_ptr, b_ptr, v_ptr, q_ptr, emit_ptr, dout_ptr, demit_ptr,
         part_ptr, db_ptr, dg_ptr, dv_ptr, dq_ptr, s0_ptr, dsfin_ptr, ds0_ptr,
         pg2g_ptr, pg2c_ptr, dvt_ptr, voff_ptr, ooff_ptr,
         T, G, NP, VT, ST, OT,
         DK: tl.constexpr, PK: tl.constexpr, CB: tl.constexpr, R: tl.constexpr, QSHARED: tl.constexpr, HAS_S0: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    b = pid // NP
    p = pid % NP
    pb = pid * T * DK              # this program's own dk partials: (t, DK)
    gi = tl.load(pg2g_ptr + p).to(tl.int64)
    j0 = tl.load(pg2c_ptr + p).to(tl.int64)
    dvg = tl.load(dvt_ptr + gi).to(tl.int64)
    vo = tl.load(voff_ptr + gi).to(tl.int64)
    oo = tl.load(ooff_ptr + gi).to(tl.int64)
    ok = tl.arange(0, PK)                  # key lanes padded to a power of two; km masks the real DK
    km = ok < DK
    oc = j0 + tl.arange(0, CB)
    cm = oc < dvg
    m2 = cm[None, :] & km[:, None]
    kb = (b * T * G + gi) * DK
    if QSHARED:
        qb = (b * T * R) * DK
        qs = R * DK
    else:
        qb = ((b * T * G + gi) * R) * DK
        qs = G * R * DK
    vb = b * T * VT + vo
    sb = b * T * ST
    ob = b * T * OT + oo
    sb0 = pid * DK * CB
    st_off = ok[:, None] * CB + tl.arange(0, CB)[None, :]
    dS = tl.load(dsfin_ptr + sb0 + st_off, mask=km[:, None], other=0.0)         # gradient arriving through the final state (zeros when unchunked)
    if HAS_S0:
        S0t = tl.load(s0_ptr + sb0 + st_off, mask=km[:, None], other=0.0)
    else:
        S0t = tl.zeros((PK, CB), dtype=tl.float32)
    t0 = T - 1
    k_n = tl.load(k_ptr + kb + t0 * G * DK + ok, mask=km, other=0.0)
    g_n = tl.load(g_ptr + vb + t0 * VT + oc, mask=cm, other=0.0)
    b_n = tl.load(b_ptr + vb + t0 * VT + oc, mask=cm, other=0.0)
    v_n = tl.load(v_ptr + vb + t0 * VT + oc, mask=cm, other=0.0)
    dE_n = tl.load(demit_ptr + sb + t0 * ST + (vo + oc)[None, :] * DK + ok[:, None], mask=m2, other=0.0)
    tp = tl.maximum(t0 - 1, 0)
    Sm_n = tl.load(emit_ptr + sb + tp * ST + (vo + oc)[None, :] * DK + ok[:, None], mask=m2 & ((ok[:, None] * 0 + t0) > 0), other=0.0)
    for i in range(0, T):
        t = T - 1 - i
        k = k_n.to(tl.float32)
        gg = g_n.to(tl.float32)
        be = b_n.to(tl.float32)
        vv = v_n.to(tl.float32)
        dE = dE_n.to(tl.float32)
        Sm1 = Sm_n.to(tl.float32)
        if HAS_S0:
            Sm1 = tl.where((ok[:, None] * 0 + t) > 0, Sm1, S0t)   # the state before token 0 is the chunk's initial state
        tn = tl.maximum(t - 1, 0)
        k_n = tl.load(k_ptr + kb + tn * G * DK + ok, mask=km, other=0.0)
        g_n = tl.load(g_ptr + vb + tn * VT + oc, mask=cm, other=0.0)
        b_n = tl.load(b_ptr + vb + tn * VT + oc, mask=cm, other=0.0)
        v_n = tl.load(v_ptr + vb + tn * VT + oc, mask=cm, other=0.0)
        dE_n = tl.load(demit_ptr + sb + tn * ST + (vo + oc)[None, :] * DK + ok[:, None], mask=m2, other=0.0)
        tp = tl.maximum(tn - 1, 0)
        Sm_n = tl.load(emit_ptr + sb + tp * ST + (vo + oc)[None, :] * DK + ok[:, None], mask=m2 & ((ok[:, None] * 0 + tn) > 0), other=0.0)
        a = tl.exp(gg)
        Sp = a[None, :] * Sm1
        kS = tl.sum(k[:, None] * Sp, axis=0)
        ur = vv - kS                              # raw residual
        u = ur * be
        St = Sp + k[:, None] * u[None, :]
        dS += dE
        for r in tl.static_range(R):
            qv = tl.load(q_ptr + qb + t * qs + r * DK + ok, mask=km, other=0.0).to(tl.float32)
            do = tl.load(dout_ptr + ob + t * OT + r * dvg + oc, mask=cm, other=0.0).to(tl.float32)
            dS += qv[:, None] * do[None, :]
            dq = tl.sum(St * do[None, :], axis=1)
            tl.atomic_add(dq_ptr + qb + t * qs + r * DK + ok, dq, mask=km)
        # S = a*Sm1 + k (be*(v - k^T (a*Sm1)))^T
        dU = tl.sum(k[:, None] * dS, axis=0)      # d/d u  (per column)
        du = dU * be                              # d/d (v - kS)
        tl.store(dv_ptr + vb + t * VT + oc, du, mask=cm)
        dbe = dU * ur                             # per column
        dSp = dS - k[:, None] * du[None, :]
        dk_ = tl.sum(dS * u[None, :], axis=1) - tl.sum(Sp * du[None, :], axis=1)
        da = tl.sum(Sm1 * dSp, axis=0)            # per column
        tl.store(part_ptr + pb + t * DK + ok, dk_, mask=km)
        tl.store(db_ptr + vb + t * VT + oc, dbe, mask=cm)          # each (b,t,column) belongs to exactly ONE program: a store, not an atomic (2.5x faster backward, bit-identical)
        tl.store(dg_ptr + vb + t * VT + oc, a * da, mask=cm)
        dS = a[None, :] * dSp
    tl.store(ds0_ptr + sb0 + st_off, dS, mask=km[:, None])                  # d loss / d initial state


def slot_layout(dv_list, dk, R, CB=16, device="cuda"):
    """Packed ragged layout for G heads with payload widths dv_list.
    Returns tables + totals used by the kernel and by the torch-side (un)packing."""
    G = len(dv_list)
    voff, ooff, pg2g, pg2c = [], [], [], []
    vt = ot = 0
    for g, dv in enumerate(dv_list):
        voff.append(vt); ooff.append(ot)
        vt += dv; ot += R * dv
        for j0 in range(0, dv, CB):
            pg2g.append(g); pg2c.append(j0)
    st = dk * vt                       # emission per token: the (VT x dk) matrix, column-major state
    i32 = lambda x: torch.tensor(x, dtype=torch.int32, device=device)
    return dict(G=G, dk=dk, R=R, CB=CB, VT=vt, ST=st, OT=ot, NP=len(pg2g),
                dv=list(dv_list), dvt=i32(dv_list), voff=i32(voff), ooff=i32(ooff),
                pg2g=i32(pg2g), pg2c=i32(pg2c),
                vsl=[(voff[g], voff[g] + dv_list[g]) for g in range(G)],
                osl=[(ooff[g], ooff[g] + R * dv_list[g]) for g in range(G)])


NUM_WARPS = 2
NUM_WARPS_BWD = 1     # measured 2026-09-03: 1 warp beats 2/4 in the backward (deep 8.9 vs 10.7 vs 14.9 ms; shadow 39.5 vs 47.5 vs 79.7)


class _KSlot(torch.autograd.Function):
    @staticmethod
    def forward(ctx, k, g, beta, v, q, L, s0=None):
        B, T, G, DK = k.shape
        R = L["R"]
        assert g.shape == (B, T, L["VT"]) and beta.shape == g.shape, \
            "column gates: g and beta are per PAYLOAD column, (B,T,VT), got %s" % (tuple(g.shape),)
        assert v.shape == (B, T, L["VT"]), v.shape
        if R == 0:
            # scan + emission only (the production mode): reads are einsums over the emission.
            # Measured: in-kernel reads cost more than the scan (3.3 -> 8.4 ms/layer at R=4) and
            # the hub's shared queries made the backward 8x the forward via 99-way atomics.
            q = k.new_empty(B, T, G, 1, DK); qshared = False
        else:
            qshared = q.dim() == 4
            assert q.shape == ((B, T, R, DK) if qshared else (B, T, G, R, DK)), q.shape
        assert g.dtype == torch.float32, "g (log-decay) must be fp32: bf16 cannot hold a ~ 0.9999"
        k, beta, v, q = [x.contiguous() for x in (k, beta, v, q)]
        g = g.contiguous()
        out = torch.empty(B, T, L["OT"], device=k.device, dtype=torch.bfloat16)
        emit = torch.empty(B, T, L["ST"], device=k.device, dtype=torch.bfloat16)
        has_s0 = s0 is not None
        s0t = s0.contiguous().float() if has_s0 else out.new_zeros(1, dtype=torch.float32)
        if has_s0: assert s0t.shape == (B * L["NP"], DK, L["CB"]), s0t.shape
        sfin = torch.empty(B * L["NP"], DK, L["CB"], device=k.device, dtype=torch.float32)
        _fwd[(B * L["NP"],)](k, g, beta, v, q, out, emit, s0t, sfin,
                             L["pg2g"], L["pg2c"], L["dvt"], L["voff"], L["ooff"],
                             T, G, L["NP"], L["VT"], L["ST"], L["OT"],
                             DK=DK, PK=triton.next_power_of_2(DK), CB=L["CB"], R=R, QSHARED=qshared, HAS_S0=has_s0, num_warps=NUM_WARPS)
        ctx.save_for_backward(k, g, beta, v, q, emit, s0t)
        ctx.L = L; ctx.qshared = qshared; ctx.has_s0 = has_s0
        return out, emit, sfin

    @staticmethod
    def backward(ctx, dout, demit, dsfin):
        k, g, beta, v, q, emit, s0t = ctx.saved_tensors
        L = ctx.L
        B, T, G, DK = k.shape
        dout = dout.contiguous()
        demit = demit.contiguous() if demit is not None else torch.zeros_like(emit)
        dsfin_t = dsfin.contiguous().float() if dsfin is not None else torch.zeros(B * L["NP"], DK, L["CB"], device=k.device, dtype=torch.float32)
        ds0 = torch.empty_like(dsfin_t)
        NP = L["NP"]
        part = torch.empty(B, NP, T, DK, device=k.device, dtype=torch.float32)
        # every (b, t, column) is stored exactly once by its program: no zero-fill needed
        dv = torch.empty(B, T, L["VT"], device=k.device, dtype=torch.float32)
        db = torch.empty(B, T, L["VT"], device=k.device, dtype=torch.float32)
        dg = torch.empty(B, T, L["VT"], device=k.device, dtype=torch.float32)
        R = L["R"]
        dq = torch.zeros((B, T, R, DK) if ctx.qshared else (B, T, G, R, DK), device=k.device, dtype=torch.float32) if R > 0 else torch.empty(1, device=k.device, dtype=torch.float32)
        _bwd[(B * L["NP"],)](k, g, beta, v, q, emit, dout, demit,
                             part, db, dg, dv, dq, s0t, dsfin_t, ds0,
                             L["pg2g"], L["pg2c"], L["dvt"], L["voff"], L["ooff"],
                             T, G, L["NP"], L["VT"], L["ST"], L["OT"],
                             DK=DK, PK=triton.next_power_of_2(DK), CB=L["CB"], R=R, QSHARED=ctx.qshared, HAS_S0=ctx.has_s0, num_warps=NUM_WARPS_BWD)
        # sum the column blocks of each head: (B,NP,T,3,DK) -> (B,G,T,3,DK) -> (B,T,G,DK) x3
        if NP == G:                                        # one program per head: the partials ARE the per-head dk
            dk = part.permute(0, 2, 1, 3)
        else:
            acc = torch.zeros(B, G, T, DK, device=k.device, dtype=torch.float32)
            acc.index_add_(1, L["pg2g"].long(), part)
            dk = acc.permute(0, 2, 1, 3)
        return (dk.to(k.dtype), dg, db.to(beta.dtype), dv.to(v.dtype), (None if R == 0 else dq.to(q.dtype)), None, (ds0 if ctx.has_s0 else None))


def kslot(k, g, beta, v, q, L):
    """k:(B,T,G,dk) bf16 unit keys · g:(B,T,VT) fp32 log-decay<=0 · beta:(B,T,VT) bf16 in [0,1]
    (COLUMN gates: one per payload dimension, so the write stays exactly along k)
    v:(B,T,VT) bf16 packed values · q:(B,T,G,R,dk) bf16, or (B,T,R,dk) to read EVERY head with the
    same R queries (the hub's context queries) · L from slot_layout.
    -> out:(B,T,OT) bf16 packed reads [head g at osl[g], r-major], emit:(B,T,ST) bf16: per token
       the state of ALL heads as one (VT x dk) matrix (column-major: row c = payload column c
       of the (dk x VT) state), head g at rows vsl[g]."""
    return _KSlot.apply(k, g, beta, v, q, L, None)[:2]


def kslot_s(k, g, beta, v, q, L, s0=None):
    """kslot with an INITIAL state and the fp32 FINAL state: -> (out, emit, sfin). s0/sfin are
    program-indexed (B*NP, dk, CB) fp32 -- feed one chunk's sfin as the next chunk's s0."""
    return _KSlot.apply(k, g, beta, v, q, L, s0)



import torch._dynamo as _dynamo
kslot = _dynamo.disable(kslot)      # measured: traced into the compiled mixer, the scan fwd+bwd ran twice per step
kslot_s = _dynamo.disable(kslot_s)

def final_state(emit, L, g):
    """The state of head g after the last token, (B, dk, dv_g), from the emission."""
    a, b = L["vsl"][g]
    return emit[:, -1].view(emit.shape[0], L["VT"], L["dk"])[:, a:b].transpose(1, 2)


def _pack_index(L, R):
    """Flat indices turning the (G*R, VT) matmul output of per-head queries into the packed
    (R*VT) vector [head g, query r, payload]. All distinct: the backward is a plain scatter."""
    key = ("pk", R)
    if key not in L:
        VT = L["VT"]
        idx = [(g * R + r) * VT + c for g, (a, b) in enumerate(L["vsl"]) for r in range(R) for c in range(a, b)]
        L[key] = torch.tensor(idx, dtype=torch.long, device=L["dvt"].device)
    return L[key]


def _pad_maps(L, R, dvmax, shared):
    """(src, dst): src gathers the real entries of the matmul output (shared: (R, VT); per-head:
    (G*R, VT)) in head-major order; dst places them in the (G, R*dvmax) zero-padded block.
    Both are permutations of distinct positions (no pad routed through a shared address)."""
    key = ("pad", R, dvmax, shared)
    if key not in L:
        VT = L["VT"]; src, dst = [], []
        for g, (a, b) in enumerate(L["vsl"]):
            for r in range(R):
                for j in range(b - a):
                    src.append((r if shared else g * R + r) * VT + a + j)
                    dst.append((g * R + r) * dvmax + j)
        dev = L["dvt"].device
        L[key] = (torch.tensor(src, dtype=torch.long, device=dev), torch.tensor(dst, dtype=torch.long, device=dev))
    return L[key]


def _mm(emit, qflat, L):
    """qflat:(N, M, dk) against the column-major emission -> (N, M, VT), computed as (S @ q^T)^T:
    the 3 GB operand on the LEFT of cuBLAS (25.5 ms fwd+bwd vs 40.1 for q @ S^T via a view)."""
    N = qflat.shape[0]
    Sc = emit.view(N, L["VT"], L["dk"])
    return torch.bmm(Sc, qflat.transpose(1, 2).to(Sc.dtype)).transpose(1, 2)


# ---------------------------------------------------------------------------------------------------------------------
# reads as ONE memory-bound pass: out[n,g,r,:] = S[n,g] (DV x DK) . q[n,(g,)r]   -- the state block is read once; the
# backward writes d(emission) = sum_r q_r (x) dout_r once and reads S once for dq. Replaces a batched GEMM of 262k tiny
# (16x32)@(32x2) products (3.4 ms/layer fwd+bwd, measured) with ~0.5 ms.
@triton.jit
def _rd_fwd(e_ptr, q_ptr, o_ptr, N, G, R: tl.constexpr, SHARED: tl.constexpr, DV: tl.constexpr, DK: tl.constexpr, PK: tl.constexpr, BM: tl.constexpr):
    pid_m = tl.program_id(0); g = tl.program_id(1).to(tl.int64)
    rows = pid_m * BM + tl.arange(0, BM); rm = rows < N; n64 = rows.to(tl.int64)
    ok = tl.arange(0, PK); km = ok < DK
    base = (n64 * G + g) * (DV * DK)
    S = tl.load(e_ptr + base[:, None, None] + tl.arange(0, DV)[None, :, None] * DK + ok[None, None, :],
                mask=rm[:, None, None] & km[None, None, :], other=0.0).to(tl.float32)
    for r in tl.static_range(R):
        if SHARED:
            qo = (n64 * R + r) * DK
        else:
            qo = ((n64 * G + g) * R + r) * DK
        q = tl.load(q_ptr + qo[:, None] + ok[None, :], mask=rm[:, None] & km[None, :], other=0.0).to(tl.float32)
        o = tl.sum(S * q[:, None, :], axis=2)
        if SHARED:
            oo = (n64 * R + r) * (G * DV) + g * DV
        else:
            oo = ((n64 * G + g) * R + r) * DV
        tl.store(o_ptr + oo[:, None] + tl.arange(0, DV)[None, :], o.to(tl.bfloat16), mask=rm[:, None])

@triton.jit
def _rd_bwd(e_ptr, q_ptr, do_ptr, de_ptr, dq_ptr, N, G, R: tl.constexpr, SHARED: tl.constexpr, DV: tl.constexpr, DK: tl.constexpr, PK: tl.constexpr, BM: tl.constexpr):
    pid_m = tl.program_id(0); g = tl.program_id(1).to(tl.int64)
    rows = pid_m * BM + tl.arange(0, BM); rm = rows < N; n64 = rows.to(tl.int64)
    ok = tl.arange(0, PK); km = ok < DK
    base = (n64 * G + g) * (DV * DK)
    off3 = base[:, None, None] + tl.arange(0, DV)[None, :, None] * DK + ok[None, None, :]
    m3 = rm[:, None, None] & km[None, None, :]
    S = tl.load(e_ptr + off3, mask=m3, other=0.0).to(tl.float32)
    dS = tl.zeros((BM, DV, PK), dtype=tl.float32)
    for r in tl.static_range(R):
        if SHARED:
            qo = (n64 * R + r) * DK
            oo = (n64 * R + r) * (G * DV) + g * DV
            dqo = ((n64 * G + g) * R + r) * DK          # per-group partial, summed over g outside
        else:
            qo = ((n64 * G + g) * R + r) * DK
            oo = ((n64 * G + g) * R + r) * DV
            dqo = qo
        q = tl.load(q_ptr + qo[:, None] + ok[None, :], mask=rm[:, None] & km[None, :], other=0.0).to(tl.float32)
        do = tl.load(do_ptr + oo[:, None] + tl.arange(0, DV)[None, :], mask=rm[:, None], other=0.0).to(tl.float32)
        dS += do[:, :, None] * q[:, None, :]
        dq = tl.sum(S * do[:, :, None], axis=1)
        tl.store(dq_ptr + dqo[:, None] + ok[None, :], dq, mask=rm[:, None] & km[None, :])
    tl.store(de_ptr + off3, dS.to(tl.bfloat16), mask=m3)

class _Reads(torch.autograd.Function):
    @staticmethod
    def forward(ctx, emit, q, G, DV, DK):
        B, T = emit.shape[0], emit.shape[1]; N = B * T
        shared = q.dim() == 4; R = q.shape[2] if shared else q.shape[3]
        q = q.contiguous(); emit = emit.contiguous()
        out = torch.empty((B, T, R, G * DV) if shared else (B, T, G, R, DV), device=emit.device, dtype=torch.bfloat16)
        BM = 16
        _rd_fwd[(triton.cdiv(N, BM), G)](emit, q, out, N, G, R=R, SHARED=shared, DV=DV, DK=DK, PK=triton.next_power_of_2(DK), BM=BM, num_warps=4)
        ctx.save_for_backward(emit, q); ctx.meta = (G, DV, DK, R, shared)
        return out
    @staticmethod
    def backward(ctx, dout):
        emit, q = ctx.saved_tensors; G, DV, DK, R, shared = ctx.meta
        B, T = emit.shape[0], emit.shape[1]; N = B * T
        dout = dout.contiguous()
        de = torch.empty_like(emit)
        dq = torch.empty((B, T, G, R, DK), device=emit.device, dtype=torch.float32)
        BM = 16
        _rd_bwd[(triton.cdiv(N, BM), G)](emit, q, dout, de, dq, N, G, R=R, SHARED=shared, DV=DV, DK=DK, PK=triton.next_power_of_2(DK), BM=BM, num_warps=4)
        dq = dq.sum(2) if shared else dq
        return de, dq.to(q.dtype), None, None, None

def reads_kernel(emit, q, L):
    """Same contract as reads() for a uniform-payload layout, via the fused kernel."""
    B, T = emit.shape[0], emit.shape[1]; G = L["G"]; DV = L["dv"][0]; DK = L["dk"]
    out = _Reads.apply(emit, q, G, DV, DK)
    if q.dim() == 4:
        R = q.shape[2]; packed = out.reshape(B, T, R * L["VT"])
        per = [out[..., a:b] for (a, b) in L["vsl"]]
    else:
        R = q.shape[3]; packed = out.reshape(B, T, G * R * DV)
        per = [out[:, :, g] for g in range(G)]
    return packed, per


def reads(emit, q, L):
    """Read the emitted states with per-head queries q:(B,T,G,R,dk) or shared queries
    q:(B,T,R,dk). ONE batched matmul against the state matrix, then ONE index_select.
    Returns the packed (B,T,R*VT) vector (per-head queries: head-major, then query, then
    payload; shared queries: query-major) and the list of per-head (B,T,R,dv_g) VIEWS (cheap
    to look at; do not build a graph on all of them -- use reads_padded for that)."""
    B, T = emit.shape[0], emit.shape[1]
    dk = L["dk"]; G = L["G"]; VT = L["VT"]
    if len(set(L["dv"])) == 1 and (q.shape[2] if q.dim() == 4 else q.shape[3]) <= 4 and (L["dv"][0] & (L["dv"][0] - 1)) == 0:
        return reads_kernel(emit, q, L)
    if q.dim() == 4:
        R = q.shape[2]
        o = _mm(emit, q.reshape(B * T, R, dk), L).reshape(B, T, R, VT)
        per = [o[..., a:b] for (a, b) in L["vsl"]]
        packed = o.reshape(B, T, R * VT)
    elif len(set(L["dv"])) == 1:
        # per-head queries, uniform payload: block-diagonal, so ONLY the diagonal blocks -- head g's (dv, dk) state
        # against head g's R queries. Same packed order as the generic path (head, query, payload); `per` are views.
        R = q.shape[3]; dv = L["dv"][0]
        S = emit.view(B * T * G, dv, dk)                                     # column-major head states
        o = torch.bmm(S, q.reshape(B * T * G, R, dk).transpose(1, 2).to(S.dtype))   # (N*G, dv, R)
        packed = o.transpose(1, 2).reshape(B, T, G * R * dv)
        pv = packed.view(B, T, G, R, dv)
        per = [pv[:, :, g] for g in range(G)]
    else:
        R = q.shape[3]
        o = _mm(emit, q.reshape(B * T, G * R, dk), L).reshape(B, T, G * R * VT)
        packed = o.index_select(-1, _pack_index(L, R))
        off = 0; per = []
        for (a, b) in L["vsl"]:
            w = (b - a) * R
            per.append(packed[..., off:off + w].view(B, T, R, b - a)); off += w
    return packed, per


def reads_padded(emit, q, L, dvmax):
    """Per-head reads zero-padded to dvmax, (B,T,G,R*dvmax), for per-head OR shared queries --
    one matmul, one index_select, one index_copy into zeros (the compressor's input)."""
    B, T = emit.shape[0], emit.shape[1]
    dk = L["dk"]; G = L["G"]; VT = L["VT"]
    shared = q.dim() == 4
    R = q.shape[2] if shared else q.shape[3]
    o = _mm(emit, q.reshape(B * T, -1, dk), L).reshape(B, T, -1)
    src, dst = _pad_maps(L, R, dvmax, shared)
    rp = o.new_zeros(B, T, G * R * dvmax).index_copy(-1, dst, o.index_select(-1, src))
    return rp.view(B, T, G, R * dvmax)
