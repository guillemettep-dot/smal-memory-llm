"""KEYLESS clean-slot scan (the user's design): slots hold VALUES only; a token writes by a learned slot
address (softmax over N with an occupancy free-slot bias) and a gate; retrieval is the compressor's job over
the emitted value buffer. No keys, no per-token read. State per subhead = N*dv (48*16 = 768 = the delta memory's).

HARD=1  ("select the row by id"): the forward writes ONE slot, a = onehot(argmax z); straight-through backward
        a = a_hard + a_soft - sg(a_soft), so the address logits still learn. A soft address smeared every write
        over all 48 slots (entropy 5.3/5.58 bits at s1 step 2000) and the loss blew up past the trained length.
HARDG=1 ("node above 0.5 = on"): the gate is binary in the forward, gh = 1[g > 0.5], so a write REPLACES the
        slot's value (no blend); straight-through g_eff = gh + g - sg(g).

Per (b,h), per step t; state V(N,DV), o(N), zero-initialised; wa already carries the address temperature:
    buf_t = V                        (pre-write state, every G steps -> the compressor)
    z = wa_t - fr*o    a = softmax(z) [or onehot(argmax z)]    w = g_t [or 1[g_t>0.5]] * a
    V <- (1-w) V + w (x) v_t         o <- (1-w) o + w
Forward: one launch, checkpoints every C. Backward per chunk (reverse): recompute launch -> adjoint launch.
Adjoint given dV' do' (post-write adjoints) and dbuf_t (a_h / g_h = the values the forward used):
    dw = sum_e (v-V) dV' + (1-o) do'     dv = sum_n w dV'     dg = a_h.dw     da = g_h dw
    dz = a_s (da - a_s.da)    dwa = dz    dfr += -o.dz
    dV = (1-w) dV' + dbuf_t    do = (1-w) do' - fr dz"""
import torch, triton, triton.language as tl

NW_FWD, NW_REC, NW_ADJ = 1, 1, 1


@triton.jit
def _vfwd(v_ptr, wa_ptr, g_ptr, fr_ptr, ckV_ptr, cko_ptr, buf_ptr,
          T, H, C, NC, G, NB,
          N: tl.constexpr, DV: tl.constexpr, NP: tl.constexpr, DVP: tl.constexpr, HARD: tl.constexpr, HARDG: tl.constexpr):
    pid = tl.program_id(0)
    b = pid // H
    h = pid % H
    on = tl.arange(0, NP)
    oe = tl.arange(0, DVP)
    nm = on < N
    em = oe < DV
    ne = nm[:, None] & em[None, :]
    fr = tl.load(fr_ptr)
    V = tl.zeros((NP, DVP), dtype=tl.float32)
    o = tl.zeros((NP,), dtype=tl.float32)
    vbase = v_ptr + (b * T * H + h) * DV
    wbase = wa_ptr + (b * T * H + h) * N
    gbase = g_ptr + (b * T * H + h)
    ckVb = ckV_ptr + pid * NC * N * DV
    ckob = cko_ptr + pid * NC * N
    for c in range(0, NC):
        tl.store(ckVb + c * N * DV + on[:, None] * DV + oe[None, :], V, mask=ne)
        tl.store(ckob + c * N + on, o, mask=nm)
        clen = tl.minimum(C, T - c * C)
        for i in range(0, clen):
            t = c * C + i
            if t % G == 0:
                tb = t // G
                tl.store(buf_ptr + ((b * NB + tb) * H + h) * N * DV + on[:, None] * DV + oe[None, :],
                         V.to(buf_ptr.dtype.element_ty), mask=ne)
            wa = tl.load(wbase + t * H * N + on, mask=nm, other=0.0)
            g = tl.load(gbase + t * H)
            if HARDG:
                g = (g > 0.5).to(tl.float32)
            z = wa - fr * o
            z = tl.where(nm, z, float("-inf"))
            if HARD:
                am = tl.argmax(z, axis=0)
                a = (on == am).to(tl.float32)
            else:
                mz = tl.max(z, axis=0)
                ez = tl.exp(z - mz)
                a = ez / tl.sum(ez, axis=0)
            w = g * a
            omw = 1.0 - w
            vv = tl.load(vbase + t * H * DV + oe, mask=em, other=0.0)
            V = omw[:, None] * V + w[:, None] * vv[None, :]
            o = omw * o + w


@triton.jit
def _vrecompute(v_ptr, wa_ptr, g_ptr, fr_ptr, ckV_ptr, cko_ptr, scV_ptr, sco_ptr,
                T, H, C, NC, c,
                N: tl.constexpr, DV: tl.constexpr, NP: tl.constexpr, DVP: tl.constexpr, HARD: tl.constexpr, HARDG: tl.constexpr):
    pid = tl.program_id(0)
    b = pid // H
    h = pid % H
    on = tl.arange(0, NP)
    oe = tl.arange(0, DVP)
    nm = on < N
    em = oe < DV
    ne = nm[:, None] & em[None, :]
    fr = tl.load(fr_ptr)
    V = tl.load(ckV_ptr + (pid * NC + c) * N * DV + on[:, None] * DV + oe[None, :], mask=ne, other=0.0)
    o = tl.load(cko_ptr + (pid * NC + c) * N + on, mask=nm, other=0.0)
    vbase = v_ptr + (b * T * H + h) * DV
    wbase = wa_ptr + (b * T * H + h) * N
    gbase = g_ptr + (b * T * H + h)
    scVb = scV_ptr + pid * C * N * DV
    scob = sco_ptr + pid * C * N
    clen = tl.minimum(C, T - c * C)
    for i in range(0, clen):
        t = c * C + i
        tl.store(scVb + i * N * DV + on[:, None] * DV + oe[None, :], V, mask=ne)
        tl.store(scob + i * N + on, o, mask=nm)
        wa = tl.load(wbase + t * H * N + on, mask=nm, other=0.0)
        g = tl.load(gbase + t * H)
        if HARDG:
            g = (g > 0.5).to(tl.float32)
        z = wa - fr * o
        z = tl.where(nm, z, float("-inf"))
        if HARD:
            am = tl.argmax(z, axis=0)
            a = (on == am).to(tl.float32)
        else:
            mz = tl.max(z, axis=0)
            ez = tl.exp(z - mz)
            a = ez / tl.sum(ez, axis=0)
        w = g * a
        omw = 1.0 - w
        vv = tl.load(vbase + t * H * DV + oe, mask=em, other=0.0)
        V = omw[:, None] * V + w[:, None] * vv[None, :]
        o = omw * o + w


@triton.jit
def _vadjoint(v_ptr, wa_ptr, g_ptr, fr_ptr, dbuf_ptr, scV_ptr, sco_ptr, cV_ptr, co_ptr,
              dv_ptr, dwa_ptr, dg_ptr, dfr_ptr,
              T, H, C, NC, c, G, NB,
              N: tl.constexpr, DV: tl.constexpr, NP: tl.constexpr, DVP: tl.constexpr, HARD: tl.constexpr, HARDG: tl.constexpr):
    pid = tl.program_id(0)
    b = pid // H
    h = pid % H
    on = tl.arange(0, NP)
    oe = tl.arange(0, DVP)
    nm = on < N
    em = oe < DV
    ne = nm[:, None] & em[None, :]
    fr = tl.load(fr_ptr)
    dV = tl.load(cV_ptr + pid * N * DV + on[:, None] * DV + oe[None, :], mask=ne, other=0.0)
    do = tl.load(co_ptr + pid * N + on, mask=nm, other=0.0)
    dfr_vec = tl.zeros((NP,), dtype=tl.float32)
    vbase = v_ptr + (b * T * H + h) * DV
    wbase = wa_ptr + (b * T * H + h) * N
    gbase = g_ptr + (b * T * H + h)
    dvb = dv_ptr + (b * T * H + h) * DV
    dwb = dwa_ptr + (b * T * H + h) * N
    dgb = dg_ptr + (b * T * H + h)
    scVb = scV_ptr + pid * C * N * DV
    scob = sco_ptr + pid * C * N
    clen = tl.minimum(C, T - c * C)
    for j in range(0, clen):
        i = clen - 1 - j
        t = c * C + i
        V = tl.load(scVb + i * N * DV + on[:, None] * DV + oe[None, :], mask=ne, other=0.0)
        o = tl.load(scob + i * N + on, mask=nm, other=0.0)
        vv = tl.load(vbase + t * H * DV + oe, mask=em, other=0.0)
        wa = tl.load(wbase + t * H * N + on, mask=nm, other=0.0)
        g = tl.load(gbase + t * H)
        if HARDG:
            gh = (g > 0.5).to(tl.float32)              # what the forward used
        else:
            gh = g
        z = wa - fr * o
        z = tl.where(nm, z, float("-inf"))
        mz = tl.max(z, axis=0)
        ez = tl.exp(z - mz)
        a_s = ez / tl.sum(ez, axis=0)                  # soft address: carries the gradient
        if HARD:
            am = tl.argmax(z, axis=0)
            a_h = (on == am).to(tl.float32)            # hard address: what the forward wrote
        else:
            a_h = a_s
        w = gh * a_h
        omw = 1.0 - w
        dw = tl.sum((vv[None, :] - V) * dV, axis=1) + (1.0 - o) * do
        dv_t = tl.sum(w[:, None] * dV, axis=0)
        dg = tl.sum(a_h * dw, axis=0)                  # straight-through: d w / d g = a_h
        da = gh * dw
        dz = a_s * (da - tl.sum(a_s * da, axis=0))
        dfr_vec += -(o * dz)
        dVn = omw[:, None] * dV
        don = omw * do - fr * dz
        if t % G == 0:
            tb = t // G
            dVn += tl.load(dbuf_ptr + ((b * NB + tb) * H + h) * N * DV + on[:, None] * DV + oe[None, :], mask=ne, other=0.0)
        tl.store(dvb + t * H * DV + oe, dv_t, mask=em)
        tl.store(dwb + t * H * N + on, dz, mask=nm)
        tl.store(dgb + t * H, dg)
        dV = dVn
        do = don
    tl.store(cV_ptr + pid * N * DV + on[:, None] * DV + oe[None, :], dV, mask=ne)
    tl.store(co_ptr + pid * N + on, do, mask=nm)
    tl.store(dfr_ptr + pid * NC + c, tl.sum(dfr_vec, axis=0))


class _SlotScanV(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v, wa, gate, fr, N, C, G, buf_dtype=torch.float32, hard=False, hard_gate=False):
        B, T, H, DV = v.shape
        assert wa.shape == (B, T, H, N) and gate.shape == (B, T, H) and fr.shape == (1,) and G >= 1
        NC = (T + C - 1) // C
        NB = (T + G - 1) // G
        dev = v.device
        f32 = torch.float32
        ckV = torch.empty(B * H, NC, N, DV, device=dev, dtype=f32)
        cko = torch.empty(B * H, NC, N, device=dev, dtype=f32)
        buf = torch.empty(B, NB, H, N, DV, device=dev, dtype=buf_dtype)
        NP, DVP = triton.next_power_of_2(N), triton.next_power_of_2(DV)
        _vfwd[(B * H,)](v, wa, gate, fr, ckV, cko, buf, T, H, C, NC, G, NB, N=N, DV=DV, NP=NP, DVP=DVP,
                        HARD=bool(hard), HARDG=bool(hard_gate), num_warps=NW_FWD)
        ctx.save_for_backward(v, wa, gate, fr, ckV, cko)
        ctx.meta = (N, C, G, NC, NB, NP, DVP, bool(hard), bool(hard_gate))
        return buf

    @staticmethod
    def backward(ctx, dbuf):
        v, wa, gate, fr, ckV, cko = ctx.saved_tensors
        N, C, G, NC, NB, NP, DVP, hard, hard_gate = ctx.meta
        B, T, H, DV = v.shape
        dev = v.device
        f32 = torch.float32
        dbuf = dbuf.contiguous().to(f32)
        dv = torch.empty_like(v); dwa = torch.empty_like(wa); dg = torch.empty_like(gate)
        dfr_buf = torch.zeros(B * H * NC, device=dev, dtype=f32)
        scV = torch.empty(B * H, C, N, DV, device=dev, dtype=f32)
        sco = torch.empty(B * H, C, N, device=dev, dtype=f32)
        cV = torch.zeros(B * H, N, DV, device=dev, dtype=f32)
        co = torch.zeros(B * H, N, device=dev, dtype=f32)
        for c in reversed(range(NC)):
            _vrecompute[(B * H,)](v, wa, gate, fr, ckV, cko, scV, sco, T, H, C, NC, c, N=N, DV=DV, NP=NP, DVP=DVP,
                                  HARD=hard, HARDG=hard_gate, num_warps=NW_REC)
            _vadjoint[(B * H,)](v, wa, gate, fr, dbuf, scV, sco, cV, co, dv, dwa, dg, dfr_buf, T, H, C, NC, c, G, NB,
                                N=N, DV=DV, NP=NP, DVP=DVP, HARD=hard, HARDG=hard_gate, num_warps=NW_ADJ)
        return dv, dwa, dg, dfr_buf.sum().reshape(1), None, None, None, None, None, None


def slot_scan_v(v, wa, gate, fr, N, C=256, G=1, buf_dtype=torch.float32, hard=False, hard_gate=False):
    """v:(B,T,H,DV) wa:(B,T,H,N) (address temperature already applied) gate:(B,T,H) in (0,1) fr: scalar tensor.
    hard: one-slot (argmax) writes; hard_gate: binary gate (write iff gate > 0.5); both straight-through.
    Returns the pre-write value buffer (B, ceil(T/G), H, N, DV) in buf_dtype (bf16 == fp32 cast)."""
    f = lambda x: x.contiguous().to(torch.float32)
    if not v.is_cuda:                                  # CPU (evals/probes): the torch reference loop, same math
        return ref_scan_v(f(v), f(wa), f(gate), fr.to(torch.float32).reshape(1), N, G=G, hard=hard, hard_gate=hard_gate).to(buf_dtype)
    return _SlotScanV.apply(f(v), f(wa), f(gate), fr.to(torch.float32).reshape(1), N, C, G, buf_dtype, hard, hard_gate)


def ref_scan_v(v, wa, gate, fr, N, G=1, hard=False, hard_gate=False):
    B, T, H, DV = v.shape
    V = v.new_zeros(B, H, N, DV); o = v.new_zeros(B, H, N)
    bufs = []
    for t in range(T):
        if t % G == 0:
            bufs.append(V)
        z = wa[:, t] - fr * o
        a = torch.softmax(z, -1)
        if hard:
            ah = torch.nn.functional.one_hot(z.argmax(-1), N).to(v.dtype)
            a = ah + a - a.detach()                    # straight-through
        g = gate[:, t]
        if hard_gate:
            g = (g > 0.5).to(v.dtype) + g - g.detach()  # straight-through
        w = g.unsqueeze(-1) * a
        ws = w.unsqueeze(-1)
        V = (1 - ws) * V + ws * v[:, t].unsqueeze(2)
        o = (1 - w) * o + w
    return torch.stack(bufs, 1)


if __name__ == "__main__":
    import time
    torch.manual_seed(0)
    dev = "cuda"
    ok_all = True

    def rel(a, b):
        return ((a - b).abs().max() / (b.abs().max() + 1e-12)).item()

    def case(name, B, T, H, DV, N, C, G, hard, hg, tol=2e-4, grad_tol=2e-3):
        global ok_all
        gen = torch.Generator(device="cpu").manual_seed(1234)
        v = torch.randn(B, T, H, DV, generator=gen).to(dev)
        wa = (1.5 * torch.randn(B, T, H, N, generator=gen)).to(dev)
        gate = torch.sigmoid(torch.randn(B, T, H, generator=gen)).to(dev)
        fr = torch.tensor([3.0], device=dev)
        ins = [x.clone().requires_grad_() for x in (v, wa, gate, fr)]
        rins = [x.clone().requires_grad_() for x in (v, wa, gate, fr)]
        bk = slot_scan_v(*ins, N, C=C, G=G, hard=hard, hard_gate=hg); br = ref_scan_v(*rins, N, G=G, hard=hard, hard_gate=hg)
        e = rel(bk, br)
        d = torch.randn_like(br)
        gk = torch.autograd.grad((bk * d).sum(), ins); gr = torch.autograd.grad((br * d).sum(), rins)
        errs = [rel(a, b) for a, b in zip(gk, gr)]
        ok = e < tol and all(x < grad_tol for x in errs) and all(bool(torch.isfinite(a).all()) for a in gk)
        ok_all = ok_all and ok
        print("%-26s buf %.1e | grads dv %.1e dwa %.1e dgate %.1e dfr %.1e | %s" % ((name, e) + tuple(errs) + ("OK" if ok else "FAIL",)), flush=True)

    for hard, hg in ((False, False), (True, False), (True, True)):
        print("=== address %s, gate %s ===" % ("HARD" if hard else "soft", "HARD" if hg else "soft"))
        case("N48 DV16 T100 C32 G1", 2, 100, 3, 16, 48, 32, 1, hard, hg)
        case("T=17<C", 2, 17, 2, 16, 48, 32, 1, hard, hg)
        case("T=300 C256 (2 chunks)", 1, 300, 2, 16, 48, 256, 1, hard, hg)
        case("G=4", 2, 100, 2, 16, 48, 32, 4, hard, hg)
        case("N16 DV16", 3, 63, 8, 16, 16, 32, 1, hard, hg)
    torch.manual_seed(5)
    B, T, H, DV, N = 2, 90, 4, 16, 48
    v = torch.randn(B, T, H, DV, device=dev); wa = torch.randn(B, T, H, N, device=dev); gate = torch.rand(B, T, H, device=dev); fr = torch.tensor([2.0], device=dev)
    r1 = [x.clone().requires_grad_() for x in (v, wa, gate, fr)]; r2 = [x.clone().requires_grad_() for x in (v, wa, gate, fr)]
    b1 = slot_scan_v(*r1, N, C=32, hard=True, hard_gate=True); b2 = slot_scan_v(*r2, N, C=32, hard=True, hard_gate=True)
    g1 = torch.autograd.grad(b1.sum(), r1); g2 = torch.autograd.grad(b2.sum(), r2)
    det = torch.equal(b1, b2) and all(torch.equal(a, b) for a, b in zip(g1, g2))
    print("HARD+HARDG bitwise deterministic fwd+bwd: %s" % det); ok_all = ok_all and det
    B, T, H, DV, N = 2, 512, 32, 16, 48
    v = torch.randn(B, T, H, DV, device=dev, requires_grad=True); wa = torch.randn(B, T, H, N, device=dev, requires_grad=True)
    gate = torch.rand(B, T, H, device=dev, requires_grad=True); fr = torch.tensor([2.0], device=dev, requires_grad=True)
    for _ in range(2):
        slot_scan_v(v, wa, gate, fr, N, buf_dtype=torch.bfloat16, hard=True, hard_gate=True).float().sum().backward()
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(5):
        slot_scan_v(v, wa, gate, fr, N, buf_dtype=torch.bfloat16, hard=True, hard_gate=True).float().sum().backward()
    torch.cuda.synchronize()
    print("speed B2 T512 H32 N48 DV16 hard+hardg: fwd+bwd %.2f ms per layer" % ((time.time() - t0) / 5 * 1e3))
    print("SLOTKERNV_OK" if ok_all else "SLOTKERNV_FAIL")
