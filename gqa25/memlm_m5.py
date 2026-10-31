"""
Pure-memory LM: no attention, no KV cache.

Every sequence mixer is a gated delta-rule compressed memory (fla chunked kernels).
State per layer = H * dk * dv, constant in context length.
Halving dk halves the memory state while leaving every other model dimension intact,
so a half-vs-full comparison isolates memory capacity and nothing else.
"""
import math
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as _ckpt
from sparse_fixed import FixedSparse
from bsparse import make_wiring, inverse_map, bs_funnel
from kprep import prep_fused as _prep_fused_raw
_prep_fused = torch._dynamo.disable(_prep_fused_raw)   # Triton autograd.Function: an eager island, like the scan
try:
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    _HAVE_FLA = True
except Exception:                       # CPU box / no triton: fall back to the reference
    chunk_gated_delta_rule = None
    _HAVE_FLA = False


def _chunks(t, C):
    """(B,T,...) -> list of contiguous (B,C,...) chunks (ragged tail allowed). One relayout copy forward and ONE stack
    in backward; a slice per chunk instead costs a full-size zero-fill + copy + add per slice (2x on the shadow, measured)."""
    B, T = t.shape[0], t.shape[1]; n = T // C
    out = list(t[:, :n * C].reshape(B, n, C, *t.shape[2:]).transpose(0, 1).contiguous().unbind(0)) if n else []
    if n * C < T: out.append(t[:, n * C:].contiguous())
    return out


def _scan_chunk(kb, gb, bb, vb, qq, s0, L):
    """One chunk of a layer's scan + its reads. Under checkpoint the emission lives only inside this call."""
    from kcol import kslot_s, reads
    _, emit, sfin = kslot_s(kb, gb, bb, vb, None, L, s0)
    o, _ = reads(emit, qq, L)
    return o, sfin


def _shadow_chunk(kall, gall, ball, vall, qrb, aske, gq, ShF, qr, Wps, s0, Lall, NH, La, dk, dv, dt, fused, bsw=None, pi=None):
    """One chunk of the compressor's all-heads scan: emission -> probe reads -> funnel -> page slice."""
    from kcol import kslot_s, reads
    _, em, sfin = kslot_s(kall, gall, ball, vall, None, Lall, s0)
    if pi is not None:
        em = em.index_select(1, pi)                     # page every K tokens: only block-end states are read and funneled
    B_, Tb = em.shape[0], em.shape[1]; G = kall.shape[2]
    oS = em.view(B_, Tb, G, dv, dk)
    pk, per = reads(em, qrb, Lall)
    if len(set(Lall["dv"])) == 1:
        rd = pk.view(B_, Tb, pk.shape[-1] // Lall["VT"], G, dv).transpose(2, 3)
    else:
        rd = torch.stack(per, 2)
    if bsw is not None:
        return _gist_funnel2_bs(oS, rd, aske, gq, ShF, qr, bsw, NH, La, dk, dv, dt), sfin
    return _gist_funnel2(oS, rd, aske, gq, ShF, qr, Wps, NH, La, dk, dv, dt, cm=True, fused=fused), sfin


def _krep_fn(kr, br, tau):
    Ts = kr.shape[1]
    Gm = torch.einsum("bthi,bshi->bhts", kr, kr)
    Wp = br.transpose(1, 2)[:, :, :, None] * br.transpose(1, 2)[:, :, None, :]
    near = torch.ones(Ts, Ts, device=kr.device, dtype=torch.bool).triu(4)
    exc = (Gm.abs() - tau).clamp(min=0) * near
    return (exc * Wp).sum() / (exc > 0).sum().clamp(min=1)


class _FoldSG(torch.autograd.Function):
    """Aw = einsum('bts,btsz->btz', w, Sbuf[:, :, :S]).  The stream buffer is
    stop-grad and APPEND-ONLY: later layers write only slots >= S, so the values
    this fold read still exist unchanged at backward time. save_for_backward would
    trip the storage version counter on those appends; a plain ctx reference is
    correct here and avoids cloning the growing prefix every layer."""
    @staticmethod
    def forward(ctx, w, Sbuf, S):
        ctx.sbuf, ctx.S = Sbuf, S
        with torch.no_grad():
            return torch.einsum('bts,btsz->btz', w, Sbuf[:, :, :S])

    @staticmethod
    def backward(ctx, dAw):
        dw = torch.einsum('btz,btsz->bts', dAw, ctx.sbuf[:, :, :ctx.S])
        return dw, None, None


def _QSCALE(dk):
    """fla scales the query by dk**-0.5 before reading the state. Verified against the
    kernel directly (beta=0, g=0, known initial_state) for dk/dv in {32/48, 64/64, 16/32}:
    raw contraction is off by rel 0.75-0.88, with the scale it matches to <1e-3. Anything
    reading an fla state -- the reference recurrence AND the cross-layer bus -- must apply
    it, or it reads a state that is sqrt(dk) too large relative to the local path."""
    return dk ** -0.5


def _delta_ref(q, k, v, g, beta, state=None, want_final=False):
    """Pure-torch gated delta rule. Slow (python loop over T) but exact and portable,
    so the model is testable without CUDA and the fla kernel has something to be
    checked against. State layout (B,H,dk,dv), matching fla; o_t reads the state
    AFTER token t has been written, per S_t = S_{t-1}(aI - b k k^T) + b v k^T, o=S q."""
    B, T, H, dk = q.shape
    dv = v.shape[-1]
    S = (torch.zeros(B, H, dk, dv, dtype=torch.float32, device=q.device)
         if state is None else state.float().clone())
    a = torch.exp(g.float())
    out = []
    for t in range(T):
        kt, vt, qt = k[:, t].float(), v[:, t].float(), q[:, t].float()
        at = a[:, t].unsqueeze(-1).unsqueeze(-1)
        bt = beta[:, t].float().unsqueeze(-1).unsqueeze(-1)
        kS = torch.einsum("bhk,bhkv->bhv", kt, S)
        # fla applies the decay to the WHOLE delta operator, not just the identity term:
        #   S_t = a_t * S_{t-1} (I - b_t k k^T) + b_t v k^T
        # not the S_{t-1}(a I - b k k^T) + b v k^T form in the project notes. Verified
        # against the kernel: this variant matches to rel 1e-3, the other is off by 8.5%.
        S = at * S + bt * kt.unsqueeze(-1) * (vt - at.squeeze(-1) * kS).unsqueeze(-2)
        out.append(_QSCALE(dk) * torch.einsum("bhk,bhkv->bhv", qt, S))
    o = torch.stack(out, dim=1).to(q.dtype)
    return o, (S if want_final else None)


def _gist_funnel3r(rp, aske, ShF, qr3, Wps, Whub, dk, dvh, dvmax, dt):
    """Hub-driven funnel over PADDED per-head reads (kslot). rp:(B,Tb,NH,NQ*dvmax) reads of
    every cabinet by the hub's NQ queries, zero-padded to the widest payload; the hub is the
    (NH+1)-th group, read by the same queries; the hub also enters densely via Whub."""
    B, Tb, NH, RW = rp.shape
    aw = aske.shape[-1]
    NQ = qr3.shape[2]
    W0 = Wps[0]
    Wr, Wa = W0[:, :RW], W0[:, RW:]
    h = torch.einsum('bthi,hij->bthj', F.gelu(rp.to(dt)), Wr[:NH].to(dt)).float()
    Sm = ShF.view(B, Tb, dk, dvh)
    hrd = torch.einsum('btqk,btkv->btqv', qr3.to(dt), Sm.to(dt)).reshape(B, Tb, NQ * dvh)
    hrd = F.pad(hrd, (0, RW - NQ * dvh))
    hh = torch.einsum('bti,ij->btj', F.gelu(hrd), Wr[NH].to(dt)).float()
    h = torch.cat([h, hh.unsqueeze(2)], dim=2)
    ga = F.gelu(aske.reshape(B, Tb, aw).to(dt))
    h = h + torch.einsum('bti,hij->bthj', ga, Wa.to(dt)).float()
    h = h + Whub(F.gelu(ShF.to(dt))).float().unsqueeze(2)
    for j in range(1, len(Wps)):
        h = torch.einsum('bthi,hij->bthj', F.gelu(h).to(dt), Wps[j].to(dt))
    return h.reshape(B, Tb, -1)


def _gist_funnel3(oS, aske, ShF, qr3, Wps, Whub, NH, La, dk, dv, dt):
    """Hub-driven funnel (gist3). The hub emits NQ context queries; those queries read
    every cabinet's emitted state and ONLY the reads go up -- no raw-state path. The hub
    also enters densely through its own layer, added to every group's hidden.

    Motivation, measured: under the additive arrangement the model closed both conditional
    channels (ask gate 0.006, probe gate -0.008) and kept the wide unconditional one, so a
    conditional path only gets used if it is the ONLY path. Per head the input is
    La*NQ*dv + aw, well under the raw-state arrangement's La*dk*dv + aw.

    oS:(B,Tb,G,dk,dv) emitted stores, aske:(B,Tb,1,aw), ShF:(B,Tb,dk*dv) hub state,
    qr3:(B,Tb,NQ,dk) hub queries, Whub: Linear(dk*dv -> w1)."""
    B, Tb = oS.shape[0], oS.shape[1]
    aw = aske.shape[-1]
    NQ = qr3.shape[2]
    W0 = Wps[0]
    RW = La * NQ * dv
    Wr, Wa = W0[:, :RW], W0[:, RW:]
    rd = torch.einsum('btqk,btgkv->btgqv', qr3.to(dt), oS.to(dt))          # (B,Tb,G,NQ,dv)
    rr = F.gelu(rd.reshape(B, Tb, NH, RW))
    h = torch.einsum('bthi,hij->bthj', rr, Wr[:NH].to(dt)).float()
    # the hub as its own group: its state read by the same queries, zero-padded to the width
    Sm = ShF.view(B, Tb, 1, dk, dv)
    hrd = torch.einsum('btqk,bthkv->bthqv', qr3.to(dt), Sm).reshape(B, Tb, 1, NQ * dv)
    hh = torch.einsum('bti,ij->btj', F.gelu(hrd.reshape(B, Tb, NQ * dv)),
                      Wr[NH, :NQ * dv].to(dt)).float()
    h = torch.cat([h, hh.unsqueeze(2)], dim=2)                             # (B,Tb,NH+1,w1)
    ga = F.gelu(aske.reshape(B, Tb, aw).to(dt))
    h = h + torch.einsum('bti,hij->bthj', ga, Wa.to(dt)).float()
    h = h + Whub(F.gelu(ShF.to(dt))).float().unsqueeze(2)                  # DENSE hub path
    for j in range(1, len(Wps)):
        h = torch.einsum('bthi,hij->bthj', F.gelu(h).to(dt), Wps[j].to(dt))
    return h.reshape(B, Tb, -1)


class BlockSparsePage(nn.Module):
    """page (all groups -> layer): block-sparse fixed-random linear, no activation (the gelu follows outside)."""
    def __init__(self, n_in, n_out, nb, seed):
        super().__init__()
        assert n_in % 32 == 0 and n_out % 32 == 0
        idx = make_wiring(1, n_in, n_out, nb, seed)
        self.register_buffer("idx", idx); self.register_buffer("inv", inverse_map(idx, n_in))
        bd = 1.0 / (idx.shape[2] * 32) ** 0.5
        self.w = nn.Parameter(torch.empty(1, n_out // 32, idx.shape[2], 32, 32).uniform_(-bd, bd))   # effective nb
        self.n_in, self.n_out = n_in, n_out
    def forward(self, x):                       # (..., n_in) -> (..., n_out), computed in fp32 out, returned in x.dtype
        sh = x.shape[:-1]
        out = bs_funnel(x.reshape(-1, 1, self.n_in).to(torch.bfloat16), None, None, self.w.to(torch.bfloat16), self.idx, self.inv, gelu=False)
        return out.reshape(*sh, self.n_out).to(x.dtype)


def _gist_funnel2_bs(oS, rd, aske, gq, ShF, qr, bsw, NH, La, dk, dv, dt):
    """gist2 funnel with BLOCK-sparse stages 1 and 2 (bsparse kernels, GELU fused on load, nothing concatenated):
    per head group [state | reads | ask] -> w1 (fp32 out) ; hub group dense ; gelu -> block-sparse w2 -> gelu -> dense w3."""
    w1, idx1, inv1, hub, w2, idx2, inv2, w3, awp = bsw
    B, Tb = oS.shape[0], oS.shape[1]; N = B * Tb
    aw = aske.shape[-1]; W = La * dk * dv; RW = La * 2 * dv
    bf = torch.bfloat16
    xs = oS.reshape(N, NH, W).to(bf)                                       # raw state, a view of the emission
    xr = (gq * rd.to(bf)).reshape(N, NH, RW)                                # probe reads (pre-gelu)
    xa = aske.reshape(N, aw).to(bf)
    if awp > aw: xa = F.pad(xa, (0, awp - aw))
    h = bs_funnel(xs, xr, xa, w1.to(bf), idx1, inv1, gelu=True, out_groups=NH + 1)   # (N, NH+1, w1) fp32, hub row left for below
    Sm = ShF.view(B, Tb, 1, dk, dv)
    hrd = torch.einsum('btqk,bthkv->bthqv', qr.to(bf), Sm).reshape(N, 2 * dv)
    xh = torch.cat([F.gelu(ShF.reshape(N, dk * dv).to(bf)), F.gelu((gq * hrd).to(bf)), F.gelu(xa)], -1)
    hh = _bmm_f32(xh.unsqueeze(0), hub.to(bf).unsqueeze(0)).squeeze(0)     # (N, w1) fp32
    h[:, NH] = hh                                                           # the hub group, in place (no cat, no strided-grad copy)
    h2 = bs_funnel(h, None, None, w2.to(bf), idx2, inv2, gelu=True)         # fp32 in, gelu on load -> (N, NH+1, w2) fp32
    h3 = torch.bmm(F.gelu(h2).to(bf).transpose(0, 1), w3.to(bf))           # (NH+1, N, 64)
    return h3.transpose(0, 1).reshape(B, Tb, -1)


class _BmmF32(torch.autograd.Function):
    """bf16 batched GEMM with fp32 accumulation AND fp32 output (torch.bmm out_dtype has no autograd derivative).
    Backward is the ordinary bf16 GEMM pair, exactly what autograd did for the bf16 einsums it replaces."""
    @staticmethod
    def forward(ctx, X, W):
        ctx.save_for_backward(X, W)
        return torch.bmm(X, W, out_dtype=torch.float32)
    @staticmethod
    def backward(ctx, g):
        X, W = ctx.saved_tensors; gb = g.to(X.dtype)
        return torch.bmm(gb, W.transpose(1, 2)), torch.bmm(X.transpose(1, 2), gb)


def _bmm_f32(X, W):
    return _BmmF32.apply(X, W)


def _gist_funnel2_cat(oS, rd, aske, gq, ShF, qr, Wps, NH, La, dk, dv, dt, cm):
    """The wide first stage as ONE GEMM per group over the concatenated sections [gelu(state) | gelu(reads) | gelu(ask)],
    bf16 inputs, fp32 accumulation AND fp32 output (no bf16 rounding of partial sums, unlike the summed-sections
    path), group-major (NH+1, N, .) so every stage is a contiguous bmm. Same math as _gist_funnel2(fused=False);
    equal to it within bf16 rounding of the partials (measured), and closer to an fp32 reference."""
    B, Tb = oS.shape[0], oS.shape[1]; N = B * Tb
    aw = aske.shape[-1]; W = La * dk * dv; RW = La * 2 * dv
    W0 = Wps[0]; w1 = W0.shape[-1]
    Ws = W0[:, :W]; Wa = W0[:, W:W + aw]; Wr = W0[:, W + aw:]
    Wsh = Ws[:NH].reshape(NH, La, dk, dv, w1).transpose(2, 3).reshape(NH, W, w1) if cm else Ws[:NH]
    Wcat = torch.cat([Wsh.to(dt), Wr[:NH].to(dt), Wa[:NH].to(dt)], 1)                       # (NH, W+RW+aw, w1)
    ga = F.gelu(aske.reshape(N, aw).to(dt))                                                 # shared by every group
    X = torch.cat([F.gelu(oS.reshape(N, NH, W).transpose(0, 1).to(dt)),                     # (NH, N, W)  contiguous
                   F.gelu((gq * rd.to(dt)).reshape(N, NH, RW).transpose(0, 1)),             # (NH, N, RW)
                   ga.unsqueeze(0).expand(NH, N, aw)], -1)                                  # (NH, N, K) bf16
    h = _bmm_f32(X, Wcat)                                                                    # (NH, N, w1) fp32
    # hub group: its store (first dk*dv state columns), its reads (first 2*dv read columns), the shared ask
    Sm = ShF.view(B, Tb, 1, dk, dv)
    hrd = torch.einsum('btqk,bthkv->bthqv', qr.to(dt), Sm).reshape(N, 2 * dv)
    Xh = torch.cat([F.gelu(ShF.reshape(N, dk * dv).to(dt)), F.gelu((gq * hrd).to(dt)), ga], -1)
    Wh = torch.cat([Ws[NH, :dk * dv].to(dt), Wr[NH, :2 * dv].to(dt), Wa[NH].to(dt)], 0)
    hh = _bmm_f32(Xh.unsqueeze(0), Wh.unsqueeze(0)).squeeze(0)                               # (N, w1) fp32
    h = torch.cat([h, hh.unsqueeze(0)], 0)                                                   # (NH+1, N, w1) fp32
    for j in range(1, len(Wps)):
        h = torch.bmm(F.gelu(h).to(dt), Wps[j].to(dt))                                       # (NH+1, N, w_j) bf16
    return h.transpose(0, 1).reshape(B, Tb, -1)


def _gist_funnel2(oS, rd, aske, gq, ShF, qr, Wps, NH, La, dk, dv, dt, cm=False, fused=True, legacy=False):
    if not fused and not legacy:
        return _gist_funnel2_cat(oS, rd, aske, gq, ShF, qr, Wps, NH, La, dk, dv, dt, cm)
    """gist2 funnel: one group per LOGICAL HEAD (its La lanes joined) plus the hub.

    oS:(B,Tb,G,dk,dv) RAW stores, rd:(B,Tb,G,2,dv) probe reads, aske:(B,Tb,1,aw),
    ShF:(B,Tb,dk*dv) hub state, qr:(B,Tb,2,dk). Per head the first stage sees
    [La*dk*dv raw | aw ask | La*2*dv reads]; the hub is the same width, zero-padded.
    Written WITHOUT materialising that concatenation: the stage is linear, so the
    three sections are three GEMMs summed, and the ask section (identical for every
    head) is a single GEMM against all heads' ask columns at once."""
    B, Tb = oS.shape[0], oS.shape[1]
    aw = aske.shape[-1]
    W = La * dk * dv
    RW = La * 2 * dv
    W0 = Wps[0]                                    # (NH+1, W+aw+RW, w1)
    Ws = W0[:, :W]                                 # state columns
    # cm=True: oS arrives (La, dv, dk) column-major instead of (La, dk, dv); permute the heads'
    # weight rows to match (a 25 MB gather) rather than copying the 3 GB activation. Same sums.
    Wsh = Ws[:NH].reshape(NH, La, dk, dv, W0.shape[-1]).transpose(2, 3).reshape(NH, W, W0.shape[-1]) if cm else Ws[:NH]
    Wa = W0[:, W:W + aw]                           # ask columns
    Wr = W0[:, W + aw:]                            # read columns
    w1 = W0.shape[-1]
    # heads: state section (B,Tb,NH,W) and reads (B,Tb,NH,RW)
    from gelumm import gelu_mm
    rr = F.gelu((gq * rd.to(dt)).reshape(B, Tb, NH, RW))
    # state section: gelu fused into the GEMM's operand load (reads the raw emission once,
    # writes no gelu'd copy); the three sections are summed in fp32
    if fused:
        h = gelu_mm(oS.reshape(B * Tb, NH, W).to(dt), Wsh.to(dt).contiguous()).view(B, Tb, NH, -1)
    else:   # plain cuBLAS: the gelu'd copy is small per chunk and any width is fine (the fused kernel wants N <= 128)
        h = torch.einsum('bthi,hij->bthj', F.gelu(oS.reshape(B, Tb, NH, W).to(dt)), Wsh.to(dt)).float()
    h = h \
        + torch.einsum('bthi,hij->bthj', rr, Wr[:NH].to(dt)).float()
    # hub: its own store (dk*dv wide -> first dk*dv state columns) and reads (2*dv)
    Sm = ShF.view(B, Tb, 1, dk, dv)
    hrd = torch.einsum('btqk,bthkv->bthqv', qr.to(dt), Sm).reshape(B, Tb, 1, 2 * dv)
    hs = F.gelu(ShF.to(dt))
    hr = F.gelu((gq * hrd).to(dt)).reshape(B, Tb, 2 * dv)
    hh = torch.einsum('bti,ij->btj', hs, Ws[NH, :dk * dv].to(dt)).float() \
        + torch.einsum('bti,ij->btj', hr, Wr[NH, :2 * dv].to(dt)).float()
    h = torch.cat([h, hh.unsqueeze(2)], dim=2)     # (B,Tb,NH+1,w1) fp32 -- small
    # ask: identical for all groups -> one GEMM against every group's ask columns
    ga = F.gelu(aske.reshape(B, Tb, aw).to(dt))
    h = h + torch.einsum('bti,hij->bthj', ga, Wa.to(dt)).float()
    for j in range(1, len(Wps)):
        h = F.gelu(h).to(dt)
        h = torch.einsum('bthi,hij->bthj', h, Wps[j].to(dt))
    return h.reshape(B, Tb, -1)


def _gist_funnel(oS, rd, aske, gq, Wa, Wr, Wps, G, sw, dv, dt):
    """Pairwise sections + funnel for the fused gist path. Pure function of tensors
    (rebound to a torch.compile'd version by the trainer). oS:(B,Tb,G,dk,dv/2) bf16,
    rd:(B,Tb,G,2,dv) bf16, aske:(B,Tb,1,500), Wa:(G,250,2), Wr:(G,dv,2), Wps: funnel mats."""
    B, Tb = oS.shape[0], oS.shape[1]
    pa = torch.einsum('btpz,gpz->btgp', aske.reshape(B, Tb, 250, 2).to(dt), Wa)
    pr = torch.einsum('btgpz,gpz->btgp', (gq * rd.to(dt)).reshape(B, Tb, G, dv, 2), Wr)
    h = F.gelu(torch.cat([oS.reshape(B, Tb, G, sw).to(dt), pa, pr], dim=-1))
    for j, Wp in enumerate(Wps):
        h = torch.einsum('bthi,hij->bthj', h, Wp.to(h.dtype))
        if j < len(Wps) - 1:
            h = F.gelu(h)
    return h.reshape(B, Tb, -1)


def _gist_inject(ft6, w_h, w_o):
    """Injection trunk: b16_o(gelu(b16_h(ft6))) as a pure function (bias-free linears)."""
    return F.linear(F.gelu(F.linear(ft6, w_h)), w_o)


def _scan(c, q, k, v, g, beta, state, want_final):
    """Dispatch one delta-rule scan to fla or to the reference implementation."""
    if _HAVE_FLA and not getattr(c, "ref", 0):
        out = chunk_gated_delta_rule(q=q, k=k, v=v, g=g, beta=beta,
                                     initial_state=state, output_final_state=want_final)
        if isinstance(out, tuple):
            return out[0], (out[1] if want_final else None)
        return out, None
    return _delta_ref(q, k, v, g, beta, state, want_final)


class Cfg:
    def __init__(self, vocab=50257, d_model=768, layers=12, heads=8,
                 dk=48, dv=96, d_ff=2048, conv=4, mixer="mem",
                 moe=0, n_exp=8, top_k=2, moe_u=0.001, nlk=0, dphi=1024, cap=1.35,
                 gbias=0, hl_min=16.0, hl_max=8192.0, bfloor=0.0, beta_u=0.001, qpaths=1,
                 bus=0, chunk=256, ref=0, bus_m=8, bus_slots=256, bus_nslot=1,
                 bus_r=1, wkey=0, wkfix=-1.0, wfreeze=0, wpreserve=0, wortho=0,
                 gtop=10, gidx=64, gread=4, kcenter=0, kdecor=0.0, krep=0.0, krep_tau=0.85, pre=0, vcenter=0,
                 gistr=64, gisth=1024, gistf="", gistph=0, ginjall=0, vwhiten=0, ginorm=0,
                 wmoe=0, gistk=1, gistpair=0, qphi=0, gistfuse=0, lanes=1, gist2=0, gist3=0, gist3_nq=8,
                 kslot=0, dv_plan="", slot_topk=0, slot_gtie=1, gist_sparse=0, gist_seed=20260905, kchunk=0, kchunk_layers=-1, gist_bs=0, gist_nb=5, kprep=0, beta_head=0, decay_head=0, shadow_shared=0, emit_ckpt=0, wbud=0.0, wbud_rho=0.3, wbud_count=0, wbud_tau=0.05):
        self.vocab, self.d_model, self.layers, self.heads = vocab, d_model, layers, heads
        self.dk, self.dv, self.d_ff, self.conv = dk, dv, d_ff, conv
        self.mixer = mixer          # "mem" = gated delta memory, "attn" = full causal attention
        self.moe, self.n_exp, self.top_k, self.moe_u = moe, n_exp, top_k, moe_u
        self.cap = cap
        self.nlk, self.dphi = nlk, dphi   # nonlinear shared key/query encoder (MQAR-style phi)
        self.gbias, self.hl_min, self.hl_max = gbias, hl_min, hl_max
        self.bfloor, self.beta_u = bfloor, beta_u   # write-gate revival floor
        self.qpaths = qpaths        # independent read directions per head
        self.bus, self.chunk = bus, chunk   # 0=off 1=linear query bus 2=full-state compressor
        self.bus_m, self.bus_slots, self.bus_nslot = bus_m, bus_slots, bus_nslot
        self.bus_r = bus_r          # read queries per token into the selected state
        self.wkey = wkey            # 1 = state-aware key placement (see MemMixer)
        self.wkfix = wkfix          # >=0: FIXED whitening blend (no learnable alpha -> no new opt params; mid-run-resume safe)
        self.wfreeze = wfreeze      # >0 = write to only top-k heads, freeze the rest
        self.wpreserve = wpreserve  # 1 = value sees what it is about to overwrite
        self.wortho = wortho        # >0 = project write key off r occupied directions
        self.gtop, self.gidx, self.gread = gtop, gidx, gread   # bus=4 global state
        self.kcenter = kcenter      # 1 = subtract running mean from keys before norm
        self.pre = pre              # plain FFN blocks BEFORE the first memory write
        self.vcenter = vcenter      # 1 = subtract running mean from VALUES
        self.gistr, self.gisth, self.gistf, self.gistph, self.ginjall = gistr, gisth, gistf, gistph, ginjall
        self.ginorm = ginorm   # compressor widths (bus=14/16)
        self.vwhiten = vwhiten      # 1 = ZCA-whiten VALUES (running cov)
        self.kdecor = kdecor        # weight on the key-decorrelation auxiliary loss
        self.krep, self.krep_tau = krep, krep_tau  # write-time key-repulsion aux loss (collision fix)
        self.wmoe = wmoe            # 1 = expert-conditional WRITES into the shared state
        self.gistk = gistk          # refresh the state gist every K tokens (1 = per token)
        self.gistpair = gistpair    # 1 = pairwise 2:1 front stage before the dense funnel
        self.qphi = qphi            # 1 = read queries via the key encoder: normalize(A.phi(x))
        self.gistfuse = gistfuse    # 1 = fused delta+pairwise gist emission (kpair.py)
        self.gist2 = gist2          # 1 = raw-state emission + per-LOGICAL-HEAD joint funnel
        self.gist3 = gist3          # 1 = hub-driven compressor: reads only, no raw-state path
        self.gist3_nq = gist3_nq    # context queries per cabinet under gist3
        self.kslot = kslot          # 1 = per-slot gates + payload plan on the kslot scan (main + shadow)
        self.dv_plan = dv_plan      # e.g. "6x24/16/16/16,3x48/32/32/16,3x96/64/48/32": per-layer payload widths
        self.slot_topk = slot_topk  # >0: each token writes to exactly this many slots per layer (straight-through)
        self.slot_gtie = slot_gtie  # 1 = an unwritten slot does not decay either (frozen)
        self.gist_sparse = int(gist_sparse)   # >0: compressor stages are fixed-random-sparse with this fan-in (last stage dense)
        self.gist_seed = int(gist_seed)       # wiring seed; the wiring itself is saved in the state_dict
        self.kchunk = int(kchunk)             # >0: scans run in chunks of this length with per-chunk recompute (no O(T x state) memory)
        self.kchunk_layers = int(kchunk_layers)   # chunk length for the per-layer scans: -1 = kchunk, 0 = whole sequence (shadow-only chunking)
        self.gist_bs = int(gist_bs)           # 1: compressor stages 1-2 and the page are BLOCK-sparse fixed-random (bsparse kernels)
        self.emit_ckpt = int(emit_ckpt)       # 1: a layer's scan emission is recomputed in backward instead of saved (96 MB/layer at dk 48, B2; the scan forward costs ~0.6 ms)
        self.wbud, self.wbud_rho = float(wbud), float(wbud_rho)   # >0: soft WRITE BUDGET -- a layer's mean gate activity above rho is penalised (weight applied by the trainer)
        self.wbud_count, self.wbud_tau = int(wbud_count), float(wbud_tau)   # count mode: penalise the FRACTION OF COLUMNS WRITTEN (beta > tau, straight-through) above rho
        self.gist_nb = int(gist_nb)           # 32-wide input blocks per 32-wide output block (gist_nb*32 inputs per unit)
        self.kprep = int(kprep)               # 1: gates + lane mix (default 0: the fused kernel measured slower on the H100) + normalisations + read queries in one fused kernel (kprep.py)
        self.beta_head = int(beta_head)       # 1: write strength per HEAD (one beta per head per token); decay stays per slot
        self.decay_head = int(decay_head)     # 1: log-decay per HEAD (one per head per token)
        self.shadow_shared = int(shadow_shared)   # 1: one shared shadow projection set for the compressor (not 12 per-layer copies)
        self.lanes = lanes          # >1: lane groups with learnable dv-merge gates
        self.ref = ref              # 1 = pure-torch recurrence (CPU / parity tests)

    @property
    def plan(self):
        """Per-layer payload widths under kslot, from dv_plan; None otherwise."""
        if not int(getattr(self, "kslot", 0)):
            return None
        out = []
        for part in str(self.dv_plan).split(","):
            part = part.strip()
            if not part:
                continue
            rep_, widths = part.split("x") if "x" in part else ("1", part)
            ws = [int(w) for w in widths.split("/")]
            assert len(ws) == self.heads, "dv_plan group %r must list %d heads" % (part, self.heads)
            out += [ws] * int(rep_)
        assert len(out) == self.layers, "dv_plan covers %d layers, model has %d" % (len(out), self.layers)
        return out

    @property
    def state_per_layer(self):
        p = self.plan
        if p is not None:
            return sum(self.dk * sum(ws) for ws in p) // len(p)
        return self.heads * self.dk * self.dv

    def __repr__(self):
        return (f"d_model{self.d_model} L{self.layers} H{self.heads} dk{self.dk} dv{self.dv} "
                f"| state/layer={self.state_per_layer} total_state={self.state_per_layer*self.layers}")


def _cap_dispatch(fe, tok, wg, E, C, dtype):
    """Static-shape capacity dispatch shared by ExpertProj and MoE (the Ampere path).

    Every (token, slot) pair is ranked inside its expert; the first C pairs of an expert get
    a row in a fixed [E*C] buffer, the rest are DROPPED into a dummy row E*C that no matmul
    ever reads. No boolean indexing anywhere, so the shapes are static (one compiled graph,
    no host sync) -- the earlier form used `tok[keep]`, which is a hidden nonzero() per layer.
    Returns buf (E*C,) token per row, gate (E*C,) gating weight per row (0 for pads),
    slot_c (M,) row of each pair (E*C if dropped), keep (M,) bool, counts (E,) long."""
    M = fe.shape[0]
    order = torch.argsort(fe, stable=True)
    rank = torch.empty_like(order)
    rank[order] = torch.arange(M, device=fe.device)
    counts = torch.zeros(E, dtype=torch.long, device=fe.device).scatter_add_(0, fe, torch.ones_like(fe))
    offs = torch.cat([counts.new_zeros(1), counts.cumsum(0)[:-1]])
    pos = rank - offs[fe]
    keep = pos < C
    slot_c = torch.where(keep, fe * C + pos, torch.full_like(pos, E * C))
    buf = torch.zeros(E * C + 1, dtype=torch.long, device=fe.device)
    buf[slot_c] = tok
    gate = torch.zeros(E * C + 1, device=fe.device, dtype=dtype)
    gate[slot_c] = (wg * keep.to(wg.dtype)).to(dtype)       # dropped pairs land in the dummy row
    return buf[:E * C], gate[:E * C], slot_c, keep, counts


def _cap_gather_back(ye, slot_c, tok, N):
    """Scatter the per-row expert outputs back to tokens: dropped pairs read the zero dummy row."""
    ye_pad = torch.cat([ye, ye.new_zeros(1, ye.shape[1])], 0)
    out = torch.zeros(N, ye.shape[1], device=ye.device, dtype=ye.dtype)
    out.index_add_(0, tok, ye_pad.index_select(0, slot_c))
    return out


class ExpertProj(nn.Module):
    # A D->O projection whose WEIGHTS are picked per token by the block router:
    #     out[t] = sum_j gate[t,j] * (x[t] @ W[expert[t,j]])
    # Zero-drop: every (token, slot) pair reaches its expert, no capacity factor.
    # Rows are sorted by expert so this is ONE grouped matmul, and the delta kernel
    # downstream never learns that the weights varied per token.
    def __init__(self, E, D, O, cap=1.35):
        super().__init__()
        self.E, self.D, self.O = E, D, O
        self.cap = cap            # capacity factor of the batched (Ampere) path; grouped_mm / loop are zero-drop
        self.use_loop = False     # True = per-expert loop (zero-drop reference)
        self.drop = None
        # torch._grouped_mm needs 16-BYTE-aligned strides; in bf16 that means the last
        # dim must be a multiple of 8. The real config wants O = H*dv + H = 132, which
        # is NOT ("strides should be multiple of 16 bytes"), so pad the bank and slice
        # the output back. The pad columns are dead weight (<0.1% of params).
        self.Opad = (O + 7) // 8 * 8
        self.w = nn.Parameter(torch.empty(E, D, self.Opad))
        nn.init.normal_(self.w, std=D ** -0.5)

    def forward(self, x, topi, topv):
        N = x.shape[0]
        k = topi.shape[1]
        fe = topi.reshape(-1)
        tok = torch.arange(N, device=x.device).repeat_interleave(k)
        wg = topv.reshape(-1)
        if _HAS_GROUPED or self.use_loop:
            # zero-drop: one grouped matmul (Hopper) or a per-expert loop (reference)
            order = torch.argsort(fe)
            fe_s, tok_s, wg_s = fe[order], tok[order], wg[order]
            counts = torch.zeros(self.E, dtype=torch.long, device=fe_s.device).scatter_add_(0, fe_s, torch.ones_like(fe_s))  # static-shape bincount (compile-clean)
            xs = x.index_select(0, tok_s)
            if _HAS_GROUPED and not self.use_loop:
                ys = torch._grouped_mm(xs.to(_GDT), self.w.to(_GDT),
                                       offs=counts.cumsum(0).to(torch.int32))
            else:
                ys = torch.empty(xs.shape[0], self.Opad, device=x.device, dtype=xs.dtype)
                st = 0
                for e, cnt in enumerate(counts.tolist()):
                    if cnt:
                        ys[st:st + cnt] = xs[st:st + cnt] @ self.w[e].to(xs.dtype)
                    st += cnt
            ys = ys * wg_s.unsqueeze(-1).to(ys.dtype)
            out = torch.zeros(N, self.Opad, device=x.device, dtype=ys.dtype)
            out.index_add_(0, tok_s, ys)
            self.drop = torch.zeros((), device=x.device)
            return out[:, :self.O]
        # Ampere: twelve separate small GEMMs run at <1 TFLOPS on a 28-SM card (launch- and
        # tail-bound), one padded bmm over [E, C, D] runs the same work 8x faster. Static
        # shapes throughout; pairs past the capacity are dropped (reported in self.drop).
        C = max(1, int(self.cap * fe.shape[0] / self.E))
        buf, gate, slot_c, keep, _ = _cap_dispatch(fe, tok, wg, self.E, C, x.dtype)
        xe = x.index_select(0, buf).view(self.E, C, x.shape[1])
        ye = torch.bmm(xe, self.w.to(x.dtype)).view(self.E * C, self.Opad) * gate.unsqueeze(-1)
        out = _cap_gather_back(ye, slot_c, tok, N)
        self.drop = (1.0 - keep.float().mean()).detach()
        return out[:, :self.O]


class MemMixer(nn.Module):
    """Gated delta-rule memory as the sequence mixer."""

    def __init__(self, c: Cfg, dvs=None):
        super().__init__()
        self.c = c
        H, dk, dv = c.heads, c.dk, c.dv
        self.kslot = int(getattr(c, "kslot", 0))
        if self.kslot:
            assert dvs is not None and len(dvs) == H, "kslot needs this layer's payload plan"
            assert not getattr(c, "wfreeze", 0), "kslot replaces wfreeze"
            self.dvs = [int(x) for x in dvs]
            self.VT = sum(self.dvs)
            self._L = None
        _VO = self.VT if self.kslot else H * dv        # packed value width
        # COLUMN GATES: one write strength and one decay per PAYLOAD dimension, so the write stays
        # exactly along k. Row gates (per key coordinate) put 49% of the write off-address.
        _BO = (H if int(getattr(c, "decay_head", 0)) else self.VT) if self.kslot else H   # decay-gate width: per slot, or per head (decay_head)
        _BOb = H if (self.kslot and int(getattr(c, "beta_head", 0))) else _BO   # write-strength width: per head under beta_head
        self._VO, self._BO, self._BOb = _VO, _BO, _BOb
        self.nlk = bool(getattr(c, "nlk", 0))
        if self.nlk:
            # shared Siamese nonlinear encoder: lets >d_model distinct keys stay separable
            # (the component that broke the width wall on MQAR). State size is UNCHANGED.
            self.phi = nn.Sequential(nn.Linear(c.d_model, c.dphi, bias=False),
                                     nn.GELU(),
                                     nn.Linear(c.dphi, H * dk, bias=False))
        else:
            self.wq = nn.Linear(c.d_model, H * dk, bias=False)
            self.wk = nn.Linear(c.d_model, H * dk, bias=False)
        self.wv = nn.Linear(c.d_model, _VO, bias=False)
        self.wb = nn.Linear(c.d_model, _BOb, bias=False)    # write strength (beta): per head, or per SLOT under kslot
        self.lanes = int(getattr(c, "lanes", 1))
        if self.lanes > 1:
            assert H % self.lanes == 0, "heads must be divisible by lanes"
            _ng = H // self.lanes
            # per logical head: lanes//2 pair gates + 1 quad gate (init 0 -> sigmoid .5)
            self.lane_gp = nn.Parameter(torch.zeros(_ng, self.lanes // 2))
            self.lane_gq = nn.Parameter(torch.zeros(_ng))
        self.wmoe = int(getattr(c, "wmoe", 0))
        if self.wmoe:
            # value + write-strength in ONE grouped matmul instead of two
            self.wvb_e = ExpertProj(c.n_exp, c.d_model, _VO + _BOb, cap=getattr(c, "cap", 1.35))
        self.bfloor = float(getattr(c, 'bfloor', 0.0))   # target min mean beta per head
        self.bu     = float(getattr(c, 'beta_u', 0.001)) # bias step size
        self.register_buffer('bbias', torch.zeros(H))    # revival bias, no gradient
        _gb = bool(getattr(c, "gbias", 0))
        self.wg = nn.Linear(c.d_model, _BO, bias=_gb)      # decay / forget gate: per head, or per SLOT under kslot
        if _gb:
            _hl = torch.logspace(math.log10(c.hl_min), math.log10(c.hl_max), H)
            with torch.no_grad():
                _lad = torch.log(torch.expm1(math.log(2.0) / _hl))
                if self.kslot:
                    assert len(set(self.dvs)) == 1, "column gates need uniform payload widths, got %s" % sorted(set(self.dvs))
                    self.wg.bias.copy_(_lad if int(getattr(c, "decay_head", 0)) else _lad.repeat_interleave(self.dvs[0]))   # a head's columns start on its rung
                else:
                    self.wg.bias.copy_(_lad)
                self.wg.weight.mul_(0.05)
        self.R = int(getattr(c, 'qpaths', 1))
        if self.R > 1:
            if int(getattr(c, "qphi", 0)):
                assert c.nlk, "qphi=1 needs nlk=1 (phi is the shared key encoder)"
                # per-head, per-path transform ON TOP of phi; identity init => q == k at init
                _A = torch.eye(dk).repeat(H, self.R, 1, 1)
                _A += 0.01 * torch.randn_like(_A)
                self.wq2p = nn.Parameter(_A)
                self.wq2 = None
            else:
                self.wq2p = None
                self.wq2 = nn.Linear(c.d_model, H * dk * self.R, bias=False)  # queries, untied from keys
        self.wo = nn.Linear(_VO * self.R, c.d_model, bias=False)
        self.conv = nn.Conv1d(c.d_model, c.d_model, c.conv, groups=c.d_model, padding=c.conv - 1)
        self.ngate = nn.LayerNorm(_VO * max(1, int(getattr(c, 'qpaths', 1))))
        # --- state-aware key placement -----------------------------------
        # Measured: keys occupy ~13 of 64 dims with |cos| 0.48, so each write erases
        # ~12% of the state and only ~8 writes survive (matching effrank(state) 8.6).
        # The writer picks its address from x_t alone, blind to which directions are
        # already occupied. Row norms of S say exactly that, so use them to suppress the
        # key's component along occupied directions and push new keys into free ones.
        # Zero-init the output so gain == 1 and the model starts identical to baseline.
        # --- selective write ------------------------------------------------
        # beta and g are already PER HEAD PER TOKEN, so freezing a head needs no kernel
        # change: beta=0 removes the write AND the erasure, g=0 removes the decay, so an
        # unselected head is preserved exactly. Writing to k of H heads cuts erasure
        # events per head by H/k and extends its effective half-life by the same factor.
        self.wfreeze = int(getattr(c, "wfreeze", 0))
        if self.wfreeze > 0:
            self.wsel = nn.Linear(c.d_model, H, bias=True)
            self.wsel_u = nn.Parameter(torch.ones(1))   # >0 = route away from full heads
        # value sees what is currently stored at the address it is about to overwrite,
        # so it can accumulate/preserve instead of always replacing. lambda init 0.
        self.wpreserve = bool(getattr(c, "wpreserve", 0))
        if self.wpreserve:
            self.wp_lam = nn.Parameter(torch.zeros(H))
        # Collateral damage IS non-orthogonality: writing at k costs a stored item at
        # k_old exactly cos(k,k_old)*(S k). Project k off the r most-occupied directions
        # and every direction orthogonal to it is provably untouched. lam init 0 = baseline.
        self.wortho = int(getattr(c, "wortho", 0))
        if self.wortho > 0:
            # Init at FULL projection, not 0. Gating it off at init is the pattern the
            # model refused three times with the bus, and the measurement says the state
            # sits at an equilibrium (writes-in = erasures-out, rank ~9 regardless of
            # context length). Forcing writes into unoccupied directions is what breaks
            # that equilibrium; it stays learnable so the model can back off if it hurts.
            self.wo_lam = nn.Parameter(torch.ones(H))
        # Keys carry a large SHARED component: centred rank ~23 of 64 but uncentred ~5.
        # The state accumulates k k^T, which sees the uncentred geometry, so every write
        # lands on the same dominant direction. Normalising fixes magnitude, not direction
        # -- a vector that is 90% "average token" is still 90% that after scaling to unit
        # length. Subtracting the running mean removes it directly. Structural, not gated:
        # the model has consistently declined mechanisms it could switch off.
        self.kdecor = float(getattr(c, "kdecor", 0.0))
        self.kdec_loss = None
        self.krep = float(getattr(c, "krep", 0.0))
        self.krep_tau = float(getattr(c, "krep_tau", 0.85))
        self.krep_loss = None
        self.wbud_loss = None
        self.kcenter = bool(getattr(c, "kcenter", 0))
        self.vcenter = int(getattr(c, 'vcenter', 0))
        if self.vcenter:
            # 47% of value ENERGY is a shared DC offset (measured p14@2500).
            self.register_buffer('vmu', torch.zeros(c.heads, c.dv))
            self.register_buffer('vmu_n', torch.zeros(()))
        self.vwhiten = int(getattr(c, 'vwhiten', 0))
        if self.vwhiten:
            # ZCA on values: lifts the state's input ceiling (span 19->~30).
            # Amplification bounded 10x via eigenvalue floor.
            self.register_buffer('vmu2', torch.zeros(c.heads, c.dv))
            self.register_buffer('vcov', torch.eye(c.dv).repeat(c.heads, 1, 1))
            self.register_buffer('vwm', torch.eye(c.dv).repeat(c.heads, 1, 1))
            self.register_buffer('vwn', torch.zeros(()))
        if self.kcenter:
            self.register_buffer("kmu", torch.zeros(H, dk))
            self.register_buffer("kmu_n", torch.zeros((), dtype=torch.long))
        self.wkey_mode = int(getattr(c, "wkey", 0))
        self.wkey_on = self.wkey_mode == 1
        if self.wkey_mode == 2:
            # EMA-WHITENED KEYS. Measured: keys span ~13 of 64 dims with |cos| 0.48, so
            # each write erases ~12% of the state and only ~8 survive. That bias is
            # CHRONIC, not within-sequence, so it can be estimated from a running
            # covariance instead of the live state -- which means no chunking and no
            # throughput cost. Whitening (not per-dim gains) because the crowded subspace
            # need not be axis-aligned, and rescaling axes cannot rotate it.
            self.register_buffer("kcov", torch.zeros(H, dk, dk))
            self.register_buffer("kw", torch.eye(dk).unsqueeze(0).repeat(H, 1, 1))
            self.register_buffer("kstep", torch.zeros((), dtype=torch.long))
            self.wkfix = float(getattr(c, "wkfix", -1.0))
            if self.wkfix < 0:
                self.wk_alpha = nn.Parameter(torch.full((H,), -6.0))   # sigmoid(-6)~0 = identity
            else:
                self.wk_alpha = None   # fixed blend: a == wkfix, no new learnable params
        if self.wkey_on:
            self.occ_mlp = nn.Sequential(nn.Linear(dk, dk), nn.GELU(), nn.Linear(dk, dk))
            # NOTE: zeroed in MemLM.__init__ AFTER self.apply(self._init), which would
            # otherwise overwrite it with normal(0.02) and break the identity start.
        # --- cross-layer state bus ---------------------------------------
        # One dk x dk adapter per SOURCE layer (this layer's query lives in its own key
        # space; reading someone else's state needs a learned change of basis). Gates are
        # zero-init so the bus contributes exactly nothing at step 0 -- the model starts
        # bit-identical to the no-bus baseline and has to learn to open each channel.
        self.bus_on = bool(getattr(c, "bus", 0))
        if self.bus_on:
            _L = int(c.layers)
            self.bus_A = nn.Parameter(torch.eye(dk).unsqueeze(0).repeat(_L, 1, 1).clone())
            self.bus_g = nn.Parameter(torch.zeros(_L))

    @torch.no_grad()
    def vw_refresh(self):
        """Recompute the value-whitening transform from the running covariance.
        Eager, trainer-driven (cadence: micro-batch n <= 5 or n % 20 == 1, the
        original in-forward schedule) so the compiled forward stays one graph."""
        if not getattr(self, 'vwhiten', 0):
            return
        eye = torch.eye(self.vcov.shape[-1], device=self.vcov.device)
        ev, U = torch.linalg.eigh(self.vcov.float() + 1e-4 * eye)
        ev = ev.clamp(min=1e-2 * ev.max(dim=-1, keepdim=True).values)
        self.vwm.copy_(U @ torch.diag_embed(ev.rsqrt()) @ U.transpose(-1, -2))

    def _bus_read(self, qq, bus):
        """qq:(B,T,H,R,dk); bus: list of L states (B,H,dk,dv) from the PREVIOUS chunk.

        The naive form is  sum_l g_l * (q @ A_l) @ S_l  -- 2L einsums per layer per chunk,
        which measured 10.5x slower than no bus (cost scales exactly with chunk count, so
        it is launch overhead, not arithmetic). By linearity that equals

            q @ ( sum_l  g_l * A_l @ S_l )  =  q @ W

        and W is a single (dk,dv) matrix built ONCE per chunk. The per-token cost is then
        one state read regardless of how many layers feed the bus. This is exact, not an
        approximation -- no capacity is given up.
        """
        if any(S is None for S in bus):
            return None
        Sb = torch.stack(list(bus), 0).to(qq.dtype)                  # (L,B,H,dk,dv)
        Ag = self.bus_A.to(qq.dtype) * self.bus_g.to(qq.dtype).view(-1, 1, 1)
        W = torch.einsum('lkj,lbhjv->bhkv', Ag, Sb)                  # collapse over layers
        return _QSCALE(qq.shape[-1]) * torch.einsum('bthrk,bhkv->bthrv', qq, W)

    def _lmix(self, t):
        # t: (B,T,H,...) -> group lanes, soft-share within pairs then quads. Mixes the LANE axis
        # only, so a per-SLOT beta/g keeps its variation across slots.
        B, T = t.shape[0], t.shape[1]
        H = self.c.heads
        _La = self.lanes; _NG = H // _La
        sh = t.shape
        # both levels are linear in the lane axis: pairs (1-gp) I + gp J2/2, then the quad (1-gq) I + gq J4/4 -- one
        # (La x La) map per group, applied as a single contraction (was ~8 elementwise kernels per call, 4 calls a layer)
        # NOT einsum: it lowers to a K=4, N=4 batched GEMM that cuBLAS runs at a crawl (measured: +290 ms/step).
        # Explicit weighted sums fuse into one pointwise kernel under compile (fp32 inside, rounded once at the store).
        M = self._lane_matrix(torch.float32)                               # (NG, La, La): out_m = sum_l M[m,l] t_l
        tg = t.reshape(B, T, _NG, _La, -1)
        lanes = tg.unbind(3)
        outs = []
        for mi in range(_La):
            acc = M[None, None, :, mi, 0, None] * lanes[0]
            for li in range(1, _La):
                acc = acc + M[None, None, :, mi, li, None] * lanes[li]
            outs.append(acc.to(t.dtype))
        return torch.stack(outs, 3).reshape(sh)

    def _shadow_inputs(self, hx):
        """(k, v, beta, g) of this layer's write path on a given stream -- what the compressor's shadow scan consumes.
        Compiled by the trainer (like forward), so the normalisation and gate elementwise chain is fused."""
        B, T = hx.shape[0], hx.shape[1]
        k = F.normalize(self.phi(hx).view(B, T, self.c.heads, self.c.dk), dim=-1)
        v = self.wv(hx)
        b, g = self.slot_gates(hx, self.wb(hx))
        return k, v, b.reshape(B, T, -1), g.reshape(B, T, -1)

    def _lane_matrix(self, dt):
        _La = self.lanes; gp = torch.sigmoid(self.lane_gp.float()); gq = torch.sigmoid(self.lane_gq.float())   # (NG, La/2), (NG,)
        _NG = gp.shape[0]; dev = gp.device
        # pairs level without loops or in-place writes (they graph-break the compiled mixer at small T):
        # M1[l, m] = delta_lm (1 - gp_pair(l)) + [pair(l) == pair(m)] gp_pair(l) / 2
        lane = torch.arange(_La, device=dev); pairmask = ((lane[:, None] // 2) == (lane[None, :] // 2)).float()
        gpl = gp.repeat_interleave(2, dim=1)                                       # (NG, La): each lane's pair gate
        M1 = torch.eye(_La, device=dev) * (1 - gpl)[:, :, None] + pairmask * (gpl / 2)[:, :, None]
        M2 = (1 - gq)[:, None, None] * torch.eye(_La, device=dev) + gq[:, None, None] * torch.full((_La, _La), 1.0 / _La, device=dev)
        return (M2 @ M1).to(dt)

    def _layout(self, device):
        if self._L is None:
            from kcol import slot_layout
            self._L = slot_layout(self.dvs, self.c.dk, 0, device=device)
        return self._L

    def slot_gates(self, x, beta_pre):
        """Per-slot write strength and log-decay from the projections (kslot).
        beta = clamp(1.2*sigmoid(z) - 0.1, 0, 1): a stretched sigmoid, so exact 0 and exact 1
        are reachable (a plain sigmoid never reaches 0, which is why every generation wrote at
        every token and had to keep its slow heads at beta ~ 0.05). slot_topk>0: each token
        writes to exactly k slots of the layer (straight-through top-k). slot_gtie: unwritten
        slots do not decay either."""
        B, T, _ = x.shape
        H, dk = self.c.heads, self.c.dk
        _dvh = self.dvs[0]
        if int(getattr(self.c, "beta_head", 0)):
            z = beta_pre.view(B, T, H, 1).float() + self.bbias.view(1, 1, H, 1)      # one beta per head, broadcast over its slots below
        else:
            z = beta_pre.view(B, T, H, _dvh).float() + self.bbias.view(1, 1, H, 1)
        beta = (1.2 * torch.sigmoid(z) - 0.1).clamp(0.0, 1.0)
        if float(getattr(self.c, "wbud", 0.0)) > 0:
            _rho = float(getattr(self.c, "wbud_rho", 0.3))
            if int(getattr(self.c, "wbud_count", 0)):
                # COUNT budget: the fraction of columns actually written (beta > tau) above rho. Forward uses the hard count;
                # backward goes through a sigmoid centred on the threshold logit, so the gates nearest the threshold feel it
                # most and the weakest columns are dropped first (m11's mean-activity budget was met by weakening all writes).
                _tau = float(getattr(self.c, "wbud_tau", 0.05)); _p = (_tau + 0.1) / 1.2
                _zt = math.log(_p / (1.0 - _p))                                   # beta == tau  <=>  z == _zt
                _s = torch.sigmoid(z - _zt)
                _active = (beta > _tau).float() + (_s - _s.detach())
                self.wbud_loss = F.relu(_active.mean() - _rho)
            else:
                # soft write budget: the layer's mean gate activity above rho is penalised (the trainer weights it). A BUDGET, not an
                # L1: once the layer writes to ~rho of its columns the pressure stops, so no slot is driven to the clamp's dead zone.
                self.wbud_loss = F.relu(torch.sigmoid(z).mean() - _rho)
        if beta.shape[-1] == 1:
            beta = beta.expand(B, T, H, _dvh)
        if int(getattr(self.c, "decay_head", 0)):
            g = -F.softplus(self.wg(x).float()).view(B, T, H, 1).expand(B, T, H, _dvh)   # one decay per head
        else:
            g = -F.softplus(self.wg(x).float()).view(B, T, H, _dvh)
        kk = int(getattr(self.c, "slot_topk", 0))
        if kk > 0:
            zf = z.reshape(B, T, H * dk)
            idx = zf.topk(min(kk, H * dk), dim=-1).indices
            hard = torch.zeros_like(zf).scatter_(-1, idx, 1.0)
            soft = torch.softmax(zf, dim=-1)
            mask = (hard + soft - soft.detach()).view(B, T, H, dk)
            beta = beta * mask
        if int(getattr(self.c, "slot_gtie", 1)):
            g = g * (beta > 0).float()
        return beta, g

    def _read_queries(self, f):
        qq = F.normalize(torch.einsum('bthd,hrde->bthre', f, self.wq2p.to(f.dtype)), dim=-1)
        if getattr(self, "lanes", 1) > 1:
            qq = F.normalize(self._lmix(qq), dim=-1)
        return qq

    def _forward_kslot(self, x, f, k, _V, _beta_pre, return_state, new_cp):
        B, T, _ = x.shape
        H, dk = self.c.heads, self.c.dk
        from kcol import kslot as _kslot, reads as _reads
        _fused = (int(getattr(self.c, "kprep", 1)) and not int(getattr(self.c, "beta_head", 0)) and not int(getattr(self.c, "decay_head", 0)) and getattr(self, "lanes", 1) > 1 and int(getattr(self.c, "slot_topk", 0)) == 0
                  and getattr(self, "wq2p", None) is not None and getattr(self, "_kmod", None) is None and not self.wkey_on and not self.kcenter)
        if _fused:
            # one kernel: beta/g from their pre-activations, lane mix, k = normalize(mix(normalize(f))), read queries -- kprep.py
            k, _b2, _g2, qq = _prep_fused(f, _beta_pre.reshape(B, T, -1), self.wg(x), self.bbias, self._lane_matrix(torch.float32), self.wq2p,
                                          self.lanes, bool(int(getattr(self.c, "slot_gtie", 1))))
            beta = _b2.view(B, T, H, -1); g = _g2.view(B, T, H, -1)
        else:
            beta, g = self.slot_gates(x, _beta_pre)
            if getattr(self, "lanes", 1) > 1:      # m4's lanes soft-share k/beta/g at WRITE time
                k = F.normalize(self._lmix(k), dim=-1)
                beta = self._lmix(beta)               # (B,T,H,dv) -- lmix works on the head axis
                g = self._lmix(g)
        if self.krep > 0 and self.training:
            # write-time key repulsion, weighted by each token's mean slot strength
            _bh = beta.mean(-1)
            _st = 4 if k.shape[1] <= 1024 else 8
            kr = k[:, ::_st].float(); br = _bh[:, ::_st].float()
            # the (B,H,Ts,Ts) fp32 Gram products are recomputed in backward (3 GB saved at T=1024)
            self.krep_loss = _krep_fn(kr, br, float(self.krep_tau))      # (recompute dropped: 3 GB is affordable at the 42 GB peak)
        else:
            self.krep_loss = None
        if getattr(self, "_wdbg", None) is not None:
            self._wdbg.append((k.detach(), beta.detach(), g.detach()))
        if not _fused:
            qq = self._read_queries(f)                              # (B,T,H,R,dk)
        L = self._layout(x.device)
        kb, gb = k.to(torch.bfloat16), g.reshape(B, T, -1)
        bb, vb = beta.reshape(B, T, -1).to(torch.bfloat16), _V.to(torch.bfloat16)
        if getattr(self, "_sdbg", None) is not None:
            self._sdbg.append((kb, gb, bb, vb, qq))
        _cl = int(getattr(self.c, "kchunk_layers", -1))
        C = (int(getattr(self.c, "kchunk", 0)) if _cl < 0 else _cl) or T
        final = None
        if C >= T and int(getattr(self.c, "emit_ckpt", 0)) and self.training and torch.is_grad_enabled() and not return_state:
            # emission NOT saved for backward: the scan + reads are recomputed there (kslot is stateless and deterministic)
            o, _ = _ckpt(_scan_chunk, kb, gb, bb, vb, qq, None, L, use_reentrant=False)
        elif C >= T:
            _, emit = _kslot(kb, gb, bb, vb, None, L)
            o, _ = _reads(emit, qq, L)                                 # (B,T,R*VT)
            if return_state:
                from kcol import final_state
                final = [final_state(emit, L, h) for h in range(H)]
        else:
            assert len(set(L["dv"])) == 1 and L["CB"] == L["dv"][0], "chunked path assumes uniform payloads"
            s0 = None; outs = []
            for kc, gc, bc, vc, qc in zip(*(_chunks(t, C) for t in (kb, gb, bb, vb, qq))):
                o_c, s0 = _ckpt(_scan_chunk, kc, gc, bc, vc, qc, s0, L, use_reentrant=False)
                outs.append(o_c)
            o = torch.cat(outs, 1)
            if return_state:
                final = [s0.view(B, L["NP"], dk, L["CB"])[:, h] for h in range(H)]
        o = self.ngate(o.to(x.dtype))
        return (self.wo(o), final, new_cp) if return_state else self.wo(o)

    def _wmask(self, x, beta, g, state=None):
        """Selective writing at HEAD granularity: each token writes to its top-k heads and
        the rest are frozen -- beta 0 (no write) AND g 0 (no decay). Straight-through so the
        selector trains. Measured motivation: replaying real writes at 1% frequency lifted
        single-cabinet retrieval 10.9% -> 42.6% at 64 tokens and 5.2% -> 17.7% at 512,
        because a dk-address store cannot survive 1,024 writes landing on top of it.
        Kept as a method so the compressor's shadow writes apply the SAME mask (a page that
        describes an unmasked store would be describing a store that does not exist)."""
        H = self.c.heads
        kk = min(self.wfreeze, H)
        sc = self.wsel(x)
        if state is not None:
            full = state.detach().float().norm(dim=(-2, -1))
            full = full / full.mean(-1, keepdim=True).clamp(min=1e-6)
            sc = sc - self.wsel_u.to(sc.dtype) * full.unsqueeze(1).to(sc.dtype)
        idx = sc.topk(kk, dim=-1).indices
        hard = torch.zeros_like(sc).scatter_(-1, idx, 1.0)     # exactly k, ties resolved by index
        soft = torch.softmax(sc, dim=-1)
        mask = hard + soft - soft.detach()                     # straight-through
        return beta * mask.to(beta.dtype), g * mask.float()

    def forward(self, x, state=None, return_state=False, bus=None, conv_prev=None, route=None):
        B, T, _ = x.shape
        H, dk, dv = self.c.heads, self.c.dk, self.c.dv

        def _wvb(xx):
            # -> (value, pre-sigmoid beta). Expert-conditional when wmoe is on.
            if self.wmoe:
                if route is None:
                    raise RuntimeError(
                        "wmoe=1 but MemMixer.forward got route=None: the block router "
                        "must pass its (topi, topv). Refusing to write with the "
                        "untrained shared self.wv.")
                vb = self.wvb_e(xx.reshape(-1, xx.shape[-1]), route[0], route[1])
                vb = vb.view(*xx.shape[:-1], self._VO + self._BOb).to(xx.dtype)
                return vb[..., :self._VO], vb[..., self._VO:]
            return self.wv(xx), self.wb(xx)
        # The depthwise conv is causal with a receptive field of c.conv. Under the chunked
        # bus loop it would otherwise see zero-padding at every chunk boundary instead of
        # the real previous tokens, corrupting the first c.conv-1 positions of each chunk.
        # Prepend the carried tail so the chunked path matches the unchunked one exactly.
        pad = self.c.conv - 1
        if conv_prev is None:
            xin, p = x, 0
        else:
            xin, p = torch.cat([conv_prev, x], dim=1), conv_prev.shape[1]
        new_cp = xin[:, -pad:] if pad > 0 else xin[:, :0]
        x = self.conv(xin.transpose(1, 2))[:, :, :p + T][:, :, p:].transpose(1, 2)
        if self.nlk:
            f = self.phi(x).view(B, T, H, dk)
            if self.kcenter:
                if self.training:
                    with torch.no_grad():
                        self.kmu_n += 1
                        mom = max(0.01, 1.0 / float(self.kmu_n))
                        self.kmu.mul_(1 - mom).add_(f.detach().float().mean(dim=(0, 1)),
                                                    alpha=mom)
                f = f - self.kmu.to(f.dtype)
            if self.wkey_mode == 2:
                if self.training:
                    with torch.no_grad():
                        # Subsample tokens for the covariance: it is a running
                        # statistic over millions of tokens, so every 4th is plenty, and
                        # the accumulation is as expensive as the whitening itself.
                        kk = F.normalize(f.detach()[:, ::4].float(), dim=-1)
                        C = torch.einsum('bthi,bthj->hij', kk, kk) / (kk.shape[0] * kk.shape[1])
                        self.kstep += 1
                        # count-based momentum: the first update sets kcov outright, then
                        # it relaxes to a 0.01 EMA. A plain 0.99 EMA from I needs ~300
                        # steps to reflect the data, so whitening would be a no-op for
                        # the whole early phase where the geometry is being set.
                        mom = max(0.01, 1.0 / float(self.kstep))
                        self.kcov.mul_(1 - mom).add_(C, alpha=mom)
                        if int(self.kstep) <= 5 or int(self.kstep) % 20 == 1:
                            eye = torch.eye(dk, device=C.device)
                            ev, U = torch.linalg.eigh(self.kcov.float() + 1e-3 * eye)
                            self.kw.copy_(U @ torch.diag_embed(
                                ev.clamp(min=1e-6).rsqrt()) @ U.transpose(-1, -2))
                if self.wk_alpha is not None:
                    a = torch.sigmoid(self.wk_alpha).view(H, 1, 1).to(f.dtype)
                else:
                    a = f.new_full((H, 1, 1), self.wkfix)
                eye = torch.eye(dk, device=f.device, dtype=f.dtype)
                Weff = (1 - a) * eye + a * self.kw.to(f.dtype)
                f = torch.einsum('bthi,hij->bthj', f, Weff)
            if self.wortho > 0 and state is not None:
                r = min(self.wortho, dk)
                M = state.detach().float()                     # (B,H,dk,dv)
                G = M @ M.transpose(-1, -2)                    # (B,H,dk,dk) key-space energy
                _, U = torch.linalg.eigh(G)
                U = U[..., -r:].to(f.dtype)                    # top-r occupied directions
                co = torch.einsum('bthk,bhkr->bthr', f, U)
                lam = self.wo_lam.view(1, 1, H, 1).to(f.dtype)
                f = f - lam * torch.einsum('bthr,bhkr->bthk', co, U)
            if self.wkey_on and state is not None:
                # occupancy of each key-direction = row norm of the state at chunk start
                occ = state.detach().float().norm(dim=-1)              # (B,H,dk)
                occ = occ / occ.mean(-1, keepdim=True).clamp(min=1e-6)
                gain = 1.0 + torch.tanh(self.occ_mlp(occ.to(f.dtype)))  # (B,H,dk)
                f = f * gain.unsqueeze(1)
            fk = f if getattr(self, '_kmod', None) is None else self._kmod(f)
            q, k = f, F.normalize(fk, dim=-1)
            # Auxiliary loss pushing the per-head key covariance toward ISOTROPIC.
            # Diagonal alone is not enough: a diagonal-but-concentrated covariance is
            # still rank-1. ||C - I/dk||^2 penalises both correlation and concentration,
            # putting gradient directly on the quantity that has never moved -- rather
            # than hoping the LM loss eventually rewards spread keys, which it has not.
            if self.kdecor > 0 and self.training:
                kk = k[:, ::4].float()
                Bk, Tk = kk.shape[0], kk.shape[1]
                Ck = torch.einsum('bthi,bthj->hij', kk, kk) / max(Bk * Tk, 1)
                Ik = torch.eye(dk, device=Ck.device, dtype=Ck.dtype) / dk
                self.kdec_loss = ((Ck - Ik) ** 2).sum()
            else:
                self.kdec_loss = None
        else:
            q = self.wq(x).view(B, T, H, dk)
            k = F.normalize(self.wk(x).view(B, T, H, dk), dim=-1)
        _V, _beta_pre = _wvb(x)          # x here is POST causal-conv, as in the original
        if self.kslot:
            return self._forward_kslot(x, f, k, _V, _beta_pre, return_state, new_cp)
        v = _V.view(B, T, H, dv)
        if getattr(self, 'vcenter', 0):
            if self.training:
                with torch.no_grad():
                    self.vmu_n += 1
                    mom = max(0.01, 1.0 / float(self.vmu_n))
                    self.vmu.mul_(1 - mom).add_(v.float().mean(dim=(0, 1)).to(self.vmu.dtype), alpha=mom)
            v = v - self.vmu.to(v.dtype)
        if getattr(self, 'vwhiten', 0):
            if self.training:
                with torch.no_grad():
                    # COMPILE-CLEAN: float()/int() on a buffer was a graph break INSIDE the
                    # compiled mixer, every layer, every step. Eigen-refresh (data-dependent
                    # cadence) lives in vw_refresh(), trainer-called at the same cadence.
                    # EXACT reproduction of the original update with a LITERAL momentum:
                    # the bf16 counter saturates at 256 (ulp), so max(0.01, 1/vwn) == 0.01
                    # for every remaining step of this lineage; same mul_/add_ ops and
                    # rounding as before, no scalar extraction. (A fresh run wanting the
                    # 1/n warm-up and a live eigen-refresh needs fp32 stat buffers.)
                    self.vwn += 1
                    vf = v.float()
                    self.vmu2.mul_(1 - 0.01).add_(vf.mean(dim=(0, 1)).to(self.vmu2.dtype), alpha=0.01)
                    vc = vf - self.vmu2.float()
                    cov = torch.einsum('bthv,bthw->hvw', vc, vc) / (v.shape[0] * v.shape[1])
                    self.vcov.mul_(1 - 0.01).add_(cov.to(self.vcov.dtype), alpha=0.01)
            v = torch.einsum('bthv,hvw->bthw', (v.float() - self.vmu2.float()),
                             self.vwm.float()).to(v.dtype)
        if self.wpreserve and state is not None:
            # (B,T,H,dk) x (B,H,dk,dv) -> what is stored at this address right now
            cur = torch.einsum('bthk,bhkv->bthv', k, state.detach().to(k.dtype))
            v = v + self.wp_lam.view(1, 1, H, 1).to(v.dtype) * cur
        beta = torch.sigmoid(_beta_pre + self.bbias)
        if self.krep > 0 and self.training:
            # WRITE-TIME KEY REPULSION. Measured (34k): 22.6% of strongly-written items
            # share a near-duplicate address (|cos|>0.9) -- the delta rule merges them.
            # Penalise only PAIRS past a high threshold, weighted by both write
            # strengths, excluding near-adjacent tokens (local similarity is natural).
            # Structurally unbiased: keys already spread => zero pairs => zero loss.
            _st = 4 if k.shape[1] <= 1024 else 8
            kr = k[:, ::_st].float()                            # (B,Ts,H,dk), unit norm
            br = beta[:, ::_st].float()
            Ts = kr.shape[1]
            Gm = torch.einsum("bthi,bshi->bhts", kr, kr)      # pairwise cos
            Wp = br.transpose(1, 2)[:, :, :, None] * br.transpose(1, 2)[:, :, None, :]
            near = torch.ones(Ts, Ts, device=kr.device, dtype=torch.bool).triu(4)
            # LINEAR penalty, averaged over ACTIVE pairs only: constant push on each
            # colliding pair, magnitude independent of how sparse collisions are.
            # (v1 divided a SQUARED penalty by ALL pairs: measured 0.0000 on real
            # batches -- the loss never fired. Collisions are 65% distant, so the
            # adjacency exclusion stays.)
            exc = (Gm.abs() - self.krep_tau).clamp(min=0) * near
            act = (exc > 0).sum().clamp(min=1)
            self.krep_loss = (exc * Wp).sum() / act
        else:
            self.krep_loss = None
        if self.training and self.bfloor > 0.0:
            # auxiliary-loss-free revival: push UP heads writing less than the
            # target, let over-target heads relax back at 1/4 speed. Never
            # negative, so a healthy head's selectivity is left alone.
            with torch.no_grad():
                obs = beta.detach().float().mean(dim=(0, 1))
                step = torch.where(obs < self.bfloor,
                                   torch.ones_like(obs),
                                   torch.full_like(obs, -0.25))
                self.bbias += self.bu * step
                self.bbias.clamp_(0.0, 6.0)
        g = -F.softplus(self.wg(x).float())               # log-decay <= 0
        if getattr(self, "lanes", 1) > 1:
            _lmix = self._lmix
            k = F.normalize(_lmix(k), dim=-1)
            beta = _lmix(beta)
            g = _lmix(g)
        if self.wfreeze > 0:
            # AFTER lane mixing: mixing averages beta/g across merged lanes and would smear a
            # head mask back onto the lane-mates (measured: 5 selected -> 8-16 written)
            beta, g = self._wmask(x, beta, g, state)
        # capture the ACTUAL k/beta/g handed to the kernel -- i.e. after wortho's
        # projection, wkey's gain and wfreeze's mask. Hooking phi instead measures the
        # keys BEFORE every write-side intervention, which makes any of them look inert.
        if getattr(self, "_wdbg", None) is not None:
            self._wdbg.append((k.detach(), beta.detach(), g.detach()))
        if getattr(self, "_selfref", None) is not None:
            # Own full state at t-1 (causal shift; t=0 sees the carried state or
            # zeros). Content is stop-grad; the refresh projections and gate learn.
            from ktriton import fused_delta_stream as _fds8
            with torch.no_grad():
                _, _SA = _fds8(
                    k.detach(), k.detach(), v.detach(), g.detach(), beta.detach(),
                    store_state=True, state_dtype=torch.bfloat16,
                    initial_state=(state.detach() if state is not None else None))
                if state is not None:
                    _S0 = state.detach().to(_SA.dtype).unsqueeze(1)
                else:
                    _S0 = _SA.new_zeros(_SA[:, :1].shape)
                _Sprev = torch.cat([_S0, _SA[:, :-1]], 1)
            if getattr(self, "_sinj", None) is not None:
                # inject the own-state summary into the INPUT and redo every write
                # decision with it. x is rebound, so the R>1 read queries (wq2) and
                # the output path downstream also become state-aware.
                x = x + self._sg2.to(x.dtype) * self._sinj(_Sprev)
                f = self.phi(x).view(B, T, H, dk)
                fk = f if getattr(self, '_kmod', None) is None else self._kmod(f)
                q, k = f, F.normalize(fk, dim=-1)
                v = _wvb(x)[0].view(B, T, H, dv)
                if getattr(self, 'vcenter', 0):
                    v = v - self.vmu.to(v.dtype)
                beta = torch.sigmoid(_wvb(x)[1] + self.bbias)
                g = -F.softplus(self.wg(x).float())
                if getattr(self, "_wdbg", None) is not None and self._wdbg:
                    self._wdbg[-1] = (k.detach(), beta.detach(), g.detach())
            v = v + self._sg.to(v.dtype) * self._selfref(_Sprev)
        if getattr(self, "_want_stream", False):
            # Full per-token state stream via the Triton fused-recurrent kernel.
            # ONE launch, ~2ms at T=2048; stop-grad by construction (bus=7 tradeoff).
            from ktriton import fused_delta_stream
            with torch.no_grad():
                _, _S7 = fused_delta_stream(
                    k.detach(), k.detach(), v.detach(), g.detach(), beta.detach(),
                    store_state=True, state_dtype=torch.bfloat16,
                    initial_state=(state.detach() if state is not None else None))
            self._stream = _S7
        R = self.R
        # k/v/g/beta -- and therefore the state trajectory -- are shared across the R query
        # paths; only q differs. r=0 is the one asked for the final state.
        if getattr(self, "wq2p", None) is not None:
            # queries in key space: same phi features the ADDRESS was made from (f is
            # the current binding -- if _sinj rebound x, f was recomputed above)
            qq = F.normalize(torch.einsum('bthd,hrde->bthre', f, self.wq2p.to(f.dtype)), dim=-1)
            if getattr(self, "lanes", 1) > 1:
                qq = F.normalize(_lmix(qq), dim=-1)
        else:
            qq = self.wq2(x).view(B, T, H, R, dk) if R > 1 else q.view(B, T, H, 1, dk)
        # FUSED multi-path read. k/v/g/beta are identical across the R paths, so instead
        # of R separate scans we run ONE call over H*R pseudo-heads (head h, path r ->
        # index h*R+r). Measured: the kernel costs ~500us of HOST time per call almost
        # independently of T, so call count is the entire cost -- this cuts it 4x at R=4
        # and raises head parallelism from H to H*R. Identical arithmetic.
        if R == 1:
            o1, final = _scan(self.c, qq[:, :, :, 0, :].contiguous(), k, v, g, beta,
                              state, bool(return_state))
            o = o1.unsqueeze(3)
        else:
            rep = lambda t: t.repeat_interleave(R, dim=2)
            sf = None if state is None else state.repeat_interleave(R, dim=1)
            of, ff = _scan(self.c, qq.reshape(B, T, H * R, dk), rep(k), rep(v),
                           rep(g), rep(beta), sf, bool(return_state))
            o = of.view(B, T, H, R, dv)
            # every path carries the same state; keep one copy
            final = None if ff is None else ff.view(B, H, R, dk, dv)[:, :, 0].contiguous()
        if self.bus_on and bus is not None:
            rd = self._bus_read(qq, bus)
            if rd is not None:
                o = o + rd
        o = self.ngate(o.reshape(B, T, H * R * dv))
        return (self.wo(o), final, new_cp) if return_state else self.wo(o)


def _rope(x, base=10000.0):
    """Rotary position embedding over the head dim."""
    B, T, H, D = x.shape
    half = D // 2
    freqs = torch.arange(half, device=x.device, dtype=torch.float32)
    inv = base ** (-freqs / half)
    t = torch.arange(T, device=x.device, dtype=torch.float32)
    ang = t[:, None] * inv[None, :]
    cos, sin = ang.cos()[None, :, None, :], ang.sin()[None, :, None, :]
    x1, x2 = x[..., :half].float(), x[..., half:].float()
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1).to(x.dtype)


class ShadowProj(nn.Module):
    """The compressor's shared write-path: its own key encoder, value, write-gate and decay projections (same shapes and
    init scheme as a mixer's, freshly initialised), evaluated once on the pre-block stream. slot_gates / _shadow_inputs are
    MemMixer's own methods bound here."""
    def __init__(self, m0):
        super().__init__()
        import copy as _cp
        self.c, self.dvs, self.VT = m0.c, m0.dvs, m0.VT
        self.phi, self.wv, self.wb, self.wg = _cp.deepcopy(m0.phi), _cp.deepcopy(m0.wv), _cp.deepcopy(m0.wb), _cp.deepcopy(m0.wg)
        self.bbias = nn.Parameter(m0.bbias.detach().clone())
        with torch.no_grad():
            for mod in (self.phi, self.wv, self.wb):
                for lin in mod.modules():
                    if isinstance(lin, nn.Linear): lin.reset_parameters()
            wg_bias = self.wg.bias.detach().clone() if self.wg.bias is not None else None
            self.wg.reset_parameters(); self.wg.weight.mul_(0.05)           # as the mixer's init; keep the half-life ladder bias
            if wg_bias is not None: self.wg.bias.copy_(wg_bias)
ShadowProj.slot_gates = MemMixer.slot_gates
ShadowProj._shadow_inputs = MemMixer._shadow_inputs


class AttnMixer(nn.Module):
    """Plain causal multi-head attention. Same shell as MemMixer (conv, norms, dims)
    so the ONLY difference is the sequence-mixing operator: attention + KV cache
    instead of a fixed-size compressed state."""

    def __init__(self, c: Cfg):
        super().__init__()
        self.c = c
        H = c.heads
        self.dh = c.d_model // H
        self.wq = nn.Linear(c.d_model, H * self.dh, bias=False)
        self.wk = nn.Linear(c.d_model, H * self.dh, bias=False)
        self.wv = nn.Linear(c.d_model, H * self.dh, bias=False)
        self.wo = nn.Linear(H * self.dh, c.d_model, bias=False)
        self.conv = nn.Conv1d(c.d_model, c.d_model, c.conv, groups=c.d_model, padding=c.conv - 1)
        self.ngate = nn.LayerNorm(H * self.dh)

    def forward(self, x, state=None, return_state=False):
        B, T, _ = x.shape
        H, dh = self.c.heads, self.dh
        x = self.conv(x.transpose(1, 2))[:, :, :T].transpose(1, 2)
        q = _rope(self.wq(x).view(B, T, H, dh)).transpose(1, 2)
        k = _rope(self.wk(x).view(B, T, H, dh)).transpose(1, 2)
        v = self.wv(x).view(B, T, H, dh).transpose(1, 2)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        o = o.transpose(1, 2).reshape(B, T, H * dh)
        o = self.ngate(o)
        out = self.wo(o)
        return (out, None) if return_state else out


_HAS_GROUPED = (hasattr(torch, "_grouped_mm") and torch.cuda.is_available()
                and torch.cuda.get_device_capability()[0] >= 9)   # Hopper+ only
_GDT = torch.bfloat16


class MoE(nn.Module):
    """Top-k routed SwiGLU experts with AUXILIARY-LOSS-FREE load balancing
    (Wang et al., arXiv:2408.15664 -- the DeepSeek-V3 strategy).

    A per-expert bias is added to the routing score for the top-k SELECTION only;
    the gating weights that scale expert outputs come from the UNBIASED scores, so
    balancing never distorts the language-modelling gradient. The bias is not learned
    by backprop -- it is nudged by a sign rule after every batch:

        b_i <- b_i + u * sign(expected_load - actual_load)

    Balance is reported as MaxVio = (max_i load_i - expected) / expected.
    """

    def __init__(self, c):
        super().__init__()
        self.E, self.k, self.u = c.n_exp, c.top_k, c.moe_u
        self.cap = getattr(c, "cap", 1.35)
        self.use_loop = False   # True = zero-drop per-expert loop (for early, unbalanced routing)
        self.router = nn.Linear(c.d_model, c.n_exp, bias=False)
        self.w1 = nn.Parameter(torch.empty(c.n_exp, c.d_model, c.d_ff))
        self.w3 = nn.Parameter(torch.empty(c.n_exp, c.d_model, c.d_ff))
        self.w2 = nn.Parameter(torch.empty(c.n_exp, c.d_ff, c.d_model))
        for w in (self.w1, self.w3, self.w2):
            nn.init.normal_(w, std=0.02)
        self.register_buffer("ebias", torch.zeros(c.n_exp))
        self.util = None
        self.maxvio = None
        self.drop = None

    def route(self, xf):
        # Routing pulled out of forward so the BLOCK can compute it once on n1(x) --
        # before the memory write -- and hand the same decision to both the write
        # projection and the FFN. The expert that thinks is the expert that writes.
        score = F.softmax(self.router(xf).float(), dim=-1)
        topi = (score + self.ebias).topk(self.k, dim=-1).indices
        topv = score.gather(-1, topi)
        topv = topv / topv.sum(-1, keepdim=True).clamp_min(1e-9)
        return topi, topv

    def forward(self, x, route=None):
        B, T, D = x.shape
        xf = x.reshape(-1, D)
        N = xf.shape[0]

        if route is None:
            topi, topv = self.route(xf)
        else:
            topi, topv = route

        if self.use_loop:
            # zero-drop path: every token reaches its experts, no capacity limit.
            # Slower, but correct while routing is still unbalanced early in training.
            out = torch.zeros_like(xf)
            counts = torch.zeros(self.E, device=x.device, dtype=torch.float32)
            for e in range(self.E):
                hit = (topi == e)
                if not hit.any():
                    continue
                idx, kpos = hit.nonzero(as_tuple=True)
                counts[e] = idx.numel()
                xe = xf[idx]
                h = F.silu(xe @ self.w1[e]) * (xe @ self.w3[e])
                ye = h @ self.w2[e]
                # AMPFIX 2026-09-14: under autocast(fp16) the expert matmuls return HALF while this
                # accumulator is the residual's FP32, and index_add_ refuses mixed dtypes.  A whole-model
                # bf16 run never sees it because everything is one dtype.  Accumulating in out.dtype is
                # also the numerically better choice, and .to() is identity when the dtypes already match,
                # so the bf16 path is bit-for-bit unchanged.
                out.index_add_(0, idx, (ye * topv[idx, kpos].unsqueeze(-1).to(ye.dtype)).to(out.dtype))
            self.drop = torch.zeros((), device=x.device)
            expected = N * self.k / self.E
            with torch.no_grad():
                self.util = (counts / max(N * self.k, 1)).detach()
                self.maxvio = ((counts.max() - expected) / expected).detach()
                if self.training:
                    self.ebias += self.u * torch.sign(expected - counts)
            return out.view(B, T, D)

        if _HAS_GROUPED and not self.use_loop:
            # variable-size per-expert matmul: no padding, no capacity factor, zero drops
            M = N * self.k
            fe = topi.reshape(-1)
            tok = torch.arange(N, device=x.device).repeat_interleave(self.k)
            wg = topv.reshape(-1)
            order = torch.argsort(fe)
            fe_s, tok_s, wg_s = fe[order], tok[order], wg[order]
            counts = torch.zeros(self.E, dtype=torch.long, device=fe_s.device).scatter_add_(0, fe_s, torch.ones_like(fe_s))  # static-shape bincount (compile-clean)
            offs = counts.cumsum(0).to(torch.int32)
            counts = counts.float()
            xs = xf[tok_s].to(_GDT)
            h1 = torch._grouped_mm(xs, self.w1.to(_GDT), offs=offs)
            h3 = torch._grouped_mm(xs, self.w3.to(_GDT), offs=offs)
            ys = torch._grouped_mm((F.silu(h1) * h3), self.w2.to(_GDT), offs=offs)
            ys = ys * wg_s.unsqueeze(-1).to(ys.dtype)
            out = torch.zeros(N, D, device=x.device, dtype=ys.dtype)
            out.index_add_(0, tok_s, ys)
            self.drop = torch.zeros((), device=x.device)
            expected = N * self.k / self.E
            with torch.no_grad():
                self.util = (counts / max(M, 1)).detach()
                self.maxvio = ((counts.max() - expected) / expected).detach()
                if self.training:
                    self.ebias += self.u * torch.sign(expected - counts)
            return out.view(B, T, D)

        # --- static-shape capacity-batched dispatch ------------------------------
        # Rank every (token, slot) pair within its expert entirely on-GPU, pack into a
        # fixed [E, C, D] buffer, run ALL experts in three bmm calls, scatter back.
        # No boolean indexing => no hidden nonzero() sync, one compiled graph; gathers go
        # through index_select so their backward is an atomic index_add, not a sort.
        M = N * self.k
        C = max(1, int(self.cap * M / self.E))
        flat_e = topi.reshape(-1)                                        # M
        tok = torch.arange(N, device=x.device).repeat_interleave(self.k)  # M
        wgt = topv.reshape(-1)                                           # M
        buf, gate, slot_c, keep, counts = _cap_dispatch(flat_e, tok, wgt, self.E, C, xf.dtype)
        counts = counts.float()

        xe = xf.index_select(0, buf).view(self.E, C, D)
        h = F.silu(torch.bmm(xe, self.w1)) * torch.bmm(xe, self.w3)
        ye = torch.bmm(h, self.w2).view(self.E * C, D) * gate.unsqueeze(-1)

        out = _cap_gather_back(ye, slot_c, tok, N)
        self.drop = (1.0 - keep.float().mean()).detach()

        expected = N * self.k / self.E
        with torch.no_grad():
            self.util = (counts / max(N * self.k, 1)).detach()
            self.maxvio = ((counts.max() - expected) / expected).detach()
            if self.training:
                self.ebias += self.u * torch.sign(expected - counts)
        return out.view(B, T, D)


class PreFFN(nn.Module):
    """Plain FFN block ahead of the memory stack: the first write is otherwise
    computed from raw embeddings + a 4-token conv + one 2-layer MLP -- the least
    informed writer in the network. ~2M params buys it a processed input."""
    def __init__(self, c):
        super().__init__()
        self.n = nn.LayerNorm(c.d_model)
        self.w1 = nn.Linear(c.d_model, c.d_ff, bias=False)
        self.w3 = nn.Linear(c.d_model, c.d_ff, bias=False)
        self.w2 = nn.Linear(c.d_ff, c.d_model, bias=False)

    def forward(self, x):
        h = self.n(x)
        return x + self.w2(F.silu(self.w1(h)) * self.w3(h))


class Block(nn.Module):
    def __init__(self, c: Cfg, dvs=None):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(c.d_model), nn.LayerNorm(c.d_model)
        self.mix = AttnMixer(c) if getattr(c, "mixer", "mem") == "attn" else MemMixer(c, dvs)
        self.use_moe = bool(getattr(c, "moe", 0))
        if getattr(c, "wmoe", 0) and not self.use_moe:
            raise ValueError("wmoe=1 requires moe=1: the write experts are routed by "
                             "the FFN router, so there must be one")
        if self.use_moe:
            self.ffn = MoE(c)
        else:
            self.w1 = nn.Linear(c.d_model, c.d_ff, bias=False)
            self.w3 = nn.Linear(c.d_model, c.d_ff, bias=False)
            self.w2 = nn.Linear(c.d_ff, c.d_model, bias=False)

    def forward(self, x, state=None, return_state=False, bus=None, conv_prev=None):
        _mem = isinstance(self.mix, MemMixer)
        h1 = self.n1(x)
        route = None
        if self.use_moe and getattr(self.mix, "wmoe", 0):
            # ONE routing decision per block, taken on n1(x) so the memory write can
            # use it, then reused by the FFN below.
            route = self.ffn.route(h1.reshape(-1, h1.shape[-1]))
        kw = {"bus": bus, "conv_prev": conv_prev, "route": route} if _mem else {}
        ncp = None
        if return_state:
            if _mem:
                d, final, ncp = self.mix(h1, state, True, **kw)
            else:
                d, final = self.mix(h1, state, True)
        else:
            d, final = self.mix(h1, state, False, **kw), None
        x = x + d
        h = self.n2(x)
        x = x + (self.ffn(h, route=route) if self.use_moe else self.w2(F.silu(self.w1(h)) * self.w3(h)))
        return (x, final, ncp) if return_state else x


class MemLM(nn.Module):
    def __init__(self, c: Cfg):
        super().__init__()
        self.c = c
        self.emb = nn.Embedding(c.vocab, c.d_model)
        _plan = c.plan if int(getattr(c, "kslot", 0)) else None
        self.blocks = nn.ModuleList([Block(c, (_plan[i] if _plan else None)) for i in range(c.layers)])
        if int(getattr(c, "kslot", 0)) and int(getattr(c, "shadow_shared", 0)):
            self.shadow = ShadowProj(self.blocks[0].mix)
            self.shadow_n1 = nn.LayerNorm(c.d_model)
        self.pre_blocks = nn.ModuleList(
            [PreFFN(c) for _ in range(int(getattr(c, 'pre', 0)))])
        self.nf = nn.LayerNorm(c.d_model)
        self.head = nn.Linear(c.d_model, c.vocab, bias=False)
        self.head.weight = self.emb.weight                 # tied
        # --- full-state compressor bus (bus=2) --------------------------------
        # ONE compressor shared by every layer. It sees each layer's ENTIRE state matrix
        # -- not R projections of it -- and distils it to a slot vector. It runs once per
        # chunk, so the expensive state-dependent work is amortised over the chunk while
        # the per-token step is only a softmax over L slots.
        # dv is contracted FIRST rather than flattening (dk,dv), which would need a
        # 16k-wide first layer (~4M params PER layer); this keeps the whole thing ~0.7M.
        if int(getattr(c, "bus", 0)) in (7, 8, 9, 10, 11):
            # PER-TOKEN cross-layer bus over LOWER layers, no chunk loop, no staleness.
            # Layer i>0 reads the per-token state streams of layers 0..i-1 (materialised
            # by the Triton stream kernel, stop-grad content): a small brain indexes each
            # head state, the token picks top-gtop heads, and the fold is read in FULL
            # (all dk*dv numbers) through one shared projection. Selection params learn;
            # what is stored does not get a bus gradient (documented tradeoff).
            _ix = int(getattr(c, "gidx", 64))
            self.b7_P = nn.Parameter(torch.randn(c.dv, 8) * (c.dv ** -0.5))
            self.b7_W1 = nn.Linear(c.dk * 8, 256, bias=False)
            self.b7_W2 = nn.Linear(256, _ix, bias=False)
            self.b7_norm = nn.LayerNorm(_ix)
            self.b7_q = nn.Parameter(torch.randn(c.layers, c.d_model, _ix) * 0.02)
            self.b7_out = nn.Linear(c.dk * c.dv, c.d_model, bias=False)
            self.b7_gate = nn.Parameter(torch.zeros(c.layers))
        if int(getattr(c, "bus", 0)) in (8, 9, 10, 11):
            # Bigger shared brain (~1M hidden params) for indexing lower states,
            # and a SELF-REFRESH path: each layer sees its OWN full state (all H
            # heads, all dk*dv numbers, at t-1, causal) and can re-emit decaying
            # content into v -- write new things AND put back what was there.
            _ix8 = int(getattr(c, "gidx", 64))
            self.b8_P = nn.Parameter(torch.randn(c.dv, 16) * (c.dv ** -0.5))
            self.b8_W1 = nn.Linear(c.dk * 16, 768, bias=False)
            self.b8_W2 = nn.Linear(768, 256, bias=False)
            self.b8_W3 = nn.Linear(256, _ix8, bias=False)
            self.b8_r = nn.Linear(c.dk * c.dv, 64, bias=False)      # per-head reduce
            self.b8_h1 = nn.Linear(c.heads * 64, 512, bias=False)
            self.b8_h2 = nn.Linear(512, c.heads * c.dv, bias=False)
            self.b8_vgate = nn.Parameter(torch.zeros(c.layers))     # refresh gate
        if int(getattr(c, "bus", 0)) in (9, 10, 11):
            # STATE-AWARE DECIDER: a summary of the layer's own full state (all H
            # heads) is injected into the layer INPUT, so k/beta/g/v -- where, how
            # strongly, how long, what -- are all decided knowing what is stored.
            assert c.nlk == 1 and not (c.kcenter or c.wkey or c.wortho
                                       or c.wfreeze or c.kdecor or c.wpreserve), \
                "bus=9 recompute path assumes the plain nlk=1 write projections"
            self.b9_r = nn.Linear(c.dk * c.dv, 128, bias=False)
            self.b9_h = nn.Linear(c.heads * 128, c.d_model, bias=False)
            self.b9_g = nn.Parameter(torch.zeros(c.layers))
        if int(getattr(c, "bus", 0)) == 12:
            # SHADOW-GIST, fully differentiable, normal parallel training.
            # Each layer's write projections scan the token stream (shadow states);
            # per-token state features come out as STATIC-query reads through the
            # SAME fla kernel (2 queries/head = 64 feats/head, linear reads of S_t);
            # shift t-1 -> MLP brain -> one injection at layer 0. No softmax, no
            # stop-grad: the loss teaches the writes what to store.
            self.b12_q = nn.Parameter(torch.randn(c.layers, c.heads, 2, c.dk) * 0.02)
            self.b12_h = nn.Linear(c.layers * c.heads * 2 * c.dv, 1024, bias=False)
            self.b12_o = nn.Linear(1024, c.d_model, bias=False)
            self.b12_g = nn.Parameter(torch.zeros(()))
        if int(getattr(c, "bus", 0)) == 13:
            # SHADOW-GIST v2: the compressor sees ALL the state of ALL heads.
            # Chunked shadow scans; at every chunk boundary fla returns the FULL
            # state matrix (differentiably) -> per-head Linear(dk*dv -> 64), the
            # true serial reduce -- no write is ever invisible to the gist
            # (<=chunk stale). A fresh per-token channel (2 static reads/head)
            # keeps immediacy. Main stack unchunked; everything gets gradients.
            self.b13_q = nn.Parameter(torch.randn(c.layers, c.heads, 2, c.dk) * 0.02)
            self.b13_red = nn.Linear(c.dk * c.dv, 64, bias=False)
            self.b13_h = nn.Linear(c.layers * c.heads * (2 * c.dv + 64), 1024,
                                   bias=False)
            self.b13_o = nn.Linear(1024, c.d_model, bias=False)
            self.b13_g = nn.Parameter(torch.zeros(()))
        if int(getattr(c, "bus", 0)) == 14:
            # FULL per-token gist: the Triton delta_stream kernel emits every
            # head's complete state at every token, DIFFERENTIABLY. The true
            # compressor (per-head reduce over all dk*dv entries) sees all 98k
            # numbers at t-1, once per token. No slice, no staleness, no
            # severed gradient. Cost is the stream emission (~12ms/layer).
            _gr = int(getattr(c, "gistr", 64))
            _gh = int(getattr(c, "gisth", 1024))
            if _gr == 0:
                # DENSE mode: the raw concatenated states (all L*H*dk*dv numbers)
                # feed the compressor first layer directly -- no per-head reduce,
                # cross-head features form from raw entries. ~100M params at
                # gisth=1024; the unfactorized compressor in its pure form.
                self.b14_red = None
                self.b14_h = nn.Linear(c.layers * c.heads * c.dk * c.dv, _gh,
                                       bias=False)
            else:
                self.b14_red = nn.Linear(c.dk * c.dv, _gr, bias=False)
                self.b14_h = nn.Linear(c.layers * c.heads * _gr, _gh, bias=False)
            self.b14_o = nn.Linear(_gh, c.d_model, bias=False)
            self.b14_g = nn.Parameter(torch.zeros(()))
        if int(getattr(c, "bus", 0)) == 16:
            # HUB-CONDITIONED compressor (p22): b14 plus a dedicated 49th "hub"
            # head. The hub is an ordinary delta head written from the token
            # stream; its per-token full state is gisted to an ask vector that
            # tells every head's reducer what the short-term context wants --
            # retrieve, don't guess. The reducer also gets 2 literal dot-product
            # reads (state^T q, queries formed from the ask) so content
            # addressing is available multiplicatively, not just by concat.
            # Ask, reads, and hub readout are all zero-gated: the model starts
            # as b14-with-a-deeper-reducer and must learn to open each channel.
            _gr = int(getattr(c, "gistr", 64))
            _gh = int(getattr(c, "gisth", 1024))
            _aw = int(getattr(c, "askw", 500))
            self.hub_k = nn.Linear(c.d_model, c.dk, bias=False)
            self.hub_v = nn.Linear(c.d_model, c.dv, bias=False)
            self.hub_b = nn.Linear(c.d_model, 1, bias=True)
            self.hub_gd = nn.Linear(c.d_model, 1, bias=True)
            self.hub_gd.bias.data.fill_(-4.5)      # start half-life ~64 tok
            self.hub_q = nn.Linear(c.d_model, c.dk, bias=False)
            self.hub_o = nn.Linear(c.dv, c.d_model, bias=False)
            self.hub_og = nn.Parameter(torch.zeros(()))
            self.ask = nn.Linear(c.dk * c.dv, _aw, bias=False)
            self.askq = nn.Linear(_aw, 2 * c.dk, bias=False)
            self.b16_ga = nn.Parameter(torch.zeros(()))
            self.b16_gq = nn.Parameter(torch.zeros(()))
            _gf = [int(w) for w in str(getattr(c, "gistf", "")).split(",") if w.strip()]
            if int(getattr(c, "gist3", 0)):
                # gist3: hub-driven. Per head the funnel sees [La*NQ*dv reads | ask]; the
                # raw-state path is gone and the hub also enters densely (see _gist_funnel3).
                import math as _m
                assert int(getattr(c, "gist2", 0)), "gist3 needs gist2=1 (raw-state emission)"
                _La3 = max(1, int(getattr(c, "lanes", 1)))
                _NH3 = (c.layers * c.heads) // _La3
                _NQ3 = int(getattr(c, "gist3_nq", 8))
                _w3 = (_NQ3 * max(max(ws) for ws in c.plan) if int(getattr(c, "kslot", 0)) else _La3 * _NQ3 * c.dv) + _aw
                _ws3 = [_w3] + _gf + [_gr]
                self.b16_pair = None
                self.b16_rf = None
                self.b16_pw = nn.ParameterList([
                    nn.Parameter(torch.empty(_NH3 + 1, _ws3[i], _ws3[i + 1])
                                 .uniform_(-1.0 / _m.sqrt(_ws3[i]), 1.0 / _m.sqrt(_ws3[i])))
                    for i in range(len(_ws3) - 1)])
                self.askq3 = nn.Linear(_aw, _NQ3 * c.dk, bias=False)
                self.b16_hub = nn.Linear(c.dk * c.dv, _ws3[1], bias=False)
                self.b16_r1 = self.b16_r2 = self.b16_r3 = None
                _npage = (_NH3 + 1) * _gr
            elif int(getattr(c, "gist2", 0)):
                # gist2: one funnel per LOGICAL HEAD (La lanes joined) + the hub.
                # Input = [La*dk*dv raw state | ask | La*2*dv reads]; no 2:1 stage.
                import math as _m
                _La2 = max(1, int(getattr(c, "lanes", 1)))
                _NH2 = ((c.heads if int(getattr(c, "shadow_shared", 0)) else c.layers * c.heads)) // _La2   # shadow groups: one layer's worth when shared
                _w2 = _La2 * c.dk * c.dv + _aw + _La2 * 2 * c.dv
                _ws2 = [_w2] + _gf + [_gr]
                self.b16_pair = None
                self.b16_rf = None
                _fs = int(getattr(c, "gist_sparse", 0))
                if int(getattr(c, "gist_bs", 0)):
                    # block-sparse: each 32-wide output block reads gist_nb random 32-wide input blocks, wiring drawn once
                    # from gist_seed and kept in the state_dict. Stage 1 input = [state La*dk*dv | reads La*2*dv | ask padded to 32].
                    _nb = int(getattr(c, "gist_nb", 5)); _sd = int(getattr(c, "gist_seed", 0))
                    assert len(_gf) == 2 and all(w % 32 == 0 for w in _gf), "gist_bs needs two 32-multiple funnel widths, e.g. gistf 512,128"
                    _awp = ((_aw + 31) // 32) * 32
                    _k1 = _La2 * c.dk * c.dv + _La2 * 2 * c.dv + _awp
                    _i1 = make_wiring(_NH2, _k1, _gf[0], _nb, _sd); _i2 = make_wiring(_NH2 + 1, _gf[0], _gf[1], _nb, _sd + 1)
                    self.register_buffer("b16_bs1_idx", _i1); self.register_buffer("b16_bs1_inv", inverse_map(_i1, _k1))
                    self.register_buffer("b16_bs2_idx", _i2); self.register_buffer("b16_bs2_inv", inverse_map(_i2, _gf[0]))
                    _bd = 1.0 / _m.sqrt(_nb * 32)
                    self.b16_bs1_w = nn.Parameter(torch.empty(_NH2, _gf[0] // 32, _i1.shape[2], 32, 32).uniform_(-_bd, _bd))
                    self.b16_bs2_w = nn.Parameter(torch.empty(_NH2 + 1, _gf[1] // 32, _i2.shape[2], 32, 32).uniform_(-_bd, _bd))
                    _hin = c.dk * c.dv + 2 * c.dv + _awp                     # the hub group: dense (its store | its reads | ask)
                    self.b16_bs1_hub = nn.Parameter(torch.empty(_hin, _gf[0]).uniform_(-1.0 / _m.sqrt(_hin), 1.0 / _m.sqrt(_hin)))
                    self.b16_bs_awp = _awp
                    self.b16_pws = None
                    self.b16_pw = nn.ParameterList([nn.Parameter(torch.empty(_NH2 + 1, _gf[1], _gr).uniform_(-1.0 / _m.sqrt(_gf[1]), 1.0 / _m.sqrt(_gf[1])))])
                elif _fs > 0:
                    # every stage but the last is fixed-random-sparse: 500 and 128 units wired to `_fs`
                    # inputs each; the final stage to the 64-wide page slot stays dense ("full")
                    self.b16_pws = nn.ModuleList([FixedSparse(_NH2 + 1, _ws2[i], _ws2[i + 1], _fs, int(getattr(c, "gist_seed", 0)) + i)
                                                  for i in range(len(_ws2) - 2)])
                    self.b16_pw = nn.ParameterList([
                        nn.Parameter(torch.empty(_NH2 + 1, _ws2[-2], _ws2[-1])
                                     .uniform_(-1.0 / _m.sqrt(_ws2[-2]), 1.0 / _m.sqrt(_ws2[-2])))])
                else:
                    self.b16_pws = None
                    self.b16_pw = nn.ParameterList([
                        nn.Parameter(torch.empty(_NH2 + 1, _ws2[i], _ws2[i + 1])
                                     .uniform_(-1.0 / _m.sqrt(_ws2[i]),
                                               1.0 / _m.sqrt(_ws2[i])))
                        for i in range(len(_ws2) - 1)])
                self.b16_r1 = self.b16_r2 = self.b16_r3 = None
                _npage = (_NH2 + 1) * _gr
            elif _gf:
                # FUNNEL reducer (p23): compute BEFORE the shrink. The reducer
                # is SHARED across all L*H+1 heads, so width is cheap in params
                # (500,128,128 costs ~1.4M once, not per head). A wide first
                # stage lets the ask-conditioned selection compute while the
                # information is still wide; one-shot 2.6k->64 wastes any
                # post-bottleneck layers (the code is already lossy).
                _w0 = c.dk * c.dv + _aw + 2 * c.dv
                self.b16_pair = None
                if int(getattr(c, "gistpair", 0)):
                    # pairwise 2:1 front stage: out[j] = w0*in[2j] + w1*in[2j+1]
                    assert _w0 % 2 == 0, "pairwise stage needs an even input width"
                    _nh0 = c.layers * c.heads + 1
                    self.b16_pair = nn.Parameter(
                        torch.empty(_nh0, _w0 // 2, 2).uniform_(-0.7071, 0.7071))
                    _w0 = _w0 // 2
                _ws = [_w0] + _gf + [_gr]
                if int(getattr(c, "gistph", 0)):
                    # PER-HEAD funnels (p24): every head owns its funnel
                    # weights. A shared reducer must stay general across all
                    # 49 heads; a private one can specialise to its head's
                    # state statistics, so it affords to be narrower.
                    # 128,128 costs ~359k/head x 49 = 17.6M params but LESS
                    # compute than the shared 500-funnel (batched einsum).
                    _nh = c.layers * c.heads + 1
                    import math as _m
                    self.b16_pw = nn.ParameterList([
                        nn.Parameter(torch.empty(_nh, _ws[i], _ws[i + 1])
                                     .uniform_(-1.0 / _m.sqrt(_ws[i]),
                                               1.0 / _m.sqrt(_ws[i])))
                        for i in range(len(_ws) - 1)])
                    self.b16_rf = None
                else:
                    self.b16_pw = None
                    self.b16_rf = nn.ModuleList(
                        [nn.Linear(_ws[i], _ws[i + 1], bias=False)
                         for i in range(len(_ws) - 1)])
                self.b16_r1 = self.b16_r2 = self.b16_r3 = None
            else:
                self.b16_rf = None
                self.b16_r1 = nn.Linear(c.dk * c.dv + _aw + 2 * c.dv, _gr, bias=False)
                self.b16_r2 = nn.Linear(_gr, _gr, bias=False)
                self.b16_r3 = nn.Linear(_gr, _gr, bias=False)
            _pin = _npage if int(getattr(c, 'gist2', 0)) or int(getattr(c, 'gist3', 0)) else (c.layers * c.heads + 1) * _gr
            if int(getattr(c, "gist_bs", 0)):
                self.b16_h = BlockSparsePage(_pin, _gh, int(getattr(c, "gist_nb", 5)), int(getattr(c, "gist_seed", 0)) + 7)
            elif int(getattr(c, "gist_sparse", 0)) > 0:
                self.b16_h = FixedSparse(1, _pin, _gh, int(c.gist_sparse), int(getattr(c, "gist_seed", 0)) + 7)   # page -> layers, sparse
            else:
                self.b16_h = nn.Linear(_pin, _gh, bias=False)
            self.b16_o = nn.Linear(_gh, c.d_model, bias=False)
            self.b16_g = nn.Parameter(torch.zeros(()))
            self.b16_ginorm = int(getattr(c, "ginorm", 0))
            if int(getattr(c, "ginjall", 0)):
                # EVERY-LAYER injection (m300): one gist per token, injected
                # before every block through per-layer zero-init gates -- each
                # depth pulls what it needs directly instead of relaying the
                # layer-0 injection through the residual stream.
                self.b16_lg = nn.Parameter(torch.zeros(c.layers))
            if int(getattr(c, "ginjall", 0)) >= 2:
                # PER-LAYER INJECTION PROJECTIONS (m300 v2, user design): the
                # per-head compressor and shared trunk are unchanged; each
                # layer OWNS its final projection of the gist into its
                # residual (KV-cache analogy: a different view per layer), so
                # layers can specialise what they read from the same state.
                import math as _m
                self.b16_ol = nn.Parameter(
                    torch.empty(c.layers, _gh, c.d_model)
                    .uniform_(-1.0 / _m.sqrt(_gh), 1.0 / _m.sqrt(_gh)))
            if int(getattr(c, "ginjall", 0)) >= 3:
                # v3 (user): EVERY layer of the model, pre-encoder included --
                # the two pre blocks get their own projections and zero-init
                # gates; the pre stage is re-encoded with injections after the
                # gist is computed (two-pass, bus=11-style approximation).
                import math as _m2
                _np = max(1, int(getattr(c, "pre", 0)))
                self.b16_olp = nn.Parameter(
                    torch.empty(_np, _gh, c.d_model)
                    .uniform_(-1.0 / _m2.sqrt(_gh), 1.0 / _m2.sqrt(_gh)))
                self.b16_plg = nn.Parameter(torch.zeros(_np))
        if int(getattr(c, "bus", 0)) == 15:
            # GROUP-FETCH compressor: the model DECIDES what to connect, per token.
            # State = 1,536 groups of 64 numbers (2 key-rows each). A shared 1k-param
            # filter describes every group; a per-token query picks top-K; the RAW
            # numbers of the winners (rank-slotted, weight-scaled, descriptor-tagged)
            # feed a small funnel. Full visibility to the decision, raw fidelity for
            # the fetch, ~1M params instead of 25M-4B of wiring.
            self.b15_d = nn.Parameter(torch.randn(64, 16) * 0.05)
            self.b15_q = nn.Parameter(torch.randn(c.d_model, 16) * 0.02)
            _K15 = int(getattr(c, "gtop", 24))
            self.b15_o = nn.Linear(_K15 * 80, c.d_model, bias=False)
            self.b15_g = nn.Parameter(torch.zeros(()))
        if int(getattr(c, "bus", 0)) == 4:
            # GLOBAL STATE. Every layer's write path sees ALL L*H head states; the read
            # path picks the top-gtop heads and queries their FULL states. Update stays
            # the standard delta rule -- only what the write path SEES changes.
            _S = c.layers * c.heads
            _m = int(getattr(c, "bus_m", 8)); _ix = int(getattr(c, "gidx", 64))
            _R = int(getattr(c, "gread", 4))
            self.g_P = nn.Parameter(torch.randn(c.dv, _m) * (c.dv ** -0.5))
            self.g_W = nn.Linear(c.dk * _m, _ix, bias=False)      # per-head index
            self.g_norm = nn.LayerNorm(_ix)
            self.gw_q = nn.Parameter(torch.randn(c.layers, c.d_model, _ix) * 0.02)
            self.gw_r = nn.Parameter(torch.randn(c.layers, c.d_model, c.dk) * 0.02)
            self.gw_out = nn.Linear(c.dv, c.d_model, bias=False)
            self.gr_q = nn.Parameter(torch.randn(c.layers, c.d_model, _ix) * 0.02)
            self.gr_r = nn.Parameter(torch.randn(c.layers, c.d_model, _R * c.dk) * 0.02)
            self.gr_out = nn.Linear(_R * c.dv, c.d_model, bias=False)
            self.gw_gate = nn.Parameter(torch.zeros(c.layers))
            self.gr_gate = nn.Parameter(torch.zeros(c.layers))
        if int(getattr(c, "bus", 0)) == 3:
            # SELECT-THEN-QUERY. The compressor is demoted to an INDEX: it only decides
            # which head to listen to. The value is q . S against that head's FULL state,
            # so nothing is compressed away and which part of the state emerges depends on
            # the token. A fixed projection can only ever expose the same 1/64 of a head no
            # matter the context; this exposes whatever the query asks for.
            _m = int(getattr(c, "bus_m", 8))
            self._idx = int(getattr(c, "bus_slots", 256)) // c.heads
            self._sd = self._idx      # _compress writes slots of this width
            self._R = int(getattr(c, "bus_r", 1))
            self._S = c.heads
            self.cmp_P = nn.Parameter(torch.randn(c.dv, _m) * (c.dv ** -0.5))
            self.cmp_W = nn.Linear(c.dk * _m, self._idx, bias=False)     # index only
            self.cmp_norm = nn.LayerNorm(self._idx)
            self.cmp_sel = nn.Parameter(torch.randn(c.layers, c.d_model, self._idx) * 0.02)
            # The read query is the widest matrix, so SHARE it across layers and give
            # each layer a cheap diagonal gain to specialise. Per-layer copies would
            # cost 12x and blow the budget; selection (cmp_sel) stays per-layer.
            self.cmp_rd = nn.Parameter(torch.randn(c.d_model, self._R * c.dk) * 0.02)
            self.cmp_rg = nn.Parameter(torch.ones(c.layers, self._R * c.dk))
            self.cmp_out = nn.Linear(self._R * c.dv, c.d_model, bias=False)
            self.bus_g = nn.Parameter(torch.zeros(c.layers))
        if int(getattr(c, "bus", 0)) == 2:
            _m = int(getattr(c, "bus_m", 8)); _ds = int(getattr(c, "bus_slots", 256))
            # Slots are PER HEAD, not per layer. Half-life is bimodal *within* a layer
            # (measured: L5 heads span 1.7 to 5,205,003 tokens), so flattening the heads
            # together blends an always-on mixer with a long-term memory before the token
            # can choose between them. One slot per head keeps them selectable.
            self._M = int(getattr(c, "bus_nslot", 1))          # slots per HEAD
            self._sd = _ds // (c.heads * self._M)              # width of one slot
            assert self._sd > 0, "bus_slots too small for heads*bus_nslot"
            self._S = c.heads * self._M                        # slots per layer
            self.cmp_P = nn.Parameter(torch.randn(c.dv, _m) * (c.dv ** -0.5))
            self.cmp_W = nn.Linear(c.dk * _m, self._M * self._sd, bias=False)
            # CONTENT-based selection: a token-derived query is scored against what is
            # actually IN each slot, over all layers*slots -- so a token can attend to
            # whichever slot happens to hold something relevant. The earlier version
            # scored slots from the token alone and could only pick a LAYER, never
            # noticing what that layer's memory contained.
            self.cmp_q = nn.Parameter(torch.randn(c.layers, c.d_model, self._sd) * 0.02)
            self.cmp_out = nn.Linear(self._sd, c.d_model, bias=False)
            # Slot norms span 25x across layers (measured), and softmax over raw dot
            # products just picks the loudest -- which made 8 of 12 layers' memories
            # unreadable no matter what they contained. Normalising each slot makes
            # the choice directional, i.e. actually about content.
            self.cmp_norm = nn.LayerNorm(self._sd)
            self.bus_g = nn.Parameter(torch.zeros(c.layers))   # zero -> starts == baseline
        self.apply(self._init)
        if int(getattr(c, "wkey", 0)):
            for b in self.blocks:                       # must run AFTER self.apply
                mx = getattr(b, "mix", None)
                if hasattr(mx, "occ_mlp"):
                    nn.init.zeros_(mx.occ_mlp[2].weight); nn.init.zeros_(mx.occ_mlp[2].bias)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def _gindex(self, Sf):
        """(S,B,dk,dv) head states -> (S,B,gidx) index vectors. Once per chunk."""
        dt = self.g_W.weight.dtype
        z = torch.einsum('sbkv,vm->sbkm', Sf.to(dt), self.g_P.to(dt))
        return self.g_norm(self.g_W(z.reshape(Sf.shape[0], Sf.shape[1], -1)))

    def _compress(self, bus):
        """(L,B,H,dk,dv) states -> (L,B,d_s) slots. Once per chunk, not per token."""
        # fla returns final_state in fp32 even for a bf16 model, so cast to the
        # compressor's dtype up front rather than inheriting the state's.
        dt = self.cmp_W.weight.dtype
        Sb = torch.stack(list(bus), 0).to(dt)
        Lc, Bc, Hc = Sb.shape[0], Sb.shape[1], Sb.shape[2]
        z = torch.einsum('lbhkv,vm->lbhkm', Sb, self.cmp_P.to(dt))
        z = F.gelu(self.cmp_W(z.reshape(Lc, Bc, Hc, -1)))      # each head compressed alone
        return self.cmp_norm(z.view(Lc, Bc, self._S, self._sd))   # (L,B,H*M,d)

    def vw_refresh(self):
        for b in self.blocks:
            mx = getattr(b, "mix", None)
            if mx is not None and hasattr(mx, "vw_refresh"):
                mx.vw_refresh()

    def set_moe_loop(self, flag: bool):
        if getattr(self.c, "moe", 0):
            for b in self.blocks:
                b.ffn.use_loop = flag

    def _funnel_weights(self, dt):
        """Dense (NH+1, in, out) matrices for _gist_funnel2: sparse stages expanded from their compact weights."""
        ws = [mod.dense(dt) for mod in self.b16_pws] if getattr(self, "b16_pws", None) is not None else []
        return ws + [p for p in self.b16_pw]

    def _page(self, ft6):
        if isinstance(self.b16_h, BlockSparsePage): return self.b16_h(ft6)
        return self.b16_h(ft6.unsqueeze(-2)).squeeze(-2) if isinstance(self.b16_h, FixedSparse) else self.b16_h(ft6)

    def _inject(self, ft6, dt):
        """b16_o(gelu(page(ft6))) -- the injection trunk, through whichever page implementation is live."""
        if isinstance(self.b16_h, BlockSparsePage): return F.linear(F.gelu(self._page(ft6)).to(dt), self.b16_o.weight)
        return _gist_inject(ft6, self._page_weight(dt), self.b16_o.weight)

    def _bs_pack(self):
        return (self.b16_bs1_w, self.b16_bs1_idx, self.b16_bs1_inv, self.b16_bs1_hub, self.b16_bs2_w, self.b16_bs2_idx, self.b16_bs2_inv, self.b16_pw[0], self.b16_bs_awp)

    def _page_weight(self, dt):
        return self.b16_h.dense(dt)[0].t() if isinstance(self.b16_h, FixedSparse) else self.b16_h.weight

    def forward(self, idx, states=None, return_states=False, skip_head=False):
        x = self.emb(idx)
        _x0 = x
        for _pb in self.pre_blocks:
            x = _pb(x)
        finals = []
        _bm = int(getattr(self.c, "bus", 0))
        _g6inj = None
        # state-aware keys need a chunk-start state, so they require the chunked loop
        _need_state = (int(getattr(self.c, "wkey", 0)) == 1
                       or int(getattr(self.c, "wortho", 0)) > 0
                       or int(getattr(self.c, "wpreserve", 0)) > 0)
        if (_bm and _bm not in (7, 8, 9, 10, 11, 12, 13, 14, 15, 16)) or _need_state:   # 7..15 unchunked
            # Chunked cross-layer state bus. Inside chunk c every layer reads the states of
            # ALL layers as they stood at the END of chunk c-1 -- strictly in the past, so
            # higher layers are readable too and no future token can leak backwards.
            # Reading per-token instead would force all layers to advance one token at a
            # time, serialising depth and forfeiting the chunked scan entirely.
            B, T, _ = x.shape
            C = max(1, int(getattr(self.c, "chunk", 256)))
            L = len(self.blocks)
            st = list(states) if states is not None else [None] * L
            cp = [None] * L                # per-layer causal-conv left context
            bus, parts = None, []
            for c0 in range(0, T, C):
                xc = x[:, c0:c0 + C]
                Z = self._compress(bus) if (_bm in (2, 3) and bus is not None) else None
                Sb = (torch.stack(list(bus), 0) if (_bm == 3 and bus is not None) else None)
                Sf = gidx_ = None
                if _bm == 4 and bus is not None:
                    _st = torch.stack(list(bus), 0)                 # (L,B,H,dk,dv)
                    Sf = _st.reshape(-1, _st.shape[1], _st.shape[-2], _st.shape[-1])
                    gidx_ = self._gindex(Sf)                        # (S,B,gidx) once per chunk
                nxt = []
                for i, b in enumerate(self.blocks):
                    if _bm == 4:
                        Bq, Tq, dt4 = xc.shape[0], xc.shape[1], xc.dtype
                        if Sf is not None:
                            Sfd = Sf.to(dt4); ixd = gidx_.to(dt4)
                            # (S,B,dk,dv) -> (B,S,dk*dv) so the fold is a bmm.
                            # einsum('bts,sbkv->btkv') materialises (B,T,S,dk,dv)
                            # = 1.6GB here; bmm contracts without the intermediate.
                            _dk4, _dv4 = Sfd.shape[-2], Sfd.shape[-1]
                            Sm = Sfd.permute(1, 0, 2, 3).reshape(Bq, Sfd.shape[0], -1)
                            sc4 = float(self.c.gidx) ** -0.5
                            # --- WRITE path sees ALL head states ---------------
                            qw = torch.einsum('btd,de->bte', xc, self.gw_q[i].to(dt4))
                            ww = torch.softmax(torch.einsum('bte,sbe->bts', qw, ixd) * sc4, -1)
                            Aw = torch.bmm(ww, Sm).view(Bq, Tq, _dk4, _dv4)
                            qwr = torch.einsum('btd,dk->btk', xc, self.gw_r[i].to(dt4))
                            rw = torch.einsum('btk,btkv->btv', qwr, Aw)
                            xc = xc + self.gw_gate[i].to(dt4) * self.gw_out(rw)
                        xc, f, cp[i] = b(xc, st[i], True, conv_prev=cp[i])
                        if Sf is not None and int(self.c.gread) > 0:
                            # --- READ path: top-gtop heads, FULL state each ----
                            # gread=0 disables it, isolating the write-side change
                            qq4 = torch.einsum('btd,de->bte', xc, self.gr_q[i].to(dt4))
                            sr = torch.einsum('bte,sbe->bts', qq4, ixd) * sc4
                            kk4 = min(int(self.c.gtop), sr.shape[-1])
                            thr = sr.topk(kk4, dim=-1).values[..., -1:]
                            wr = torch.softmax(sr.masked_fill(sr < thr, float('-inf')), -1)
                            Ar = torch.bmm(wr, Sm).view(Bq, Tq, _dk4, _dv4)
                            qr4 = torch.einsum('btd,de->bte', xc, self.gr_r[i].to(dt4))
                            qr4 = qr4.view(Bq, Tq, int(self.c.gread), -1)
                            rr = torch.einsum('btrk,btkv->btrv', qr4, Ar).reshape(Bq, Tq, -1)
                            xc = xc + self.gr_gate[i].to(dt4) * self.gr_out(rr)
                    elif _bm == 3:
                        xc, f, cp[i] = b(xc, st[i], True, conv_prev=cp[i])
                        if Z is not None:
                            Bq, Tq = xc.shape[0], xc.shape[1]
                            dt2 = xc.dtype
                            zc = Z.to(dt2)                       # (L,B,H,idx)
                            qs = torch.einsum('btd,de->bte', xc, self.cmp_sel[i].to(dt2))
                            sc = torch.einsum('bte,lbse->btls', qs, zc) * (self._idx ** -0.5)
                            w = torch.softmax(sc.reshape(Bq, Tq, -1), dim=-1)
                            S2 = Sb.reshape(-1, Bq, Sb.shape[-2], Sb.shape[-1]).to(dt2)
                            qr = torch.einsum('btd,de->bte', xc, self.cmp_rd.to(dt2))
                            qr = qr * self.cmp_rg[i].to(dt2)
                            qr = qr.view(Bq, Tq, self._R, -1)
                            # query the FULL state of every head, then weight by selection
                            # Fold the selection into a weighted state FIRST, then read
                            # it. Materialising (B,T,S,R,dv) instead costs ~19GB at R=8;
                            # this is (B,T,dk,dv) and is independent of R. Same result.
                            A = torch.einsum('bts,sbkv->btkv', w, S2)
                            o2 = torch.einsum('btrk,btkv->btrv', qr, A).reshape(Bq, Tq, -1)
                            contrib = self.bus_g[i].to(dt2) * self.cmp_out(o2)
                            if getattr(self, "_bus_dbg", None) is not None:
                                self._bus_dbg.append((i, w.detach().view(Bq, Tq, Z.shape[0], -1),
                                                      contrib.detach(), xc.detach()))
                            xc = xc + contrib
                    elif _bm == 2:
                        xc, f, cp[i] = b(xc, st[i], True, conv_prev=cp[i])
                        if Z is not None:
                            Zc = Z.to(xc.dtype)
                            qb, Tb = xc.shape[0], xc.shape[1]
                            qq2 = torch.einsum('btd,de->bte', xc, self.cmp_q[i].to(xc.dtype))
                            sc = torch.einsum('bte,lbme->btlm', qq2, Zc) * (self._sd ** -0.5)
                            w = torch.softmax(sc.reshape(qb, Tb, -1), dim=-1).view_as(sc)
                            r = torch.einsum('btlm,lbme->bte', w, Zc)
                            contrib = self.bus_g[i].to(xc.dtype) * self.cmp_out(r)
                            if getattr(self, "_bus_dbg", None) is not None:
                                self._bus_dbg.append((i, w.detach(), contrib.detach(),
                                                      xc.detach()))
                            xc = xc + contrib
                    else:
                        xc, f, cp[i] = b(xc, st[i], True, bus=bus, conv_prev=cp[i])
                    nxt.append(f)
                st = nxt
                bus = st                       # visible only to the NEXT chunk
                parts.append(xc)
            x = torch.cat(parts, dim=1)
            finals = st
        else:
            _b7 = (_bm in (7, 8, 9, 10, 11))
            if _bm in (8, 9, 10, 11):
                _H8, _dv8 = self.c.heads, self.c.dv
                def _sref(Sp):
                    z8 = self.b8_r(Sp.flatten(3))
                    h8 = F.gelu(self.b8_h1(z8.flatten(2)))
                    return self.b8_h2(h8).view(Sp.shape[0], Sp.shape[1], _H8, _dv8)
                if _bm in (9, 10, 11):
                    def _sinj9(Sp):
                        z9 = self.b9_r(Sp.flatten(3))
                        return self.b9_h(F.gelu(z9.flatten(2)))
            if _bm == 12 and _HAVE_FLA:
                _B12, _T12, _ = x.shape
                _H, _dk, _dv = self.c.heads, self.c.dk, self.c.dv
                fe = []
                for _i, _b in enumerate(self.blocks):
                    mx = _b.mix
                    hx = _b.n1(x)
                    f12 = mx.phi(hx).view(_B12, _T12, _H, _dk)
                    k12 = F.normalize(f12, dim=-1)
                    v12 = mx.wv(hx).view(_B12, _T12, _H, _dv)
                    b12 = torch.sigmoid(mx.wb(hx) + mx.bbias)
                    g12 = -F.softplus(mx.wg(hx).float())
                    # 2 static reads/head via pseudo-heads through the SAME kernel
                    k2 = k12.repeat_interleave(2, dim=2)
                    v2 = v12.repeat_interleave(2, dim=2)
                    b2 = b12.repeat_interleave(2, dim=2)
                    g2 = g12.repeat_interleave(2, dim=2)
                    q2 = self.b12_q[_i].reshape(1, 1, _H * 2, _dk).expand(
                        _B12, _T12, _H * 2, _dk).to(k2.dtype)
                    o12 = chunk_gated_delta_rule(q=q2, k=k2, v=v2, g=g2, beta=b2,
                                                 output_final_state=False)
                    if isinstance(o12, tuple):
                        o12 = o12[0]
                    fe.append(o12.reshape(_B12, _T12, -1))
                ft = torch.cat(fe, dim=-1)
                ft = torch.cat([ft.new_zeros(_B12, 1, ft.shape[-1]),
                                ft[:, :-1]], dim=1)          # strictly t-1
                x = x + self.b12_g * self.b12_o(F.gelu(self.b12_h(ft)))
            if _bm == 13 and _HAVE_FLA:
                _B3, _T3, _ = x.shape
                _H, _dk, _dv = self.c.heads, self.c.dk, self.c.dv
                _C = min(256, _T3)
                frs, bds = [], []
                for _i, _b in enumerate(self.blocks):
                    mx = _b.mix
                    hx = _b.n1(x)
                    f13 = mx.phi(hx).view(_B3, _T3, _H, _dk)
                    k13 = F.normalize(f13, dim=-1)
                    v13 = mx.wv(hx).view(_B3, _T3, _H, _dv)
                    be13 = torch.sigmoid(mx.wb(hx) + mx.bbias)
                    g13 = -F.softplus(mx.wg(hx).float())
                    k2 = k13.repeat_interleave(2, dim=2)
                    v2 = v13.repeat_interleave(2, dim=2)
                    b2 = be13.repeat_interleave(2, dim=2)
                    g2 = g13.repeat_interleave(2, dim=2)
                    q2 = self.b13_q[_i].reshape(1, 1, _H * 2, _dk).to(k2.dtype)
                    carry, fr_l, bd_l = None, [], []
                    for _c0 in range(0, _T3, _C):
                        sl3 = slice(_c0, _c0 + _C)
                        Tc = min(_C, _T3 - _c0)
                        out13 = chunk_gated_delta_rule(
                            q=q2.expand(_B3, Tc, _H * 2, _dk), k=k2[:, sl3],
                            v=v2[:, sl3], g=g2[:, sl3], beta=b2[:, sl3],
                            initial_state=carry, output_final_state=True)
                        o13, cnew = out13
                        fr_l.append(o13.reshape(_B3, Tc, -1))
                        # full state of the H real heads: pseudo-head pairs share
                        # identical k/v/g/beta, so take every 2nd state
                        Sfull = cnew.to(x.dtype)[:, ::2]          # (B,H,dk,dv)
                        bd_l.append(self.b13_red(Sfull.reshape(_B3, _H, -1)))
                        carry = cnew
                    fr = torch.cat(fr_l, dim=1)                    # (B,T,H*2*dv)
                    # boundary features: tokens of chunk c see chunk c-1's END state
                    bstack = torch.stack(bd_l, dim=1)              # (B,nc,H,64)
                    bstack = torch.cat([torch.zeros_like(bstack[:, :1]),
                                        bstack[:, :-1]], dim=1)
                    bd = bstack.repeat_interleave(_C, dim=1)[:, :_T3]
                    frs.append(fr)
                    bds.append(bd.reshape(_B3, _T3, -1))
                ft3 = torch.cat([torch.cat(frs, -1), torch.cat(bds, -1)], dim=-1)
                ft3 = torch.cat([ft3.new_zeros(_B3, 1, ft3.shape[-1]),
                                 ft3[:, :-1]], dim=1)              # strictly t-1
                x = x + self.b13_g * self.b13_o(F.gelu(self.b13_h(ft3)))
            if _bm == 15:
                from ktriton import delta_stream as _ds5
                _B5, _T5, _ = x.shape
                _H, _dk, _dv = self.c.heads, self.c.dk, self.c.dv
                st5 = []
                for _b in self.blocks:
                    mx = _b.mix
                    hx = _b.n1(x)
                    if getattr(mx, "nlk", 1):
                        k15 = F.normalize(mx.phi(hx).view(_B5, _T5, _H, _dk), dim=-1)
                    else:
                        k15 = F.normalize(mx.wk(hx).view(_B5, _T5, _H, _dk), dim=-1)
                    v15 = mx.wv(hx).view(_B5, _T5, _H, _dv)
                    be15 = torch.sigmoid(mx.wb(hx) + mx.bbias)
                    g15 = -F.softplus(mx.wg(hx).float())
                    _o5, S5 = _ds5(k15, k15, v15, g15, be15)
                    st5.append(S5.to(x.dtype).reshape(_B5, _T5, -1))
                G5 = torch.cat(st5, dim=-1).reshape(_B5, _T5, -1, 64)   # (B,T,1536,64)
                D5 = torch.einsum('btgn,nm->btgm', G5, self.b15_d.to(G5.dtype))
                q5 = torch.einsum('btd,dm->btm', x, self.b15_q.to(x.dtype))
                sc5 = torch.einsum('btm,btgm->btg', q5, D5) * (16 ** -0.5)
                _K = self.b15_o.weight.shape[1] // 80
                top5 = sc5.topk(_K, dim=-1)
                w5 = torch.softmax(top5.values, dim=-1)
                ixe = top5.indices.unsqueeze(-1)
                Gs = torch.gather(G5, 2, ixe.expand(-1, -1, -1, 64))
                Ds = torch.gather(D5, 2, ixe.expand(-1, -1, -1, 16))
                slot = torch.cat([Gs * w5.unsqueeze(-1), Ds], dim=-1)   # (B,T,K,80)
                ft5 = slot.reshape(_B5, _T5, -1)
                ft5 = torch.cat([ft5.new_zeros(_B5, 1, ft5.shape[-1]),
                                 ft5[:, :-1]], dim=1)                    # strictly t-1
                x = x + self.b15_g * self.b15_o(ft5)
            if _bm == 14:
                from ktriton import delta_stream as _dstr
                _B4, _T4, _ = x.shape
                _H, _dk, _dv = self.c.heads, self.c.dk, self.c.dv
                fe4 = []
                for _i, _b in enumerate(self.blocks):
                    mx = _b.mix
                    hx = _b.n1(x)
                    if getattr(mx, "nlk", 1):
                        k14 = F.normalize(mx.phi(hx).view(_B4, _T4, _H, _dk), dim=-1)
                    else:   # untied q/k (nlk=0): shadow keys from wk
                        k14 = F.normalize(mx.wk(hx).view(_B4, _T4, _H, _dk), dim=-1)
                    v14 = mx.wv(hx).view(_B4, _T4, _H, _dv)
                    if getattr(mx, 'vcenter', 0):
                        v14 = v14 - mx.vmu.to(v14.dtype)
                    if getattr(mx, 'vwhiten', 0):
                        v14 = torch.einsum('bthv,hvw->bthw',
                                           (v14.float() - mx.vmu2.float()), mx.vwm.float()).to(v14.dtype)
                    be14 = torch.sigmoid(mx.wb(hx) + mx.bbias)
                    g14 = -F.softplus(mx.wg(hx).float())
                    _o4, S4 = _dstr(k14, k14, v14, g14, be14)
                    if self.b14_red is None:
                        fe4.append(S4.to(x.dtype).reshape(_B4, _T4, -1))
                    else:
                        fe4.append(self.b14_red(S4.to(x.dtype).reshape(
                            _B4, _T4, _H, -1)).reshape(_B4, _T4, -1))
                ft4 = torch.cat(fe4, dim=-1)
                ft4 = torch.cat([ft4.new_zeros(_B4, 1, ft4.shape[-1]),
                                 ft4[:, :-1]], dim=1)          # strictly t-1
                x = x + self.b14_g * self.b14_o(F.gelu(self.b14_h(ft4)))
            if _bm == 16:
                from ktriton import delta_stream as _dstr
                _B6, _T6, _ = x.shape
                _H, _dk, _dv = self.c.heads, self.c.dk, self.c.dv
                # hub head: ordinary delta write from the token stream; its
                # readout joins the residual like any head (zero-gated).
                hk = F.normalize(self.hub_k(x).view(_B6, _T6, 1, _dk), dim=-1)
                hv = self.hub_v(x).view(_B6, _T6, 1, _dv)
                hbe = torch.sigmoid(self.hub_b(x)).view(_B6, _T6, 1)
                hg = (-F.softplus(self.hub_gd(x).float())).view(_B6, _T6, 1)
                hq = F.normalize(self.hub_q(x).view(_B6, _T6, 1, _dk), dim=-1)
                _oh, Sh = _dstr(hq, hk, hv, hg, hbe)
                x = x + self.hub_og * self.hub_o(_oh.reshape(_B6, _T6, _dv).to(x.dtype))
                _K6 = max(1, int(getattr(self.c, "gistk", 1)))
                _pi6 = None
                if _K6 > 1:
                    _Tb6 = (_T6 + _K6 - 1) // _K6
                    _pi6 = (torch.arange(1, _Tb6 + 1, device=x.device) * _K6).clamp(max=_T6) - 1
                _ShF = Sh.to(x.dtype).reshape(_B6, _T6, -1)
                if _pi6 is not None:
                    _ShF = _ShF[:, _pi6]
                ask = self.ask(_ShF)
                qr = self.askq(ask).view(_B6, ask.shape[1], 2, _dk) * (_dk ** -0.5)
                aske = (self.b16_ga * ask).unsqueeze(2)
                gq = self.b16_gq
                def _red16(Sflat, off=0):
                    _Ts = Sflat.shape[1]
                    Sm = Sflat.view(_B6, _Ts, -1, _dk, _dv)
                    rd = torch.einsum('btqk,bthkv->bthqv', qr.to(Sflat.dtype), Sm)
                    rd = rd.reshape(_B6, _Ts, Sm.shape[2], 2 * _dv)
                    inp = torch.cat([Sflat.view(_B6, _Ts, Sm.shape[2], -1),
                                     aske.expand(-1, -1, Sm.shape[2], -1),
                                     gq * rd], dim=-1)
                    _pw = getattr(self, "b16_pw", None)
                    if _pw is not None:
                        h = inp
                        if getattr(self, "b16_pair", None) is not None:
                            _P = self.b16_pair[off:off + Sm.shape[2]].to(h.dtype)
                            h = (h.view(*h.shape[:-1], h.shape[-1] // 2, 2) * _P).sum(-1)
                            h = F.gelu(h)
                        for _j, _Wp in enumerate(_pw):
                            _W = _Wp[off:off + Sm.shape[2]].to(h.dtype)
                            h = torch.einsum('bthi,hij->bthj', h, _W)
                            if _j < len(_pw) - 1:
                                h = F.gelu(h)
                        return h.reshape(_B6, _Ts, -1)
                    if self.b16_rf is not None:
                        h = inp
                        for _i, _lin in enumerate(self.b16_rf):
                            h = _lin(h)
                            if _i < len(self.b16_rf) - 1:
                                h = F.gelu(h)
                        return h.reshape(_B6, _Ts, -1)
                    h = F.gelu(self.b16_r1(inp))
                    h = F.gelu(self.b16_r2(h))
                    return self.b16_r3(h).reshape(_B6, _Ts, -1)
                fe6 = []
                _ks, _vs, _bs, _gs = [], [], [], []
                _ksl = int(getattr(self.c, "kslot", 0))
                _shs = _ksl and int(getattr(self.c, "shadow_shared", 0))
                if _shs:
                    # shared shadow: one projection set on the pre-block stream -> a single 32-head shadow memory
                    _k6, _v6, _b6, _g6 = self.shadow._shadow_inputs(self.shadow_n1(x))
                    _ks.append(_k6); _vs.append(_v6); _bs.append(_b6); _gs.append(_g6)
                for _li, _b in (enumerate(self.blocks) if not _shs else []):
                    mx = _b.mix
                    hx = _b.n1(x)
                    if _ksl:
                        # same projections and the same per-slot gates as the real path, on the
                        # pre-block stream; values stay PACKED (ragged payloads). One compiled method per layer.
                        _k6, _v6, _b6, _g6 = mx._shadow_inputs(hx)
                        _ks.append(_k6); _vs.append(_v6); _bs.append(_b6); _gs.append(_g6)
                        continue
                    if getattr(mx, "nlk", 1):
                        k16 = F.normalize(mx.phi(hx).view(_B6, _T6, _H, _dk), dim=-1)
                    else:
                        k16 = F.normalize(mx.wk(hx).view(_B6, _T6, _H, _dk), dim=-1)
                    v16 = mx.wv(hx).view(_B6, _T6, _H, _dv)
                    if getattr(mx, 'vcenter', 0):
                        v16 = v16 - mx.vmu.to(v16.dtype)
                    if getattr(mx, 'vwhiten', 0):
                        v16 = torch.einsum('bthv,hvw->bthw',
                                           (v16.float() - mx.vmu2.float()), mx.vwm.float()).to(v16.dtype)
                    _ks.append(k16); _vs.append(v16)
                    _b16 = torch.sigmoid(mx.wb(hx) + mx.bbias)
                    _g16 = -F.softplus(mx.wg(hx).float())
                    if getattr(mx, "wfreeze", 0) > 0:
                        # the same selective-write mask the real path applies, so the page
                        # describes the store that actually exists
                        _b16, _g16 = mx._wmask(hx, _b16, _g16, None)
                    _bs.append(_b16); _gs.append(_g16)
                # FUSED emission (H100 fix): one delta_stream call over all L*H
                # shadow heads instead of L sequential calls -- identical math
                # (heads are independent in the scan), but launch count drops
                # 12->1 and scan occupancy rises 12x. The funnel then runs as a
                # single L*H-head einsum too (off=0 slices the per-head weights).
                _g2 = int(getattr(self.c, "gist2", 0))
                if _ksl and not int(getattr(self.c, "gist3", 0)):
                    # m4's gist2 funnel, fed from the kslot emission: the RAW per-head stores are
                    # exactly what the emission holds (column-major, so transpose to dk x dv).
                    from kcol import slot_layout, kslot as _kslot, reads as _reads
                    if getattr(self, "_Lall", None) is None:
                        self._Lall = slot_layout(list(self.c.plan[0]) if _shs else [w for ws in self.c.plan for w in ws], _dk, 0, device=x.device)
                    _Lall = self._Lall
                    _kall = torch.cat(_ks, 2).to(torch.bfloat16); _gall = torch.cat(_gs, -1)
                    _ball = torch.cat(_bs, -1).to(torch.bfloat16); _vall = torch.cat(_vs, -1).to(torch.bfloat16)
                    _G6 = _H if _shs else self.c.layers * _H
                    _La6 = max(1, int(getattr(self.c, "lanes", 1)))
                    _NH6 = _G6 // _La6                       # gist2 funnels one group per LOGICAL head
                    _Tb6 = _kall.shape[1]
                    _qrb = qr.to(torch.bfloat16)
                    _bs = int(getattr(self.c, "gist_bs", 0)) > 0
                    _Wps = None if _bs else self._funnel_weights(x.dtype); _bsw = self._bs_pack() if _bs else None
                    C6 = int(getattr(self.c, "kchunk", 0)) or _Tb6
                    _fused = (not _bs) and (C6 >= _Tb6) and (_Wps[0].shape[-1] <= 128)
                    if C6 >= _Tb6:
                        fe6.append(_shadow_chunk(_kall, _gall, _ball, _vall, _qrb, aske, gq, _ShF, qr, _Wps, None, _Lall, _NH6, _La6, _dk, _dv, x.dtype, _fused, _bsw, _pi6)[0])
                    else:
                        assert _pi6 is None and aske.shape[1] == _Tb6 and _ShF.shape[1] == _Tb6 and qr.shape[1] == _Tb6, "chunked compressor needs per-position inputs (gistk=1)"
                        s0 = None; parts = []
                        for kc, gc, bc, vc, qc, ac, sc, rc in zip(*(_chunks(t, C6) for t in (_kall, _gall, _ball, _vall, _qrb, aske, _ShF, qr))):
                            f_c, s0 = _ckpt(_shadow_chunk, kc, gc, bc, vc, qc, ac, gq, sc, rc, _Wps, s0, _Lall, _NH6, _La6, _dk, _dv, x.dtype, _fused, _bsw, use_reentrant=False)
                            parts.append(f_c)
                        fe6.append(torch.cat(parts, 1))
                elif _ksl:
                    from kcol import slot_layout, kslot as _kslot, reads_padded as _reads_padded
                    if getattr(self, "_Lall", None) is None:
                        self._Lall = slot_layout([w for ws in self.c.plan for w in ws], _dk, 0, device=x.device)
                    _Lall = self._Lall
                    _NQ3 = int(getattr(self.c, "gist3_nq", 8))
                    _qr3 = self.askq3(ask).view(_B6, ask.shape[1], _NQ3, _dk) * (_dk ** -0.5)
                    _, _em = _kslot(torch.cat(_ks, 2).to(torch.bfloat16), torch.cat(_gs, 2),
                                    torch.cat(_bs, 2).to(torch.bfloat16), torch.cat(_vs, -1).to(torch.bfloat16), None, _Lall)
                    _dvmax = max(_Lall["dv"])
                    _rp = _reads_padded(_em, _qr3.to(torch.bfloat16), _Lall, _dvmax)   # (B,T,48,NQ*dvmax): one matmul + one gather
                    fe6.append(_gist_funnel3r(_rp, aske, _ShF, _qr3, list(self.b16_pw), self.b16_hub,
                                              _dk, _dv, _dvmax, x.dtype))
                elif _g2:
                    assert self.b16_pw is not None, "gist2 needs gistph=1 (per-head funnels)"
                    _G = self.c.layers * _H
                    _La2 = max(1, int(getattr(self.c, "lanes", 1)))
                    _NH2 = _G // _La2
                    if _K6 == 1:
                        from kpair3 import delta_raw3 as _draw3
                        _oS, _rd = _draw3(torch.cat(_ks, 2), torch.cat(_vs, 2),
                                          torch.cat(_gs, 2), torch.cat(_bs, 2),
                                          qr.to(x.dtype))
                    else:
                        from kpair2 import delta_raw as _draw
                        _oS, _rd = _draw(torch.cat(_ks, 2), torch.cat(_vs, 2),
                                         torch.cat(_gs, 2), torch.cat(_bs, 2),
                                         qr.to(x.dtype), 128, _K6)
                    _Tb = _oS.shape[1]
                    if int(getattr(self.c, "gist3", 0)):
                        _NQ3 = int(getattr(self.c, "gist3_nq", 8))
                        _qr3 = self.askq3(ask).view(_B6, ask.shape[1], _NQ3, _dk) * (_dk ** -0.5)
                        fe6.append(_gist_funnel3(_oS, aske, _ShF, _qr3, list(self.b16_pw),
                                                 self.b16_hub, _NH2, _La2, _dk, _dv, x.dtype))
                    else:
                        fe6.append(_gist_funnel2(_oS, _rd, aske, gq, _ShF, qr,
                                                 list(self.b16_pw), _NH2, _La2,
                                                 _dk, _dv, x.dtype))
                elif int(getattr(self.c, "gistfuse", 0)):
                    assert getattr(self, "b16_pair", None) is not None \
                        and self.b16_pw is not None, "gistfuse needs gistpair=1, gistph=1"
                    from kpair import delta_pair as _dpair
                    _G = self.c.layers * _H
                    _sw = _dk * _dv // 2
                    _W0 = self.b16_pair[:_G, :_sw, 0].reshape(_G, _dk, _dv // 2)
                    _W1 = self.b16_pair[:_G, :_sw, 1].reshape(_G, _dk, _dv // 2)
                    # gistk>1: the kernel scans every token (state is per-token) but
                    # EMITS only at block ends (emit_every=_K6): outputs are block-level
                    # (B, ceil(T/K), ...). qr is already block-level -- it derives from
                    # the block-sliced hub state -- and its block query is read exactly
                    # at its own block end. K=1 is the original per-token kernel.
                    _oS, _rd = _dpair(torch.cat(_ks, 2), torch.cat(_vs, 2),
                                      torch.cat(_gs, 2), torch.cat(_bs, 2),
                                      qr.to(x.dtype), _W0, _W1, 128, _K6)
                    _Tb = _oS.shape[1]
                    # aske + probe-read sections + funnel: one pure function (compiled by
                    # the trainer via memlm._gist_funnel rebinding; eager == inline code)
                    _Wa = self.b16_pair[:_G, _sw:_sw + 250, :].to(x.dtype)
                    _Wr = self.b16_pair[:_G, _sw + 250:_sw + 250 + _dv, :].to(x.dtype)
                    _Wps = [_Wp[:_G] for _Wp in self.b16_pw]
                    fe6.append(_gist_funnel(_oS, _rd, aske, gq, _Wa, _Wr, _Wps,
                                            _G, _sw, _dv, x.dtype))
                else:
                    _o6, S6a = _dstr(torch.cat(_ks, 2), torch.cat(_ks, 2),
                                     torch.cat(_vs, 2), torch.cat(_gs, 2),
                                     torch.cat(_bs, 2))
                    _S6F = S6a.to(x.dtype).reshape(_B6, _T6, -1)
                    if _pi6 is not None:
                        _S6F = _S6F[:, _pi6]
                    fe6.append(_red16(_S6F, 0))
                if not (_g2 or _ksl):
                    fe6.append(_red16(_ShF, self.c.layers * _H))
                ft6 = torch.cat(fe6, dim=-1)
                ft6 = torch.cat([ft6.new_zeros(_B6, 1, ft6.shape[-1]),
                                 ft6[:, :-1]], dim=1)          # strictly t-1 (block level if K>1)
                # K>1: the trunk chain is pointwise per position, so it runs at BLOCK
                # level and the result is expanded after -- identical math, 1/K the GEMM.
                # block j = state after token j*K-1, held for tokens [j*K, (j+1)*K)
                _x6pand = ((lambda _t: _t.repeat_interleave(_K6, dim=1)[:, :_T6])
                           if _pi6 is not None else (lambda _t: _t))
                if getattr(self, "b16_ol", None) is not None:
                    _g6inj = F.gelu(self._page(ft6))              # shared trunk, BLOCK-level rows; expanded to tokens after each layer's GEMM
                    if getattr(self, 'b16_ginorm', 0):
                        _Kinj = float(self.b16_ol.shape[0] + (self.b16_olp.shape[0] if getattr(self, 'b16_olp', None) is not None else 0))
                        _g6inj = _g6inj / _Kinj + (_g6inj * ((_Kinj - 1.0) / _Kinj)).detach()
                elif getattr(self, "b16_lg", None) is not None:
                    _g6inj = _x6pand(self._inject(ft6, x.dtype))
                else:
                    x = x + self.b16_g * _x6pand(self._inject(ft6, x.dtype))
            gS = gI = None
            if _bm == 11:
                # MEMORY-ONLY shadow pass: every layer's write projections read the
                # (pre-encoded) token stream directly -- no FFNs, no depth chain, no
                # second full forward. States approximate the true t-1 states at
                # ~1/3 the cost and feed the same everything-to-everything read.
                with torch.no_grad():
                    _shs, _shi = [], []
                    for _b in self.blocks:
                        _b.mix._want_stream = True
                        _ = _b.mix(_b.n1(x), None, False)
                        _b.mix._want_stream = False
                        _S = _b.mix._stream; _b.mix._stream = None
                        _z = torch.einsum('bthkv,vm->bthkm', _S,
                                          self.b8_P.to(_S.dtype)).flatten(3)
                        _ix = self.b7_norm(self.b8_W3(F.gelu(self.b8_W2(
                            F.gelu(self.b8_W1(_z))))))
                        _shs.append(_S.flatten(3)); _shi.append(_ix.to(x.dtype))
                    gS = torch.cat(_shs, dim=2)
                    gI = torch.cat(_shi, dim=2)
                    gS = torch.cat([torch.zeros_like(gS[:, :1]), gS[:, :-1]], 1)
                    gI = torch.cat([torch.zeros_like(gI[:, :1]), gI[:, :-1]], 1)
            if _bm == 10 and not getattr(self, "_collecting", False):
                # pass 1 (no grad): same architecture minus the global bus, run only
                # to materialise every layer's per-token states and index vectors
                self._collecting = True
                try:
                    with torch.no_grad():
                        gS, gI = self.forward(idx, states=states)
                finally:
                    self._collecting = False
                # previous token: strictly t-1, zeros at t=0 -- causal for ALL layers
                gS = torch.cat([torch.zeros_like(gS[:, :1]), gS[:, :-1]], 1)
                gI = torch.cat([torch.zeros_like(gI[:, :1]), gI[:, :-1]], 1)
            if _b7:
                # ONE preallocated buffer; each layer reads a VIEW of the filled prefix.
                # A per-layer torch.cat would copy the growing prefix 11 times (~4.4GB
                # of autograd-saved duplicates); views share the single storage.
                _B, _T, _ = x.shape
                _H, _dk, _dv = self.c.heads, self.c.dk, self.c.dv
                _L = len(self.blocks)
                Sbuf = torch.empty(_B, _T, _L * _H, _dk * _dv,
                                   device=x.device, dtype=x.dtype)
                idxs = []                   # per-layer (B,T,H,gidx), keeps grads
            if _g6inj is not None and getattr(self, "b16_olp", None) is not None:
                # v3: re-encode the pre stage from embeddings WITH per-layer
                # gist injections (gist itself came from the pass-A stream).
                x = _x0
                _olp = self.b16_olp.unbind(0)      # unbind once: indexing the Parameter per layer costs a full-size zero-fill + add per layer in backward
                for _pj, _pb in enumerate(self.pre_blocks):
                    x = x + self.b16_plg[_pj].to(x.dtype) * _x6pand(torch.einsum('btg,gd->btd', _g6inj, _olp[_pj].to(x.dtype)))
                    x = _pb(x)
                # WIRING FIX: the hub readout was added to the pass-A stream,
                # which this re-encode just discarded -- re-apply it here so
                # the hub keeps its ordinary-head role in the v3 path.
                x = x + self.hub_og * self.hub_o(_oh.reshape(_B6, _T6, _dv).to(x.dtype))
            _ol = self.b16_ol.unbind(0) if (_g6inj is not None and getattr(self, "b16_ol", None) is not None) else None   # see _olp
            for i, b in enumerate(self.blocks):
                s = states[i] if states is not None else None
                if _g6inj is not None:
                    if _ol is not None:
                        x = x + self.b16_lg[i].to(x.dtype) * _x6pand(torch.einsum('btg,gd->btd', _g6inj, _ol[i].to(x.dtype)))
                    else:
                        x = x + self.b16_lg[i].to(x.dtype) * _g6inj
                if gS is not None:
                    # EVERY layer (L0 included) reads ALL L*H head states at t-1:
                    # transfer from everything to everything, one token stale
                    qw = torch.einsum('btd,de->bte', x, self.b7_q[i].to(x.dtype))
                    sc = torch.einsum('bte,btse->bts', qw, gI)
                    sc = sc * (float(gI.shape[-1]) ** -0.5)
                    kk = min(int(getattr(self.c, "gtop", 10)), sc.shape[-1])
                    thr = sc.topk(kk, dim=-1).values[..., -1:]
                    w = torch.softmax(sc.masked_fill(sc < thr, float('-inf')), -1)
                    if getattr(self, "_b7_dbg", None) is not None:
                        self._b7_dbg.append((i, w.detach()))
                    Aw = torch.einsum('bts,btsz->btz', w, gS)   # gS static, no-grad
                    x = x + self.b7_gate[i].to(x.dtype) * self.b7_out(Aw)
                elif _b7 and i > 0:
                    Ix = torch.cat(idxs, dim=2)             # small: (B,T,S,gidx)
                    qw = torch.einsum('btd,de->bte', x, self.b7_q[i].to(x.dtype))
                    sc = torch.einsum('bte,btse->bts', qw, Ix)
                    sc = sc * (float(Ix.shape[-1]) ** -0.5)
                    kk = min(int(getattr(self.c, "gtop", 10)), sc.shape[-1])
                    thr = sc.topk(kk, dim=-1).values[..., -1:]
                    w = torch.softmax(sc.masked_fill(sc < thr, float('-inf')), -1)
                    if getattr(self, "_b7_dbg", None) is not None:
                        self._b7_dbg.append((i, w.detach()))
                    Aw = _FoldSG.apply(w, Sbuf, i * _H)
                    x = x + self.b7_gate[i].to(x.dtype) * self.b7_out(Aw)
                if _b7:
                    b.mix._want_stream = True
                    # pass 1 only materialises streams -- the refresh/state-aware
                    # machinery is redone in pass 2 anyway, so skip it there
                    if not getattr(self, "_collecting", False):
                        if _bm in (8, 9, 10, 11):
                            b.mix._selfref = _sref
                            b.mix._sg = self.b8_vgate[i]
                        if _bm in (9, 10, 11):
                            b.mix._sinj = _sinj9
                            b.mix._sg2 = self.b9_g[i]
                if return_states:
                    x, f, _ = b(x, s, True)
                    finals.append(f)
                else:
                    x = b(x, s, False)
                if _b7:
                    b.mix._want_stream = False
                    b.mix._selfref = None
                    b.mix._sinj = None
                    S7 = b.mix._stream          # (B,T,H,dk,dv) bf16, stop-grad
                    b.mix._stream = None
                    if _bm in (8, 9, 10, 11):
                        z = torch.einsum('bthkv,vm->bthkm', S7,
                                         self.b8_P.to(S7.dtype)).flatten(3)
                        ix = self.b7_norm(self.b8_W3(F.gelu(self.b8_W2(
                            F.gelu(self.b8_W1(z))))))
                    else:
                        z = torch.einsum('bthkv,vm->bthkm', S7,
                                         self.b7_P.to(S7.dtype)).flatten(3)
                        ix = self.b7_norm(self.b7_W2(F.gelu(self.b7_W1(z))))
                    Sbuf[:, :, i * _H:(i + 1) * _H] = S7.flatten(3)
                    idxs.append(ix.to(x.dtype))
            if getattr(self, "_collecting", False):
                return Sbuf, torch.cat(idxs, dim=2)
        if float(getattr(self.c, "kdecor", 0.0)) > 0:
            _ls = [b.mix.kdec_loss for b in self.blocks
                   if getattr(b.mix, "kdec_loss", None) is not None]
            self.kdec = torch.stack(_ls).sum() if _ls else None
        else:
            self.kdec = None
        if float(getattr(self.c, "krep", 0.0)) > 0:
            _lr = [b.mix.krep_loss for b in self.blocks
                   if getattr(b.mix, "krep_loss", None) is not None]
            self.krep = torch.stack(_lr).sum() if _lr else None
        else:
            self.krep = None
        if float(getattr(self.c, "wbud", 0.0)) > 0:
            _lw = [b.mix.wbud_loss for b in self.blocks if getattr(b.mix, "wbud_loss", None) is not None]
            self.wbud = torch.stack(_lw).sum() if _lw else None
        else:
            self.wbud = None
        x = self.nf(x)
        if not skip_head:            # the trainer's fused loss takes the hidden states + head weight
            x = self.head(x)
        if getattr(self.c, "moe", 0):
            us = [b.ffn.util for b in self.blocks if b.ffn.util is not None]
            self.util = torch.stack(us).mean(0) if us else None
            mv = [b.ffn.maxvio for b in self.blocks if b.ffn.maxvio is not None]
            self.maxvio = torch.stack(mv).max() if mv else None
            dr = [b.ffn.drop for b in self.blocks if b.ffn.drop is not None]
            self.drop = torch.stack(dr).max() if dr else None
        return (x, finals) if return_states else x


if __name__ == "__main__":
    import time, argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dk", type=int, default=48)
    ap.add_argument("--T", type=int, default=1024)
    ap.add_argument("--bs", type=int, default=8)
    a = ap.parse_args()

    c = Cfg(dk=a.dk)
    print(c)
    m = MemLM(c).cuda().to(torch.bfloat16)
    n = sum(p.numel() for p in m.parameters())
    print(f"params {n/1e6:.1f}M")
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4)
    x = torch.randint(0, c.vocab, (a.bs, a.T + 1), device="cuda")

    def step():
        logits = m(x[:, :-1])
        loss = F.cross_entropy(logits.float().reshape(-1, c.vocab), x[:, 1:].reshape(-1))
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()

    for _ in range(3): step()
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(10): step()
    torch.cuda.synchronize(); dt = (time.time() - t0) / 10
    tok = a.bs * a.T / dt
    print(f"step {dt*1000:.1f} ms   {tok:,.0f} tok/s   VRAM {torch.cuda.max_memory_allocated()/2**30:.2f} GiB")
    print(f"5B tokens => {5e9/tok/86400:.2f} days")
