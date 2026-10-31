"""memlm_slot.py -- the KEYLESS clean-slot memory LM (the user's design). MEMLM_MOD=memlm_slot.

Reuses memlm_m5's MoE + PreFFN and runs under train_memlm.py unchanged (same model contract:
MemLM(cfg) -> .to(bf16); forward(idx, skip_head=...); .blocks[i].mix / .ffn; .pre_blocks; .head;
set_moe_loop; util/maxvio/drop; kdec/krep/wbud = None).

Per layer:  x + SlotMixer(LN(x), bus)  ;  x + FFN(LN(x))
SlotMixer (one per layer):
  * slots hold VALUES only (N slots x dv per subhead; 48 x 16 = 768 = the delta memory's state). A token WRITES
    by a learned slot address (softmax over N with an occupancy free-slot bias) and a gate: erase-replace of the
    chosen slot's value. No keys, no per-token read -- the exact Triton scan in slotkern_v.py.
  * a short EXACT KV cache (sliding-window attention over the last kv_window tokens).
  * RETRIEVAL is the compressor's job: the per-token slot buffer (La*N*dv = 3k per big head, emitted in bf16 by
    the kernel) + the KV-cache reads feed the per-big-head-per-layer funnel (3k+reads -> 512 -> 128 -> 64):
    128 = own-layer tap -> residual; 64 per big head = cross-layer broadcast.
  * lanes: m14's pair/quad merge gates soft-share the ADDRESS and GATE across a big head's lanes (values stay per
    lane), so four 16-dim lanes can write one slot index together = a 32/64-dim fact.
  * cross-layer bus: this layer receives the 64-wide broadcasts of every EARLIER layer through a zero-init gate.
  * the input projections are ONE GEMM (w_in, split afterwards).
emit_ckpt=1 checkpoints ONLY the compressor (recomputed in backward)."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as _ckpt
import memlm_m5 as _M5
from slotkern_v import slot_scan_v
from slotkern_kv import ref_scan_kv, scan_kv
from compressor import Compressor
try:
    # block-sparse sliding-window attention: only the band's blocks are computed (6.4x the band-masked
    # SDPA at T=1024, which does full TxT work; test_flexwin.py). Falls back to SDPA off-GPU.
    from torch.nn.attention.flex_attention import flex_attention as _flex_attention, create_block_mask as _create_block_mask
    _FLEX = torch.compile(_flex_attention, dynamic=False)
except Exception:
    _FLEX, _create_block_mask = None, None


class Cfg:
    def __init__(self, vocab=50304, d_model=1024, layers=12, heads=32, dk=48, dv=16, d_ff=704, lanes=4, kdim=0,
                 moe=1, n_exp=12, top_k=2, moe_u=0.001, cap=2.0, pre=2,
                 n_slots=48, scan_chunk=256, buf_stride=1, kv_window=16, kv_dim=32, kv_heads=0,
                 comp_nb=5, gist_seed=20260905, xbus=1, emit_ckpt=1, slot_hard=1,
                 slot_hard_gate=1, gate_init=0.3, wbud=0.0, wbud_rho=0.3, wbud_count=1, wbud_tau=0.05, sel_heads=4, rowsel="none", rs_heads=2, free_bias=1, possel=0, top2=0, shsc=0, tksel=0, tksel_dk=16, wbud_lo=0.0, comp_off=0, tksel_hard=0, cone=0, cone_ctx=64, cone_src="x",
                 osel=0, osel_k=8, osel_lam=1.0,
                 xsel=0, xsel_lam=1.0, xsel_tau=0.1,
                 hsel=0, hsel_merge=0, hsel_hid="400,400,200,100", hsel_floor=1e-3,
                 hsel_lam=1.0, hsel_tau=1.0, hsel_prev=1, krep=0.0, krep_tau=0.85, kdetach=0, wq=0):
        assert heads % lanes == 0, "heads must be a multiple of lanes (heads = subheads, heads/lanes = big heads)"
        self.vocab, self.d_model, self.layers, self.heads = vocab, d_model, layers, heads
        self.kdim = int(kdim)                      # width of the STORED key (0 -> same as dv)
        self.krep, self.krep_tau = float(krep), float(krep_tau)   # write-time key repulsion
        self.kdetach = int(kdetach)   # 1 = the write address reads the key buffer as a constant
        self.wq = int(wq)             # 1 = the write SEARCHES with its own projection
        self.dk, self.dv, self.d_ff, self.lanes = dk, dv, d_ff, lanes      # dk is accepted for the trainer's sake; slots have no keys
        self.moe, self.n_exp, self.top_k, self.moe_u, self.cap = int(moe), n_exp, top_k, moe_u, cap
        self.pre = int(pre)
        self.n_slots, self.scan_chunk, self.buf_stride = int(n_slots), int(scan_chunk), int(buf_stride)
        self.kv_window, self.kv_dim, self.comp_nb = int(kv_window), int(kv_dim), int(comp_nb)
        self.kv_heads = int(kv_heads)      # 0 = same as heads. Lets the WINDOW keep its head count when the
                                           # slot memory drops to one head (they used to be forced equal).
        self.gist_seed, self.xbus, self.emit_ckpt = int(gist_seed), int(xbus), int(emit_ckpt)
        self.slot_hard = int(slot_hard)     # 1 = one-slot (argmax) writes with a straight-through address gradient
        self.slot_hard_gate = int(slot_hard_gate)   # 1 = binary write gate (write iff gate > 0.5), straight-through
        self.gate_init = float(gate_init)   # initial write probability (per-subhead gate bias)
        self.sel_heads = int(sel_heads)     # selection-attention heads inside the compressor (0 = funnel only)
        self.rowsel, self.rs_heads, self.free_bias = str(rowsel), int(rs_heads), int(free_bias)
        # HSEL -- a LEARNED row selector trained by HINDSIGHT supervision rather than by the read's own
        # softmax gradient.  Why: measured in softgrad2.py, the soft read's gradient on row n carries a
        # factor att_n, so a row the model has stopped attending to gets ~0 signal even when it was the
        # RIGHT row.  At s3's measured 0.855 row-row cosine with a peaked read that is ~10^4 attenuation,
        # and it is self-reinforcing -- the read can only reinforce rows it already looks at.  A
        # cross-entropy toward a target instead gives every row a full-strength (att_n - t_n) gradient.
        # The target comes from the model itself: with g = dL/d(read), u_n = g.row_n is to first order how
        # much reading row n WOULD have helped, so no data annotation is needed and plain text supervises
        # the selector too.  (A first-order stand-in for perplexity distillation, minus the N forwards.)
        # XSEL -- supervise the match with an EXACT target taken from the data, and keep the FIXED
        # cosine read.  Measured why: the loss-derived target is hopeless (its pick matches the true
        # best row 10.9% of the time in a run whose memory WORKS, below chance in one whose memory is
        # idle) because the loss is nearly flat across rows -- median 2 of 24 rows tie within 0.01 nats.
        # But on a recall example the answer token appeared EARLIER in the context, so the row that
        # should be read is the one holding that position's write, and that is known exactly.  The term
        # is a cross-entropy on the read's OWN cosine scores toward that row: it trains the query and the
        # stored content to line up (learn what goes INTO the match) and never learns the match itself,
        # which is the only thing that has survived here.
        # OSEL -- supervise the cosine read toward the row that genuinely MINIMISES the loss, found by
        # a two-stage oracle the trainer computes before the training pass (see train_memlm.oracle_target):
        # rank all N rows by g.row_n (top-8 contains the true best 100% of the time, measured), then rerank
        # those 8 EXACTLY by running the layers above the read.  Two-pass rather than injected into the
        # backward: slower (~1.44x a step vs 1.11x) but it needs no custom autograd, which is where the
        # previous attempt went quietly wrong.
        self.osel, self.osel_k, self.osel_lam = int(osel), int(osel_k), float(osel_lam)
        self.xsel, self.xsel_lam, self.xsel_tau = int(xsel), float(xsel_lam), float(xsel_tau)
        self.hsel, self.hsel_merge = int(hsel), int(hsel_merge)
        self.hsel_hid = tuple(int(v) for v in str(hsel_hid).split(",") if v.strip())
        self.hsel_floor, self.hsel_lam, self.hsel_tau = float(hsel_floor), float(hsel_lam), float(hsel_tau)
        self.hsel_prev = int(hsel_prev)
        self.shsc = int(shsc)
        self.tksel, self.tksel_dk = int(tksel), int(tksel_dk)   # per-lane TIED-KEY read: row's own first tksel_dk dims are its address               # SHARED ROW SCORER read: s_j = w2.gelu(W1[q_x ; row_j]) shared across rows, SOFT mix of all rows (every row graded), shsc = heads
        self.tksel_hard = int(tksel_hard)  # 1 = the read injects ONE row (hard argmax, straight-through), not a soft mix
        self.cone, self.cone_ctx, self.cone_src = int(cone), int(cone_ctx), str(cone_src)   # the SHRINKING CONE read: [all rows | context] -> 256 -> 128 -> 64 -> one logit per row, SOFT softmax mix
        self.comp_off = int(comp_off)      # 1 = NO compressor: the tied-key read + the exact window go straight to the residual (measured: the funnel adds 0 retrieval)
        if self.comp_off:
            self.xbus = 0                   # the cross-layer bus carries the compressor's 64-wide output; without a compressor there is nothing to broadcast
        self.top2 = int(top2)               # the user's selector: per big head, rows of La*dv (lanes merged); funnel [rows|ctx] 3k->256->128->64->N logits; TOP-2 rows injected (2*La*dv)
        self.possel = int(possel)           # POSITION selector: a dense MLP over [all rows | context] emits one logit per row (the compressor picks the position)   # ROW-ID read: select a row by index from the token (tied = the write address itself)
        self.wbud, self.wbud_rho = float(wbud), float(wbud_rho)   # >0: penalise the FRACTION OF TOKENS THAT WRITE above rho (count mode)
        self.wbud_lo = float(wbud_lo)      # >0: ALSO penalise the write rate BELOW this (a FLOOR). 0 = ceiling only: push down, never up.
        self.wbud_count, self.wbud_tau = int(wbud_count), float(wbud_tau)   # accepted for the trainer; the gate is binary so count mode is the only mode

    @property
    def big_heads(self):
        return self.heads // self.lanes

    @property
    def state_per_layer(self):
        return self.heads * self.n_slots * self.dv

    def __repr__(self):
        return (f"SLOT(keyless) d_model{self.d_model} L{self.layers} H{self.heads}({self.big_heads}x{self.lanes}) "
                f"N{self.n_slots}x{self.dv} kv{self.kv_window}x{self.kv_dim} G{self.buf_stride} xbus{self.xbus} "
                f"| state/layer={self.state_per_layer} total_state={self.state_per_layer * self.layers}")


import os as _os
_ROPE = int(_os.environ.get("SLOT_ROPE", "1"))    # 0 = the pre-fix behaviour (no positional signal in the window), for A/B only
_ROPE_FRAC = float(_os.environ.get("SLOT_ROPE_FRAC", "0.5"))   # fraction of each window head's dims that carry POSITION; the rest stay pure CONTENT.
# 0.5 measured best on BOTH jobs (tiny model, 2 seeds): copy-2 loss 0.003 (same as full rotation; 0.386 with none)
# AND fact-retrieval lift +0.942/+0.766 vs +0.576/+0.714 at none and +0.454/+0.645 at full. Rotating EVERY dim
# mixes position into the very dot product the window uses to match CONTENT, and that costs retrieval.


def _rope(x, base=10000.0, frac=1.0):
    """Rotary position embedding over the head dim (identical to memlm_m5's).

    WHY THIS EXISTS: without it the 16-token exact window is softmax attention with NO positional signal, so
    within the window it cannot tell the token immediately before it from the one 15 back. That makes a
    previous-token head impossible, and without a previous-token head an INDUCTION head cannot form -- so the
    model cannot copy a token it has already seen. Measured on s1/s2 before this fix: a random sequence repeated
    twice cost 11.34 nats on copy 1 and 11.04 on copy 2, a 0.30-nat gain where copying is worth ~11; and no better
    at repeat distance 6, which is INSIDE the window. The delta line (memlm_m5), which ropes its window, gains
    2.28 nats at step 6000 and 3.43 at 8000. The slot READ is deliberately NOT roped: it addresses by CONTENT,
    and rotating a row's key by its write position would make a match depend on how long ago it was written.
    """
    B, T, H, D = x.shape
    R = D if frac >= 1.0 else (int(D * frac) // 2) * 2          # PARTIAL rotation: rotate R dims, leave D-R for content
    if R <= 0:
        return x
    if R < D:
        return torch.cat([_rope(x[..., :R], base, 1.0), x[..., R:]], -1)
    half = D // 2
    freqs = torch.arange(half, device=x.device, dtype=torch.float32)
    inv = base ** (-freqs / half)
    t = torch.arange(T, device=x.device, dtype=torch.float32)
    ang = t[:, None] * inv[None, :]
    cos, sin = ang.cos()[None, :, None, :], ang.sin()[None, :, None, :]
    x1, x2 = x[..., :half].float(), x[..., half:].float()
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1).to(x.dtype)


class _HSelAtt(torch.autograd.Function):
    """softmax -> floor -> read, with the HINDSIGHT cross-entropy injected in the backward.

    Written as an autograd.Function rather than tensor hooks for a measured reason: a hook closure
    that captures the buffer forms a reference cycle (tensor -> hook -> closure -> tensor) that
    refcounting cannot free, and across gradient accumulation it grew memory by 2.38 GB per
    micro-batch -- 198 MB per layer, exactly one buf.  save_for_backward is released with the graph.

    The injection itself: the read's own gradient on the selector logits is
        d/d sc_n = att_n (u_n - sum_m att_m u_m)
    which carries a factor att_n, so a row the model has stopped attending to receives almost no
    signal even when u_n says it was the right row.  The cross-entropy term (att_n - t_n) has no
    such factor.  And u_n = g . row_n IS dL/d att_n, already on hand -- no second forward.
    """
    @staticmethod
    def forward(ctx, sc, buf, valid, eps, lam, tau):
        soft = torch.nan_to_num(torch.softmax(sc, -1))
        nval = valid.to(soft.dtype).sum(-1, keepdim=True).clamp_min(1.0)
        att = (1.0 - eps) * soft + eps * (valid.to(soft.dtype) / nval)
        rd = torch.einsum("bthn,bthnv->bthv", att.to(buf.dtype), buf)
        ctx.save_for_backward(att, buf, valid, soft)
        ctx.eps, ctx.lam, ctx.tau, ctx.scdt = float(eps), float(lam), float(tau), sc.dtype
        return rd, att.detach()

    @staticmethod
    def backward(ctx, g_rd, _g_att):
        att, buf, valid, soft = ctx.saved_tensors
        bf = buf.float()
        u = torch.einsum("bthv,bthnv->bthn", g_rd.float(), bf)        # == dL/d att_n
        d_buf = (att.unsqueeze(-1).to(g_rd.dtype) * g_rd.unsqueeze(-2))
        d_soft = u * (1.0 - ctx.eps)
        d_sc = soft.float() * (d_soft - (soft.float() * d_soft).sum(-1, keepdim=True))
        if ctx.lam > 0:
            un = u / bf.norm(dim=-1).clamp_min(1e-6)                  # else a big-norm row always wins
            t = torch.nan_to_num(torch.softmax((-un / ctx.tau).masked_fill(~valid, float("-inf")), -1))
            d_sc = d_sc + ctx.lam * (att.float() - t)
        return d_sc.to(ctx.scdt), d_buf, None, None, None, None


class SlotMixer(nn.Module):
    def __init__(self, c: Cfg, layer_idx: int):
        super().__init__()
        D, H, BH, La = c.d_model, c.heads, c.big_heads, c.lanes
        dv, N, kvd = c.dv, c.n_slots, c.kv_dim
        self.H, self.BH, self.La, self.dv, self.N, self.kvd = H, BH, La, dv, N, kvd
        self.KVH = int(getattr(c, "kv_heads", 0)) or H      # window heads; defaults to the memory head count
        self.C, self.G, self.W = c.scan_chunk, c.buf_stride, c.kv_window
        self.emit_ckpt = int(c.emit_ckpt)
        self.hard = bool(c.slot_hard)
        self.hard_gate = bool(c.slot_hard_gate)
        self.wbud, self.wbud_rho = float(c.wbud), float(c.wbud_rho)
        self.wbud_lo = float(getattr(c, "wbud_lo", 0.0))
        gi = min(max(float(c.gate_init), 1e-3), 1 - 1e-3)
        self.g_bias = nn.Parameter(torch.full((H,), math.log(gi / (1 - gi))))   # per-subhead gate bias: starts writing at gate_init
        self.wbud_loss = None
        self.wrate = None
        # one fused input GEMM: [v | address | gate | ck | cq | cv]
        self.nsel = int(c.sel_heads)
        self.rowsel, self.nrs, self.free_bias = c.rowsel, (c.rs_heads if c.rowsel == "free" else (1 if c.rowsel == "tied" else 0)), float(c.free_bias)
        self.kdim = int(c.kdim) or dv              # pa's budget is now the KEY, not N write logits
        self.wq = int(_os.environ.get("KV_WQ", getattr(c, "wq", 0)))
        # One vector currently does two jobs: the query that SEARCHES for a row, and the key that
        # later readers must MATCH. Those objectives differ, and a vector pulled toward both has
        # less freedom to spread -- measured 40-54%% of key pairs above |cos| 0.85. wq gives the
        # search its own projection. Appended LAST so every existing part index is unchanged.
        self.sizes = [H * dv, H * self.kdim, H, self.KVH * kvd, self.KVH * kvd, self.KVH * kvd] + ([BH * self.nsel * dv] if self.nsel > 0 else []) + ([H * self.nrs * N] if c.rowsel == "free" else []) + ([H * self.kdim] if self.wq else [])   # + selection queries / row-read logits / write query
        self.w_in = nn.Linear(D, sum(self.sizes), bias=False)
        self.log_addr = nn.Parameter(torch.tensor(0.0))    # (unused by the keyed write; kept so old ckpts load)
        # WRITE SHARPNESS. Measured on the reference scan with 24 well-separated keys and fr=3:
        #   temp 2-4  -> 67-100% of rows become blends (this IS s3's 0.855 smear)
        #   temp 16-32 -> zero leakage, all 24 rows holding one entity each
        #   temp 64 / hard top-1 -> weak similarities start winning over a free row, 20/24 rows
        # so it starts at 16 and is LEARNED, because the optimum depends on how correlated the
        # learned keys turn out to be -- random keys are a best case.
        self.log_wsharp = nn.Parameter(torch.tensor(math.log(16.0)))
        self.log_free = nn.Parameter(torch.tensor(1.5))    # occupancy penalty: push writes to FREE slots
        assert La == 1 or La % 2 == 0, "lanes must be 1 or even (pair merging)"
        if La >= 2:
            self.lane_gp = nn.Parameter(torch.zeros(BH * (La // 2)))   # 1-D -> fp32 masters + the trainer's --lane_clamp
        if La >= 4:
            self.lane_gq = nn.Parameter(torch.zeros(BH))
        self.comp_off = int(getattr(c, "comp_off", 0))
        if self.comp_off:
            # No funnel at all. The exact recent window still needs a path to the residual -- it used to reach the
            # model ONLY as an input to the compressor, and on its own it is worth about as much as plain attention.
            self.comp = None
            self.wo_kv = nn.Linear(self.KVH * kvd, D, bias=False)
        else:
            self.comp = Compressor(BH, state_dim=La * N * dv, kv_dim=La * kvd + 32, nb=c.comp_nb, dv=dv, n_sel=c.sel_heads,   # +32: fill fraction per lane (padded)
                                   seed=c.gist_seed + 1000 * (layer_idx + 1))
            self.wo_own = nn.Linear(BH * self.comp.own_dim, D, bias=False)
        if self.nsel > 0:
            self.wo_sel = nn.Linear(BH * self.nsel * dv, D, bias=False)   # the selected values -> residual DIRECTLY (a read path)
        if self.nrs > 0:
            self.wo_rs = nn.Linear(H * self.nrs * dv, D, bias=False)      # row-id reads -> residual DIRECTLY
        self.cone = int(getattr(c, "cone", 0)); self.cone_c = int(getattr(c, "cone_ctx", 64)); self.cone_src = str(getattr(c, "cone_src", "x"))
        if self.cone:
            # The user's funnel: every row of a big head, plus a context vector, shrink to ONE LOGIT PER ROW.
            # Soft softmax over rows (not top-k), the mixed row injected. cone_src 'x' feeds the token's own
            # hidden state as the context; 'kv' feeds the exact-window reads instead.
            cin = La * N * dv + (self.cone_c if self.cone_src == "x" else La * kvd)
            dims = [cin, 256, 128, 64, N]
            self.cw = nn.ParameterList([nn.Parameter(torch.empty(BH, a_, b_).normal_(0, a_ ** -0.5)) for a_, b_ in zip(dims[:-1], dims[1:])])
            self.cb = nn.ParameterList([nn.Parameter(torch.zeros(BH, b_)) for b_ in dims[1:-1]])
            if self.cone_src == "x":
                self.cone_qx = nn.Linear(D, BH * self.cone_c, bias=False)
            self.wo_cone = nn.Linear(BH * La * dv, D, bias=False)
        self.possel = int(c.possel)
        self.top2 = int(c.top2)
        self.shsc = int(c.shsc)
        self.hsel = int(getattr(c, "hsel", 0))
        if self.hsel:
            # ONE weight set per LAYER, applied to every group in that layer.  Group size = the merge level:
            # 1 subhead (no merge), 2 subheads (a bit merged), or a whole big head's lanes (fully merged).
            grp = 1 if c.hsel_merge == 0 else (2 if c.hsel_merge == 1 else La)
            assert H % grp == 0, f"hsel group {grp} does not divide {H} subheads"
            self.hs_grp, self.hs_prev = grp, int(c.hsel_prev)
            self.hs_floor, self.hs_lam, self.hs_tau = float(c.hsel_floor), float(c.hsel_lam), float(c.hsel_tau)
            # Linear over cat([x, rows]) == Linear_x(x) + Linear_rows(rows), so project the n-1 state ONCE
            # and broadcast-add it instead of materialising (B,T,ng,D) copies of it -- exactly equivalent,
            # and it was the whole reason the selector doubled activation memory (measured: merge=0 needed
            # 31.1 GB at 4096 tokens vs the control's 15.8, and more merging cost LESS because ng was smaller).
            _h1 = c.hsel_hid[0]
            self.hs_in_rows = nn.Linear(grp * N * dv, _h1)
            self.hs_in_x = nn.Linear(D, _h1, bias=False) if self.hs_prev else None
            _ls, _prev = [], _h1
            for _h in c.hsel_hid[1:]:
                _ls += [nn.GELU(), nn.Linear(_prev, _h)]; _prev = _h
            _ls += [nn.GELU()]
            self.hs_mlp = nn.Sequential(*_ls)
            self.hs_out = nn.Linear(_prev, grp * N)             # one logit per (subhead in group, row)
            self.wo_hs = nn.Linear(H * dv, D, bias=False)
            self.hs_fired, self.hs_ce, self.hs_agree = 0, None, None
        self.osel = int(getattr(c, "osel", 0))
        self._osel_tgt = None          # (B,T,H) int target, or None; set by the trainer each micro-batch
        self._last_g = None            # gradient at the read, kept for NEXT step's shortlist
        self.osel_loss = None
        self.xsel = int(getattr(c, "xsel", 0))
        self.xs_lam, self.xs_tau = float(getattr(c, "xsel_lam", 1.0)), float(getattr(c, "xsel_tau", 0.1))
        self.xs_loss = None
        self.tksel = int(c.tksel); self.tkdk = self.kdim   # the read matches the STORED key, so it is kdim wide
        self.krep = float(getattr(c, "krep", 0.0))
        self.krep_tau = float(getattr(c, "krep_tau", 0.85))
        self.krep_loss = None
        # env knob so this can be selected without editing train_memlm.py, which the live p1
        # run imports; a trainer edit would be picked up by p1 on any guard restart.
        self.kdetach = int(_os.environ.get("KV_DETACH", getattr(c, "kdetach", 0)))
        self.tk_hard = int(getattr(c, "tksel_hard", 0))   # 1 = inject ONE row (argmax, straight-through) instead of the softmax mix
        if self.tksel:
            # PER-LANE TIED-KEY READ: each SUBHEAD scores its OWN N rows by cosine(row[:tkdk], q) x a learned
            # temperature and injects the mixed row straight into the residual. No stored key: the row IS the key.
            self.tk_q = nn.Linear(D, H * self.tkdk, bias=False)
            self.wo_tk = nn.Linear(H * dv, D, bias=False)
            self.log_rsel = nn.Parameter(torch.tensor(math.log(8.0)))   # read sharpness, init 8 (attention's circuit)
        if self.shsc:
            R = La * dv
            self.ss_q = nn.Linear(D, BH * self.shsc * R, bias=False)                     # per big head, per head: a 64-dim query from x
            self.ss_w1 = nn.Parameter(torch.empty(BH, self.shsc, 2 * R, 64).normal_(0, (2 * R) ** -0.5))
            self.ss_b1 = nn.Parameter(torch.zeros(BH, self.shsc, 64))
            self.ss_w2 = nn.Parameter(torch.empty(BH, self.shsc, 64).normal_(0, 64 ** -0.5))
            self.wo_ss = nn.Linear(BH * self.shsc * R, D, bias=False)                    # the soft-selected rows -> residual DIRECTLY
        if self.top2:
            din = La * N * dv + La * kvd + 32
            dims = [din, 256, 128, 64, N]
            self.t2w = nn.ParameterList([nn.Parameter(torch.empty(BH, a, b).normal_(0, a ** -0.5)) for a, b in zip(dims[:-1], dims[1:])])
            self.t2b = nn.ParameterList([nn.Parameter(torch.zeros(BH, b)) for b in dims[1:-1]])
            self.wo_t2 = nn.Linear(BH * 2 * La * dv, D, bias=False)             # the two selected 64-dim rows -> residual DIRECTLY
        if self.possel:
            din = La * N * dv + La * kvd + 32
            self.ps1 = nn.Parameter(torch.empty(BH, din, 512).normal_(0, din ** -0.5)); self.ps1b = nn.Parameter(torch.zeros(BH, 512))
            self.ps2 = nn.Parameter(torch.empty(BH, 512, La * N).normal_(0, 512 ** -0.5))   # -> one logit per row of the big head
            self.wo_ps = nn.Linear(BH * dv, D, bias=False)                     # the selected row's value -> residual DIRECTLY
        self.n_prev = layer_idx if c.xbus else 0
        if self.n_prev > 0:
            self.bus_in = nn.Linear(self.n_prev * BH * self.comp.out_dim, D, bias=False)
            self.bus_g = nn.Parameter(torch.zeros(()))
        self._masks = {}

    def _band(self, T, device):
        key = (T, str(device))
        m = self._masks.get(key)
        if m is None:
            i = torch.arange(T, device=device)
            m = (i[:, None] >= i[None, :]) & ((i[:, None] - i[None, :]) < self.W)   # True = attend
            self._masks[key] = m
        return m

    def _window(self, cq, ck, cv, T):
        """(B,H,T,kvd) -> (B,H,T,kvd): causal attention over the last kv_window tokens."""
        if _FLEX is not None and cq.is_cuda:
            key = ("flex", T, str(cq.device))
            bm = self._masks.get(key)
            if bm is None:
                W = self.W
                bm = _create_block_mask(lambda b, h, qi, ki: (qi >= ki) & (qi - ki < W),
                                        B=None, H=None, Q_LEN=T, KV_LEN=T, device=cq.device)
                self._masks[key] = bm
            return _FLEX(cq, ck, cv, block_mask=bm)
        return F.scaled_dot_product_attention(cq, ck, cv, attn_mask=self._band(T, cq.device))

    def _lane_merge(self, x, d):
        """x (B,T,H*d) -> soft-shared across each big head's lanes: pairs first, then the quad."""
        La, BH = self.La, self.BH
        if La < 2:
            return x
        B, T, _ = x.shape
        xp = x.reshape(B, T, BH, La // 2, 2, d)
        gp = torch.sigmoid(self.lane_gp.to(x.dtype)).view(1, 1, BH, La // 2, 1, 1)
        xp = xp + gp * (xp.mean(4, keepdim=True) - xp)
        x = xp.reshape(B, T, BH, La, d)
        if La >= 4:
            gq = torch.sigmoid(self.lane_gq.to(x.dtype)).view(1, 1, BH, 1, 1)
            x = x + gq * (x.mean(3, keepdim=True) - x)
        return x.reshape(B, T, BH * La * d)

    def _hselread(self, buf, x):
        """Learned selector over rows, read with a FLOOR so no row is ever exactly zero, supervised in
        hindsight.  buf: (B,T,H,N,dv) -> the mixed row per subhead."""
        B, T, _ = x.shape
        H, dv, N, grp = self.H, self.dv, self.N, self.hs_grp
        ng = H // grp
        rws = buf.reshape(B, T, ng, grp * N * dv)                  # a VIEW of the buffer, no copy
        h = self.hs_in_rows(rws.to(self.hs_in_rows.weight.dtype))
        if self.hs_in_x is not None:
            h = h + self.hs_in_x(x.to(h.dtype)).unsqueeze(2)       # broadcast over groups, never replicated
        sc = self.hs_out(self.hs_mlp(h)).reshape(B, T, H, N).float()

        valid = buf.float().pow(2).sum(-1) > 1e-12            # an all-zero row was never written
        sc = sc.masked_fill(~valid, float("-inf"))
        lam = self.hs_lam if (self.training and torch.is_grad_enabled()) else 0.0
        rd, att = _HSelAtt.apply(sc, buf, valid, self.hs_floor, lam, self.hs_tau)
        ent = -(att.clamp_min(1e-9) * att.clamp_min(1e-9).log2()).sum(-1).mean()
        self.hs_min = att.masked_fill(~valid, 1.0).min().detach()
        return rd, ent, valid.float().sum(-1).mean()

    def _xselloss(self, buf, v, x, src):
        """Cross-entropy on the READ's own cosine scores toward the row that holds the needed fact.

        src[b,t] = the earlier position whose write holds what position t must answer, or -1.
        The row is found by matching the stored rows at time t against the VALUE written at that
        source position -- with a hard write that value landed in exactly one row and persists."""
        B, T, _ = x.shape
        H, dv, N, dks = self.H, self.dv, self.N, self.tkdk
        sel = src >= 0
        if not bool(sel.any()):
            return None
        p = src.clamp_min(0)
        vp = v.gather(1, p[:, :, None, None].expand(B, T, H, dv))          # what was written at the source
        ky = buf[..., :dks]
        nsq = ky.float().pow(2).sum(-1)
        valid = nsq > 1e-12
        tgt = torch.einsum("bthnd,bthd->bthn", ky.float(), vp[..., :dks].float())
        tgt = tgt.masked_fill(~valid, float("-inf")).argmax(-1)            # (B,T,H) the correct row
        q = self.tk_q(x).reshape(B, T, H, dks)
        q = q / q.float().norm(dim=-1, keepdim=True).clamp_min(1e-6).to(q.dtype)
        sc = torch.einsum("bthnd,bthd->bthn", ky, q).float() / nsq.clamp_min(1e-12).sqrt()
        sc = (sc / max(self.xs_tau, 1e-4)).masked_fill(~valid, float("-inf"))
        sc, tg = sc[sel].reshape(-1, N), tgt[sel].reshape(-1)
        keep = torch.isfinite(sc).any(-1) & torch.isfinite(sc.gather(1, tg[:, None]).squeeze(1))
        if not bool(keep.any()):
            return None
        return F.cross_entropy(sc[keep], tg[keep])

    def _tkread(self, kbuf, buf, x):
        """Per-subhead TIED-KEY read: cos(row[:tkdk], q(x)) x learned temperature, softmax over that subhead's own
        rows (empty rows masked), value = the mixed row. No stored key: a row's own leading dims are its address."""
        B, T, _ = x.shape; H, dv, N, dks = self.H, self.dv, self.N, self.tkdk
        q = self.tk_q(x).reshape(B, T, H, dks)
        q = q / q.float().norm(dim=-1, keepdim=True).clamp_min(1e-6).to(q.dtype)
        ky = kbuf                                                                 # the STORED key (no longer a slice of the value)
        # cosine WITHOUT materialising a normalised copy of the keys: divide the raw dot by the key norm afterwards.
        # Same value; it saves a (B,T,H,N,dk) temporary per layer and one full pass writing it.
        nsq = ky.float().pow(2).sum(-1)                                           # (B,T,H,N)
        # NANFIX 2026-09-13: the denominator clamps BEFORE the sqrt, not after.  A HARD write leaves an
        # unwritten row EXACTLY zero, so nsq == 0; d/dnsq sqrt(nsq) is +inf there and clamp_min hands it an
        # incoming gradient of 0, so the old sqrt-then-clamp form produced 0 * inf = NaN in the BACKWARD
        # (the forward was finite, which is why s4 trained step 0 and died on step 1's first sample).
        # Soft writes never hit it: every row gets a fraction of every write, so nsq is never exactly 0.
        # Verified: forward and att bitwise identical, gradients identical on written rows, NaN gone.
        sc = torch.einsum("bthnd,bthd->bthn", ky, q).float() / nsq.clamp_min(1e-12).sqrt() * torch.exp(self.log_rsel.float())
        valid = nsq > 1e-12                                                       # a row with an all-zero address was never written (and could never be matched)
        att = torch.nan_to_num(torch.softmax(sc.masked_fill(~valid, float("-inf")), -1))
        if self.tk_hard:
            att = F.one_hot(att.argmax(-1), att.shape[-1]).to(att.dtype) + att - att.detach()   # ONE row, straight-through
        rd = torch.einsum("bthn,bthnv->bthv", att.to(buf.dtype), buf)
        if self.osel:
            if rd.requires_grad:
                # store only a DETACHED tensor on the module: no graph is captured, so this cannot
                # create the reference cycle that leaked 198 MB per layer in the earlier attempt.
                rd.register_hook(lambda gg, _s=self: setattr(_s, "_last_g", gg.detach()))
            if self._osel_tgt is not None:
                tg, msk = self._osel_tgt
                if msk.any():
                    scm = sc.masked_fill(~valid, float("-inf"))
                    self.osel_loss = F.cross_entropy(scm[msk].reshape(-1, N), tg[msk].reshape(-1))
                else:
                    self.osel_loss = None
        ent = -(att.clamp_min(1e-9) * att.clamp_min(1e-9).log2()).sum(-1).mean()
        return rd, ent, valid.float().sum(-1).mean()

    def _compress(self, state, kv_in, qsel):
        return self.comp(state, kv_in, qsel)

    def forward(self, x, bus_prev=None, src=None):
        B, T, D = x.shape
        H, BH, La, dv, N, kvd, G = self.H, self.BH, self.La, self.dv, self.N, self.kvd, self.G
        parts = self.w_in(x).split(self.sizes, -1)
        pv, pa, pg, pck, pcq, pcv = parts[:6]
        qsel = parts[6].reshape(B, T, BH, self.nsel * dv) if self.nsel > 0 else None
        prs = parts[6 + (1 if self.nsel > 0 else 0)] if self.rowsel == "free" else None
        pa, pg = self._lane_merge(pa, self.kdim), self._lane_merge(pg, 1)
        # ---- keyless slot write-scan (exact fp32; buffer emitted in the model dtype) ----
        v = pv.reshape(B, T, H, dv)
        kk = pa.reshape(B, T, H, self.kdim)                      # the key written alongside the value
        qq = self._lane_merge(parts[-1], self.kdim).reshape(B, T, H, self.kdim) if self.wq else None
        g = torch.sigmoid(pg.float() + self.g_bias.float())                        # (B,T,H) write probability
        if self.krep > 0 and self.training:
            # WRITE-TIME KEY REPULSION, ported from the delta line (memlm_m5._krep_fn), where its own
            # comment records the identical failure: "22.6% of strongly-written items share a
            # near-duplicate address (|cos|>0.9) -- the delta rule merges them."
            # Measured here: with nothing keeping keys apart, the write softmax never peaks (top-row
            # weight 0.19-0.45 instead of ~1), so a SOFT write bleeds into every row -- median
            # retention 0.0000 over a 512-token sequence -- and a HARD write's straight-through bias
            # is huge. Both diverged or erased the memory.
            # Penalise only PAIRS above tau, weighted by both write strengths, excluding near-adjacent
            # tokens. Structurally unbiased: keys already spread => no pairs over tau => zero loss.
            # Stride 4 keeps the (B,H,Ts,Ts) Gram small, exactly as the delta line does.
            _st = 4 if T <= 1024 else 8
            _kr = kk[:, ::_st].float()
            _kr = _kr / _kr.norm(dim=-1, keepdim=True).clamp_min(1e-6)      # _krep_fn wants unit norm
            self.krep_loss = _M5._krep_fn(_kr, g[:, ::_st].float(), self.krep_tau)
        else:
            self.krep_loss = None
        act = (g > 0.5).float()
        self.wrate = act.mean().detach()                                             # fraction of (token, subhead) that write
        if self.wbud > 0:                                                       # count budget, straight-through
            _rate = (act + (g - g.detach())).mean()
            self.wbud_loss = F.relu(_rate - self.wbud_rho)                      # CEILING: pushes down only when above rho
            if self.wbud_lo > 0:
                self.wbud_loss = self.wbud_loss + F.relu(self.wbud_lo - _rate)   # optional FLOOR: also pushes up when below
        else:
            self.wbud_loss = None
        # KEY-MATCHED WRITE: the address is a cosine match of the new key against the keys already
        # stored, so re-writing the same key lands on the row that holds it.  `- fr*o` (the learned
        # free-row preference) and the occupancy `o` are the originals, unchanged.
        kbuf, buf = scan_kv(kk.float(), v.float(), g, torch.exp(self.log_free.float()) * self.free_bias,
                                N, torch.exp(self.log_wsharp.float()), G=G,
                                hard=self.hard, hard_gate=self.hard_gate,
                                detach_addr=bool(self.kdetach),
                                q=(None if qq is None else qq.float()))
        kbuf, buf = kbuf.to(x.dtype), buf.to(x.dtype)
        if self.xsel and self.training and src is not None:
            self.xs_loss = self._xselloss(buf, v, x, src)
        # ---- exact recent window ----
        _rp = (lambda z: _rope(z, frac=_ROPE_FRAC)) if _ROPE else (lambda z: z)
        cq = _rp(pcq.reshape(B, T, self.KVH, kvd)).transpose(1, 2)     # ROPE on the window only: it needs to know WHERE
        ck = _rp(pck.reshape(B, T, self.KVH, kvd)).transpose(1, 2)     # (the slot read stays unroped: it matches by CONTENT)
        cv = pcv.reshape(B, T, self.KVH, kvd).transpose(1, 2)
        kvr = self._window(cq, ck, cv, T).transpose(1, 2)                          # (B,T,H,kvd)
        # ---- retrieval: the compressor over the slot buffer + the window reads, every G tokens ----
        TB = buf.shape[1]
        # the buffer starts EMPTY and fills over ~N/write-rate tokens; the funnel's output must not scale with how
        # many slots are filled (measured: |own128| x2 between the first and last 128 tokens of a 1024 window at a
        # checkpoint trained only at T=128). The funnel LayerNorms its stage outputs (compressor.py) -- an input-side RMS
        # inflated near-empty buffers x14 and killed learning -- and gets each lane's fill fraction as a feature (32-block).
        if self.comp_off:
            own = cross = sel = None
            delta = self.wo_kv(kvr.reshape(B, T, self.KVH * kvd))          # the exact recent window straight to the residual
        else:
            state = buf.reshape(B, TB, BH, La * N * dv)                                      # raw values; the funnel normalises INSIDE (LayerNorm after its stages)
            fill = (buf.reshape(B, TB, BH, La, N, dv).float().abs().amax(-1) > 0).float().mean(-1)     # (B,TB,BH,La) filled fraction
            fill = F.pad(fill, (0, 32 - La)).to(x.dtype)
            kv_in = torch.cat([(kvr[:, ::G] if G > 1 else kvr).reshape(B, TB, BH, La * kvd), fill], -1)
            qs = None if qsel is None else (qsel[:, ::G] if G > 1 else qsel)
            if self.emit_ckpt and self.training and torch.is_grad_enabled():
                own, cross, sel = _ckpt(self._compress, state, kv_in, qs, use_reentrant=False)
            else:
                own, cross, sel = self.comp(state, kv_in, qs)                            # (B,TB,BH,128), (B,TB,BH,64), (B,TB,BH,nsel*dv)
            if G > 1:                                                                    # hold the page for G tokens
                own = own.repeat_interleave(G, dim=1)[:, :T]
                cross = cross.repeat_interleave(G, dim=1)[:, :T]
                sel = None if sel is None else sel.repeat_interleave(G, dim=1)[:, :T]
            delta = self.wo_own(own.reshape(B, T, BH * self.comp.own_dim))
        if sel is not None:
            delta = delta + self.wo_sel(sel.reshape(B, T, BH * self.nsel * dv))
        if self.shsc and G == 1:
            # SHARED ROW SCORER: the same small network scores every row against the token's query; soft attention over
            # rows (empty rows masked) so EVERY row's score receives a gradient about its own content; the mix is injected.
            R = La * dv; S = self.shsc
            rows = buf.float().reshape(B, T, BH, La, N, dv).permute(0, 1, 2, 4, 3, 5).reshape(B, T, BH, N, R)      # (B,T,BH,N,64)
            qx = self.ss_q(x).float().reshape(B, T, BH, S, R)                                                      # (B,T,BH,S,64)
            hq = torch.einsum("btgsi,gsih->btgsh", qx, self.ss_w1[:, :, :R].float())                                # (B,T,BH,S,64)
            hr = torch.einsum("btgni,gsih->btgsnh", rows, self.ss_w1[:, :, R:].float())                             # (B,T,BH,S,N,64)
            hsc = F.gelu(hq.unsqueeze(4) + hr + self.ss_b1.float()[None, None, :, :, None, :])
            sc = torch.einsum("btgsnh,gsh->btgsn", hsc, self.ss_w2.float())                                         # (B,T,BH,S,N)
            valid = (rows.abs().amax(-1) > 0).unsqueeze(3)                                                          # (B,T,BH,1,N)
            att = torch.nan_to_num(torch.softmax(sc.masked_fill(~valid, float("-inf")), -1))
            mix = torch.einsum("btgsn,btgnr->btgsr", att, rows).reshape(B, T, BH * S * R)
            delta = delta + self.wo_ss(mix.to(x.dtype))
        if self.hsel and G == 1:
            # checkpoint it exactly as the cosine read is checkpointed, or the comparison is unfair AND it
            # does not fit: 12 layers each storing a 400/400/200/100 MLP's activations measured +15 GB over
            # the control at 4096 tokens.  The hindsight hooks are registered under `torch.is_grad_enabled()`,
            # so they are skipped in the no-grad forward pass and attach during the backward recomputation --
            # verified by hs_fired still counting up.
            if self.emit_ckpt and self.training and torch.is_grad_enabled():
                rd, _e, _f = _ckpt(self._hselread, buf, x, use_reentrant=False)
            else:
                rd, _e, _f = self._hselread(buf, x)
            self.tk_ent, self.tk_fill = _e.detach(), _f.detach()
            delta = delta + self.wo_hs(rd.reshape(B, T, -1).to(x.dtype))
        if self.tksel and G == 1:
            if self.emit_ckpt and self.training and torch.is_grad_enabled():
                rd, _e, _f = _ckpt(self._tkread, kbuf, buf, x, use_reentrant=False)     # recompute the read in backward (it was the only unchecked path once the funnel went)
            else:
                rd, _e, _f = self._tkread(kbuf, buf, x)
            self.tk_ent, self.tk_fill = _e.detach(), _f.detach()
            delta = delta + self.wo_tk(rd.reshape(B, T, H * dv))
        if self.cone and G == 1:
            rows = buf.reshape(B, T, BH, La, N, dv).permute(0, 1, 2, 4, 3, 5).reshape(B, T, BH, N, La * dv)   # a big head's row = that slot across its lanes
            ctx = (self.cone_qx(x).reshape(B, T, BH, self.cone_c) if self.cone_src == "x"
                   else kvr.reshape(B, T, BH, La * kvd))
            h = torch.cat([buf.reshape(B, T, BH, La * N * dv).float(), ctx.float()], -1)
            for li, w in enumerate(self.cw):
                h = torch.einsum("btgi,gio->btgo", h, w.float())
                if li < len(self.cw) - 1:
                    h = F.gelu(h + self.cb[li].float())
            a = torch.softmax(h, -1)                                                       # (B,T,BH,N) SOFT selection
            self.cone_ent = -(a.clamp_min(1e-9) * a.clamp_min(1e-9).log2()).sum(-1).mean().detach()
            mix = torch.einsum("btgn,btgnr->btgr", a, rows.float()).reshape(B, T, BH * La * dv)
            delta = delta + self.wo_cone(mix.to(x.dtype))
        if self.top2 and G == 1:
            # rows of a big head = its La lanes' slot values side by side (N rows x La*dv); the funnel scores every row
            rows = buf.float().reshape(B, T, BH, La, N, dv).permute(0, 1, 2, 4, 3, 5).reshape(B, T, BH, N, La * dv)
            hcur = torch.cat([state.float(), kv_in.float()], -1)                                  # (B,T,BH, 3k+160)
            for li, w in enumerate(self.t2w):
                hcur = torch.einsum("btgi,gio->btgo", hcur, w.float())
                if li < len(self.t2w) - 1:
                    hcur = F.gelu(hcur + self.t2b[li].float())
            sc2 = hcur                                                                            # (B,T,BH,N) row scores
            soft = torch.softmax(sc2, -1)
            top = sc2.topk(2, dim=-1).indices                                                      # (B,T,BH,2)
            picked = []
            for kk in range(2):
                idx = top[..., kk]
                wgt = 1.0 + soft.gather(-1, idx.unsqueeze(-1)).squeeze(-1) - soft.gather(-1, idx.unsqueeze(-1)).squeeze(-1).detach()   # ST: value 1, gradient of the soft score
                picked.append(wgt.unsqueeze(-1) * rows.gather(3, idx[..., None, None].expand(B, T, BH, 1, La * dv)).squeeze(3))
            sel2 = torch.cat(picked, -1).reshape(B, T, BH * 2 * La * dv)
            delta = delta + self.wo_t2(sel2.to(x.dtype))
        if self.possel and G == 1:
            # POSITION SELECTOR: the compressor sees every row and the context and picks a position (dense MLP -> row logits)
            inp = torch.cat([state.float(), kv_in.float()], -1)                                   # (B,T,BH, 3k+160)
            hpos = F.gelu(torch.einsum("btgi,gio->btgo", inp, self.ps1.float()) + self.ps1b.float())
            plog = torch.einsum("btgo,gor->btgr", hpos, self.ps2.float())                         # (B,T,BH, La*N)
            ap = torch.softmax(plog, -1)
            if self.hard:
                ap = F.one_hot(plog.argmax(-1), La * N).to(ap.dtype) + ap - ap.detach()
            rows = state.float().reshape(B, T, BH, La * N, dv)
            rdp = torch.einsum("btgr,btgrd->btgd", ap, rows).reshape(B, T, BH * dv)
            delta = delta + self.wo_ps(rdp.to(x.dtype))
        if self.nrs > 0 and G == 1:
            # ROW-ID READ: choose a row by index from the token (tied: the write address logits themselves, so an entity
            # is read back under the id it was written under), read that row's value, straight to the residual.
            logits = wa.reshape(B, T, H, 1, N) if self.rowsel == "tied" else (prs.reshape(B, T, H, self.nrs, N).float() * torch.exp(self.log_addr.float()))
            ar = torch.softmax(logits, -1)
            if self.hard:
                ar = F.one_hot(logits.argmax(-1), N).to(ar.dtype) + ar - ar.detach()
            rd = torch.einsum("bthrn,bthnd->bthrd", ar, buf.float()).reshape(B, T, H * self.nrs * dv)
            delta = delta + self.wo_rs(rd.to(x.dtype))
        if bus_prev is not None:
            delta = delta + self.bus_g.to(x.dtype) * self.bus_in(bus_prev)
        return delta, (None if cross is None else cross.reshape(B, T, BH * self.comp.out_dim))


class SlotBlock(nn.Module):
    def __init__(self, c: Cfg, layer_idx: int):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(c.d_model), nn.LayerNorm(c.d_model)
        self.mix = SlotMixer(c, layer_idx)
        self.use_moe = bool(c.moe)
        if self.use_moe:
            self.ffn = _M5.MoE(c)
        else:
            self.w1 = nn.Linear(c.d_model, c.d_ff, bias=False)
            self.w3 = nn.Linear(c.d_model, c.d_ff, bias=False)
            self.w2 = nn.Linear(c.d_ff, c.d_model, bias=False)

    def forward(self, x, bus_prev=None, src=None):
        d, cross = self.mix(self.n1(x), bus_prev, src)
        x = x + d
        h = self.n2(x)
        x = x + (self.ffn(h) if self.use_moe else self.w2(F.silu(self.w1(h)) * self.w3(h)))
        return x, cross


class MemLM(nn.Module):
    def __init__(self, c: Cfg):
        super().__init__()
        self.c = c
        self.emb = nn.Embedding(c.vocab, c.d_model)
        self.pre_blocks = nn.ModuleList([_M5.PreFFN(c) for _ in range(c.pre)])
        self.blocks = nn.ModuleList([SlotBlock(c, i) for i in range(c.layers)])
        self.nf = nn.LayerNorm(c.d_model)
        self.head = nn.Linear(c.d_model, c.vocab, bias=False)
        if _os.environ.get("MEMLM_UNTIE") == "1":
            # UNTIE (env-gated, default OFF so every banked run is bit-identical).  Measured
            # 2026-09-16: reverting this ONE tied tensor from the diverging step-4000 weights to
            # the working step-2971 weights recovers 39% of the +1.46 CE gap, and the damage is
            # DIRECTIONAL -- rescaling it to the old norm recovers only 6%.  A tied tensor serves
            # both the input lookup and the output projection, so output-side gradient rotates the
            # input embedding.  Untied, the head can rotate without dragging the embedding with it.
            with torch.no_grad():
                self.head.weight.copy_(self.emb.weight)    # starts identical to the tied model
        else:
            self.head.weight = self.emb.weight             # tied
        self.kdec = self.krep = self.wbud = None           # set per-forward below
        self.xsel = None
        self.osel = None
        self.util = self.maxvio = self.drop = None
        self.apply(self._init)                             # same as memlm_m5: N(0, 0.02) for every Linear/Embedding
        for b in self.blocks:                              # the cross-layer gates start closed (apply() must not touch them)
            if hasattr(b.mix, "bus_g"):
                nn.init.zeros_(b.mix.bus_g)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def set_moe_loop(self, flag: bool):
        if self.c.moe:
            for b in self.blocks:
                b.ffn.use_loop = flag

    def forward(self, idx, states=None, return_states=False, skip_head=False, src=None):
        assert states is None and not return_states, "memlm_slot v1: no state carry across calls"
        x = self.emb(idx)
        for pb in self.pre_blocks:
            x = pb(x)
        bus = []
        for b in self.blocks:
            bp = torch.cat(bus, -1) if (self.c.xbus and bus) else None
            x, cross = b(x, bp, src)
            if self.c.xbus:
                bus.append(cross)
        if getattr(self.c, "osel", 0):
            _lo = [b.mix.osel_loss for b in self.blocks if b.mix.osel_loss is not None]
            self.osel = torch.stack(_lo).mean() if _lo else None    # trainer adds osel_lam * model.osel
        if getattr(self.c, "xsel", 0):
            _lx = [b.mix.xs_loss for b in self.blocks if b.mix.xs_loss is not None]
            self.xsel = torch.stack(_lx).mean() if _lx else None   # the trainer adds xsel_lam * model.xsel
        if self.c.wbud > 0:
            _lw = [b.mix.wbud_loss for b in self.blocks if b.mix.wbud_loss is not None]
            self.wbud = torch.stack(_lw).sum() if _lw else None      # the trainer adds a.wbud * model.wbud
        if float(getattr(self.c, "krep", 0.0)) > 0:
            _lr = [b.mix.krep_loss for b in self.blocks if b.mix.krep_loss is not None]
            self.krep = torch.stack(_lr).sum() if _lr else None   # the trainer adds a.krep * model.krep
        else:
            self.krep = None
        x = self.nf(x)
        if not skip_head:
            x = self.head(x)
        if self.c.moe:
            us = [b.ffn.util for b in self.blocks if b.ffn.util is not None]
            self.util = torch.stack(us).mean(0) if us else None
            mv = [b.ffn.maxvio for b in self.blocks if b.ffn.maxvio is not None]
            self.maxvio = torch.stack(mv).max() if mv else None
            dr = [b.ffn.drop for b in self.blocks if b.ffn.drop is not None]
            self.drop = torch.stack(dr).max() if dr else None
        return x
