# -*- coding: utf-8 -*-
"""memlm_sh -- ONE shared slot memory, written once, read by every layer at or above the writer.

The DeepSeek-V4.1-Flash idea (arXiv 2609.19969, 'Causal Encoder-Decoder') adapted to this model: there,
a 20-layer encoder is followed by a 20-layer decoder whose global KV is projected ONCE from the hidden
state of layer L/2 and reused by every decoder layer; all layers keep their own sliding window.  memlm_kv gives every one of the 12 layers
its own memory (2 heads x 96 slots x (64+64) each, 294,912 numbers in total) and runs 12 slot scans
in sequence, each occupying 2 of the GPU's 28 SMs.  Here the SAME total budget is spent on a single
memory with more slots per head, written by ONE layer and read by all layers above it:

    layers < sh_write    the 16-token window + MoE only (no long-range memory)
    layer  == sh_write   computes key / value / write gate from ITS input, runs the one slot scan
    layers >= sh_write   read the shared memory with their own queries (sh_nq per head)

WHY ONE WRITER, NOT ALL LAYERS: if every layer wrote into one memory, layer 0 at token t would read
what layer 11 wrote at token t-1 -- a recurrence across BOTH depth and time, which can only be
computed one token at a time, ~T x slower to train.  A single writer keeps the graph a plain DAG
(write once per sequence, every reader above it sees the state from earlier positions), which is
exactly what DeepSeek-V4.1 does with its encoder->decoder split.  It also keeps writes per slot sane:
12 writers would put ~12k writes per head into 192 slots, the regime where this project already
measured every soft-written row converging to the same blend.

Causality: the scan emits the state BEFORE token t's write at position t (see slotkern_kvt), so a
reader at t sees writes from positions < t only -- the same contract the per-layer memory had.

Built as a separate module (MEMLM_MOD=memlm_sh) so memlm_kv, and every checkpoint and battery that
depends on it, is untouched.
"""
import inspect
import math
import os as _os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as _ckpt

import memlm_kv as _K
import memlm_m5 as _M5
from slotkern_kv import scan_kv, ref_scan_kv
try:
    from slotkern_kvg import kv_scan_g
except Exception:          # no triton (CPU-only box): the reference handles any stride
    kv_scan_g = None
# KV_CUDA=1: the CUDA scan (slotkern_cuda.kv_scan_c) -- same contract, hard + detached address in CUDA, every other
# mode falls back to kv_scan_g inside it.  Imported OUTSIDE the try on purpose: a failure here must be loud, not a
# silent fall-back to the reference scan.
if _os.environ.get("KV_CUDA", "0") == "1":
    from slotkern_cuda import kv_scan_c as kv_scan_g
    print("memlm_sh: memory scan = CUDA (slotkern_cuda.kv_scan_c)", flush=True)

_SH_DEFAULTS = dict(sh_write=4, sh_heads=12, sh_slots=192, sh_kdim=64, sh_dv=64, sh_nq=2, sh_stride=16,
                    sh_qbits=0, sh_qgroup=16, sh_enc_write=-1, sh_enc_slots=96,
                    sh_gate_floor=0.0, sh_wq=0, sh_wq_std=0.0, sh_wq_sign=1.0, sh_fr=0.0, sh_nogate=0,
                    kv_nowin_from=-1, sh_pos=0, sh_pos_nfreq=8, sh_pos_wmin=64.0, sh_pos_wmax=4096.0, sh_pos_scale=1.0, sh_vnorm=0,
                    sh_wq_detach=0, sh_occbin=0, sh_frtok=0.0, sh_frmax=2.0, sh_wqk=0, sh_rsharp=0.0, kv_nq=1)


class Cfg(_K.Cfg):
    """memlm_kv.Cfg plus the shared-memory shape.  The per-layer memory fields of the base config
    (heads, n_slots, dv, lanes, ...) are accepted and ignored: in this model only the window and the
    shared memory exist."""

    def __init__(self, sh_write=4, sh_heads=12, sh_slots=192, sh_kdim=64, sh_dv=64, sh_nq=2, sh_stride=16,
                 sh_qbits=0, sh_qgroup=16, sh_enc_write=-1, sh_enc_slots=96,
                 sh_gate_floor=0.0, sh_wq=0, sh_wq_std=0.0, sh_wq_sign=1.0, sh_fr=0.0, sh_nogate=0,
                 kv_nowin_from=-1, sh_pos=0, sh_pos_nfreq=8, sh_pos_wmin=64.0, sh_pos_wmax=4096.0, sh_pos_scale=1.0, sh_vnorm=0,
                 sh_wq_detach=0, sh_occbin=0, sh_frtok=0.0, sh_frmax=2.0, sh_wqk=0, sh_rsharp=0.0, kv_nq=1, **kw):
        super().__init__(**kw)
        self.sh_write, self.sh_heads, self.sh_slots = int(sh_write), int(sh_heads), int(sh_slots)
        self.sh_kdim, self.sh_dv, self.sh_nq = int(sh_kdim), int(sh_dv), max(1, int(sh_nq))
        assert 0 <= self.sh_write < self.layers, "sh_write must be a layer index"
        # OUTPUT STRIDE.  The memory is copied out every sh_stride tokens and a reader at token t reads
        # the copy from the start of its block, so it is blind to the last <= sh_stride-1 writes.  Those
        # must all fall inside the exact window or a fact would be unreachable for a few tokens.
        self.sh_stride = max(1, int(sh_stride))
        assert self.sh_stride <= self.kv_window, \
            "sh_stride %d > kv_window %d: writes in the gap would be visible to neither path" % (self.sh_stride, self.kv_window)
        # STORAGE PRECISION, trained (QAT) and evaluated alike.  0 = full precision.
        self.sh_qbits, self.sh_qgroup = int(sh_qbits), int(sh_qgroup)
        assert self.sh_qbits in (0, 2, 3, 4, 8), "sh_qbits must be 0 (off) or 2/3/4/8"
        assert self.sh_qbits == 0 or (self.sh_kdim % self.sh_qgroup == 0 and self.sh_dv % self.sh_qgroup == 0), \
            "sh_qgroup must divide sh_kdim and sh_dv"
        # ENCODER MEMORY (DeepSeek-V4.1's encoder layers have long-range attention of their own; with only the
        # 16-token window below layer sh_write, the main writer would build its keys from local context
        # alone).  A second shared memory, same heads/dims, written at layer sh_enc_write and read by layers
        # sh_enc_write .. sh_write-1.  -1 = off (sh1/sh2 behaviour, bit-identical).
        self.sh_enc_write, self.sh_enc_slots = int(sh_enc_write), int(sh_enc_slots)
        assert self.sh_enc_write < self.sh_write, "the encoder memory must be written below the main writer"
        # GATE FLOOR: write-gate values below this are zeroed in the FORWARD (the token writes nothing) while the
        # BACKWARD passes the gradient straight through, so a closed token can still learn to open.  With top-1
        # writes (slot_hard) a weak write lands entirely on one slot and overwrites part of a stored fact;
        # measured on sh6 @7000, 82% of tokens wrote more than 0.01.  0 = off.
        self.sh_gate_floor = float(sh_gate_floor)
        # WRITE QUERY (two keys at write time): sh_wq=1 gives each writer its OWN projection used only to
        # CHOOSE the slot (cosine against the stored keys); the stored key stays the key projection.  With one
        # vector doing both jobs, "where should this go" and "what will match this later" fight each other --
        # measured on sh7 @8000 (top-1): 13.5 of 192 slots used, 96.6% of writes overwrite, median life 6 tokens.
        # The write query is initialised as an exact COPY of the key projection, so switching it on does not
        # change the function; it then learns its own job.  Registered LAST in the model (see MemLM) so a resumed
        # optimizer state can be remapped exactly.
        self.sh_wq = int(sh_wq)
        # sh_wq_std > 0: the write query starts as an INDEPENDENT N(0, std) matrix, not a copy of the key.
        # Measured on a fresh model (gate 0.999, T=128): the copy init sends 87% of main-memory writes into an
        # already-used row (9-25 rows per head); random std 0.01 sends 22% (88-112 rows).  A SCALED copy does
        # nothing -- the address is a cosine, so 0.01*copy picks the same rows and only multiplies its gradient
        # by 100.  Only affects a fresh model; a loaded state dict overrides it.
        self.sh_wq_std = float(sh_wq_std)
        # sh_wq_sign (used only when sh_wq_std == 0): the write query starts as sign * the key projection.  -1 points
        # it AWAY from the keys, which share a common direction, so a used row scores below an empty one.  Measured on a
        # fresh model (T=128, gate 0.999): main writes into a used row copy 86.5%, random 0.01 21.7%, -copy 3.3%.
        self.sh_wq_sign = float(sh_wq_sign)
        # sh_fr > 0: the free-row preference fr is FIXED at this value (no gradient).  A used row beats an empty one only
        # if temp*cos > fr*o; with fr/temp > 1 (fr 16, temp 15.9) no cosine can, so a write takes a new row until the
        # memory is full.  Learned fr barely moved in sh9/sh10 (4.48 -> 4.2-4.4) and drifted DOWN.  0 = learned, as before.
        self.sh_fr = float(sh_fr)
        # sh_nogate=1: NO write gate, every token writes with g = 1.  Measured: a learnable gate SHUTS between steps
        # 1000 and 2000 in every run (sh9 soft 0.013, sh10 soft 0.09, sh12 hard: main 0.0% of tokens writing at 1750)
        # before the memory is useful, so nothing is ever stored worth reading.  With sh_fr 16 each token takes its own
        # row, so at T=128 the main memory holds the whole sequence.
        self.sh_nogate = int(sh_nogate)
        # kv_nowin_from L >= 0: layers >= L have NO rolling window -- their only token mixing is the shared-memory read
        # (asked 2026-10-12: decoder layers after the writer read the memory alone).  Layers < L keep the window, and the
        # residual stream carries their local context up.  Readers there still read the block-start snapshot, so the
        # last <= sh_stride-1 tokens reach them only through the residual.  -1 = off (window in every layer).
        self.kv_nowin_from = int(kv_nowin_from)
        # sh_pos=1: ABSOLUTE POSITION for the memories.  Measured 2026-10-12 (linear probe, step 3000): the stored main key
        # of sh5 (recalls) decodes the write position with R2 0.683, sh4 (same architecture, the new training, does not
        # recall) 0.113 -- the model had to COMPUTE position, and did so only under heavy recall pressure, and only at
        # wavelengths 128-256.  Here it is given for free: a fixed sin/cos code added to stored keys and read queries
        # through zero-initialised learned projections (see SharedMixer).  0 = off, no parameters.
        self.sh_pos, self.sh_pos_nfreq = int(sh_pos), int(sh_pos_nfreq)
        self.sh_pos_wmin, self.sh_pos_wmax = float(sh_pos_wmin), float(sh_pos_wmax)
        # sh_pos_scale S: the code is multiplied by S before the projections.  Step-0 function unchanged (projections
        # are zero); the position part learns S x faster.  sh15 (S=1) reached only 1.9% of the query by step 2000.
        self.sh_pos_scale = float(sh_pos_scale)
        # sh_pos with sh_wq: the write query gets its OWN zero-initialised position projection (pos_wq), so it starts as
        # exactly the content-only query and learns how much position to use (added 2026-10-17; before, this was refused).
        # sh_vnorm=1: OCCUPANCY-NORMALISED READ (asked 2026-10-17).  A soft write stores w*v, so a row written once at
        # gate g returns g*v: the read matches it at full cosine (keys are normalised) and gets a faint value back.
        # Measured on sh18 @12000: occupied rows hold o ~0.86 and only ~45 of 192 are used -- strength comes only
        # from MERGING many writes into one row.  Here the writer's value is (sh_dv - 1) learned channels plus a
        # constant 1, which the scan turns into exactly o (sum_t c_tn); each row is read as V / max(o, 0.01) with o
        # itself as the last channel.  A row written once reads at full strength whatever the gate.  0 = off.
        self.sh_vnorm = int(sh_vnorm)
        assert self.sh_vnorm == 0 or self.sh_dv >= 2
        # sh_wq_detach=1 (2026-10-18): the write query reads x.detach().  Measured on sh19 @2000: the top-1 address's
        # straight-through gradient, through the write query INTO the trunk, made layers 0-6 carry 8-10x sh18's
        # gradient (total 13.0 vs 1.7) before the loss turned; detaching only the query's input gave the trunk exactly
        # the gradient of cutting the address entirely, while the query's own weights keep learning.
        # sh_occbin=1 (2026-10-18): the free-row penalty counts a row as taken once written, fr*[o >= 1e-4], not fr*o.
        # With a soft gate o ~ g (median 0.019 at sh19 @2000), so fr*o ~ 0.6 against a write sharpness of 16.
        self.sh_wq_detach, self.sh_occbin = int(sh_wq_detach), int(sh_occbin)
        # sh_frtok c0 > 0 (sh20, asked 2026-10-20): the FREE-ROW PREFERENCE IS DECIDED PER TOKEN.  Each writer head
        # computes fr_t = temp * frmax * sigmoid(w . x_t.detach() + b), w zero-initialised and b set so the ratio
        # fr/temp starts at c0 for every token.  A taken row then beats the first empty row iff its address cosine
        # exceeds fr_t/temp, so a token can MERGE into a matching row while free rows remain (sh19c: fr 32 > temp,
        # so no token could merge until all rows were full).  c0 must sit inside the range of cosines the address
        # actually produces, or the straight-through gradient on fr is ~exp(-temp*(c0-cos)) and it never moves.
        # 0 = off: the scalar fr path, bit-identical.
        self.sh_frtok, self.sh_frmax = float(sh_frtok), float(sh_frmax)
        assert self.sh_frtok == 0 or 0 < self.sh_frtok < self.sh_frmax, "need 0 < sh_frtok < sh_frmax"
        assert self.sh_wq_detach == 0 or self.sh_wq, 'sh_wq_detach needs a write query (sh_wq=1)'
        # sh_wqk=1 (sh21, asked 2026-10-23): the write query takes a LEARNED share of the token's OWN stored key,
        #   q' = q + a * (|q| / |k|) * k.detach(),   a = tanh(w . x.detach() + b)   per token and per head, in (-1, 1).
        # a > 0 aims the write at rows whose key resembles the token's own, a < 0 away from them; the model decides.
        # w and b are ZERO-initialised, so a = 0 exactly and the first step is the sh20 function bit for bit.  k is
        # detached and the (|q|/|k|) scale too: the only new gradient path is into w and b (an address gradient into
        # the trunk is what diverged sh19).  WHY: measured on sh20/sh19c @12k (kvval, kvgate, kvvict): the row a write
        # replaces sits at the 51st percentile by value similarity and 43-46th by key; the decision margin is ~0.016 in
        # cosine; the write query never sees the token's key.
        self.sh_wqk = int(sh_wqk)
        # sh_rsharp C > 0 (sh23, asked 2026-10-25): each reader's score is multiplied by exp(C tanh(s/C)), s = w . x.detach()
        # per (token, head, read query), w zero-init -> x1 exactly at the switch.  One scalar rsel per layer forces every
        # query to the same sharpness; a dot product would give each query its own (|q|) but drops the normalisation that
        # keeps the scores bounded, so the sharpness is learned per query instead, bounded to exp(+-C).  0 = off.
        self.sh_rsharp = float(sh_rsharp)
        assert self.sh_rsharp >= 0, 'sh_rsharp must be >= 0'
        # kv_nq Q > 1 (sh25, asked 2026-10-28): every rolling-window K/V head gets Q queries instead of 1 (grouped-query
        # attention inside the 16-token window), so a layer can pull up to Q x kv_heads distinct recent tokens.  The
        # original query of each K/V head is untouched; the Q-1 extra ones live in MemLM.win_q2 / win_o2 (registered
        # LAST), and win_o2 starts at ZERO, so the model is the kv_nq=1 function exactly at the switch.  1 = off.
        self.kv_nq = int(kv_nq)
        assert self.kv_nq >= 1, 'kv_nq must be >= 1'
        assert self.sh_wqk == 0 or self.sh_wq, 'sh_wqk needs a write query (sh_wq=1)'
        assert self.sh_pos == 0 or (self.sh_pos_nfreq >= 1 and 0 < self.sh_pos_wmin <= self.sh_pos_wmax)

    @property
    def shared_state(self):
        n = self.sh_slots + (self.sh_enc_slots if self.sh_enc_write >= 0 else 0)
        return self.sh_heads * n * (self.sh_kdim + self.sh_dv)

    def __repr__(self):
        return (f"SHARED d_model{self.d_model} L{self.layers} write@{self.sh_write} "
                f"mem {self.sh_heads}h x {self.sh_slots} slots x ({self.sh_kdim}+{self.sh_dv}) nq{self.sh_nq} G{self.sh_stride} "
                f"{('q%db/g%d ' % (self.sh_qbits, self.sh_qgroup)) if self.sh_qbits else ''}"
                f"{('+ enc mem write@%d %d slots ' % (self.sh_enc_write, self.sh_enc_slots)) if self.sh_enc_write >= 0 else ''}"
                f"= {self.shared_state} | window {self.kv_window}x{self.kv_heads or self.heads}x{self.kv_dim}"
                f"{(' x%d queries' % self.kv_nq) if self.kv_nq > 1 else ''}")


# The trainer and the battery build a Cfg by filtering a dict against inspect.signature(Cfg.__init__).
# A **kw passthrough would hide every base parameter from that filter, so publish the FULL signature:
# the base class's explicit parameters followed by the shared-memory ones.
_base_sig = inspect.signature(_K.Cfg.__init__)
_extra = [inspect.Parameter(n, inspect.Parameter.KEYWORD_ONLY, default=d) for n, d in _SH_DEFAULTS.items()]
Cfg.__init__.__signature__ = _base_sig.replace(parameters=list(_base_sig.parameters.values()) + _extra)


# STORAGE PRECISION of the shared memory -- the copies the readers see, not the scan's own fp32 state.
# Scheme after DeepSeek-V4.1-Flash's FP4 global KV (one scale per 16 channels), at 3 bits: per group
# of `group` channels, MIDRISE uniform with 2^bits levels at +-(i+0.5)*s, s = absmax/2^(bits-1).  All 8
# codes are used (a midtread -3..3 wastes one), so the step is absmax/4 instead of absmax/3.  A group
# that is exactly zero (a never-written row) stays exactly zero, so `valid` is unchanged.
# Training passes the gradient straight through (QAT); cfg.sh_qbits applies it in train AND eval.
# SH_QBITS=<b> forces a precision at EVAL only -- for scoring a model trained at full precision.
# Bits per stored number: bits + 8/group if the scale is stored fp8 (3 + 0.5 = 3.5 at group 16); the scale
# is kept fp32 here, not rounded to fp8.
_QBITS_EVAL = int(_os.environ.get("SH_QBITS", "0"))

def _fq(x, bits, group):
    sh = x.shape
    xg = x.float().reshape(*sh[:-1], sh[-1] // group, group)
    a = xg.abs().amax(-1, keepdim=True)
    half = 2 ** (bits - 1)
    s = (a / half).clamp_min(1e-30)
    q = ((torch.floor(xg / s).clamp(-half, half - 1) + 0.5) * s)
    q = torch.where(a > 0, q, torch.zeros_like(q)).reshape(sh)
    return x + (q.to(x.dtype) - x).detach()          # straight-through: forward q, gradient identity

def _quantize_mem(kbuf, buf, bits, group):
    kq = _fq(kbuf, bits, group)
    nsq = kq.float().pow(2).sum(-1)
    return kq, _fq(buf, bits, group), torch.rsqrt(nsq.clamp_min(1e-12)), nsq > 1e-12


_VNORM_EPS = 1e-2   # rows with o below this read attenuated: bounds 1/o (and its gradient) at 100x


def _vnorm(buf):
    """buf (..., N, dv) whose LAST channel is the row occupancy o (a constant-1 value channel through the scan).
    Returns the first dv-1 channels divided by max(o, _VNORM_EPS), with o kept as the last channel."""
    o = buf[..., -1:]
    return torch.cat([buf[..., :-1] / o.clamp_min(_VNORM_EPS), o], -1)


def _pos_code(T, device, nfreq, wmin, wmax):
    """(T, 2*nfreq) fp32: sin and cos of 2*pi*t/w for nfreq wavelengths w geometric from wmin to wmax tokens."""
    t = torch.arange(T, device=device, dtype=torch.float32)[:, None]
    if nfreq == 1:
        w = torch.tensor([wmin], device=device)
    else:
        w = wmin * (wmax / wmin) ** (torch.arange(nfreq, device=device, dtype=torch.float32) / (nfreq - 1))
    a = 2 * math.pi * t / w[None, :]
    return torch.cat([torch.sin(a), torch.cos(a)], -1)


class SharedMixer(nn.Module):
    """One layer's mixing: the exact recent window always; the shared-memory WRITE at layer sh_write;
    a shared-memory READ at every layer >= sh_write."""

    def __init__(self, c: Cfg, layer_idx: int):
        super().__init__()
        D = c.d_model
        self.KVH = int(getattr(c, "kv_heads", 0)) or c.heads
        self.kvd, self.W = c.kv_dim, c.kv_window
        self.Hs, self.N, self.dk, self.dv, self.NQ = c.sh_heads, c.sh_slots, c.sh_kdim, c.sh_dv, c.sh_nq
        # which memory this layer belongs to: "main" (>= sh_write), "enc" (sh_enc_write .. sh_write-1), or none
        enc = getattr(c, "sh_enc_write", -1)
        if layer_idx >= c.sh_write:
            self.mid, wl = "main", c.sh_write
        elif enc >= 0 and layer_idx >= enc:
            self.mid, wl, self.N = "enc", enc, c.sh_enc_slots
        else:
            self.mid, wl = None, None
        self.G = c.sh_stride
        self.qbits, self.qgroup = c.sh_qbits, c.sh_qgroup
        self.gfloor = float(getattr(c, "sh_gate_floor", 0.0))
        self.nogate = bool(int(getattr(c, "sh_nogate", 0)))
        self.is_writer = (self.mid is not None and layer_idx == wl)
        self.is_reader = (self.mid is not None)
        _nw = int(getattr(c, "kv_nowin_from", -1))
        self.nowin = _nw >= 0 and layer_idx >= _nw
        self.emit_ckpt = int(c.emit_ckpt)
        self.hard, self.hard_gate = bool(c.slot_hard), bool(c.slot_hard_gate)
        self.kdetach = int(_os.environ.get("KV_DETACH", getattr(c, "kdetach", 0)))
        self.krep, self.krep_tau = float(getattr(c, "krep", 0.0)), float(getattr(c, "krep_tau", 0.85))
        self.free_bias = float(c.free_bias)
        # one fused input GEMM: [window k | window q | window v] (+ [value | key | gate] at the writer)
        self.sizes = [0 if self.nowin else self.KVH * self.kvd] * 3
        if self.is_writer:
            self.vnorm = int(getattr(c, "sh_vnorm", 0)) > 0
            self.wq_detach = int(getattr(c, "sh_wq_detach", 0)) > 0
            self.occbin = int(getattr(c, "sh_occbin", 0)) > 0
            self.sizes += [self.Hs * (self.dv - (1 if self.vnorm else 0)), self.Hs * self.dk, self.Hs]
            gi = min(max(float(c.gate_init), 1e-3), 1 - 1e-3)
            self.g_bias = nn.Parameter(torch.full((self.Hs,), math.log(gi / (1 - gi))))
            self.log_wsharp = nn.Parameter(torch.tensor(math.log(16.0)))   # write sharpness, as memlm_kv
            self.fr_fixed = float(getattr(c, "sh_fr", 0.0)) > 0
            _lf = math.log(float(c.sh_fr) / self.free_bias) if self.fr_fixed else 1.5
            self.log_free = nn.Parameter(torch.tensor(_lf))                # free-row preference, as memlm_kv
            self.frtok = float(getattr(c, "sh_frtok", 0.0))
            if self.frtok > 0:
                self.frmax = float(getattr(c, "sh_frmax", 2.0))
                self.fr_proj = nn.Linear(D, self.Hs, bias=False)            # ZEROED by MemLM after its global init
                self.fr_b = nn.Parameter(torch.full((self.Hs,), math.log(self.frtok / (self.frmax - self.frtok))))
            self.fr_c = None                                                # mean fr/temp of the last forward (frtok)
        # a reader with no window has no input columns at all (an empty Linear would reach the 8-bit optimizer)
        self.w_in = nn.Linear(D, sum(self.sizes), bias=False) if sum(self.sizes) > 0 else None
        self.wo_kv = None if self.nowin else nn.Linear(self.KVH * self.kvd, D, bias=False)
        if self.is_reader:
            self.tk_q = nn.Linear(D, self.Hs * self.NQ * self.dk, bias=False)
            self.wo_tk = nn.Linear(self.Hs * self.NQ * self.dv, D, bias=False)
            self.log_rsel = nn.Parameter(torch.tensor(math.log(8.0)))      # read sharpness, as memlm_kv
        # sh_pos: position code -> stored key (writer) / read query (reader).  ZEROED by MemLM after its global init.
        self.pos = int(getattr(c, "sh_pos", 0)) > 0 and self.mid is not None
        if self.pos:
            self.pos_args = (int(c.sh_pos_nfreq), float(c.sh_pos_wmin), float(c.sh_pos_wmax))
            self.pos_scale = float(getattr(c, "sh_pos_scale", 1.0))
            P = 2 * self.pos_args[0]
            if self.is_writer:
                self.pos_k = nn.Linear(P, self.Hs * self.dk, bias=False)
                if int(getattr(c, "sh_wq", 0)):
                    self.pos_wq = nn.Linear(P, self.Hs * self.dk, bias=False)   # position for the write query
            self.pos_q = nn.Linear(P, self.Hs * self.NQ * self.dk, bias=False)
        # attributes MemLM / the trainer collect from every mixer
        self.krep_loss = self.wbud_loss = self.osel_loss = self.xs_loss = None
        self._wq = []          # the write-query Linear lives in MemLM.write_q (registered last); held here unregistered
        self._wqk = []         # sh_wqk: the key-share Linear lives in MemLM.write_qk (registered after write_q)
        self._rsq = []         # sh_rsharp: the read-sharpness Linear lives in MemLM.read_s (registered LAST)
        self._wq2 = []         # kv_nq: [extra-query Linear, its output Linear], in MemLM.win_q2 / win_o2 (after read_s)
        self.kvnq = int(getattr(c, 'kv_nq', 1))
        self.rsharp = float(getattr(c, 'sh_rsharp', 0.0))
        self.rs_a = None       # sh_rsharp: detached mean and std of the log-multiplier of the last forward (for the log)
        self.wqk_a = None      # sh_wqk: detached mean |a| and mean a of the last forward (for the log)
        self.wrate = self.tk_ent = self.tk_fill = None
        self._masks = {}

    # ---- exact recent window (identical maths to memlm_kv.SlotMixer._window) ----
    def _band(self, T, device):
        key = (T, str(device))
        m = self._masks.get(key)
        if m is None:
            i = torch.arange(T, device=device)
            m = (i[:, None] >= i[None, :]) & ((i[:, None] - i[None, :]) < self.W)
            self._masks[key] = m
        return m

    def _window(self, cq, ck, cv, T):
        if _K._FLEX is not None and cq.is_cuda:
            key = ("flex", T, str(cq.device))
            bm = self._masks.get(key)
            if bm is None:
                W = self.W
                bm = _K._create_block_mask(lambda b, h, qi, ki: (qi >= ki) & (qi - ki < W),
                                           B=None, H=None, Q_LEN=T, KV_LEN=T, device=cq.device)
                self._masks[key] = bm
            return _K._FLEX(cq, ck, cv, block_mask=bm)
        return F.scaled_dot_product_attention(cq, ck, cv, attn_mask=self._band(T, cq.device))

    # ---- shared-memory read, at the memory's output stride ----
    def _read(self, kbuf, buf, kinv, valid, x):
        """kbuf (B,NB,Hs,N,dk), buf (B,NB,Hs,N,dv): the PRE-write memory at t = 0, G, 2G, ...
        kinv / valid (B,NB,Hs,N): 1/|key| and 'row was ever written', computed ONCE at the writer --
        per reader they were 8 redundant fp32 passes over the whole memory, measured as the bulk of the
        shared model's elementwise time.  The G positions of a block share one copy, so the read is a
        real (G*NQ x dk) @ (dk x N) matmul per (block, head) instead of T*Hs tiny ones."""
        B, T, _ = x.shape
        Hs, NQ, dk, G = self.Hs, self.NQ, self.dk, self.G
        NB = kbuf.shape[1]
        Tp = NB * G
        q = self.tk_q(x)
        if self.pos:
            _pc = _pos_code(T, x.device, *self.pos_args)
            if self.pos_scale != 1.0:
                _pc = _pc * self.pos_scale
            q = q + self.pos_q(_pc.to(self.pos_q.weight.dtype))[None]
        q = q.reshape(B, T, Hs, NQ, dk)
        q = q / q.float().norm(dim=-1, keepdim=True).clamp_min(1e-6).to(q.dtype)
        if Tp != T:
            q = F.pad(q, (0, 0, 0, 0, 0, 0, 0, Tp - T))
        q = q.reshape(B, NB, G, Hs, NQ, dk).permute(0, 1, 3, 2, 4, 5).reshape(B, NB, Hs, G * NQ, dk)
        sc = torch.einsum("bjhmd,bjhnd->bjhmn", q, kbuf).float() * (kinv.unsqueeze(-2) * torch.exp(self.log_rsel.float()))
        if self._rsq:
            # sh_rsharp: per-token log-multiplier, laid out exactly as q (pad T -> Tp, then (B,NB,Hs,G*NQ)); pad = 0 -> x1
            _s = self._rsq[0](x.detach()).float()
            _s = self.rsharp * torch.tanh(_s / self.rsharp)                     # (B,T,Hs*NQ), |s| < C
            self.rs_a = (_s.detach().mean(), _s.detach().std())
            _s = _s.reshape(B, T, Hs, NQ)
            if Tp != T:
                _s = F.pad(_s, (0, 0, 0, 0, 0, Tp - T))
            _s = _s.reshape(B, NB, G, Hs, NQ).permute(0, 1, 3, 2, 4).reshape(B, NB, Hs, G * NQ)
            sc = sc * torch.exp(_s).unsqueeze(-1)
        att = torch.nan_to_num(torch.softmax(sc.masked_fill(~valid.unsqueeze(-2), float("-inf")), -1))
        rd = torch.einsum("bjhmn,bjhnv->bjhmv", att.to(buf.dtype), buf)                  # (B,NB,Hs,G*NQ,dv)
        rd = rd.reshape(B, NB, Hs, G, NQ, self.dv).permute(0, 1, 3, 2, 4, 5).reshape(B, Tp, Hs, NQ, self.dv)
        return rd[:, :T]

    def forward(self, x, mems=None):
        mems = {} if mems is None else mems
        mem = mems.get(self.mid) if self.mid is not None else None
        B, T, D = x.shape
        KVH, kvd = self.KVH, self.kvd
        if self.w_in is None:
            parts = pck = pcq = pcv = None
        else:
            parts = self.w_in(x).split(self.sizes, -1)
            pck, pcq, pcv = parts[0], parts[1], parts[2]
        if self.is_writer:
            pv, pk, pg = parts[3], parts[4], parts[5]
            if self.vnorm:
                _pv = pv.reshape(B, T, self.Hs, self.dv - 1)
                v = torch.cat([_pv, torch.ones_like(_pv[..., :1])], -1)            # last channel -> occupancy o
            else:
                v = pv.reshape(B, T, self.Hs, self.dv)
            if self.pos:
                _pc = _pos_code(T, x.device, *self.pos_args)
                if self.pos_scale != 1.0:
                    _pc = _pc * self.pos_scale
                pk = pk + self.pos_k(_pc.to(self.pos_k.weight.dtype))[None]
            kk = pk.reshape(B, T, self.Hs, self.dk)
            g = torch.sigmoid(pg.float() + self.g_bias.float())                    # (B,T,Hs)
            if self.nogate:
                g = torch.ones_like(g)                                              # every token writes, fully
            if self.krep > 0 and self.training:
                # write-time key repulsion, exactly as memlm_kv: penalise near-duplicate keys only
                _st = 4 if T <= 1024 else 8
                _kr = kk[:, ::_st].float()
                _kr = _kr / _kr.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                self.krep_loss = _M5._krep_fn(_kr, g[:, ::_st].float(), self.krep_tau)
            else:
                self.krep_loss = None
            self.wrate = (g > 0.5).float().mean().detach()
            if self.gfloor > 0:
                # forward: exactly 0 below the floor; backward: identity (straight-through)
                g = g + (g * (g >= self.gfloor).to(g.dtype) - g).detach()
            fr = torch.exp(self.log_free.float().detach() if self.fr_fixed else self.log_free.float()) * self.free_bias
            temp = torch.exp(self.log_wsharp.float())
            if self.frtok > 0:
                _c = self.frmax * torch.sigmoid(self.fr_proj(x.detach()).float() + self.fr_b.float())   # (B,T,Hs)
                fr = _c * temp.detach()                                       # threshold ratio is the only thing learned
                self.fr_c = _c.detach().mean()
            qq = self._wq[0](x.detach() if self.wq_detach else x).reshape(B, T, self.Hs, self.dk).float() if self._wq else None   # the address-only key
            if qq is not None and hasattr(self, "pos_wq"):
                qq = qq + self.pos_wq(_pc.to(self.pos_wq.weight.dtype)).reshape(1, T, self.Hs, self.dk).float()
            if qq is not None and self._wqk:
                # sh_wqk: q' = q + a (|q|/|k|) k.detach(); a = 0 exactly at init (tanh(0) = 0, 0 * finite = 0), so q' == q bitwise
                _a = torch.tanh(self._wqk[0](x.detach()).float())                 # (B,T,Hs)
                _kd = kk.detach().float()
                _r = (qq.norm(dim=-1, keepdim=True) / _kd.norm(dim=-1, keepdim=True).clamp_min(1e-6)).detach()
                qq = qq + _a.unsqueeze(-1) * _r * _kd
                self.wqk_a = (_a.detach().abs().mean(), _a.detach().mean())
            if kv_scan_g is not None and kk.is_cuda:
                kbuf, buf = kv_scan_g(kk.float(), v.float(), g, fr, self.N, temp, G=self.G,
                                      hard=self.hard, hard_gate=self.hard_gate, detach_addr=bool(self.kdetach), q=qq,
                                      occ_bin=self.occbin)
            else:
                # pin=False: compute what the Triton kernel computes.  The reference's default pins the
                # score of a near-zero row to 0 and the kernel does not -- measured 20% apart on real
                # inputs at this shape, so a CPU battery would score a model it was never trained as.
                kbuf, buf = ref_scan_kv(kk.float(), v.float(), g, fr, self.N, temp, G=self.G,
                                        hard=self.hard, hard_gate=self.hard_gate, detach_addr=bool(self.kdetach),
                                        pin=False, q=qq, occ_bin=self.occbin)
            if self.vnorm:
                buf = _vnorm(buf)
            # NANFIX (memlm_kv): clamp the squared norm BEFORE the sqrt, so a never-written (exactly
            # zero) row has no infinite derivative in the backward.
            qb = self.qbits or (_QBITS_EVAL if not self.training else 0)
            if qb:
                # quantised ONCE here, in fp32, then cast: every reader sees the same stored copy
                mem = _quantize_mem(kbuf, buf, qb, self.qgroup)
                mem = (mem[0].to(x.dtype), mem[1].to(x.dtype), mem[2], mem[3])
            else:
                nsq = kbuf.pow(2).sum(-1)                                          # fp32, once
                mem = (kbuf.to(x.dtype), buf.to(x.dtype), torch.rsqrt(nsq.clamp_min(1e-12)), nsq > 1e-12)
        # window
        if self.nowin:
            assert self.is_reader, "a layer with no window and no memory read would not mix tokens at all"
            delta = None
        else:
            _rp = (lambda z: _K._rope(z, frac=_K._ROPE_FRAC)) if _K._ROPE else (lambda z: z)
            cq = _rp(pcq.reshape(B, T, KVH, kvd)).transpose(1, 2)
            ck = _rp(pck.reshape(B, T, KVH, kvd)).transpose(1, 2)
            cv = pcv.reshape(B, T, KVH, kvd).transpose(1, 2)
            kvr = self._window(cq, ck, cv, T).transpose(1, 2)
            delta = self.wo_kv(kvr.reshape(B, T, KVH * kvd))
            if self._wq2:
                # kv_nq: Q-1 more queries per K/V head, a SEPARATE window call so the original heads' maths is unchanged
                _q2l, _o2l = self._wq2
                _r = self.kvnq - 1
                cq2 = _rp(_q2l(x).reshape(B, T, KVH * _r, kvd)).transpose(1, 2)
                kvr2 = self._window(cq2, ck.repeat_interleave(_r, dim=1), cv.repeat_interleave(_r, dim=1), T).transpose(1, 2)
                delta = delta + _o2l(kvr2.reshape(B, T, KVH * _r * kvd))
        if self.is_reader:
            assert mem is not None, "a reader layer ran before the writer layer"
            kbuf, buf, kinv, valid = mem
            # no activation checkpoint: at stride G the read's saved tensors are G x smaller and the
            # recompute would cost more than the memory it frees
            rd = self._read(kbuf, buf, kinv, valid, x)
            _dr = self.wo_tk(rd.reshape(B, T, self.Hs * self.NQ * self.dv))
            delta = _dr if delta is None else delta + _dr
        if self.is_writer:
            mems = dict(mems); mems[self.mid] = mem
        return delta, mems


class _PermRows(torch.autograd.Function):
    """y = x[idx] for a PERMUTATION idx; the backward is the inverse gather g[inv].  Autograd's own
    backward for indexing is an atomic bf16 index_add, which does not know idx is a permutation."""
    @staticmethod
    def forward(ctx, x, idx, inv):
        ctx.save_for_backward(inv)
        return x.index_select(0, idx)

    @staticmethod
    def backward(ctx, g):
        (inv,) = ctx.saved_tensors
        return g.index_select(0, inv), None, None


class ExactMoE(_M5.MoE):
    """memlm_m5.MoE with the SAME maths and none of the padding.

    On this GPU (Ampere) memlm_m5 cannot use torch._grouped_mm (Hopper only) and falls to a fixed
    [E, C, D] capacity buffer with C = cap * N * k / E.  At the trainer's cap 2.0 with 4 experts top-2
    that is C = N: every expert computes ALL N rows while its real load is ~N/2, so half the FFN
    matmul work is padding.  cap 2.0 is also exactly what makes that path zero-drop (no expert can
    receive more than N), so the function computed is: every (token, expert) pair, weighted, summed.
    This computes that same function on the real rows only -- sort the N*k pairs by expert, one
    matmul chain per expert on its own slice, un-sort, sum the k contributions per token -- with both
    permutations as gathers (no atomics).  The four slice sizes need one host sync per layer, and
    vary every call, so this stays OUT of torch.compile."""

    @torch.compiler.disable
    def forward(self, x, route=None):
        if self.use_loop:
            return super().forward(x, route)
        B, T, D = x.shape
        xf = x.reshape(-1, D)
        N = xf.shape[0]
        topi, topv = self.route(xf) if route is None else route
        flat_e = topi.reshape(-1)                                      # (N*k,), token-major
        order = torch.argsort(flat_e, stable=True)
        inv = torch.empty_like(order); inv[order] = torch.arange(order.numel(), device=order.device)
        counts = torch.bincount(flat_e, minlength=self.E)
        sizes = counts.tolist()                                        # the one host sync
        xs = _PermRows.apply(xf.repeat_interleave(self.k, 0), order, inv)
        # unbind, not w1[e]: indexing makes autograd build a FULL-size zero gradient per expert and add
        # the four (measured: the bf16 add went 18 -> 44 ms and a fill kernel appeared); unbind's
        # backward is a single stack
        w1, w3, w2 = self.w1.unbind(0), self.w3.unbind(0), self.w2.unbind(0)
        ys = []
        for e, ch in enumerate(xs.split(sizes, 0)):
            if ch.shape[0] == 0:
                continue
            ys.append((F.silu(ch @ w1[e]) * (ch @ w3[e])) @ w2[e])
        ys = torch.cat(ys, 0) * topv.reshape(-1)[order].unsqueeze(-1).to(xs.dtype)
        out = _PermRows.apply(ys, inv, order).view(N, self.k, D).sum(1)
        self.drop = torch.zeros((), device=x.device)
        expected = N * self.k / self.E
        with torch.no_grad():
            c = counts.float()
            self.util = (c / max(N * self.k, 1)).detach()
            self.maxvio = ((c.max() - expected) / expected).detach()
            if self.training:
                self.ebias += self.u * torch.sign(expected - c)
        return out.view(B, T, D)


class SharedBlock(nn.Module):
    def __init__(self, c: Cfg, layer_idx: int):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(c.d_model), nn.LayerNorm(c.d_model)
        self.mix = SharedMixer(c, layer_idx)
        self.use_moe = bool(c.moe)
        if self.use_moe:
            self.ffn = ExactMoE(c)
        else:
            self.w1 = nn.Linear(c.d_model, c.d_ff, bias=False)
            self.w3 = nn.Linear(c.d_model, c.d_ff, bias=False)
            self.w2 = nn.Linear(c.d_ff, c.d_model, bias=False)

    def forward(self, x, mem=None):
        d, mem = self.mix(self.n1(x), mem)       # mem: {memory id: (kbuf, buf, kinv, valid)}
        x = x + d
        h = self.n2(x)
        x = x + (self.ffn(h) if self.use_moe else self.w2(F.silu(self.w1(h)) * self.w3(h)))
        return x, mem


class MemLM(nn.Module):
    def __init__(self, c: Cfg):
        super().__init__()
        self.c = c
        self.emb = nn.Embedding(c.vocab, c.d_model)
        self.pre_blocks = nn.ModuleList([_M5.PreFFN(c) for _ in range(c.pre)])
        self.blocks = nn.ModuleList([SharedBlock(c, i) for i in range(c.layers)])
        self.nf = nn.LayerNorm(c.d_model)
        self.head = nn.Linear(c.d_model, c.vocab, bias=False)
        self.head.weight = self.emb.weight                     # tied, as memlm_kv's default
        self.kdec = self.krep = self.wbud = None
        self.xsel = self.osel = None
        self.util = self.maxvio = self.drop = None
        self.apply(self._init)
        # sh_pos: the position projections start at ZERO (after the global init, which would overwrite them), so the
        # model computes exactly the sh_pos=0 function at step 0 and learns how much position to use.
        for b in self.blocks:
            for _n in ("pos_k", "pos_q", "pos_wq", "fr_proj"):
                if hasattr(b.mix, _n):
                    nn.init.zeros_(getattr(b.mix, _n).weight)
        # write queries: registered AFTER everything else so they come last in model.parameters(); each starts as
        # an exact copy of its writer's key projection (same function at the switch)
        if int(getattr(c, "sh_wq", 0)):
            self.write_q = nn.ModuleDict()
            for i, b in enumerate(self.blocks):
                mx = b.mix
                if mx.is_writer:
                    lin = nn.Linear(c.d_model, mx.Hs * mx.dk, bias=False)
                    k0 = sum(mx.sizes[:4]); k1 = k0 + mx.Hs * mx.dk                # the key rows of the fused w_in
                    with torch.no_grad():
                        if float(getattr(c, "sh_wq_std", 0.0)) > 0:
                            lin.weight.normal_(0.0, float(c.sh_wq_std))
                        else:
                            lin.weight.copy_(mx.w_in.weight[k0:k1] * float(getattr(c, "sh_wq_sign", 1.0)))
                    self.write_q[str(i)] = lin
                    mx._wq = [lin]
        # sh_wqk: registered AFTER write_q so its weight is the LAST 2-D parameter and its bias the LAST 1-D one;
        # zero weight AND bias -> a = tanh(0) = 0 for every token: the sh20 function exactly at the switch
        if int(getattr(c, "sh_wqk", 0)):
            assert int(getattr(c, "sh_wq", 0)), "sh_wqk needs sh_wq"
            self.write_qk = nn.ModuleDict()
            for i, b in enumerate(self.blocks):
                mx = b.mix
                if mx.is_writer:
                    lin = nn.Linear(c.d_model, mx.Hs, bias=True)
                    nn.init.zeros_(lin.weight); nn.init.zeros_(lin.bias)
                    self.write_qk[str(i)] = lin
                    mx._wqk = [lin]
        # sh_rsharp: registered LAST (after write_q / write_qk) so the new weights are the LAST ids of the decay group and a
        # resumed optimizer state can be re-indexed exactly (convert_rs.py); weight ZERO -> multiplier exp(0) = 1, bitwise
        if float(getattr(c, "sh_rsharp", 0.0)) > 0:
            self.read_s = nn.ModuleDict()
            for i, b in enumerate(self.blocks):
                mx = b.mix
                if mx.is_reader:
                    lin = nn.Linear(c.d_model, mx.Hs * mx.NQ, bias=False)
                    nn.init.zeros_(lin.weight)
                    self.read_s[str(i)] = lin
                    mx._rsq = [lin]
        # kv_nq: registered LAST (after read_s): all q2 weights, then all o2 weights -> the last 2-D ids of the decay group,
        # so a resumed optimizer state can be re-indexed exactly (convert_kvnq.py).  o2 ZERO -> the kv_nq=1 function.
        if int(getattr(c, 'kv_nq', 1)) > 1:
            self.win_q2 = nn.ModuleDict(); self.win_o2 = nn.ModuleDict()
            _wl = [(i, b.mix) for i, b in enumerate(self.blocks) if b.mix.wo_kv is not None]
            for i, mx in _wl:
                lin = nn.Linear(c.d_model, mx.KVH * (c.kv_nq - 1) * mx.kvd, bias=False)
                nn.init.normal_(lin.weight, std=0.02)
                self.win_q2[str(i)] = lin
            for i, mx in _wl:
                lin = nn.Linear(mx.KVH * (c.kv_nq - 1) * mx.kvd, c.d_model, bias=False)
                nn.init.zeros_(lin.weight)
                self.win_o2[str(i)] = lin
                mx._wq2 = [self.win_q2[str(i)], lin]

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def set_moe_loop(self, flag: bool):
        if self.c.moe:
            for b in self.blocks:
                b.ffn.use_loop = flag

    def forward(self, idx, states=None, return_states=False, skip_head=False, src=None):
        assert states is None and not return_states, "memlm_sh: no state carry across calls"
        x = self.emb(idx)
        for pb in self.pre_blocks:
            x = pb(x)
        mem = {}
        for b in self.blocks:
            x, mem = b(x, mem)
        if float(getattr(self.c, "krep", 0.0)) > 0:
            _lr = [b.mix.krep_loss for b in self.blocks if b.mix.krep_loss is not None]
            self.krep = torch.stack(_lr).sum() if _lr else None
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
