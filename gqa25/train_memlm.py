"""
Train the pure-memory LM (no attention, zero KV cache) with a progressive context curriculum.

Curriculum: start on short contexts, widen in stages, and keep every earlier length in the
mix (uniform over active lengths) so nothing is catastrophically forgotten.

Two things make the long-range signal real rather than assumed:
  * doc-aware windows -- a training sequence stays inside ONE document, so the early tokens
    are actually related to the late ones (a random window across shard boundaries splices
    unrelated documents together and contains no long-range dependency to learn).
  * repeat-passage task -- on a fraction of steps, a passage from early in the sequence is
    repeated at the end, so the late tokens CANNOT be predicted without carrying the early
    ones forward in the state.
"""
import os, sys, time, math, json, argparse, glob
import numpy as np
import torch, torch.nn.functional as F
import torch._dynamo as _dynamo   # compile recompile-limit knobs (set in main before compile)
import importlib as _il
_mm = _il.import_module(os.environ.get("MEMLM_MOD", "memlm"))   # MEMLM_MOD=memlm_ks for the kslot run
Cfg, MemLM = _mm.Cfg, _mm.MemLM

DEV = "cuda"
MEMPROBE_ROOT = os.environ.get("MEMPROBE_ROOT",
                               os.path.expanduser("~/memprobe"))
DATA = os.path.join(MEMPROBE_ROOT, "data")
EOT = 50256


class Shards:
    """Random windows out of the uint16 token shards. Picks up new shards as they land."""

    def __init__(self, hold_out_first=10_000_000, doc_aware=True, tries=24):
        self.hold = hold_out_first
        self.doc_aware = doc_aware
        self.tries = tries
        self.rejects = 0
        self.taken = 0
        self.refresh()

    def refresh(self):
        self.files = sorted(glob.glob(os.path.join(DATA, "shard_*.npy")))
        self.mm = [np.load(f, mmap_mode="r") for f in self.files]

    def _window(self, T, rng, val):
        """One window of T+1 tokens, kept inside a single document when possible."""
        for attempt in range(self.tries if self.doc_aware else 1):
            if val:
                a = self.mm[0]; lo, hi = 0, self.hold - T - 2
            else:
                j = int(rng.integers(0, len(self.mm)))
                a = self.mm[j]
                lo = self.hold if j == 0 else 0
                hi = len(a) - T - 2
            s = int(rng.integers(lo, hi))
            w = a[s:s + T + 1]
            if not self.doc_aware:
                return w
            # reject windows that straddle a document boundary
            if not np.any(w[:-1] == EOT):
                self.taken += 1
                return w
            self.rejects += 1
        return w                      # give up: accept a spliced window

    @staticmethod
    def _mqar(w, T, rng, npairs=32, vocab=50257, qfrac=0.40, want_mask=False):
        """Multi-query associative recall.

        [k1 v1 ... kN vN] [filler] [k? v? k? v? ...  queries in RANDOM order]

        The repeat task is one span retrieved by continuation: once the first token
        is right the rest follows from local context, so it is solvable with a
        low-rank memory. Here the queries arrive shuffled, so knowing the previous
        answer tells you nothing about the next -- the model must hold N separately
        addressable facts at once, which puts a hard floor of ~N on the state rank.
        """
        n = T + 1
        npairs = int(min(npairs, max(4, n // 16)))
        nq = (int(qfrac * n) // 2) * 2
        fill = n - 2 * npairs - nq
        if fill < 32:
            return (w, None) if want_mask else w
        ks = rng.choice(vocab, size=npairs, replace=False).astype(np.int64)
        vs = rng.choice(vocab, size=npairs, replace=False).astype(np.int64)
        facts = np.empty(2 * npairs, dtype=np.int64)
        facts[0::2] = ks
        facts[1::2] = vs
        pick = rng.integers(0, npairs, size=nq // 2)
        q = np.empty(nq, dtype=np.int64)
        q[0::2] = ks[pick]
        q[1::2] = vs[pick]
        row = np.concatenate([facts, w[:fill], q])
        if not want_mask:
            return row
        # the queries are [k? v? k? v? ...]: the VALUES are the answers; the keys are given
        m = np.zeros(n, dtype=np.float32)
        m[2 * npairs + fill + 1::2] = 1.0
        return row, m

    def batch(self, bs, T, rng, val=False, p_synth=0.0, p_mqar=0.0, mqar_pairs=32,
              p_broad=0.25, long_frac=0.5, p_tasks=0.0, task_names=None, task_weights=None,
              ans_w=1.0):
        if not val and len(self.files) != len(self.mm):
            self.refresh()
        out = np.empty((bs, T + 1), dtype=np.int64)
        wt = np.ones((bs, T), dtype=np.float32)
        for i in range(bs):
            w = self._window(T, rng, val)
            _syn = False
            _m = None
            if (not val) and p_synth > 0 and T >= 128 and rng.random() < p_synth:
                _syn = True
                if p_tasks > 0 and task_names and rng.random() < p_tasks:
                    import task_lib as _TL
                    _cues = self._cues()
                    w, _m = _TL.compose(w, T, rng, _cues[int(rng.integers(0, len(_cues)))],
                                    task_names, task_weights, want_mask=True)
                elif p_mqar > 0 and rng.random() < p_mqar:
                    w, _m = self._mqar(w, T, rng, npairs=mqar_pairs, want_mask=True)
                else:
                    w, _m = self._repeat_passage(w, T, rng, p_broad=p_broad,
                                             long_frac=long_frac, want_mask=True)
            out[i] = w
            if _syn and ans_w != 1.0 and _m is not None:
                # the GENERATOR says which tokens are answers. Recovering them by matching a cue
                # string missed every uncued exercise (mqar always, _repeat_passage half the
                # time) and weighted the given query keys, which are fresh random symbols that
                # nothing can predict. wt[j] weights the prediction of out[j+1], so shift by one.
                wt[i] = 1.0 + (ans_w - 1.0) * _m[1:T + 1]
        x = torch.from_numpy(out).to(DEV, non_blocking=True)
        return x[:, :-1], x[:, 1:], torch.from_numpy(wt).to(DEV, non_blocking=True)

    _CUES = None

    @staticmethod
    def _cues():
        if Shards._CUES is None:
            import tiktoken
            e = tiktoken.get_encoding("gpt2")
            Shards._CUES = [np.array(e.encode_ordinary(c), dtype=np.int64) for c in (
                "\n\nTo repeat what I said at the beginning: ",
                "\n\nAs I said at the start, ",
                "\n\nI will repeat what I said earlier.\n\n",
                "\n\nHere is that passage again:\n\n",
                "\n\nRepeating the passage above:\n\n",
                "\n\nTo quote it again: ",
            )]
        return Shards._CUES

    @staticmethod
    def _tail_mask(n, k):
        m = np.zeros(n, dtype=np.float32)
        m[n - k:] = 1.0
        return m

    @staticmethod
    def _repeat_passage(w, T, rng=None, p_cue=0.5, p_broad=0.25, long_frac=0.5,
                        want_mask=False):
        """A span from earlier in the window is repeated at the end.

        rng=None -> original fixed [A][B][A] (always the first T//5 tokens).  Kept
        byte-identical so recall_probe stays comparable across every run.

        rng given -> the DISTANCE is sampled first and the span is then placed to
        achieve it.  The old code sampled span start/length and let the distance fall
        out of the geometry, which put 28% of copies under 50 tokens and under 1%
        past 800.  Sampling the gap directly, with 75% of draws from the upper half
        of the available range, roughly doubles mean distance at no cost."""
        n = T + 1
        if rng is None:
            la = max(32, T // 5)
            rest = n - 2 * la
            if rest <= 0:
                return (w, None) if want_mask else w
            A = w[:la]
            B = w[la:la + rest]
            row = np.concatenate([A, B, A])
            return (row, Shards._tail_mask(n, la)) if want_mask else row
        la = int(rng.integers(24, max(25, n // 4)))
        cue = np.empty(0, dtype=np.int64)
        if rng.random() < p_cue:
            cs = Shards._cues()
            cue = cs[int(rng.integers(0, len(cs)))]
        head = n - la - len(cue)
        gmax = head - la
        if gmax < 8:
            return (w, None) if want_mask else w
        if rng.random() < p_broad:
            g = int(rng.integers(0, gmax + 1))          # broad coverage of all scales
        else:
            g = int(rng.integers(int(long_frac * gmax), gmax + 1))   # long-range bias
        s = head - la - g
        row = np.concatenate([w[:head], cue, w[s:s + la]])
        # the repeated span is the answer whether or not a cue was planted -- which is the
        # point: half these rows draw no cue, and the old cue-matching weighter saw none of them
        return (row, Shards._tail_mask(n, la)) if want_mask else row


@torch.no_grad()
def oracle_target(model, x, y, src, LY, K):
    """Two-stage oracle, computed BEFORE the training pass so it can be a plain loss term.

      stage 1  rank the rows by u_n = g.row_n, using the gradient kept from the PREVIOUS micro-batch.
               Only the SHORTLIST needs it, and 8 of 24 held the true best row 100% of the time, so one
               step of staleness is cheap.  Before any gradient exists, fall back to the read's own score.
      stage 2  rerank those K EXACTLY by running the layers above the read.

    Cheap because delta = d_other + wo_tk(read) is linear (so d_other comes free by subtraction, no second
    scan) and because only ANSWER tokens are evaluated -- above the last block everything is position-wise,
    so restricting to them is exact, not a sample."""
    blk = model.blocks[LY]; mix = blk.mix
    cap = {}
    hk = blk.register_forward_pre_hook(lambda m_, i_: cap.__setitem__("xin", i_[0].detach()))
    hm = mix.register_forward_hook(lambda m_, i_, o_: cap.__setitem__("delta", o_[0].detach()))
    _orig = mix._tkread
    def spy(buf, xx):
        r, e, f = _orig(buf, xx); cap["buf"] = buf.detach(); cap["rd"] = r.detach(); return r, e, f
    mix._tkread = spy
    was = model.training; model.eval()
    model(x)
    mix._tkread = _orig; hk.remove(); hm.remove()
    if was: model.train()
    buf, xin, rd, delta = cap["buf"], cap["xin"], cap["rd"], cap["delta"]
    B, T, H, N, dv = buf.shape
    msk = src >= 0
    bi, ti = msk.nonzero(as_tuple=True)
    if bi.numel() == 0:
        return None
    nrm = buf.float().norm(dim=-1).clamp_min(1e-6)
    valid = buf.float().pow(2).sum(-1) > 1e-12
    g = getattr(mix, "_last_g", None)
    if g is not None and g.shape == rd.shape:
        u = torch.einsum("bthv,bthnv->bthn", g.float(), buf.float()) / nrm
        rank = (-u).masked_fill(~valid, float("-inf"))            # lower u is better
    else:
        rank = torch.zeros(B, T, H, N, device=buf.device).masked_fill(~valid, float("-inf"))
    cand = rank[bi, ti].topk(min(K, N), dim=-1).indices           # (n,H,K)
    n, _, Kk = cand.shape
    WD = mix.wo_tk.weight.dtype
    d_other = delta.float() - mix.wo_tk(rd.reshape(B, T, -1).to(WD)).float()
    z0 = (xin.float() + d_other)[bi, ti]
    rows = torch.gather(buf[bi, ti].float(), 2, cand.unsqueeze(-1).expand(-1, -1, -1, dv))
    yy = y[bi, ti]
    Ls = []
    for j in range(Kk):
        z = (z0 + mix.wo_tk(rows[:, :, j, :].reshape(n, H * dv).to(WD)).float()).to(WD).unsqueeze(1)
        z = z + (blk.ffn(blk.n2(z)) if blk.use_moe else blk.w2(F.silu(blk.w1(blk.n2(z))) * blk.w3(blk.n2(z))))
        for bb in model.blocks[LY + 1:]:
            d, _ = bb.mix(bb.n1(z), None); z = z + d
            z = z + (bb.ffn(bb.n2(z)) if bb.use_moe else bb.w2(F.silu(bb.w1(bb.n2(z))) * bb.w3(bb.n2(z))))
        lg = model.head(model.nf(z)).float().reshape(n, -1)
        Ls.append(F.cross_entropy(lg, yy, reduction="none"))
    win = torch.stack(Ls, 0).argmin(0)                            # (n,)
    best = torch.gather(cand, 2, win[:, None, None].expand(-1, H, 1)).squeeze(-1)   # (n,H)
    tgt = torch.zeros(B, T, H, dtype=torch.long, device=buf.device)
    tgt[bi, ti] = best
    return tgt, msk


def source_pos(x, y, w):
    """For every answer token, the most recent EARLIER position carrying that same token -- i.e. where
    the fact was stated.  -1 elsewhere.  Needs no change to the data generators: on a recall example
    the answer appears verbatim in the fact, so the last earlier occurrence IS the source."""
    B, T = x.shape
    ar = torch.arange(T, device=x.device)
    match = x.unsqueeze(1) == y.unsqueeze(2)                    # (B, T_answer, T_source)
    earlier = ar.view(1, 1, T) < ar.view(1, T, 1)
    cand = torch.where(match & earlier, ar.view(1, 1, T).expand(B, T, T), torch.full((1,), -1, device=x.device, dtype=ar.dtype))
    src = cand.max(-1).values
    if w is not None:
        src = torch.where(w > 1.0, src, torch.full_like(src, -1))   # answer tokens only
    return src


def student_force(model, x, rng, span_lo=0.15, span_hi=0.60, top_k=50, temp=1.0):
    """Free generation inside the training stack.

    A random span of the INPUT is replaced by the model's OWN sampled tokens
    while the targets stay ground truth, so the model must keep predicting the
    true continuation after being fed its own (possibly wrong) text -- exactly
    the situation teacher forcing never shows it.  Start and length are random
    so every offset in the window gets covered.
    """
    B, T = x.shape
    lo = max(8, int(span_lo * T))
    hi = max(lo + 1, int(span_hi * T))
    ln = int(rng.integers(lo, hi))
    s = int(rng.integers(0, max(1, T - ln)))
    n = min(ln, T - 1 - s)
    if n <= 0:
        return x
    was = model.training
    model.eval()               # keep the MoE bias controller from ticking twice
    with torch.no_grad():
        lg = model(x)[:, s:s + n, :]
        flat = lg.reshape(-1, lg.size(-1))
        outs = []
        for c in flat.split(8192, 0):
            c = c.float()
            if top_k > 0:
                k = min(top_k, c.size(-1))
                v, _ = torch.topk(c, k, dim=-1)
                c = c.masked_fill(c < v[..., -1:], float("-inf"))
            outs.append(torch.multinomial(torch.softmax(c / max(temp, 1e-6), -1), 1))
        samp = torch.cat(outs, 0).view(B, n)
        del lg, flat, outs
    if was:
        model.train()
    xs = x.clone()
    xs[:, s + 1:s + 1 + n] = samp      # logits at t feed input position t+1
    return xs


def _ce_chunk(lc, tc):
    return F.cross_entropy(lc.float(), tc, reduction="none")


def chunked_ce(logits, tgt, chunks=8, w=None):
    """fp32 cross-entropy without materialising a full fp32 logits copy.
    w: optional per-token weight. Returns (weighted mean, plain mean, share of the weighted
    loss carried by tokens with w>1) -- the last is the fraction of the GRADIENT spent on recall."""
    logits = logits.reshape(-1, logits.size(-1))
    tgt = tgt.reshape(-1)
    wf = None if w is None else w.reshape(-1).float()
    tot = logits.new_zeros((), dtype=torch.float32)
    plain = logits.new_zeros((), dtype=torch.float32)
    hot = logits.new_zeros((), dtype=torch.float32)
    den = logits.new_zeros((), dtype=torch.float32)
    n = 0
    for i, (lc, tc) in enumerate(zip(logits.chunk(chunks, 0), tgt.chunk(chunks, 0))):
        # recomputed in backward: the fp32 chunk copies (3 GB per micro-batch at T=1024) are not saved
        ce = torch.utils.checkpoint.checkpoint(_ce_chunk, lc, tc, use_reentrant=False)
        plain = plain + ce.sum()
        if wf is None:
            tot = tot + ce.sum(); den = den + ce.numel()
        else:
            wc = wf.chunk(chunks, 0)[i]
            tot = tot + (ce * wc).sum(); den = den + wc.sum()
            hot = hot + (ce * wc * (wc > 1).float()).sum()
        n += tc.numel()
    return tot / den.clamp(min=1), plain / n, (hot / tot.clamp(min=1e-9))


class _ChunkedCE(torch.autograd.Function):
    """Per-token fp32 cross-entropy straight from the hidden states.

    Same math as head(h) -> chunked fp32 CE, without autograd's bookkeeping: the plain path
    assembles the logits' gradient from 8 checkpointed chunks with 8 zero-fills + 7 adds of
    the full (N, V) buffer (18 ms per micro-batch on the 3060). Here the ONE bf16 logits
    tensor is kept from the forward, each chunk's softmax gradient is written back INTO it
    in the backward, and dh / dW are the same two GEMMs the Linear layer would have run.
    Precision is identical: fp32 log-softmax per chunk, one bf16 cast, one GEMM per gradient."""
    @staticmethod
    def forward(ctx, h, W, tgt, chunks):
        # AMPCE 2026-09-14: a custom autograd.Function is NOT autocast-aware -- inside it, ops run in
        # whatever dtype they are handed.  Under amp=fp16 the hidden states arrive HALF while the tied head
        # weight is FP32, and this GEMM raises "mat1 and mat2 have the same dtype".  Cast the weight to the
        # activation dtype here, and cast the weight gradient BACK in the backward (autograd requires a
        # gradient to match its input's dtype).  Both casts are identity in the bf16/fp32-matched case.
        lg = torch.mm(h, (W if W.dtype == h.dtype else W.to(h.dtype)).t())   # (N, V) in h.dtype, one GEMM
        ce = torch.empty(h.shape[0], device=h.device, dtype=torch.float32)
        for lc, tc, cc in zip(lg.chunk(chunks, 0), tgt.chunk(chunks, 0), ce.chunk(chunks, 0)):
            cc.copy_(F.cross_entropy(lc.float(), tc, reduction="none"))
        ctx.save_for_backward(h, W, lg, tgt)
        ctx.chunks = chunks
        return ce

    @staticmethod
    def backward(ctx, dce):
        h, W, lg, tgt = ctx.saved_tensors
        dce = dce.contiguous()
        for lc, tc, gc in zip(lg.chunk(ctx.chunks, 0), tgt.chunk(ctx.chunks, 0), dce.chunk(ctx.chunks, 0)):
            # Same operation order as autograd's nll_loss_backward -> _log_softmax_backward_data
            # (grad_out - exp(logp) * grad_out.sum(-1), grad_out = -dce at the target), so the
            # bf16 dlogits are BITWISE those of the plain path -- verified on the whole model.
            valid = tc != -100
            tcl = tc.clamp_min(0).unsqueeze(1)
            s = -(gc * valid.to(gc.dtype))                         # row sum of grad_out (0 for ignored rows)
            res = torch.exp(F.log_softmax(lc.float(), dim=-1)).mul_(s.unsqueeze(1)).neg_()
            res.scatter_add_(1, tcl, s.unsqueeze(1))               # + grad_out at the target
            lc.copy_(res)                                          # logits buffer becomes dlogits
        # autocast is OFF in backward, so nothing casts for us here: lg came out of an autocast'd mm
        # and may be HALF while h and W are FP32.  Do both GEMMs in lg's dtype, then put each gradient
        # back into its input's dtype -- autograd requires grad.dtype == input.dtype.  All four casts
        # are identity when the dtypes already agree, so bf16 is bit-for-bit unchanged.
        _dt = lg.dtype
        dh = torch.mm(lg, W if W.dtype == _dt else W.to(_dt))
        if dh.dtype != h.dtype:
            dh = dh.to(h.dtype)
        dW = torch.mm(lg.t(), h if h.dtype == _dt else h.to(_dt))
        if dW.dtype != W.dtype:
            dW = dW.to(W.dtype)          # the optimiser holds W in fp32; autograd checks this
        return dh, dW, None, None


def chunked_ce_h(h, W, tgt, chunks=8, w=None):
    """chunked_ce computed from the final hidden states and the (tied) head weight -- same
    (weighted mean, plain mean, recall share) triple, via _ChunkedCE."""
    h = h.reshape(-1, h.size(-1))
    tgt = tgt.reshape(-1)
    ce = _ChunkedCE.apply(h, W, tgt, chunks)
    plain = ce.sum() / tgt.numel()
    if w is None:
        return plain, plain, ce.new_zeros(())
    wf = w.reshape(-1).float()
    tot = (ce * wf).sum()
    den = wf.sum()
    hot = (ce * wf * (wf > 1).float()).sum()
    return tot / den.clamp(min=1), plain, hot / tot.clamp(min=1e-9)


@torch.no_grad()
def evaluate(model, sh, rng, T=1024, bs=8, batches=6, nb=16):
    """Val loss + loss at every position bucket. A curve that keeps falling past the
    trained window = genuine long-range use; one that flattens = only the local window."""
    model.eval()
    pos = torch.zeros(T, device=DEV, dtype=torch.float32)
    tot = 0.0; n = 0
    for _ in range(batches):
        x, y, _ = sh.batch(bs, T, rng, val=True)
        lg = model(x)
        for i in range(0, bs, 2):
            per = F.cross_entropy(lg[i:i + 2].float().transpose(1, 2), y[i:i + 2],
                                  reduction="none")
            pos += per.sum(0)
            tot += per.sum().item(); n += per.numel()
        del lg
    pos /= (batches * bs)
    buckets = pos.view(nb, T // nb).mean(1).tolist()
    model.train()
    return tot / n, buckets


@torch.no_grad()
def recall_probe(model, sh, rng, T=1024, bs=4, batches=4):
    """Loss on the REPEATED copy of a passage seen ~800 tokens earlier.
    This is the direct test: can the state carry information that far?"""
    model.eval()
    la = max(32, T // 5)
    copy_l = fresh_l = 0.0
    for _ in range(batches):
        rows = np.empty((bs, T + 1), dtype=np.int64)
        for i in range(bs):
            rows[i] = Shards._repeat_passage(sh._window(T, rng, True), T)
        x = torch.from_numpy(rows).to(DEV)
        xi, y = x[:, :-1], x[:, 1:]
        lg = model(xi)
        c = f = 0.0
        nch = 0
        for i in range(0, bs, 2):                   # chunk rows: fp32 logits stay small
            per = F.cross_entropy(lg[i:i + 2].float().transpose(1, 2), y[i:i + 2],
                                  reduction="none")
            c += per[:, -la:].mean().item()
            f += per[:, la:2 * la].mean().item()
            nch += 1
            del per
        copy_l += c / max(nch, 1)                   # the repeated span
        fresh_l += f / max(nch, 1)                  # ordinary text, same-ish depth
        del lg
    model.train()
    return copy_l / batches, fresh_l / batches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="memlm5b")
    ap.add_argument("--dk", type=int, default=48)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--dv", type=int, default=96)
    ap.add_argument("--gbias", type=int, default=0)
    ap.add_argument("--hl_min", type=float, default=16.0)
    ap.add_argument("--hl_max", type=float, default=8192.0)
    ap.add_argument("--mixer", default="mem", choices=["mem", "attn"])
    ap.add_argument("--moe", type=int, default=0)
    ap.add_argument("--compile", type=int, default=0,
                    help="1 = torch.compile the eager submodules (kernels graph-break)")
    ap.add_argument("--lanes", type=int, default=1,
                    help=">1: lane groups with learnable dv-merge gates")
    ap.add_argument("--lane_clamp", type=float, default=0.0)     # |logit| bound on the lane merge gates (0 = off)
    ap.add_argument("--lane_clamp_until", type=int, default=0)   # step at which the bound is released
    ap.add_argument("--kslot", type=int, default=0)        # per-slot gates + payload plan on the kslot scan
    ap.add_argument("--dv_plan", type=str, default="")     # e.g. 6x24/16/16/16,3x48/32/32/16,3x96/64/48/32
    ap.add_argument("--slot_topk", type=int, default=0)    # each token writes to exactly k slots per layer
    ap.add_argument("--slot_gtie", type=int, default=1)    # unwritten slots do not decay
    ap.add_argument("--gist3", type=int, default=0)        # hub-driven compressor
    ap.add_argument("--gist3_nq", type=int, default=8)
    ap.add_argument("--gist_sparse", type=int, default=0)    # compressor fan-in (0 = dense stages)
    ap.add_argument("--gist_seed", type=int, default=20260905)
    ap.add_argument("--kchunk", type=int, default=0)         # chunked scans with per-chunk recompute (0 = off)
    ap.add_argument("--kchunk_layers", type=int, default=-1) # per-layer scan chunk (-1 = kchunk, 0 = whole sequence)
    ap.add_argument("--gist_bs", type=int, default=0)        # 1: block-sparse compressor stages + page (bsparse kernels)
    ap.add_argument("--gist_nb", type=int, default=5)        # 32-wide input blocks per output block
    ap.add_argument("--beta_head", type=int, default=0)      # write strength per head (decay stays per slot)
    ap.add_argument("--emit_ckpt", type=int, default=0)      # 1 = recompute each layer's scan emission in backward (saves ~1.2 GB at B2, +2% time)
    ap.add_argument("--wbud", type=float, default=0.0)       # weight of the soft write budget (mean gate activity above --wbud_rho per layer)
    ap.add_argument("--wbud_rho", type=float, default=0.3)   # target fraction of columns written per token
    ap.add_argument("--wbud_count", type=int, default=0)     # 1 = budget on the COUNT of written columns (beta > --wbud_tau, straight-through) instead of mean activity
    ap.add_argument("--wbud_tau", type=float, default=0.05)
    ap.add_argument("--decay_head", type=int, default=0)     # log-decay per head
    ap.add_argument("--shadow_shared", type=int, default=0)  # one shared shadow projection set for the compressor
    ap.add_argument("--kprep", type=int, default=0)          # fused gate/lane-mix/query kernel (kprep.py): exact, but measured SLOWER on the H100 (353 vs 322.6 ms) -> off
    ap.add_argument("--gist2", type=int, default=0)   # raw-state emission + per-logical-head joint funnel
    ap.add_argument("--gistfuse", type=int, default=0,
                    help="1 = fused delta+pairwise gist emission (no full state stream)")
    ap.add_argument("--qphi", type=int, default=0,
                    help="1 = read queries through the key encoder (normalized)")
    ap.add_argument("--gistpair", type=int, default=0,
                    help="1 = pairwise 2:1 front stage in the gist funnel")
    ap.add_argument("--gistk", type=int, default=1,
                    help="refresh the state gist every K tokens (1 = per token)")
    ap.add_argument("--wmoe", type=int, default=0,
                    help="1 = expert-conditional WRITES into the shared per-layer state")
    ap.add_argument("--n_exp", type=int, default=8)
    ap.add_argument("--top_k", type=int, default=2)
    ap.add_argument("--moe_u", type=float, default=0.001)
    ap.add_argument("--cap", type=float, default=1.35)
    ap.add_argument("--cap_warmup", type=int, default=500)
    ap.add_argument("--nlk", type=int, default=0)
    ap.add_argument("--dphi", type=int, default=1024)
    ap.add_argument("--d_ff", type=int, default=2048)
    ap.add_argument("--d_model", type=int, default=768)
    ap.add_argument("--layers", type=int, default=12)
    ap.add_argument("--tokens", type=float, default=5e9)
    ap.add_argument("--tok_per_micro", type=int, default=8192)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--min_lr_frac", type=float, default=0.1)
    ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--stages", default="128,256,512,1024")
    ap.add_argument("--ramp_frac", type=float, default=0.20)
    ap.add_argument("--p_synth", type=float, default=0.08)
    ap.add_argument("--vocab", type=int, default=50257)
    ap.add_argument("--p_synth_start", type=float, default=-1.0)
    ap.add_argument("--p_synth_ramp", type=float, default=0.25)
    ap.add_argument("--stage_weight", type=int, default=1)
    ap.add_argument("--qpaths", type=int, default=1)
    ap.add_argument("--bus", type=int, default=0)      # cross-layer state bus
    ap.add_argument("--chunk", type=int, default=256)  # bus refresh period, tokens
    ap.add_argument("--ref", type=int, default=0)      # 1 = pure-torch recurrence
    ap.add_argument("--bus_m", type=int, default=8)        # dv contraction in compressor
    ap.add_argument("--bus_slots", type=int, default=256)  # compressor slot width
    ap.add_argument("--bus_nslot", type=int, default=1)    # slots per head
    ap.add_argument("--bus_r", type=int, default=8)        # read queries per token
    ap.add_argument("--wkey", type=int, default=0)
    ap.add_argument("--wkfix", type=float, default=-1.0)         # state-aware key placement
    ap.add_argument("--wfreeze", type=int, default=0)      # write to top-k heads only
    ap.add_argument("--wpreserve", type=int, default=0)    # value sees current content
    ap.add_argument("--wortho", type=int, default=0)       # project key off occupied dirs
    ap.add_argument("--gtop", type=int, default=10)        # heads the read path selects
    ap.add_argument("--gidx", type=int, default=64)        # per-head index width
    ap.add_argument("--gread", type=int, default=4)        # read queries per token
    ap.add_argument("--kcenter", type=int, default=0)      # subtract running key mean
    ap.add_argument("--p_mqar", type=float, default=0.0)   # share of synth that is MQAR
    ap.add_argument("--p_tasks", type=float, default=0.0)  # share of synth routed to task_lib
    ap.add_argument("--task_mix", type=str, default="keyed_span:2,updated:1,tagged:1,list:1,multihop:1,reverse:1,pred:1")
    ap.add_argument("--p_broad", type=float, default=0.25)    # frac of gaps drawn uniformly
    ap.add_argument("--long_frac", type=float, default=0.5)   # rest drawn from [lf*max, max]
    ap.add_argument("--tbias", type=float, default=0.0)       # stage draw weight ~ (i+1)**tbias
    ap.add_argument("--pre", type=int, default=0)        # FFN blocks before layer 0
    # memlm_slot (clean-slot memory) knobs; memlm_m5 ignores them (signature-filtered below)
    ap.add_argument("--n_slots", type=int, default=48)     # clean slots per subhead
    ap.add_argument("--scan_chunk", type=int, default=64)  # slot-scan checkpoint interval (backward recompute chunk)
    ap.add_argument("--buf_stride", type=int, default=1)   # compressor runs every G tokens on the slot buffer (1 = per token)
    ap.add_argument("--kv_window", type=int, default=16)   # exact short-term KV cache window
    ap.add_argument("--kv_dim", type=int, default=32)      # KV-cache head dim (q/k/v)
    ap.add_argument("--kv_heads", type=int, default=0)    # 0 = same as --heads; lets the local window keep its head count when the memory drops to one head
    ap.add_argument("--tk_nq", type=int, default=0)       # 0 = 1. Query heads per memory head: a grouped-query read over one rack
    ap.add_argument("--comp_nb", type=int, default=5)      # block-sparse fan-in (32-blocks per output block) in the compressor
    ap.add_argument("--xbus", type=int, default=1)         # 1 = cross-layer 64-wide bus from earlier layers
    ap.add_argument("--slot_hard", type=int, default=1)    # 1 = one-slot (argmax) writes, straight-through address gradient
    ap.add_argument("--slot_hard_gate", type=int, default=1)   # 1 = binary write gate (write iff gate > 0.5), straight-through
    ap.add_argument("--gate_init", type=float, default=0.3)    # initial write probability (per-subhead gate bias)
    ap.add_argument("--sel_heads", type=int, default=4)       # selection-attention heads inside the compressor (0 = funnel only)
    # HSEL: learned row selector trained by HINDSIGHT supervision rather than by the read's own softmax
    # gradient, which carries a factor att_n and so starves exactly the rows the model has stopped looking
    # at.  Target comes from the model itself (u_n = dL/d(read) . row_n), so no annotation is needed.
    # OSEL: supervise the cosine read toward the row that actually MINIMISES the loss.
    ap.add_argument("--osel", type=int, default=0)
    ap.add_argument("--osel_k", type=int, default=8)      # shortlist size; 8 of 24 held the true best 100%
    ap.add_argument("--osel_lam", type=float, default=0.3)
    ap.add_argument("--osel_layer", type=int, default=-1) # -1 = the last block (everything above is position-wise)
    # XSEL: exact-target contrastive supervision on the COSINE read (the read itself is unchanged).
    ap.add_argument("--xsel", type=int, default=0)
    ap.add_argument("--xsel_lam", type=float, default=1.0)
    ap.add_argument("--xsel_tau", type=float, default=0.1)
    ap.add_argument("--hsel", type=int, default=0)          # 1 = use the hindsight selector instead of the cosine read
    ap.add_argument("--hsel_merge", type=int, default=0)    # 0 per subhead, 1 per 2 subheads, 2 per big head
    ap.add_argument("--hsel_hid", type=str, default="400,400,200,100")
    ap.add_argument("--hsel_floor", type=float, default=1e-3)   # no valid row's read weight may reach 0
    ap.add_argument("--hsel_lam", type=float, default=1.0)      # weight on the injected selection gradient
    ap.add_argument("--hsel_tau", type=float, default=1.0)      # temperature of the hindsight target
    ap.add_argument("--hsel_prev", type=int, default=1)         # feed the n-1 layer state into the selector
    ap.add_argument("--tksel", type=int, default=0)         # 1 = per-subhead TIED-KEY read: cos(row[:tksel_dk], q) x learned temp, softmax over the subhead's own rows, straight to the residual
    ap.add_argument("--tksel_dk", type=int, default=16)     # how many of a row's leading dims are its ADDRESS (the rest are payload)
    ap.add_argument("--comp_off", type=int, default=0)      # 1 = no compressor/funnel at all (forces xbus off); the exact window gets its own path to the residual
    ap.add_argument("--wbud_lo", type=float, default=0.0)   # >0 = also penalise a write rate BELOW this (a floor); 0 = ceiling only
    ap.add_argument("--vcenter", type=int, default=0)    # center VALUES (running mean)
    ap.add_argument("--gistr", type=int, default=64)
    ap.add_argument("--gistf", type=str, default="")
    ap.add_argument("--gistph", type=int, default=0)
    ap.add_argument("--ginjall", type=int, default=0)
    ap.add_argument("--ginorm", type=int, default=0)
    ap.add_argument("--relift", type=int, default=0)     # steps to ramp lr after a resume (progressive re-lift)
    ap.add_argument("--relift_lr0", type=float, default=5e-5)
    ap.add_argument("--gisth", type=int, default=1024)
    ap.add_argument("--vwhiten", type=int, default=0)   # ZCA-whiten values
    ap.add_argument("--kdecor", type=float, default=0.0)   # key-decorrelation loss weight
    ap.add_argument("--krep", type=float, default=0.0)    # key-repulsion (collision) loss weight
    ap.add_argument("--krep_tau", type=float, default=0.85)
    ap.add_argument("--mqar_pairs", type=int, default=32)  # facts held simultaneously
    ap.add_argument("--bfloor", type=float, default=0.0)
    ap.add_argument("--beta_u", type=float, default=0.001)
    ap.add_argument("--p_synth_lo", type=float, default=-1.0)
    ap.add_argument("--p_synth_hi", type=float, default=-1.0)
    ap.add_argument("--p_free", type=float, default=0.0)
    ap.add_argument("--free_lo", type=float, default=0.15)
    ap.add_argument("--free_hi", type=float, default=0.60)
    ap.add_argument("--free_topk", type=int, default=50)
    ap.add_argument("--doc_aware", type=int, default=1)
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--eval_every", type=int, default=2000)
    ap.add_argument("--eval_bs", type=int, default=8)          # eval batch (1 on the 12 GB card)
    ap.add_argument("--opt8bit", type=int, default=0)         # bitsandbytes 8-bit AdamW
    # FP16PORT 2026-09-14: V100 (sm_70) has NO bf16.  A straight bf16->fp16 swap is unsafe: bf16 carries
    # fp32 EXPONENT range, fp16 has 5 exponent bits, so pure-fp16 weights underflow the Adam update.
    # amp=fp16 therefore keeps the model in FP32 and uses autocast + a GradScaler instead.
    # NOTE: do NOT gate on torch.cuda.is_bf16_supported() -- measured, it returns True on a V100.
    ap.add_argument("--amp", type=str, default="bf16", choices=["bf16", "fp16", "auto"])
    ap.add_argument("--ckpt_every", type=int, default=1000)
    ap.add_argument("--ckpt_dir", type=str, default="")     # where <tag>_ckpt.pt lives (default: MEMPROBE_ROOT)
    ap.add_argument("--ans_w", type=float, default=1.0)     # weight on tokens after a recall cue (1 = off)
    ap.add_argument("--ce_fused", type=int, default=1)      # 1 = loss from hidden states via _ChunkedCE (no logits-grad buffer churn)
    a = ap.parse_args()

    # DDPPORT 2026-09-14: data parallel WITHOUT wrapping the module.  A DistributedDataParallel wrapper
    # would break the many `model.blocks` / `model.util` / `model.set_moe_loop` accesses in this file, so
    # instead the gradients are all-reduced ONCE per optimizer step -- which is also the minimum possible
    # communication (DDP overlaps per-bucket reductions with backward; here there is exactly one pass).
    # Every rank builds the model under the SAME torch seed, so the replicas start identical with no broadcast.
    _WORLD = int(os.environ.get("WORLD_SIZE", "1"))
    _RANK  = int(os.environ.get("RANK", "0"))
    _LOCAL = int(os.environ.get("LOCAL_RANK", "0"))
    _DDP   = _WORLD > 1
    if _DDP:
        import torch.distributed as dist
        torch.cuda.set_device(_LOCAL)          # makes bare "cuda" mean THIS rank's device
        dist.init_process_group("nccl")
        if _RANK != 0:                          # one log, not N interleaved copies
            import builtins as _bi
            _bi.print = lambda *a, **k: None
        print(f"[ddp] world {_WORLD}, manual grad all-reduce once per step", flush=True)

    torch.manual_seed(0)
    # each rank must draw DIFFERENT windows, or the second GPU just recomputes the first one's batch
    rng = np.random.default_rng([0, _RANK] if _DDP else 0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    _cfg_kw = dict(vocab=a.vocab, dk=a.dk, mixer=a.mixer, moe=a.moe, wmoe=a.wmoe, gistk=a.gistk, gistpair=a.gistpair, qphi=a.qphi, gistfuse=a.gistfuse, gist2=a.gist2, gist3=a.gist3, gist3_nq=a.gist3_nq, gist_sparse=a.gist_sparse, gist_seed=a.gist_seed, kchunk=a.kchunk, kchunk_layers=a.kchunk_layers, gist_bs=a.gist_bs, gist_nb=a.gist_nb, kprep=a.kprep, beta_head=a.beta_head, decay_head=a.decay_head, shadow_shared=a.shadow_shared, emit_ckpt=a.emit_ckpt, wbud=a.wbud, wbud_rho=a.wbud_rho, wbud_count=a.wbud_count, wbud_tau=a.wbud_tau, kslot=a.kslot, dv_plan=a.dv_plan, slot_topk=a.slot_topk, slot_gtie=a.slot_gtie, lanes=a.lanes, n_exp=a.n_exp, top_k=a.top_k,
              moe_u=a.moe_u, d_ff=a.d_ff, d_model=a.d_model, layers=a.layers,
              nlk=a.nlk, dphi=a.dphi, cap=a.cap,
              heads=a.heads, dv=a.dv, gbias=a.gbias,
              hl_min=a.hl_min, hl_max=a.hl_max,
              bfloor=a.bfloor, beta_u=a.beta_u, qpaths=a.qpaths,
              bus=a.bus, chunk=a.chunk, ref=a.ref,
              bus_m=a.bus_m, bus_slots=a.bus_slots, bus_nslot=a.bus_nslot,
              bus_r=a.bus_r, wkey=a.wkey, wkfix=a.wkfix,
              wfreeze=a.wfreeze, wpreserve=a.wpreserve, wortho=a.wortho,
              gtop=a.gtop, gidx=a.gidx, gread=a.gread, kcenter=a.kcenter,
              kdecor=a.kdecor, krep=a.krep, krep_tau=a.krep_tau, pre=a.pre, vcenter=a.vcenter,
              gistr=a.gistr, gisth=a.gisth, gistf=a.gistf, gistph=a.gistph, ginjall=a.ginjall, vwhiten=a.vwhiten, ginorm=a.ginorm,
              n_slots=a.n_slots, scan_chunk=a.scan_chunk, buf_stride=a.buf_stride, kv_window=a.kv_window,
              kv_dim=a.kv_dim, kv_heads=a.kv_heads, tk_nq=a.tk_nq, comp_nb=a.comp_nb, xbus=a.xbus, slot_hard=a.slot_hard,
              slot_hard_gate=a.slot_hard_gate, gate_init=a.gate_init, sel_heads=a.sel_heads,
              tksel=a.tksel, tksel_dk=a.tksel_dk, comp_off=a.comp_off, wbud_lo=a.wbud_lo,
              osel=a.osel, osel_k=a.osel_k, osel_lam=a.osel_lam,
              xsel=a.xsel, xsel_lam=a.xsel_lam, xsel_tau=a.xsel_tau,
              hsel=a.hsel, hsel_merge=a.hsel_merge, hsel_hid=a.hsel_hid, hsel_floor=a.hsel_floor,
              hsel_lam=a.hsel_lam, hsel_tau=a.hsel_tau, hsel_prev=a.hsel_prev)
    import inspect as _insp
    _ok = set(_insp.signature(Cfg.__init__).parameters) - {"self"}
    _dropped = sorted(k for k in _cfg_kw if k not in _ok)
    if _dropped: print("[cfg] this module ignores:", " ".join(_dropped))
    cfg = Cfg(**{k: v for k, v in _cfg_kw.items() if k in _ok})
    if a.amp == "auto":
        a.amp = "bf16" if torch.cuda.get_device_capability(0)[0] >= 8 else "fp16"
    _FP16 = (a.amp == "fp16")
    if _FP16:
        model = MemLM(cfg).to(DEV)                 # FP32 weights; autocast casts the math to fp16
        scaler = torch.amp.GradScaler("cuda")
        print(f"[amp] fp16: fp32 weights + autocast + GradScaler (device cc "
              f"{torch.cuda.get_device_capability(0)})", flush=True)
    else:
        model = MemLM(cfg).to(DEV).to(torch.bfloat16)
        scaler = None
        print("[amp] bf16: whole-model bf16, no autocast, no scaler", flush=True)
    nparam = sum(p.numel() for p in model.parameters())

    if _DDP and a.accum % _WORLD:
        raise SystemExit(f"--accum {a.accum} must divide WORLD_SIZE {_WORLD}")
    _accum_local = a.accum // _WORLD if _DDP else a.accum   # --accum is the GLOBAL micro-batch count
    tok_per_step = a.tok_per_micro * a.accum
    total_steps = int(a.tokens / tok_per_step)
    stages = [int(s) for s in a.stages.split(",")]
    t_ref = sum(stages) / len(stages)

    decay = [p for p in model.parameters() if p.dim() >= 2]
    nodecay = [p for p in model.parameters() if p.dim() < 2]
    # patch26: fp32 master copies for all dim<2 params (scalar gates, biases, LN
    # weights). In pure bf16 their ulp (0.008-0.03 at gate-init magnitudes) exceeds the
    # ~lr-sized Adam step, so every update rounded to zero -- the optimizer steps the
    # fp32 masters, which are written back to the bf16 model params each step.
    nodecay32 = nodecay if _FP16 else [p.detach().float().clone() for p in nodecay]
    torch.backends.cuda.matmul.allow_tf32 = True   # fp32 einsums (krep Gram, vwhiten) on tensor cores
    if getattr(a, "compile", 0):
        for _k, _v in (("recompile_limit", 64), ("cache_size_limit", 64), ("accumulated_recompile_limit", 4096)):
            if hasattr(_dynamo.config, _k):
                setattr(_dynamo.config, _k, _v)   # 4 stage lengths x guard variants > 8 -> silent eager fallback
        # wrap the BOUND FORWARDS, not the modules: state_dict keys, checkpoints,
        # resumes and hooks stay untouched; inductor still fuses the eager glue.
        if getattr(model, "shadow", None) is not None:
            model.shadow._shadow_inputs = torch.compile(model.shadow._shadow_inputs, dynamic=False)
        for _blk in model.blocks:
            _blk.mix.forward = torch.compile(_blk.mix.forward, dynamic=False)
            if hasattr(_blk.mix, "_shadow_inputs"):
                _blk.mix._shadow_inputs = torch.compile(_blk.mix._shadow_inputs, dynamic=False)   # the compressor's per-layer projections
            _blk.ffn.forward = torch.compile(_blk.ffn.forward, dynamic=False)
        for _pb in model.pre_blocks:
            _pb.forward = torch.compile(_pb.forward, dynamic=False)
        print("compile: forwards wrapped (kernels graph-break; ckpts unaffected)", flush=True)
    if getattr(a, "opt8bit", 0):
        import bitsandbytes as bnb                 # 8-bit Adam states: ~1.15 GB instead of 2.3 GB for this model (12 GB card)
        opt = bnb.optim.AdamW8bit([{"params": decay, "weight_decay": 0.1},
                                 {"params": nodecay32, "weight_decay": 0.0}],
                                lr=a.lr, betas=(0.9, 0.95), eps=1e-8, )
    else:
        opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1},
                                 {"params": nodecay32, "weight_decay": 0.0}],
                                lr=a.lr, betas=(0.9, 0.95), eps=1e-8, fused=True)

    _cdir = a.ckpt_dir or MEMPROBE_ROOT
    os.makedirs(_cdir, exist_ok=True)
    ck = os.path.join(_cdir, f"{a.tag}_ckpt.pt")
    print(f"checkpoints: {ck}", flush=True)
    step0 = 0
    # RNGFIX 2026-09-13: a checkpoint used to store only model/opt/step/cfg, so a RESUME restarted the
    # data stream from default_rng(0) and replayed the batches the run had already trained on -- silently.
    # Measured: m12 crash-looped 6 times off its step-8000 checkpoint and replayed the same windows each
    # time; s3 replayed its first ~740 steps after one restart at 2000.  Checkpointing itself was never the
    # problem (torch.save does not touch the rng) -- only a process restart was.  Now the sampler state
    # rides along.  Reading it back is best-effort: an OLD checkpoint has no "rng" key and must still load.
    def _rng_state():
        d = {"np": rng.bit_generator.state}
        try: d["torch"] = torch.get_rng_state()
        except Exception: pass
        try: d["cuda"] = torch.cuda.get_rng_state_all()
        except Exception: pass
        return d

    if os.path.exists(ck):
        st = torch.load(ck, map_location=DEV, weights_only=False)
        _miss = model.load_state_dict(st["model"], strict=False)
        if getattr(_miss, "missing_keys", None):
            print("load: new buffers initialised ->", _miss.missing_keys[:6],
                  f"({len(_miss.missing_keys)} total)", flush=True)
        if getattr(_miss, "unexpected_keys", None):
            print("load: UNEXPECTED keys ->", _miss.unexpected_keys[:6], flush=True)
        if os.environ.get("MEMLM_FRESH_OPT") == "1":
            # Needed by any arm that CHANGES THE PARAMETER SET or the optimizer type: untying adds
            # a tensor (so the saved positional indices no longer line up) and --opt8bit 0 cannot
            # read bitsandbytes' qmap/absmax state at all.  Such an arm must therefore be compared
            # only against a control that ALSO starts fresh -- never across this boundary.
            print("FRESH_OPT: optimizer moments deliberately NOT restored", flush=True)
        else:
            opt.load_state_dict(st["opt"])
        step0 = st["step"]
        with torch.no_grad():   # patch26: re-sync masters to the resumed weights
            for _p32, _p in zip(nodecay32, nodecay):
                _p32.data.copy_(_p.data.float())
        # RANKGUARD 2026-09-14: under DDP each rank needs its OWN sampler state.  Only rank 0 writes the
        # checkpoint, so restoring st["rng"] on every rank would hand them all rank 0's stream and every
        # GPU would then train on identical batches -- the exact waste the per-rank seeding avoids.
        # Each rank therefore drops a small sidecar beside the checkpoint and reads its own back.
        _rs = None
        if _DDP:
            _side = ck + f".rng{_RANK}"
            if os.path.exists(_side):
                try: _rs = torch.load(_side, map_location="cpu", weights_only=False)
                except Exception as _e: print(f"WARNING: rng sidecar unreadable ({_e})", flush=True)
            if _rs is None:
                print(f"WARNING: no rng sidecar for rank {_RANK}; its data stream restarts from the top",
                      flush=True)
        if _rs is None and not _DDP:
            _rs = st.get("rng")
        if _rs:
            # one try per generator: restoring the CUDA state fails when the device COUNT differs
            # (a 2-GPU checkpoint resumed on 1 GPU raises IndexError), and a single try-block would
            # then report the DATA stream as lost when numpy had in fact already been restored.
            try:
                rng.bit_generator.state = _rs["np"]
                print("resumed the DATA STREAM too (numpy rng restored)", flush=True)
            except Exception as _e:
                print(f"WARNING: numpy rng not restorable ({type(_e).__name__}: {_e}); "
                      f"THE DATA STREAM RESTARTS FROM THE TOP", flush=True)
            try:
                if _rs.get("torch") is not None:
                    torch.set_rng_state(_rs["torch"].cpu() if hasattr(_rs["torch"], "cpu") else _rs["torch"])
            except Exception as _e:
                print(f"note: torch cpu rng not restored ({type(_e).__name__})", flush=True)
            try:
                if _rs.get("cuda") is not None and torch.cuda.is_available():
                    _cs = [t.cpu() if hasattr(t, "cpu") else t for t in _rs["cuda"]]
                    if len(_cs) == torch.cuda.device_count():
                        torch.cuda.set_rng_state_all(_cs)
                    else:
                        print(f"note: checkpoint holds {len(_cs)} cuda rng states but this process sees "
                              f"{torch.cuda.device_count()}; skipping (data stream is unaffected)", flush=True)
            except Exception as _e:
                print(f"note: cuda rng not restored ({type(_e).__name__})", flush=True)
        else:
            print("WARNING: this checkpoint predates rng-state saving -- the data stream restarts from the "
                  "top, so this run will replay batches it has already seen", flush=True)
        print(f"resumed from step {step0}", flush=True)

    _vwn_prev = int(model.blocks[0].mix.vwn.item()) if getattr(model.blocks[0].mix, "vwhiten", 0) else 0
    _task_names, _task_weights = [], []
    if a.p_tasks > 0:
        for _it in a.task_mix.split(","):
            _nm, _wt = _it.split(":"); _task_names.append(_nm.strip()); _task_weights.append(float(_wt))
        print(f"=== retrieval task suite: p_tasks {a.p_tasks} mix {dict(zip(_task_names,_task_weights))} ===", flush=True)
    print(f"=== {a.tag} | {cfg} | params {nparam/1e6:.1f}M ===", flush=True)
    print(f"=== {a.tokens/1e9:.1f}B tokens | {total_steps:,} steps | {tok_per_step:,} tok/step "
          f"| stages {stages} | doc_aware {a.doc_aware} | p_synth {a.p_synth} ===", flush=True)

    sh = Shards(doc_aware=bool(a.doc_aware))
    while len(sh.files) < 2:
        print("waiting for shards...", flush=True); time.sleep(60); sh.refresh()

    def lr_at(s):
        if s < a.warmup:
            return a.lr * (s + 1) / a.warmup
        p = (s - a.warmup) / max(1, total_steps - a.warmup)
        return a.lr * (a.min_lr_frac + (1 - a.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * min(p, 1.0))))

    def active_lengths(s):
        frac = s / max(1, total_steps)
        k = 1 + int(len(stages) * frac / max(a.ramp_frac, 1e-9))
        return stages[:max(1, min(k, len(stages)))]

    model.train()
    t0 = time.time(); seen = 0
    for step in range(step0, total_steps):
        lr = lr_at(step)
        if a.relift > 0 and step < step0 + a.relift:   # patch29: glide from lr0 to schedule
            _f = (step - step0) / max(1, a.relift)
            lr = a.relift_lr0 + (lr - a.relift_lr0) * _f
        for g in opt.param_groups:
            g["lr"] = lr
        act = active_lengths(step)
        if a.moe:
            model.set_moe_loop(step < a.cap_warmup)   # zero-drop until routing balances
        opt.zero_grad(set_to_none=True)
        lsum = 0.0
        nfree = 0
        _plain_acc = 0.0; _hot_acc = 0.0; _nmb = 0; _wb_acc = 0.0; _xs_acc = 0.0; _xs_n = 0; _src_hit = 0; _src_tot = 0; _os_acc = 0.0; _os_n = 0
        for _ in range(_accum_local):
            if a.tbias > 0 and len(act) > 1:
                # longer contexts drawn more often: recall spans are the point,
                # and a uniform draw spends 80% of steps below the target length
                _tw = np.arange(1, len(act) + 1, dtype=np.float64) ** a.tbias
                T = int(act[rng.choice(len(act), p=_tw / _tw.sum())])
            else:
                T = int(act[rng.integers(0, len(act))])
            bs = max(1, a.tok_per_micro // T)
            if a.p_synth_lo >= 0:        # asymmetric floor, log-interpolated over stages
                _l0, _l1 = math.log(stages[0]), math.log(stages[-1])
                _fT = 0.0 if _l1 <= _l0 else (math.log(T) - _l0) / (_l1 - _l0)
                floor_T = a.p_synth_lo + (a.p_synth_hi - a.p_synth_lo) * _fT
            else:
                floor_T = a.p_synth
            if a.p_synth_start >= 0:      # recall-heavy -> text, never to zero
                fr_ = min(1.0, step / max(1, int(a.p_synth_ramp * total_steps)))
                p_syn = a.p_synth_start + (floor_T - a.p_synth_start) * fr_
            else:
                p_syn = floor_T
            if a.stage_weight and a.p_synth_lo < 0:   # asym floor supersedes legacy weighting
                p_syn = min(0.95, p_syn * T / (sum(act) / len(act)))
            p_syn = min(0.95, max(0.0, p_syn))
            if step % 1000 == 0:
                print(f"CURRICULUM step {step} p_synth {p_syn:.3f} T {T} free {nfree}/{_accum_local}", flush=True)
            x, y, wt = sh.batch(bs, T, rng, p_synth=p_syn, ans_w=a.ans_w,
                            p_mqar=a.p_mqar, mqar_pairs=a.mqar_pairs,
                            p_broad=a.p_broad, long_frac=a.long_frac,
                            p_tasks=a.p_tasks, task_names=_task_names, task_weights=_task_weights)
            if a.p_free > 0 and T >= 128 and rng.random() < a.p_free:
                with torch.autocast("cuda", dtype=torch.float16, enabled=_FP16):
                    x = student_force(model, x, rng, a.free_lo, a.free_hi, a.free_topk)
                nfree += 1
            _src = source_pos(x, y, wt) if (a.xsel or a.osel) else None
            if a.osel:
                _LY = (len(model.blocks) - 1) if a.osel_layer < 0 else a.osel_layer
                _ot = oracle_target(model, x, y, _src, _LY, a.osel_k)
                for _b in model.blocks: _b.mix._osel_tgt = None
                if _ot is not None: model.blocks[_LY].mix._osel_tgt = _ot
            if _src is not None:
                _ans = (wt > 1.0) if wt is not None else torch.ones_like(_src, dtype=torch.bool)
                _src_tot += int(_ans.sum()); _src_hit += int((_src >= 0).sum())
            with torch.autocast("cuda", dtype=torch.float16, enabled=_FP16):
                if a.ce_fused:
                    hid = model(x, skip_head=True, src=_src)
                    loss, _plain, _hot = chunked_ce_h(hid, model.head.weight, y, w=(wt if a.ans_w != 1.0 else None))
                else:
                    logits = model(x, src=_src)
                    loss, _plain, _hot = chunked_ce(logits, y, w=(wt if a.ans_w != 1.0 else None))
            _plain_acc += float(_plain); _hot_acc += float(_hot); _nmb += 1
            loss = loss / _accum_local
            if a.kdecor > 0 and getattr(model, "kdec", None) is not None:
                loss = loss + (a.kdecor * model.kdec) / _accum_local
            if a.krep > 0 and getattr(model, "krep", None) is not None:
                loss = loss + (a.krep * model.krep) / _accum_local
            if a.osel and getattr(model, "osel", None) is not None:
                loss = loss + (a.osel_lam * model.osel) / _accum_local
                _os_acc += float(model.osel); _os_n += 1
            if a.xsel and getattr(model, "xsel", None) is not None:
                loss = loss + (a.xsel_lam * model.xsel) / _accum_local
                _xs_acc += float(model.xsel); _xs_n += 1
            if a.wbud > 0 and getattr(model, "wbud", None) is not None:
                loss = loss + (a.wbud * model.wbud) / _accum_local
                _wb_acc += float(model.wbud)
            (scaler.scale(loss) if _FP16 else loss).backward()
            lsum += loss.item()
            # value-whitening eigen-refresh, trainer-driven at the original in-forward
            # cadence (micro-batch count n <= 5 or n % 20 == 1); vwn counts every
            # training forward, so the cadence is read from the buffer itself.
            _vwn_now = int(model.blocks[0].mix.vwn.item()) if getattr(model.blocks[0].mix, "vwhiten", 0) else 0
            if _vwn_now:
                for _m in range(_vwn_prev + 1, _vwn_now + 1):
                    if _m <= 5 or _m % 20 == 1:
                        model.vw_refresh(); break
                _vwn_prev = _vwn_now
            seen += bs * T
        if _DDP:
            # reduce the SCALED grads: unscale_ then computes found_inf on identical data on every rank,
            # so scaler.step()/update() make the same skip decision everywhere and the scales never diverge
            for _p in model.parameters():
                if _p.grad is not None:
                    dist.all_reduce(_p.grad, op=dist.ReduceOp.SUM)
                    _p.grad.div_(_WORLD)
        if _FP16:
            scaler.unscale_(opt)          # grads must be unscaled BEFORE the norm clip, or the clip is meaningless
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not _FP16:
            for _p32, _p in zip(nodecay32, nodecay):   # patch26: route grads to masters
                _p32.grad = None if _p.grad is None else _p.grad.float()
        if _FP16:
            scaler.step(opt)              # skips the step entirely if any grad was inf/nan
            scaler.update()
        else:
            opt.step()
        if a.lane_clamp > 0 and step < a.lane_clamp_until:
            with torch.no_grad():
                _gate_ids = set()
                for _b in model.blocks:
                    _mx = getattr(_b, "mix", None)
                    for _nm in ("lane_gq", "lane_gp"):
                        _p = getattr(_mx, _nm, None)
                        if _p is not None:
                            _p.clamp_(-a.lane_clamp, a.lane_clamp)
                            _gate_ids.add(id(_p))
                # the optimizer steps fp32 MASTERS and writes them back below: bound those too,
                # or the write-back undoes the clamp every step (measured: |logit| 2.70 at 8k)
                if not _FP16:
                    for _p32, _p in zip(nodecay32, nodecay):
                        if id(_p) in _gate_ids:
                            _p32.clamp_(-a.lane_clamp, a.lane_clamp)
        if not _FP16:
            with torch.no_grad():                   # patch26: bf16 write-back
                for _p32, _p in zip(nodecay32, nodecay):
                    _p.data.copy_(_p32.data.to(_p.dtype))

        if _DDP:
            # ebias is a BUFFER that steers expert routing and is updated from each rank's own token
            # counts; nothing syncs buffers here, so without this the two replicas would drift into
            # routing differently while sharing gradients.  Rank 0's copy wins, which is what stock
            # DDP's broadcast_buffers does.  (Ranks drift slightly WITHIN a step; that is accepted.)
            with torch.no_grad():
                for _m in model.modules():
                    _eb = getattr(_m, "ebias", None)
                    if _eb is not None:
                        dist.broadcast(_eb, 0)

        if os.environ.get("MEMLM_PARAMHASH"):
            # PARAMHASH: every rank prints the same scalar summary of its weights.  If the gradient
            # all-reduce works, the replicas started identical and stepped identically, so these MUST
            # agree digit for digit; a divergence means the ranks are training different models.
            # stderr, because rank!=0 has stdout silenced.
            if step % int(os.environ["MEMLM_PARAMHASH"]) == 0:
                _ph = sum(float(_p.detach().float().abs().sum()) for _p in model.parameters())
                sys.stderr.write(f"[paramhash] rank {_RANK} step {step} {_ph:.6f}\n"); sys.stderr.flush()

        if step % a.log_every == 0:
            el = time.time() - t0
            tps = seen / max(el, 1e-9)
            eta = (total_steps - step) * tok_per_step / max(tps, 1) / 86400
            rej = sh.rejects / max(sh.rejects + sh.taken, 1)
            ustr = ""
            if a.moe and getattr(model, "util", None) is not None:
                uu = model.util
                ustr = (f" exp_min {uu.min().item():.3f} exp_max {uu.max().item():.3f}"
                        f" maxvio {float(model.maxvio):.3f}"
                        f" drop {float(model.drop):.4f}")
            if a.ans_w != 1.0 and _nmb:
                ustr += f" ce {_plain_acc/_nmb:.4f} recall {100*_hot_acc/_nmb:.1f}%"
            if a.osel and _os_n:
                ustr += f" osel {_os_acc/_os_n:.3f}"      # CE of the read's scores vs the EXACT best row
            if a.xsel and _xs_n:
                # xsel = CE of the read's own cosine scores against the row the data says is correct.
                # src% = fraction of ANSWER tokens for which a source position was found at all; if that
                # is ~0 the term is silently doing nothing, which is the failure mode to watch for.
                ustr += f" xsel {_xs_acc/_xs_n:.3f} src {100*_src_hit/max(1,_src_tot):.1f}%"
            if a.wbud > 0 and _nmb:
                ustr += f" wbud {_wb_acc/_nmb:.3f}"      # sum over layers of relu(mean gate activity - rho): 0 = every layer inside its budget
            print(f"step {step:6d}/{total_steps} loss {lsum:.4f} ppl {math.exp(min(lsum,20)):8.2f}{ustr} "
                  f"lr {lr:.2e} T{act[-1]:5d} {tps:8,.0f} tok/s eta {eta:.2f}d splice {rej:.1%}",
                  flush=True)
            t0 = time.time(); seen = 0

        if step > step0 and step % a.eval_every == 0:
            vl, bk = evaluate(model, sh, np.random.default_rng(1), bs=a.eval_bs)
            cp, fr = recall_probe(model, sh, np.random.default_rng(2))
            print(f"EVAL step {step} val_loss {vl:.4f} val_ppl {math.exp(min(vl,20)):.2f} "
                  f"p64 {bk[1]:.4f} p512 {bk[len(bk)//2]:.4f} p1000 {bk[-1]:.4f} "
                  f"longgain {bk[1]-bk[-1]:+.4f} | RECALL copy {cp:.4f} fresh {fr:.4f} "
                  f"benefit {fr-cp:+.4f}", flush=True)
            print("CURVE step " + str(step) + " " + " ".join(f"{b:.4f}" for b in bk), flush=True)

        # ON-DEMAND CHECKPOINT: `touch <ckpt_dir>/CKPT_NOW` and the next step writes one, then clears the flag.
        # Without this the only checkpoints are every --ckpt_every steps, so a probe cannot be run when something
        # interesting happens -- which cost a 4-hour wait on s3 at step 980.
        _now = os.path.join(os.path.dirname(ck), "CKPT_NOW")
        _asked = os.path.exists(_now)
        if _DDP:
            # both ranks see the same flag file, and whichever removed it first would leave the other
            # thinking no checkpoint was asked for -- so agree on it collectively.  Unconditional every
            # step, so it can never deadlock against a rank that took a different branch.
            _at = torch.tensor([1.0 if _asked else 0.0], device=DEV)
            dist.all_reduce(_at, op=dist.ReduceOp.MAX)
            _asked = bool(_at.item() > 0)
        if _asked and (not _DDP or _RANK == 0):
            try: os.remove(_now)
            except OSError: pass
        if (step > step0 and step % a.ckpt_every == 0) or _asked:
            try:
                if _DDP:
                    # every rank saves its own sampler state; ONLY rank 0 writes the model, or the
                    # ranks race on the same .tmp file and the checkpoint is corrupt
                    torch.save(_rng_state(), ck + f".rng{_RANK}.tmp")
                    os.replace(ck + f".rng{_RANK}.tmp", ck + f".rng{_RANK}")
                if not _DDP or _RANK == 0:
                    torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                                "step": step, "cfg": vars(a), "rng": _rng_state()}, ck + ".tmp")
                    os.replace(ck + ".tmp", ck)
            except Exception as _e:      # a failed save must not kill a run that is training fine
                print(f"WARNING: checkpoint at step {step} FAILED ({_e}); continuing", flush=True)
                try: os.remove(ck + ".tmp")      # an orphan .tmp makes the next save fail sooner
                except OSError: pass

    vl, bk = evaluate(model, sh, np.random.default_rng(1), bs=a.eval_bs)
    cp, fr = recall_probe(model, sh, np.random.default_rng(2))
    if _DDP:
        torch.save(_rng_state(), ck + f".rng{_RANK}")
    if not _DDP or _RANK == 0:
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "step": total_steps, "cfg": vars(a), "rng": _rng_state()}, ck)
    print("RESULT " + json.dumps({"tag": a.tag, "dk": a.dk, "params": nparam,
          "state_per_layer": cfg.state_per_layer, "tokens": a.tokens,
          "val_loss": vl, "val_ppl": math.exp(min(vl, 20)),
          "pos_curve": [round(b, 4) for b in bk],
          "recall_copy": cp, "recall_fresh": fr}), flush=True)


if __name__ == "__main__":
    main()
