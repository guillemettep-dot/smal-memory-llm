"""Do the reading layers read the SAME thing for the same token, and does each one find the fact?
CPU, model run exactly as the battery runs it (ref_scan_kv, pin=False).  usage: layerread.py CKPT [NTRIAL=48] [GAP=1024]

A  plain held-out text: per reading layer, read sharpness; per layer pair, how much their read distributions
   over slots overlap for the SAME token (Bhattacharyya sum sqrt(p q)), against a baseline of two DIFFERENT
   tokens of the same 8-token block (same memory snapshot, so any agreement there is the memory, not the token).
B  battery `code` trials: the share of each slot's content that came from the fact's code-digit tokens is
   tracked exactly through the soft writes (F = (1-w)F + w*isfact).  At the answer-predicting positions, the
   share of what each layer READ that came from the fact digits.  CONTROL: the same in the control trial,
   whose fact line names a DIFFERENT person -- reading it there is reading the wrong fact.
C  per layer: the battery answer with ONLY that layer reading the CONTROL trial's memory (identical layout, the
   other person's fact), every other layer reading its own -- which layers' reads the answer depends on."""
import os, sys, inspect, time
os.environ.update(MEMPROBE_ROOT="/home/paul/memprobe", MEMLM_MOD="memlm_sh", KV_DETACH="1", KV_WQ="0", SLOT_ROPE="1", SLOT_ROPE_FRAC="0.5", CUDA_VISIBLE_DEVICES="")
sys.path.insert(0, "/home/paul/memprobe/gqa")
import numpy as np, torch, torch.nn.functional as F
import memlm_sh as S, slotkern_kv as SK
torch.set_num_threads(int(os.environ.get("THREADS", "12")))
NTRIAL = int(sys.argv[2]) if len(sys.argv) > 2 else 48
GAP = int(sys.argv[3]) if len(sys.argv) > 3 else 1024

st = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
ok = set(inspect.signature(S.Cfg.__init__).parameters) - {"self"}
cfg = S.Cfg(**{k: v for k, v in st["cfg"].items() if k in ok})
m = S.MemLM(cfg)
miss, unexp = m.load_state_dict(st["model"], strict=False)
assert not miss and not unexp, (miss, unexp)
m.float().eval(); m.requires_grad_(False)
STEP = st["step"]; del st
for i, b in enumerate(m.blocks):
    b.mix._li = i
READERS = {mid: [i for i, b in enumerate(m.blocks) if b.mix.mid == mid] for mid in ("enc", "main")}
G = cfg.sh_stride
print("ckpt step %s | readers %s | stride %d" % (STEP, READERS, G), flush=True)

# ---- spies -------------------------------------------------------------------------------------------
CAUGHT = []                       # per forward: writer calls in order (enc first, then main)
def spy(k, v, g, fr, N, temp, **kw):
    kw = dict(kw); kw["return_w"] = True
    kb, vb, a = SK.ref_scan_kv(k, v, g, fr, N, temp, **kw)
    CAUGHT.append(dict(g=g.detach().float(), a=a.detach().float(), N=N))
    return kb, vb
S.ref_scan_kv = spy

ATT = {}                          # layer -> (B,T,Hs,NQ,N) read distribution
RECORD = {}                       # mid -> memory tuple as read (for C)
OVERRIDE = {}                     # layer -> memory tuple to read instead of its own
_orig_read = S.SharedMixer._read
def _att(self, kbuf, kinv, valid, x):
    """the read attention, line for line as SharedMixer._read computes it"""
    B, T, _ = x.shape
    Hs, NQ, dk, Gs = self.Hs, self.NQ, self.dk, self.G
    NB = kbuf.shape[1]; Tp = NB * Gs
    q = self.tk_q(x).reshape(B, T, Hs, NQ, dk)
    q = q / q.float().norm(dim=-1, keepdim=True).clamp_min(1e-6).to(q.dtype)
    if Tp != T:
        q = F.pad(q, (0, 0, 0, 0, 0, 0, 0, Tp - T))
    q = q.reshape(B, NB, Gs, Hs, NQ, dk).permute(0, 1, 3, 2, 4, 5).reshape(B, NB, Hs, Gs * NQ, dk)
    sc = torch.einsum("bjhmd,bjhnd->bjhmn", q, kbuf).float() * (kinv.unsqueeze(-2) * torch.exp(self.log_rsel.float()))
    return torch.nan_to_num(torch.softmax(sc.masked_fill(~valid.unsqueeze(-2), float("-inf")), -1))
MAXERR = [0.0]
def _read(self, kbuf, buf, kinv, valid, x):
    if self._li in OVERRIDE:
        kbuf, buf, kinv, valid = OVERRIDE[self._li]
    RECORD.setdefault(self.mid, (kbuf, buf, kinv, valid))
    rd = _orig_read(self, kbuf, buf, kinv, valid, x)
    att = _att(self, kbuf, kinv, valid, x)
    B, T, _ = x.shape; NB = kbuf.shape[1]
    # self-check: the captured attention must reproduce the model's own read
    rd2 = torch.einsum("bjhmn,bjhnv->bjhmv", att.to(buf.dtype), buf)
    rd2 = rd2.reshape(B, NB, self.Hs, self.G, self.NQ, self.dv).permute(0, 1, 3, 2, 4, 5).reshape(B, NB * self.G, self.Hs, self.NQ, self.dv)[:, :T]
    MAXERR[0] = max(MAXERR[0], (rd2 - rd).abs().max().item())
    ATT[self._li] = att.reshape(B, NB, self.Hs, self.G, self.NQ, -1).permute(0, 1, 3, 2, 4, 5).reshape(B, NB * self.G, self.Hs, self.NQ, -1)[:, :T]
    return rd
S.SharedMixer._read = _read

def run(x):
    CAUGHT.clear(); ATT.clear(); RECORD.clear()
    with torch.no_grad():
        lg = m(x)
    assert [c["N"] for c in CAUGHT] == [cfg.sh_enc_slots, cfg.sh_slots], [c["N"] for c in CAUGHT]
    return lg

def bhat(p, q):                   # (...,N) x (...,N) -> (...)
    return (p.clamp_min(0) * q.clamp_min(0)).sqrt().sum(-1)

# ---- A: plain text ------------------------------------------------------------------------------------
import train_sh as TS
TS.DEV = "cpu"
sh = TS.Shards(doc_aware=True)
rng = np.random.default_rng(20261009)
W = np.stack([sh._window(1024, rng, True) for _ in range(8)]).astype(np.int64)
X = torch.from_numpy(W[:, :1024])
assert not (X == TS.EOT).any()
P = {}
for i in range(0, 8, 4):
    run(X[i:i + 4])
    for L, a in ATT.items():
        P.setdefault(L, []).append(a.mean(3))          # mean over the NQ queries: (B,T,Hs,N)
P = {L: torch.cat(v) for L, v in P.items()}
print("selfcheck A: captured attention reproduces the model's read, max |d rd| = %.1e  %s" % (MAXERR[0], "OK" if MAXERR[0] < 1e-3 else "FAIL"), flush=True)
assert MAXERR[0] < 1e-3
t0 = 64                                               # past the window and the empty-memory start
Tn = P[READERS["main"][0]].shape[1]
# baseline partner: the token G/2 positions away inside the same block (same memory snapshot)
idx = torch.arange(Tn); blk = idx // G; partner = blk * G + (idx % G + G // 2) % G
partner = partner.clamp_max(Tn - 1)
for mid in ("main", "enc"):
    Ls = READERS[mid]
    print("\n== A  %s memory, plain text, tokens %d-1024, 8 seqs" % (mid, t0))
    print("   layer  top weight  eff. slots read   (valid slots ~%d)" % int((RECORD[mid][3][:, -1].float().sum(-1)).mean()))
    for L in Ls:
        p = P[L][:, t0:]
        ent = -(p.clamp_min(1e-12) * p.clamp_min(1e-12).log()).sum(-1)
        print("   %5d  %10.3f  %15.1f" % (L, p.max(-1).values.mean(), ent.exp().mean()))
    print("   overlap of two layers' reads, same token  [same layer, different token of the same block = baseline]")
    print("        " + "".join("%8d" % L for L in Ls))
    for La in Ls:
        row = []
        for Lb in Ls:
            pa = P[La][:, t0:]; pb = P[Lb][:, t0:] if La != Lb else P[Lb][:, partner][:, t0:]
            row.append(bhat(pa, pb).mean().item())
        print("   %5d" % La + "".join(("%8.3f" if La != Lb else "   [%.3f]") % v for Lb, v in zip(Ls, row)))
    # cross-layer, different token baseline (layer a token t vs layer b partner token)
    base = np.mean([bhat(P[La][:, t0:], P[Lb][:, partner][:, t0:]).mean().item() for La in Ls for Lb in Ls if La != Lb])
    same = np.mean([bhat(P[La][:, t0:], P[Lb][:, t0:]).mean().item() for La in Ls for Lb in Ls if La != Lb])
    top = np.mean([(P[La][:, t0:].argmax(-1) == P[Lb][:, t0:].argmax(-1)).float().mean().item() for La in Ls for Lb in Ls if La != Lb])
    print("   mean over layer pairs: same token %.3f | different token, same block %.3f | same top-1 slot %.1f%%" % (same, base, 100 * top))

# ---- B / C: battery code trials ---------------------------------------------------------------------------
src = open("/home/paul/memprobe/gqa/battery3.py").read().split("# ---- model")[0]
os.environ["TASKS"] = "code"
BT = {}
exec(compile(src, "battery3_head", "exec"), BT)
trial, filler, E = BT["trial"], BT["filler"], BT["E"]

def fact_share(c, T, isfact):
    """slot content share from fact tokens, snapshotted where the readers read it (t % G == 0)"""
    g, a = c["g"][0], c["a"][0]                      # (T,H), (T,H,N); a one-hot in the forward
    H, N = a.shape[1], a.shape[2]
    Fm = torch.zeros(H, N); O = torch.zeros(H, N); snaps = []
    for t in range(T):
        if t % G == 0:
            snaps.append(torch.where(O > 0, Fm / O.clamp_min(1e-12), torch.zeros_like(O)))
        w = g[t].unsqueeze(-1) * a[t]
        Fm = (1 - w) * Fm + w * float(isfact[t]); O = (1 - w) * O + w
    return torch.stack(snaps)                         # (NB,H,N)

def answer_positions(lc, k):                          # logits at these positions predict the k answer tokens
    return list(range(lc - 1, lc - 1 + k))

stats = {mid: {L: dict(saw=[], ctl=[]) for L in READERS[mid]} for mid in ("main", "enc")}
pairsame = {mid: [] for mid in ("main", "enc")}
abl = {L: [0, 0] for L in READERS["main"] + READERS["enc"]}; abl_all = {"own": [0, 0], "ctlmem_all_main": [0, 0]}; nC = 0
t_start = time.time()
for t in range(NTRIAL):
    f, of, cue, ans, r = trial(t, "code")
    mid_ = filler(r, GAP)
    # the code digits inside the fact line: tokens after the prefix, before the closing ".\n"
    n_name = BT["NAMES"]; tx = BT["enc"].decode(f.tolist())
    pre = tx[:tx.rindex(" is ") + 3]
    npre = len(E(pre)); ndig = len(ans) - 1        # ans = " 1234." -> digits = all but the final "."
    assert BT["enc"].decode(f[npre:npre + ndig].tolist()) == BT["enc"].decode(ans[:-1].tolist()), (tx, ans)
    out = {}
    for which, fl in (("saw", f), ("ctl", of)):
        seq = np.concatenate([fl, mid_, cue, ans])[:-1]
        T = len(seq)
        lg = run(torch.tensor(seq[None]))
        isfact = np.zeros(T, bool); isfact[npre:npre + ndig] = True    # same span in both (same template)
        if which == "ctl":
            # control's fact line has a different name; locate its digits the same way
            txc = BT["enc"].decode(of.tolist()); prec = txc[:txc.rindex(" is ") + 3]
            npc = len(E(prec)); tail = len(f) - npre - ndig            # ".\n" tokens, same template
            ndc = len(of) - npc - tail
            dig = BT["enc"].decode(of[npc:npc + ndc].tolist()).strip()
            assert dig.isdigit() and len(dig) == 4, (txc, dig)
            isfact[:] = False; isfact[npc:npc + ndc] = True
        lc = len(fl) + len(mid_) + len(cue)
        pos = answer_positions(lc, len(ans))
        g_ok = lg[0, pos].argmax(-1).tolist() == ans.tolist()
        shares = {"enc": fact_share(CAUGHT[0], T, isfact), "main": fact_share(CAUGHT[1], T, isfact)}
        for mid in ("main", "enc"):
            for L in READERS[mid]:
                a = ATT[L][0, pos].mean(2)              # (k,Hs,N)
                sh_ = shares[mid][[p // G for p in pos]] # (k,Hs,N)
                stats[mid][L][which].append((a * sh_).sum(-1).mean().item())
            if which == "saw":
                Ls = READERS[mid]
                pairsame[mid].append(np.mean([bhat(ATT[La][0, pos].mean(2), ATT[Lb][0, pos].mean(2)).mean().item() for La in Ls for Lb in Ls if La < Lb]))
        out[which] = (seq, lc, pos, g_ok, {k: v for k, v in RECORD.items()})
    # C: only one layer reads the control trial's memory (needs identical layouts)
    if len(f) == len(of):
        nC += 1
        seq, lc, pos, g_ok, _ = out["saw"]
        abl_all["own"][0] += int(g_ok)
        ctlmem = out["ctl"][4]
        def acc(over):
            OVERRIDE.clear(); OVERRIDE.update(over)
            lg = run(torch.tensor(seq[None])); OVERRIDE.clear()
            g = lg[0, pos].argmax(-1).tolist()
            return int(g == ans.tolist()), float(np.mean([a_ == b_ for a_, b_ in zip(g, ans.tolist())]))
        for L in abl:
            e, tk = acc({L: ctlmem[m.blocks[L].mix.mid]}); abl[L][0] += e; abl[L][1] += tk
        e, tk = acc({L: ctlmem["main"] for L in READERS["main"]}); abl_all["ctlmem_all_main"][0] += e; abl_all["ctlmem_all_main"][1] += tk
    if t % 8 == 7:
        print("  trial %d/%d  %.0fs" % (t + 1, NTRIAL, time.time() - t_start), flush=True)

print("selfcheck B: max |d rd| over all runs = %.1e  %s" % (MAXERR[0], "OK" if MAXERR[0] < 1e-3 else "FAIL"))
for mid in ("main", "enc"):
    print("\n== B  %s memory, battery code, gap %d, %d trials: share of each layer's read that came from the fact's digits" % (mid, GAP, NTRIAL))
    print("   layer   right fact (saw)   wrong person's fact (control)")
    for L in READERS[mid]:
        s, c = stats[mid][L]["saw"], stats[mid][L]["ctl"]
        print("   %5d   %15.3f   %15.3f" % (L, np.mean(s), np.mean(c)))
    print("   overlap between layers' reads at the answer (saw): %.3f" % np.mean(pairsame[mid]))
print("\n== C  answer exact match with ONE layer reading the control trial's memory (%d trials with equal layout)" % nC)
print("   all layers own memory: %d/%d" % (abl_all["own"][0], nC))
for L in abl:
    print("   layer %2d (%s) swapped: exact %d/%d  tok acc %.1f%%" % (L, m.blocks[L].mix.mid, abl[L][0], nC, 100 * abl[L][1] / max(nC, 1)))
print("   all main readers swapped: exact %d/%d  tok acc %.1f%%" % (abl_all["ctlmem_all_main"][0], nC, 100 * abl_all["ctlmem_all_main"][1] / max(nC, 1)))
