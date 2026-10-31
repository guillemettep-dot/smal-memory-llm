"""WHY does the main memory stop at ~128 of 192 slots?  Replays the top-1 write address token by token.
Address score of slot n:  z_n = temp * cos(q_t, K_n) - fr * o_n ; an empty slot scores exactly 0
(only the first empty one is eligible).  So an empty slot wins only if EVERY used slot has
temp*cos < fr*o.  Reports temp, fr, the margins, and how the write query q differs from the stored key k.
Self-check: the replay must pick the same slot as the model's own scan."""
import os, sys, inspect
os.environ.update(MEMPROBE_ROOT="/home/paul/memprobe", MEMLM_MOD="memlm_sh", KV_DETACH="1", KV_WQ="0", SLOT_ROPE="1", SLOT_ROPE_FRAC="0.5", CUDA_VISIBLE_DEVICES="")
sys.path.insert(0, "/home/paul/memprobe/gqa")
import numpy as np, torch
import memlm_sh as S, slotkern_kv as SK
torch.set_num_threads(12)
st = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
ok = set(inspect.signature(S.Cfg.__init__).parameters) - {"self"}
cfg = S.Cfg(**{k: v for k, v in st["cfg"].items() if k in ok})
m = S.MemLM(cfg); m.load_state_dict(st["model"], strict=False); m.float().eval()
caught = []
def spy(k, v, g, fr, N, temp, **kw):
    q = kw.get("q")
    kb, vb, a = SK._ref_scan_kv_impl(k.float(), v.float(), g.float(), fr, N, temp, 1, kw.get("hard", False), kw.get("hard_gate", False),
                                     1e-4, True, kw.get("detach_addr", False), None if q is None else q.float(), pin=kw.get("pin", True))
    caught.append(dict(k=k.detach().float(), q=None if q is None else q.detach().float(), g=g.detach().float(),
                       a=a.detach().float(), fr=float(fr), temp=float(temp), N=N, pin=kw.get("pin", True)))
    G = kw.get("G", 1)
    return kb[:, ::G], vb[:, ::G]
S.ref_scan_kv = spy
import train_sh as TS
TS.DEV = "cpu"
sh = TS.Shards(doc_aware=True)
rng = np.random.default_rng(3)
x = torch.from_numpy(np.stack([sh._window(1024, rng, True) for _ in range(2)])[:, :1024].astype(np.int64))
with torch.no_grad():
    m(x)
print("step %s | %d memories caught" % (st["step"], len(caught)))
def unit(z): return z / z.norm(dim=-1, keepdim=True).clamp_min(1e-6)
pct = lambda z, p: float(np.percentile(np.asarray(z), p))
for name, c in zip(("ENCODER (layer 2)", "MAIN (layer 6)"), caught):
    k, q, g, a, N, temp, fr = c["k"], c["q"], c["g"], c["a"], c["N"], c["temp"], c["fr"]
    B, T, H, D = k.shape
    assert q is not None and not c["pin"]
    print("\n== %s  N=%d  temp(write sharpness)=%.2f  fr(free preference)=%.3f" % (name, N, temp, fr))
    # ---- how the two keys differ
    ku, qu = unit(k), unit(q)
    same = (ku * qu).sum(-1)                                            # cos(q_t, k_t), same token
    print("  cos(write query, stored key) SAME token: median %.3f  [p10 %.3f, p90 %.3f]" % (same.median(), pct(same, 10), pct(same, 90)))
    for nm, u in (("stored key k", ku), ("write query q", qu)):
        mean = u.mean(1)                                                # (B,H,D) mean direction over the sequence
        aniso = mean.norm(dim=-1).mean()                                # 1 = all identical, ~1/sqrt(T) = isotropic
        idx = torch.randperm(T)[:256]
        pc = torch.einsum("bihd,bjhd->bhij", u[:, idx], u[:, idx])
        off = pc[:, :, ~torch.eye(256, dtype=bool)]
        print("  %-14s |mean unit vector| %.3f   pairwise cos between tokens: median %.3f  p90 %.3f  frac>0 %.2f"
              % (nm, aniso, off.median(), pct(off, 90), float((off > 0).float().mean())))
    ncos = torch.norm(k, dim=-1); nq = torch.norm(q, dim=-1)
    print("  norms: |k| median %.2f   |q| median %.2f" % (ncos.median(), nq.median()))
    # ---- replay the address decision
    K = torch.zeros(B, H, N, D); o = torch.zeros(B, H, N)
    agree = tot = 0; rec = []
    for t in range(T):
        kn = unit(K); qn = unit(q[:, t])
        cos = torch.einsum("bhnd,bhd->bhn", kn, qn)
        z = cos * temp - fr * o
        elig = SK._first_empty_mask(o, 1e-4)
        z = z.masked_fill(~elig, -1e30)
        r = z.argmax(-1)                                                # (B,H)
        gt = g[:, t]
        wr = gt > 0
        mine = a[:, t].argmax(-1)
        agree += int(((r == mine) | ~wr).sum()); tot += B * H
        used = o >= 1e-4
        zu = (cos * temp - fr * o).masked_fill(~used, -1e30)
        bu = zu.argmax(-1)
        nused = used.sum(-1)
        for b in range(B):
            for h in range(H):
                if not wr[b, h] or nused[b, h] == 0: continue
                j = int(bu[b, h])
                rec.append((t, int(nused[b, h]), float(zu[b, h, j]), float(cos[b, h, j]), float(o[b, h, j]),
                            int(r[b, h]) == j and bool(used[b, h, j]), float(cos[b, h][used[b, h]].max())))
        w = (gt.unsqueeze(-1) * torch.nn.functional.one_hot(r, N).float())
        K = (1 - w.unsqueeze(-1)) * K + w.unsqueeze(-1) * k[:, t].unsqueeze(2)
        o = (1 - w) * o + w
    print("  replay self-check: same slot as the model's scan on %.4f of decisions" % (agree / tot))
    R = np.array(rec)
    t_, nu, zb, cb, ob, over, cmax = R.T
    print("  writes analysed: %d  | went to a USED slot: %.1f%%" % (len(R), 100 * over.mean()))
    print("  best used slot: score z median %.2f  (empty slot scores 0)  | its cos median %.3f | its occupancy o median %.3f"
          % (np.median(zb), np.median(cb), np.median(ob)))
    print("  penalty for being used, fr*o: median %.2f  vs  temp*cos of best match: median %.2f" % (np.median(fr * ob), np.median(temp * cb)))
    print("  cos threshold an empty slot needs to win: every used cos < fr*o/temp  ~ %.3f at o=1" % (fr / temp))
    print("  max cos to any used slot, by how many slots are used:")
    for lo, hi in ((1, 16), (16, 64), (64, 96), (96, 128), (128, 160), (160, 193)):
        s = (nu >= lo) & (nu < hi)
        if s.sum() < 20: continue
        print("    used %3d-%3d: n=%6d  max-cos median %.3f  p10 %.3f | empty won %.1f%%"
              % (lo, hi - 1, s.sum(), np.median(cmax[s]), pct(cmax[s], 10), 100 * (1 - over[s].mean())))
    print("  final slots used per head: median %d of %d" % (int(np.median((o >= 1e-4).sum(-1).numpy())), N))
    fin = (o >= 1e-4).sum(-1)                                            # (B,H)
    wn = (g > 0).float().sum(1)                                          # writes per (B,H)
    qa = unit(q).mean(1).norm(dim=-1); ka = unit(k).mean(1).norm(dim=-1)
    same_h = (unit(k) * unit(q)).sum(-1).median(1).values                # (B,H)
    print("  per head (seq0 / seq1): slots used | writes | write-query clustering |mean q| | key clustering |mean k| | cos(q,k) same token")
    for h in range(H):
        print("    head %2d: used %3d / %3d | writes %4d / %4d | q %.2f / %.2f | k %.2f / %.2f | qk %.2f / %.2f"
              % (h, fin[0, h], fin[1, h], wn[0, h], wn[1, h], qa[0, h], qa[1, h], ka[0, h], ka[1, h], same_h[0, h], same_h[1, h]))
    print("  mean slots used over heads x seqs: %.1f" % fin.float().mean())
