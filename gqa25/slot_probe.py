"""How does sh7 (top-1 writes + gate floor) USE its slots?  Real held-out text, T=1024, CPU.
Per memory: tokens that write at all (gate > 0 after the floor), strong writes, distinct slots used,
writes landing on an EMPTY slot vs OVERWRITING a used one, and how long a written token survives
before the next write to the same slot (its lifetime) -- what a two-token answer needs is BOTH tokens alive."""
import os, sys, inspect
os.environ.update(MEMPROBE_ROOT="/home/paul/memprobe", MEMLM_MOD="memlm_sh", KV_DETACH="1", KV_WQ="0", SLOT_ROPE="1", SLOT_ROPE_FRAC="0.5", CUDA_VISIBLE_DEVICES="")
sys.path.insert(0, "/home/paul/memprobe/gqa")
import numpy as np, torch
import memlm_sh as S, slotkern_kv as SK
torch.set_num_threads(12)
st = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
ok = set(inspect.signature(S.Cfg.__init__).parameters) - {"self"}
cfg = S.Cfg(**{k: v for k, v in st["cfg"].items() if k in ok})
print("config: slot_hard=%s gate_floor=%s step=%s" % (cfg.slot_hard, getattr(cfg, "sh_gate_floor", None), st["step"]))
m = S.MemLM(cfg); m.load_state_dict(st["model"], strict=False); m.float().eval()
caught = []
def spy(k, v, g, fr, N, temp, **kw):
    kb, vb, a = SK._ref_scan_kv_impl(k.float(), v.float(), g.float(), fr, N, temp, 1, kw.get("hard", False), kw.get("hard_gate", False),
                                     1e-4, True, kw.get("detach_addr", False), (None if kw.get("q") is None else kw.get("q").float()), pin=kw.get("pin", True))
    caught.append(dict(g=g.detach().float(), a=a.detach().float(), N=N))
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
for name, c in zip(("encoder memory (layer 2)", "main memory (layer 6)"), caught):
    g, a, N = c["g"], c["a"], c["N"]                  # g (B,T,H) after the floor; a (B,T,H,N): one-hot under top-1
    B, T, H = g.shape
    row = a.argmax(-1)                                # the slot each token writes into
    writes = g > 0
    print("== %s, %d slots/head, %d tokens x %d heads x %d seqs" % (name, N, T, H, B))
    print("   tokens that write at all (gate>0 after floor): %.1f%% | strong (>0.5): %.1f%% | mean gate of writers %.3f"
          % (100 * writes.float().mean(), 100 * (g > 0.5).float().mean(), g[writes].mean() if writes.any() else 0))
    lifes, empty_hits, over_hits, distinct, per_slot = [], 0, 0, [], []
    for b in range(B):
        for h in range(H):
            last = {}                                  # slot -> time of its current occupant
            cnt = np.zeros(N, int)
            for t in range(T):
                if not writes[b, t, h]: continue
                r = int(row[b, t, h])
                cnt[r] += 1
                if r in last:
                    over_hits += 1; lifes.append(t - last[r])
                else:
                    empty_hits += 1
                last[r] = t
            for r, t0 in last.items():                # still alive at the end: censored lifetime
                lifes.append(-(T - t0))
            distinct.append(len(last)); per_slot.append(cnt[cnt > 0].mean() if (cnt > 0).any() else 0)
    L = np.array(lifes); done = L[L > 0]; cens = -L[L < 0]
    nw = empty_hits + over_hits
    print("   writes per head per sequence: %.0f | slots ever used: %.1f of %d | writes per used slot: %.1f"
          % (nw / (B * H), np.mean(distinct), N, np.mean(per_slot)))
    print("   writes landing in an EMPTY slot: %.1f%% | OVERWRITING a used slot: %.1f%%" % (100 * empty_hits / max(nw, 1), 100 * over_hits / max(nw, 1)))
    if len(done):
        print("   lifetime of a stored token before it is overwritten: median %d tokens | 25%% die within %d | 75%% within %d"
              % (np.median(done), np.percentile(done, 25), np.percentile(done, 75)))
        print("   overwritten within 64 tokens: %.1f%% of writes | within 256: %.1f%% | still alive at token 1024: %.1f%%"
              % (100 * (done <= 64).sum() / len(L), 100 * (done <= 256).sum() / len(L), 100 * len(cens) / len(L)))
