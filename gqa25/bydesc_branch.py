# -*- coding: utf-8 -*-
"""Same by-description trials, but scored PER BRANCH, to find where the ceiling comes from.
factgen picks _simple (60%), _multihop (20%) or _inference (20%). Only _simple is pure retrieval;
the others need two hops or arithmetic. If _simple is much higher, the ceiling is the reasoning
branches, not the memory. Runs either the memlm arm (MEMLM_MOD) or GPT-2 (GPT2=<name>)."""
import os, sys, torch, numpy as np, importlib, inspect
sys.path.insert(0, "."); os.environ.setdefault("MEMPROBE_ROOT", os.path.expanduser("~/memprobe"))
import factgen as F
torch.set_num_threads(int(os.environ.get("THREADS", "5")))
G = os.environ.get("GPT2", "")
if G:
    from transformers import GPT2LMHeadModel
    m = GPT2LMHeadModel.from_pretrained(G).eval(); m.requires_grad_(False)
    fwd = lambda x: m(x).logits
    tag = G
else:
    M = importlib.import_module(os.environ.get("MEMLM_MOD", "memlm_kv"))
    import cpu_shim; cpu_shim.install(M)
    st = torch.load(os.environ["CKPT"], map_location="cpu", weights_only=False)
    ok = set(inspect.signature(M.Cfg.__init__).parameters) - {"self"}
    mm = M.MemLM(M.Cfg(**{k: v for k, v in st["cfg"].items() if k in ok}))
    miss, unex = mm.load_state_dict(st["model"], strict=False)
    assert not miss and not unex, "checkpoint/model mismatch: %d missing %d unexpected" % (len(miss), len(unex))
    mm = mm.float().eval(); mm.requires_grad_(False)
    try: mm.set_moe_loop(False)
    except Exception: pass
    fwd = lambda x: mm(x); tag = "step %s" % st["step"]
print("model: %s" % tag, flush=True)

N, NF, SEED = int(os.environ.get("N","256")), 8, 20260823
agg = {}
def acc(br, key, h, t, e):
    d = agg.setdefault(br, {"sh":0,"st":0,"se":0,"ch":0,"ct":0,"ce":0,"n":0})
    d[key[0]] += h; d[key[1]] += t; d[key[2]] += e

def build(rng, pick, ts, tq, ans, stmts, repl):
    out = []
    for body in (stmts + [ts], stmts + [repl]):
        bs = list(body); rng.shuffle(bs)
        tt = F.enc.encode(" ".join(bs) + " " + tq); at = F.enc.encode(" " + ans)
        toks = np.array(tt + at, dtype=np.int64)
        mask = np.zeros(len(toks), dtype=np.float32); mask[len(tt):] = 1.0
        out.append((toks, mask))
    return out

@torch.no_grad()
def sc(toks, mask):
    lg = fwd(torch.tensor(toks[None])[:, :-1]).float()
    pred = lg[0].argmax(-1); pos = np.nonzero(mask)[0]
    h = t = 0
    for p in pos:
        if p == 0: continue
        t += 1; h += int(pred[p-1]) == int(toks[p])
    return h, t, (t > 0 and h == t)

with torch.no_grad():
    for i in range(N):
        rng = np.random.default_rng([SEED, i])
        stmts = [F._simple(rng)[0] for _ in range(NF - 1)]
        gp = rng.random()
        pick, br = (F._simple, "simple") if gp < 0.6 else ((F._multihop, "multihop") if gp < 0.8 else (F._inference, "inference"))
        ts, tq, ans, _ = pick(rng)
        repl = F._simple(rng)[0]
        (sa, sm), (ca, cm) = build(rng, pick, ts, tq, ans, stmts, repl)
        h, t, e = sc(sa, sm); acc(br, ("sh","st","se"), h, t, e)
        h, t, e = sc(ca, cm); acc(br, ("ch","ct","ce"), h, t, e)
        agg[br]["n"] += 1
        if (i+1) % 64 == 0: print("  %d/%d" % (i+1, N), flush=True)

print()
print("  %-10s %5s %10s %10s %8s %9s" % ("branch","n","tok saw","tok ctl","delta","exact"))
for br in ("simple","multihop","inference"):
    d = agg.get(br)
    if not d: continue
    ts_, tc = 100*d["sh"]/max(d["st"],1), 100*d["ch"]/max(d["ct"],1)
    print("  %-10s %5d %9.1f%% %9.1f%% %+8.1f %8.1f%%" % (br, d["n"], ts_, tc, ts_-tc, 100*d["se"]/d["n"]))
