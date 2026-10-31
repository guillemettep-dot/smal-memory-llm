import numpy as np, task_lib, factgen_task, factgen
enc = factgen.enc
ORIG_GAP = task_lib._gap
def gap_loguni(rng, lo, hi):
    if hi <= lo: return lo
    return int(np.clip(round(np.exp(rng.uniform(np.log(lo), np.log(hi + 1)))), lo, hi))

rec = []          # (task, gap, len_facts, len_queries)
CUR = {"task": None}
def spl(orig):
    def f(w, facts, queries, gap, cue, qmask=None):
        out = orig(w, facts, queries, gap, cue, qmask)
        if out[0] is not None: rec.append((CUR["task"], gap, len(facts), len(queries)))
        return out
    return f
task_lib._splice = spl(task_lib._splice); factgen_task._splice = task_lib._splice

ORIG_FT = factgen_task.factgen_task
def make_ft(kmode):
    src = ORIG_FT
    def ft(w, T, rng, cue, want_mask=False, tok_per_fact=40, q_frac=0.5):
        if kmode == "full":
            return src(w, T, rng, cue, want_mask, tok_per_fact, q_frac)
        n = len(w); kmax = int(np.clip(n // tok_per_fact, 3, 24))
        if kmode == "mix" and rng.random() < 0.5:
            k = kmax
        else:
            k = int(np.clip(round(np.exp(rng.uniform(np.log(2), np.log(kmax + 1)))), 2, kmax))
        return src(w, T, rng, cue, want_mask, tok_per_fact=max(1, n // k), q_frac=q_frac)
    return ft

names = ["keyed_span","updated","tagged","list","multihop","reverse","pred","factgen"]
wts   = [2,1,1,1,1,1,1,4]
cue = np.array(enc.encode("\nQ:"), dtype=np.int64)
T = 1024
def run(label, gapfn, kmode):
    rec.clear()
    task_lib._gap = gapfn; factgen_task._gap = gapfn
    task_lib.GENERATORS["factgen"] = make_ft(kmode)
    ans = {"factgen": [], "other": []}
    for i in range(2500):
        rng = np.random.default_rng([5, i])
        w = np.random.default_rng(i).integers(1000, 40000, size=T + 1).astype(np.int64)
        nm = rng.choice(names, p=np.array(wts) / sum(wts))
        CUR["task"] = nm
        out, m = task_lib.GENERATORS[nm](w, T, rng, cue, want_mask=True)
        if m is not None: ans["factgen" if nm == "factgen" else "other"].append(float(m.sum()))
    r = np.random.default_rng(0); d = {"factgen": [], "other": []}
    for t, g, lf, lq in rec:
        key = "factgen" if t == "factgen" else "other"
        for _ in range(8): d[key].append(g + r.integers(0, max(lf, 1)) + r.integers(0, max(lq, 1)))
    allv = np.array(d["factgen"] + d["other"]); fg = np.array(d["factgen"])
    edges = [16, 32, 64, 128, 256, 512, 1100]
    h, _ = np.histogram(allv, bins=edges)
    print("%-38s <=64: all %5.1f%%  factgen %5.1f%% | <=256: all %5.1f%%  factgen %5.1f%% | ans tok/window factgen %5.1f other %4.1f"
          % (label, 100*np.mean(allv <= 64), 100*np.mean(fg <= 64), 100*np.mean(allv <= 256), 100*np.mean(fg <= 256),
             np.mean(ans["factgen"]), np.mean(ans["other"])))
    print("   per-octave share:  " + "  ".join("%d-%d %4.1f%%" % (a, b, 100*c/len(allv)) for a, b, c in zip(edges[:-1], edges[1:], h)))

run("V0 current (75% upper half, K=full)", ORIG_GAP, "full")
run("V1 log-uniform gap, K=full",           gap_loguni, "full")
run("V2 log-uniform gap + log-uniform K",   gap_loguni, "loguni")
run("V3 log-uniform gap + K 50/50 mix",     gap_loguni, "mix")

def make_ft_q(kmode):
    src = ORIG_FT
    def ft(w, T, rng, cue, want_mask=False, tok_per_fact=40, q_frac=0.5):
        n = len(w); kmax = int(np.clip(n // tok_per_fact, 3, 24))
        k = int(np.clip(round(np.exp(rng.uniform(np.log(2), np.log(kmax + 1)))), 2, kmax))
        qf = 1.0 if k <= kmax // 2 else 0.5          # ask every fact when the block is short
        return src(w, T, rng, cue, want_mask, tok_per_fact=max(1, n // k), q_frac=qf)
    return ft
print()
_old = make_ft
make_ft = lambda kmode: make_ft_q(kmode)
run("V4 log-uni gap + log-uni K + ask-all", gap_loguni, "loguni")
