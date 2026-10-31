import sys, os, numpy as np
sys.path.insert(0, "/home/paul/memprobe/gqa")
import task_lib, factgen_task, factgen
enc = factgen.enc
rec = []
def spl(orig):
    def f(w, facts, queries, gap, cue, qmask=None):
        out = orig(w, facts, queries, gap, cue, qmask)
        if out[0] is not None: rec.append((gap, len(facts), len(queries)))
        return out
    return f
task_lib._splice = spl(task_lib._splice); factgen_task._splice = task_lib._splice
names = ["keyed_span","updated","tagged","list","multihop","reverse","pred","factgen"]
wts = [2,1,1,1,1,1,1,4]; cue = np.array(enc.encode("\nQ:"), dtype=np.int64); T = 1024
oct_edges = [32, 64, 128, 256, 512, 1100]
psingle, smax = float(os.environ["PS"]), int(os.environ["SM"])
factgen_task.P_SINGLE = psingle; factgen_task.S_MAX = smax; task_lib.P_TOP = float(os.environ.get("PT", "0"))
rec.clear(); ans = []
for i in range(1500):
    rng = np.random.default_rng([5, i])
    w = np.random.default_rng(i).integers(1000, 40000, size=T + 1).astype(np.int64)
    out, m = task_lib.compose(w, T, rng, cue, names, wts, want_mask=True); ans.append(float(m.sum()))
r = np.random.default_rng(0); d = []
for g, lf, lq in rec:
    for _ in range(8): d.append(g + r.integers(0, max(lf, 1)) + r.integers(0, max(lq, 1)))
d = np.array(d); dd = d[(d >= 32) & (d < 1100)]
h, _ = np.histogram(dd, bins=oct_edges); sh = 100 * h / max(len(d), 1)
dev = np.max(np.abs(100 * h / max(len(dd), 1) - 20))
print("P_TOP %.2f P_SINGLE %.2f S_MAX %d | <32 %4.1f%% | %s | worst octave off-flat %4.1f pts | ans/win %.1f"
      % (task_lib.P_TOP, psingle, smax, 100*np.mean(d < 32), " ".join("%5.1f" % x for x in sh), dev, np.mean(ans)))
