import sys, os, numpy as np, collections
sys.path.insert(0, os.environ["SRC"])
import task_lib, factgen_task, factgen
enc = factgen.enc
rec = []; used_S = collections.Counter()
def spl(orig):
    def f(w, facts, queries, gap, cue, qmask=None):
        out = orig(w, facts, queries, gap, cue, qmask)
        if out[0] is not None: rec.append((gap, len(facts), len(queries)))
        return out
    return f
task_lib._splice = spl(task_lib._splice); factgen_task._splice = task_lib._splice
if hasattr(factgen_task, "_segmented"):
    _seg = factgen_task._segmented
    def segw(w, items, hard, rng, q_frac, S):
        r = _seg(w, items, hard, rng, q_frac, S)
        if r is not None: used_S[S] += 1
        return r
    factgen_task._segmented = segw
names = ["keyed_span","updated","tagged","list","multihop","reverse","pred","factgen"]
wts   = [2,1,1,1,1,1,1,4]
cue = np.array(enc.encode("\nQ:"), dtype=np.int64)
T = 1024; ans = []; built = 0; N = 2500
fg_ans = []
for i in range(N):
    rng = np.random.default_rng([5, i])
    w = np.random.default_rng(i).integers(1000, 40000, size=T + 1).astype(np.int64)
    out, m = task_lib.compose(w, T, rng, cue, names, wts, want_mask=True)
    assert out.shape == w.shape and m.shape == w.shape
    if m.sum() > 0: built += 1
    ans.append(float(m.sum()))
for i in range(800):                                   # factgen alone, for its own answer budget
    rng = np.random.default_rng([9, i])
    w = np.random.default_rng(10**6 + i).integers(1000, 40000, size=T + 1).astype(np.int64)
    o, m = factgen_task.factgen_task(w, T, rng, cue, want_mask=True)
    fg_ans.append(0.0 if m is None else float(m.sum()))
r = np.random.default_rng(0); d = []
for g, lf, lq in rec:
    for _ in range(8): d.append(g + r.integers(0, max(lf, 1)) + r.integers(0, max(lq, 1)))
d = np.array(d)
edges = [0, 32, 64, 128, 256, 512, 768, 1100]
h, _ = np.histogram(d, bins=edges)
print("[%s]" % os.environ["TAG"])
print("  windows with answers %d/%d | answer tokens/window mean %.1f | factgen alone %.1f"
      % (built, N, np.mean(ans), np.mean(fg_ans)))
print("  <=64: %.1f%%   <=128: %.1f%%   <=256: %.1f%%   median distance %d"
      % (100*np.mean(d <= 64), 100*np.mean(d <= 128), 100*np.mean(d <= 256), np.median(d)))
for a, b, c in zip(edges[:-1], edges[1:], h):
    print("     %4d-%-5d %5.1f%%  %s" % (a, b, 100*c/len(d), "#" * int(60*c/len(d))))
if used_S: print("  factgen segments used: " + "  ".join("S=%d:%d" % (k, v) for k, v in sorted(used_S.items())))
