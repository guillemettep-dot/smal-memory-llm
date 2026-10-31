import sys, numpy as np
sys.path.insert(0, "/home/paul/memprobe/gqa")
import factgen as F, factgen2 as F2, factgen_task, task_lib
r = np.random.default_rng(123)
fails = []
# 1. anchor knob binds on mid-sentence cues
for p in (0.0, 0.5, 1.0):
    ms = [m for m in (F2.sample(r, "train", p, q_frac=0.25)[4] for _ in range(2000)) if not m["question"]]
    got = np.mean([m["anchored"] for m in ms])
    print("anchor_p %.1f -> %.3f" % (p, got)); abs(got - p) < 0.05 or fails.append("anchor %.1f" % p)
# 2. held-out split still disjoint
tr = {(m["rel"], m["si"], m["qi"]) for m in (F2.sample(r, "train", .5)[4] for _ in range(4000)) if not m["novel"]}
ev = {(m["rel"], m["si"], m["qi"]) for m in (F2.sample(r, "eval", .5)[4] for _ in range(4000)) if not m["novel"]}
print("curated split overlap: %d" % len(tr & ev)); (not (tr & ev)) or fails.append("split")
# 3. the older eval scripts' seam still works
s, q, a, e = F._simple(r); print("_simple ok:", bool(s and q and a))
s, q, a, e = F._multihop(r); print("_multihop ok:", bool(s and q and a))
# 4. every task still builds windows of the right shape at every stage length
names = ["keyed_span","updated","tagged","list","multihop","reverse","pred","factgen"]; wts = [2,1,1,1,1,1,1,4]
cue = np.array(F.enc.encode("\nQ:"), dtype=np.int64)
for T in (128, 256, 512, 1024):
    bad = 0; withans = 0
    for i in range(400):
        rng = np.random.default_rng([T, i]); w = np.random.default_rng(i).integers(1000, 40000, size=T + 1).astype(np.int64)
        out, m = task_lib.compose(w, T, rng, cue, names, wts, want_mask=True)
        bad += int(out.shape != w.shape or m.shape != w.shape); withans += int(m.sum() > 0)
    print("T=%4d  shape errors %d  windows with answers %d/400" % (T, bad, withans)); bad == 0 or fails.append("shape T=%d" % T)
print("\nALL CHECKS PASS" if not fails else "FAILURES: %s" % fails)
