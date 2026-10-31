import sys, numpy as np, collections
sys.path.insert(0, "/home/paul/memprobe/gqa")
import factgen as F, factgen2 as F2

r = np.random.default_rng(3)
print("=== _simple (what training samples) ===")
for _ in range(6):
    s, q, a, e = F._simple(r); print("  %-50s | %-38s -> %r" % (s[:50], q[:38], a))
print("=== _multihop (two-hop, cue entity differs from fact entity) ===")
for _ in range(4):
    s, q, a, e = F._multihop(r); print("  %-58s | %-32s -> %r" % (s[:58], q[:32], a))
print("=== _inference (untouched) ===")
for _ in range(2):
    s, q, a, e = F._inference(r); print("  %-58s | %-32s -> %r" % (s[:58], q[:32], a))

# the real training path: factgen_task splices into a token window
import factgen_task
w = np.random.randint(1000, 40000, size=1024).astype(np.int64)
cue = np.array(F.enc.encode("\nQ:"), dtype=np.int64)
ok = 0; masks = []
for i in range(60):
    rng = np.random.default_rng([99, i])
    out, m = factgen_task.factgen_task(w.copy(), 1024, rng, cue, want_mask=True)
    assert out.shape == w.shape, "window shape changed: %s vs %s" % (out.shape, w.shape)
    assert out.dtype == w.dtype, "dtype changed"
    assert m is None or m.shape == (len(w),), "mask shape %s" % (m.shape,)
    if m is not None: masks.append(float(m.sum()))
    ok += 1
print("\nfactgen_task: %d/60 windows built, shape+dtype preserved -> OK" % ok)
print("   answer-mask tokens per window: mean %.1f, min %.0f, max %.0f, zero-mask windows %d"
      % (np.mean(masks), min(masks), max(masks), sum(1 for x in masks if x == 0)))

# split + anchor behaviour through the factgen seam
combos = collections.Counter()
anch = []
for _ in range(4000):
    st, cu, v, _e, m = F2.sample(r, "train", 0.5)
    combos[(m["rel"], m["si"], m["qi"])] += 1; anch.append(m["anchored"])
ev = set()
for _ in range(2000):
    _, _, _, _, m = F2.sample(r, "eval", 0.5); ev.add((m["rel"], m["si"], m["qi"]))
print("\ntrain combos seen %d | eval combos seen %d | OVERLAP %d -> %s"
      % (len(combos), len(ev), len(set(combos) & ev), "OK" if not (set(combos) & ev) else "BROKEN"))
print("anchored fraction at anchor_p=0.5: %.3f" % np.mean(anch))
print("every training cue ends mid-sentence (no '?'): %s"
      % ("OK" if not any(F2.sample(r, "train", 0.5)[1].rstrip().endswith("?") for _ in range(2000)) else "SOME ARE QUESTIONS"))
