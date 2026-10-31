import sys, numpy as np, collections
sys.path.insert(0, "/home/paul/memprobe/gqa")
import factgen2 as F

r = np.random.default_rng(7)
print("=== 12 TRAIN samples (statement | cue -> answer) ===")
for _ in range(12):
    s, c, a, e, m = F.sample(r, "train", anchor_p=0.5)
    print("  %-52s | %-44s -> %-8r %s" % (s[:52], c[:44], a, "ANCHORED" if m["anchored"] else "reworded"))

print("\n=== 6 EVAL samples (frames the model is never trained on) ===")
for _ in range(6):
    s, c, a, e, m = F.sample(r, "eval", anchor_p=0.5)
    print("  %-52s | %-44s -> %-8r %s" % (s[:52], c[:44], a, "ANCHORED" if m["anchored"] else "reworded"))

# --- 1. split really disjoint?
tr = set(); ev = set()
for _ in range(6000):
    _, _, _, _, m = F.sample(r, "train", 0.5); tr.add((m["rel"], m["si"], m["qi"]))
for _ in range(6000):
    _, _, _, _, m = F.sample(r, "eval", 0.5);  ev.add((m["rel"], m["si"], m["qi"]))
print("\nsplit: %d train combos, %d eval combos, OVERLAP %d  ->  %s"
      % (len(tr), len(ev), len(tr & ev), "OK" if not (tr & ev) else "BROKEN"))

# --- 2. anchor_p honoured?
print("\nanchor rate actually delivered:")
for p in (0.0, 0.25, 0.5, 0.75, 1.0):
    got = np.mean([F.sample(r, "train", p)[4]["anchored"] for _ in range(1500)])
    print("   asked %.2f -> got %.3f  %s" % (p, got, "OK" if abs(got - p) < 0.12 else "OFF"))

# --- 3. surface diversity: how memorisable is the cue set?
cues = collections.Counter(F.sample(r, "train", 0.5)[1] for _ in range(8000))
print("\ndiversity: %d distinct cue strings in 8000 draws; most common appears %d times (%.2f%%)"
      % (len(cues), cues.most_common(1)[0][1], 100 * cues.most_common(1)[0][1] / 8000))
print("   (the old generator had 16 fixed cue strings in total)")

# --- 4. answer must be the natural continuation of the cue
bad = [(s, c, a) for s, c, a, _, _ in (F.sample(r, "train", 0.5) for _ in range(3000)) if a not in s]
print("\nanswer present in its own statement: %d/3000 failures -> %s" % (len(bad), "OK" if not bad else "BROKEN"))
for s, c, a in bad[:3]: print("   ", repr(s), "|", repr(c), "->", repr(a))
