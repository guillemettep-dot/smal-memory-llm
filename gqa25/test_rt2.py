import sys, time, collections
sys.path.insert(0, "/home/paul/memprobe/gqa")
import numpy as np
import recall_tasks2 as R
import task_lib as TL
enc = R.enc
X = enc.encode(" x")[0]
fails = []
def check(c, msg):
    if not c: fails.append(msg); print("  FAIL " + msg)
def runs(m):
    out, i = [], 0
    while i < len(m):
        if m[i] > 0:
            j = i
            while j < len(m) and m[j] > 0: j += 1
            out.append((i, j)); i = j
        else: i += 1
    return out
N_DIST = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
print("%-10s %6s %8s %9s %10s %10s %8s" % ("family", "T", "planted", "ans tok", "uniq text", "uniq struct", "us/call"))
for name, gen in R.GENERATORS.items():
    for T in (128, 256, 512, 1024):
        rng = np.random.default_rng([7, T, len(name)])
        n_ok = 0; atok = 0; texts = set(); skels = set(); t0 = time.time()
        reps = N_DIST if T == 1024 else 400
        for t in range(reps):
            w = np.full(T + 1, X, dtype=np.int64)
            if name == "quote":     # quote copies real spans from the window: give it varied tokens
                w = rng.integers(1000, 50000, size=T + 1).astype(np.int64)
            out, m = gen(w.copy(), T, rng, None, want_mask=True)
            check(len(out) == T + 1, "%s T=%d length %d" % (name, T, len(out)))
            if m is None:
                continue
            check(len(m) == T + 1 and m.dtype == np.float32, "%s mask shape/dtype" % name)
            n_ok += 1; atok += int(m.sum())
            sk = R._LAST_SKELETON[0]; skels.add(sk)
            if name == "quote":
                # the query region is at the end; every answer run must equal a span planted earlier
                q0 = runs(m)[0][0]
                for a, b in runs(m):
                    seg = out[a:b].tolist(); hay = out[:q0].tolist()
                    check(any(hay[i:i + len(seg)] == seg for i in range(len(hay) - len(seg) + 1)),
                          "quote answer not found in facts")
                texts.add(tuple(out[np.flatnonzero(m)].tolist()) + (sk,))
                continue
            txt = enc.decode([int(t) for t in out])
            planted = txt.replace(" x", "")
            texts.add(planted)
            for f in R._BATTERY_FRAMES:
                check(f not in planted, "%s contains battery frame %r" % (name, f))
            words = set(planted.replace(".", " ").replace(",", " ").replace('"', " ").replace("-", " ").split())
            check(not (words & R._BATTERY_WORDS), "%s uses battery words %s" % (name, words & R._BATTERY_WORDS))
            rr = runs(m)
            check(len(rr) >= 1, "%s no answer run" % name)
            q0 = rr[0][0]
            # the query block begins at the last newline before the first answer
            facts_txt = enc.decode([int(t) for t in out[:q0]])
            for a, b in rr:
                ans = enc.decode([int(t) for t in out[a:b]]).strip()
                check(len(ans) > 0 and ans in facts_txt, "%s T=%d answer %r not in the facts" % (name, T, ans))
        us = (time.time() - t0) / reps * 1e6
        print("%-10s %6d %7.0f%% %9.1f %10d %10d %8.0f" % (name, T, 100 * n_ok / reps, atok / max(n_ok, 1),
              len(texts), len(skels), us), flush=True)
        if T == 1024:
            check(len(texts) >= 10000 or reps < 20000, "%s: only %d distinct texts in %d" % (name, len(texts), reps))
            check(len(skels) >= 10000 or reps < 20000, "%s: only %d distinct structures in %d" % (name, len(skels), reps))
print("\nALL PASS" if not fails else "\n%d FAILURES, first: %s" % (len(fails), fails[:5]))
