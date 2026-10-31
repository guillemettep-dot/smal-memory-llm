import sys
sys.path.insert(0, "/home/paul/memprobe/gqa")
import numpy as np, recall_tasks2 as R
enc = R.enc; X = enc.encode(" x")[0]
for name, gen in R.GENERATORS.items():
    rng = np.random.default_rng([11, len(name)])
    for ex in range(2):
        T = 512
        w = np.full(T + 1, X, dtype=np.int64)
        if name == "quote":
            w = np.array((enc.encode(" The council met on Tuesday to discuss the bridge repairs, which had been delayed twice by flooding and a shortage of steel.") * 40)[:T + 1], dtype=np.int64)
        out, m = gen(w, T, rng, None, want_mask=True)
        s, cur = "", None
        for t, mm in zip(out.tolist(), m.tolist()):
            if mm > 0 and cur != 1: s += "⟦"
            if mm == 0 and cur == 1: s += "⟧"
            s += enc.decode([t]); cur = 1 if mm > 0 else 0
        if cur == 1: s += "⟧"
        import re
        s = re.sub(r"( x){3,}", lambda g: " …[%d filler]… " % (len(g.group(0)) // 2), s)
        if name == "quote":
            s = s.replace("The council met on Tuesday to discuss the bridge repairs, which had been delayed twice by flooding and a shortage of steel.", "<txt>")
        print("=" * 20, name, ex, "=" * 20); print(s.strip()[-1400:])
