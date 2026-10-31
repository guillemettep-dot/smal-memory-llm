import json, sys, numpy as np
D = "/home/paul/memprobe/needle_out/"
def load(n):
    j = json.load(open(D + f"needle_{n}_pl.json"))
    r = j["per_trial"]
    out = {int(k): v for k, v in r.items()}; assert len(out) == 6, out.keys(); return out
A = load("sh23_6000"); B = load("sh20_6000"); C = load("sh23_5000")
rng = np.random.default_rng(0)
def ci(x):
    x = np.asarray(x); bs = rng.integers(0, len(x), (10000, len(x)))
    m = x[bs].mean(1); return x.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)
names = ["digit1", "digit2", "."]
for dist in sorted(A):
    a, b, c = A[dist], B[dist], C[dist]
    assert a["a"] == b["a"] == c["a"], "answer tokens differ"
    idx = [i for i, t in enumerate(a["a"]) if len(t) == 3]
    for r in (a, b, c):  # self-check: per-token mean == ls/lo
        for i in idx:
            assert abs(np.mean(r["p1"][i]) - r["ls"][i]) < 1e-3 and abs(np.mean(r["p2"][i]) - r["lo"][i]) < 1e-3
    print(f"dist {dist}  n={len(idx)} (3-token answers)")
    for key, lab in (("p1", "saw"), ("p2", "ctl")):
        for p in range(3):
            va = np.array([a[key][i][p] for i in idx]); vb = np.array([b[key][i][p] for i in idx]); vc = np.array([c[key][i][p] for i in idx])
            d1 = ci(va - vb); d2 = ci(va - vc)
            print(f"  {lab} {names[p]:6s} sh23@6k {va.mean():6.3f} sh20@6k {vb.mean():6.3f} sh23@5k {vc.mean():6.3f} | 23-20 {d1[0]:+.3f} [{d1[1]:+.3f},{d1[2]:+.3f}] | 6k-5k {d2[0]:+.3f} [{d2[1]:+.3f},{d2[2]:+.3f}]")
