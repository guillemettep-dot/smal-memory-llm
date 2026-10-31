import json, numpy as np
D = "/home/paul/memprobe/needle_out/"
A = json.load(open(D + "needle_sh23_9000_pl.json"))["per_trial"]; B = json.load(open(D + "needle_sh20_9000_pl.json"))["per_trial"]
rng = np.random.default_rng(0)
def boot(x):
    bs = rng.integers(0, len(x), (20000, len(x))); m = x[bs].mean(1)
    return x.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5), (m <= 0).mean()
print(" dist | digits saw sh23 / sh20 | ctl | diff [CI] p | 2nd-digit loss with fact sh23-sh20 [CI] | ctl")
for d in ["64", "128", "256", "512", "768", "1024"]:
    a, b = A[d], B[d]; assert a["a"] == b["a"]
    def dig(r, key):
        return np.array([np.mean([g[i] == t[i] for i in range(len(t) - 1)]) for g, t in zip(r[key], r["a"])])
    sa, sb, ca, cb = dig(a, "g1"), dig(b, "g1"), dig(a, "g2"), dig(b, "g2")
    m, lo, hi, p = boot(100 * (sa - sb))
    idx = [i for i, t in enumerate(a["a"]) if len(t) == 3]
    l1 = np.array([a["p1"][i][1] - b["p1"][i][1] for i in idx]); l2 = np.array([a["p2"][i][1] - b["p2"][i][1] for i in idx])
    m1, lo1, hi1, _ = boot(l1); m2, lo2, hi2, _ = boot(l2)
    print(f"{d:>5} | {100*sa.mean():5.1f} / {100*sb.mean():5.1f} | {100*ca.mean():.1f} / {100*cb.mean():.1f} | {m:+5.1f} [{lo:+.1f},{hi:+.1f}] p {p:.4f} | {m1:+.2f} [{lo1:+.2f},{hi1:+.2f}] | {m2:+.2f} [{lo2:+.2f},{hi2:+.2f}]")
