# digits-only paired comparison, truncated to the shared first n trials (same seed => same trials, asserted by answers) of two needle_pl jsons: python digab.py A B  (names like sh24_4000)
import json, sys, numpy as np
D = "/home/paul/memprobe/needle_out/"
na, nb = sys.argv[1], sys.argv[2]
A = json.load(open(D + f"needle_{na}_pl.json"))["per_trial"]; B = json.load(open(D + f"needle_{nb}_pl.json"))["per_trial"]
rng = np.random.default_rng(0)
def boot(x):
    bs = rng.integers(0, len(x), (20000, len(x))); m = x[bs].mean(1)
    return x.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5), min((m <= 0).mean(), (m >= 0).mean()) * 2
def dig(r, key):
    return np.array([np.mean([g[i] == t[i] for i in range(len(t) - 1)]) for g, t in zip(r[key], r["a"])])
print(f"{na} vs {nb}: dist | digits {na} / {nb} | ctl | diff [95% CI] two-sided p")
for d in ["64", "128", "256", "512", "768", "1024"]:
    if d not in A or d not in B: continue
    n = min(len(A[d]["a"]), len(B[d]["a"])); a = {k: v[:n] for k, v in A[d].items() if isinstance(v, list)}; b = {k: v[:n] for k, v in B[d].items() if isinstance(v, list)}; assert a["a"] == b["a"] and n > 0
    sa, sb, ca, cb = dig(a, "g1"), dig(b, "g1"), dig(a, "g2"), dig(b, "g2")
    m, lo, hi, p = boot(100 * (sa - sb))
    print(f"{d:>5} n{n} | {100*sa.mean():5.1f} / {100*sb.mean():5.1f} | {100*ca.mean():.1f} / {100*cb.mean():.1f} | {m:+5.1f} [{lo:+.1f},{hi:+.1f}] p {p:.4f}")
