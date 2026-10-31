"""lenan.py REF OTHER...  (names like sh25_11000) -- loss by context length from lenloss.py outputs, and each
OTHER minus REF paired by document (bootstrap over docs, two-sided p). Context length of a prediction at index i = i+1."""
import sys, numpy as np
D = "/home/paul/memprobe/len_out/"
names = sys.argv[1:]
R = {n: np.load(D + f"len_{n}.npz") for n in names}
sha = {str(R[n]["sha"]) for n in names}; assert len(sha) == 1, sha
L = int(R[names[0]]["L"]); ce = {n: R[n]["ce"].astype(np.float64) for n in names}
edges = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 768, 1024, 1536, 2048]
B = [(a, b) for a, b in zip(edges[:-1], edges[1:]) if b <= L]   # context lengths (a, b]
rng = np.random.default_rng(0)
def boot(x):
    m = x[rng.integers(0, len(x), (20000, len(x)))].mean(1)
    return x.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5), min(1.0, 2 * min((m <= 0).mean(), (m >= 0).mean()))
print(f"{len(ce[names[0]])} docs, tokens sha {sha.pop()}; trained context 1024 (beyond it = extrapolation)")
print("context  " + "".join(f"{n:>12s}" for n in names) + "".join(f"   {n} − {names[0]} [95% CI] p" for n in names[1:]))
for a, b in B:
    row = f"{a+1:>4}-{b:<4}" + "".join(f"{ce[n][:, a:b].mean():12.4f}" for n in names)
    for n in names[1:]:
        m, lo, hi, p = boot(ce[n][:, a:b].mean(1) - ce[names[0]][:, a:b].mean(1))
        row += f"   {m:+.4f} [{lo:+.4f},{hi:+.4f}] p {p:.4f}"
    print(row)
print("all      " + "".join(f"{ce[n].mean():12.4f}" for n in names))
