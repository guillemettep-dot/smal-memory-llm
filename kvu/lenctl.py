"""lenctl.py NAME...  -- for each model: full-context loss at positions 1025..2048 (len_NAME.npz) vs the SAME tokens with the
context restarted at 1024 (len_NAME_r1024.npz). full − restarted < 0: the first 1024 tokens still help past the trained
length; > 0: the long context hurts (extrapolation failure). Also restarted vs the model's own fresh-start curve on 1..1024."""
import sys, numpy as np
D = "/home/paul/memprobe/len_out/"
rng = np.random.default_rng(0)
def boot(x):
    m = x[rng.integers(0, len(x), (20000, len(x)))].mean(1)
    return x.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5), min(1.0, 2 * min((m <= 0).mean(), (m >= 0).mean()))
edges = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 768, 1024]
for n in sys.argv[1:]:
    F, R = np.load(D + f"len_{n}.npz"), np.load(D + f"len_{n}_r1024.npz")
    assert (F["docs"] == R["docs"]).all() and int(F["L"]) == 2048 and int(R["L"]) == 1024 and int(R["OFF"]) == 1024
    f, r, f0 = F["ce"][:, 1024:].astype(np.float64), R["ce"].astype(np.float64), F["ce"][:, :1024].astype(np.float64)
    # same tokens: F index 1024+j predicts doc token 1025+j; R index j predicts doc token 1024+1+j. Identical targets.
    print(f"\n{n}: positions 1025-2048, full context (pos) vs restarted at 1024 (context = pos-1024)")
    print("restarted ctx |   full   restart | full − restart [95% CI] p")
    for a, b in zip(edges[:-1], edges[1:]):
        m, lo, hi, p = boot(f[:, a:b].mean(1) - r[:, a:b].mean(1))
        print(f"  {a+1:>4}-{b:<4}    | {f[:, a:b].mean():.4f}  {r[:, a:b].mean():.4f} | {m:+.4f} [{lo:+.4f},{hi:+.4f}] p {p:.4f}")
    m, lo, hi, p = boot(f.mean(1) - r.mean(1)); print(f"  all          | {f.mean():.4f}  {r.mean():.4f} | {m:+.4f} [{lo:+.4f},{hi:+.4f}] p {p:.4f}")
    print(f"  fresh-start curve on doc tokens 1..1024: {f0.mean():.4f}; restarted (doc tokens 1025..2048, 1..1024 of context): {r.mean():.4f}  -> text difficulty of the later half {r.mean()-f0.mean():+.4f}")
