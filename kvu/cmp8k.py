import json, numpy as np
D = "/home/paul/memprobe/needle_out/"
A = json.load(open(D + "needle_sh23_8000_pl.json"))["per_trial"]
B = json.load(open(D + "needle_sh20_8000.json"))["per_trial"]
rng = np.random.default_rng(0)
def boot(x):
    bs = rng.integers(0, len(x), (20000, len(x))); m = x[bs].mean(1)
    return x.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5), (m <= 0).mean()
for d in ["64", "128", "256", "512", "768", "1024"]:
    a, b = A[d], B[d]
    assert len(a["ts"]) == len(b["ts"]) == 256
    g = (np.array(a["ts"]) - np.array(a["to"])) - (np.array(b["ts"]) - np.array(b["to"]))
    l = (np.array(a["lo"]) - np.array(a["ls"])) - (np.array(b["lo"]) - np.array(b["ls"]))
    m, lo, hi, p = boot(100 * g); m2, lo2, hi2, p2 = boot(l)
    print(f"{d:>5} gap23 {100*np.mean(np.array(a['ts'])-a['to']):+5.1f} gap20 {100*np.mean(np.array(b['ts'])-b['to']):+5.1f} diff {m:+5.1f} [{lo:+.1f},{hi:+.1f}] p {p:.4f} | lift diff {m2:+.3f} [{lo2:+.3f},{hi2:+.3f}] p {p2:.4f}")
