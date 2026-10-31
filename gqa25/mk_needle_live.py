import io, ast
src = "/home/paul/memprobe/gqa/needle_cpu.py"; dst = "/home/paul/memprobe/gqa/needle_live.py"
s = io.open(src, encoding="utf-8").read()
a = '''        if (t + 1) % 64 == 0:
            print(f"    dist {dist}: {t+1}/{TRIALS}", flush=True)
'''
b = '''        if (t + 1) % 32 == 0:
            n_ = t + 1
            lift = float(np.mean(np.asarray(lo) - np.asarray(ls)))
            msg = (f"    dist {dist}: {n_:3d}/{TRIALS}  loss lift {lift:+.3f} nats  "
                   f"tok-acc {np.mean(ts):.1%} (ctl {np.mean(to):.1%})  exact {sum(es)}/{n_} (ctl {sum(eo)})")
            for _rn, _rd in _REFS:           # the SAME trials from a reference checkpoint: paired
                _r = _rd.get(str(dist))
                if _r is not None:
                    _rl = float(np.mean(np.asarray(_r["lo"][:n_]) - np.asarray(_r["ls"][:n_])))
                    msg += f" | {_rn} lift {_rl:+.3f} tok {np.mean(_r['ts'][:n_]):.1%}"
            print(msg, flush=True)
'''
assert s.count(a) == 1; s = s.replace(a, b)
a2 = '''    print(f"  dist {dist} done  exact={sum(es)}/{TRIALS} ctl={sum(eo)}/{TRIALS}", flush=True)'''
b2 = '''    print(f"  dist {dist} done  loss lift {float(np.mean(np.asarray(lo) - np.asarray(ls))):+.3f} nats  "
          f"tok-acc {np.mean(ts):.1%} (ctl {np.mean(to):.1%})  exact={sum(es)}/{TRIALS} ctl={sum(eo)}/{TRIALS}", flush=True)'''
assert s.count(a2) == 1; s = s.replace(a2, b2)
a3 = '''# ---------------------------------------------------------------- run
results = {}'''
b3 = '''# ---------------------------------------------------------------- run
# REFS="name=path.json,..." : per-trial results of other checkpoints on the SAME seed, printed beside
# the live numbers over the same trials (a paired comparison, as the trials are identical)
_REFS = []
for _it in [x for x in os.environ.get("REFS", "").split(",") if x]:
    _n, _p = _it.split("=", 1)
    _b = json.load(open(_p))
    assert _b["seed"] == SEED, "reference %s used seed %s, this run %s: not paired" % (_n, _b["seed"], SEED)
    _REFS.append((_n, _b["per_trial"]))
results = {}'''
assert s.count(a3) == 1; s = s.replace(a3, b3)
ast.parse(s); io.open(dst, "w", encoding="utf-8").write(s); print("needle_live.py written")
