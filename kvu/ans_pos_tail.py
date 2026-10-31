# ---- per-position answer accuracy (appended to needle_live.py lines 1-215 by mk_pos.sh)
import collections
NPOS = int(os.environ.get("NPOS_TRIALS", "64"))
res = {}
for dist in DISTS:
    cs, co, n3 = [0, 0, 0], [0, 0, 0], 0
    wrong_dot = collections.Counter()
    tot_s = tot_o = 0.0
    for t in range(NPOS):
        fact, ofact, q, ans, pre, stream = trial_content(t)
        mid = stream[:dist]
        _, g1 = ans_loss(np.concatenate([pre, fact, mid, q]), ans)
        _, g2 = ans_loss(np.concatenate([pre, ofact, mid, q]), ans)
        a = ans.tolist()
        tot_s += np.mean([x == y for x, y in zip(g1, a)]); tot_o += np.mean([x == y for x, y in zip(g2, a)])
        if len(a) != 3:
            continue
        n3 += 1
        for p in range(3):
            cs[p] += int(g1[p] == a[p]); co[p] += int(g2[p] == a[p])
        if g2[2] != a[2]:
            wrong_dot[enc.decode([g2[2]])] += 1
    res[dist] = dict(n3=n3, saw=[c / n3 for c in cs], ctl=[c / n3 for c in co],
                     tok_saw=tot_s / NPOS, tok_ctl=tot_o / NPOS, ctl_dot_wrong=wrong_dot.most_common(5))
    print(f"dist {dist:5d} n3={n3}  saw pos0/1/2(.) " + " ".join(f"{x:.1%}" for x in res[dist]['saw'])
          + "  ctl " + " ".join(f"{x:.1%}" for x in res[dist]['ctl'])
          + f"  | tok saw {tot_s/NPOS:.1%} ctl {tot_o/NPOS:.1%} | ctl '.' misses {wrong_dot.most_common(4)}", flush=True)
json.dump(dict(ckpt=CK, step=STEP, ntrials=NPOS, res=res), open(os.environ["OUT"], "w"))
