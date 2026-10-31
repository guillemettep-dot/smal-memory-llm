# build needle_pos.py = needle_live.py + per-trial greedy tokens (g1 saw, g2 control, a answer). Extra JSON keys only.
import sys
d = sys.argv[1]; src = open(f"/home/paul/memprobe/{d}/needle_live.py").read()
def rep(o, n):
    global src; c = src.count(o); assert c == 1, (c, o); src = src.replace(o, n)
rep("    ls, lo, es, eo, ts, to = [], [], [], [], [], []\n",
    "    ls, lo, es, eo, ts, to = [], [], [], [], [], []\n    G1, G2, AA = [], [], []\n")
rep("        ls.append(l1); lo.append(l2)\n",
    "        ls.append(l1); lo.append(l2)\n        G1.append(list(map(int, g1))); G2.append(list(map(int, g2))); AA.append(list(map(int, ans.tolist())))\n")
rep("    results[dist] = dict(ls=ls, lo=lo, es=es, eo=eo, ts=ts, to=to)\n",
    "    results[dist] = dict(ls=ls, lo=lo, es=es, eo=eo, ts=ts, to=to, g1=G1, g2=G2, a=AA)\n")
open(f"/home/paul/memprobe/{d}/needle_pos.py", "w").write(src); print("built", d)
