# build needle_pl.py = needle_pos.py + per-token answer CE (p1 saw, p2 control). Extra JSON keys only.
import sys
d = sys.argv[1]; src = open(f"/home/paul/memprobe/{d}/needle_pos.py").read()
def rep(o, n):
    global src; c = src.count(o); assert c == 1, (c, o); src = src.replace(o, n)
rep("    greedy = lg[0, -n:].argmax(-1).tolist()\n",
    "    greedy = lg[0, -n:].argmax(-1).tolist()\n    global _LAST_PER\n    _LAST_PER = [float(x) for x in per.tolist()]\n")
rep("        l1, g1 = ans_loss(np.concatenate([pre, fact, mid, q]), ans)\n",
    "        l1, g1 = ans_loss(np.concatenate([pre, fact, mid, q]), ans)\n        P1.append(_LAST_PER)\n")
rep("        l2, g2 = ans_loss(np.concatenate([pre, ofact, mid, q]), ans)\n",
    "        l2, g2 = ans_loss(np.concatenate([pre, ofact, mid, q]), ans)\n        P2.append(_LAST_PER)\n")
rep("    G1, G2, AA = [], [], []\n", "    G1, G2, AA = [], [], []\n    P1, P2 = [], []\n")
rep("g1=G1, g2=G2, a=AA)\n", "g1=G1, g2=G2, a=AA, p1=P1, p2=P2)\n")
open(f"/home/paul/memprobe/{d}/needle_pl.py", "w").write(src); print("built", d)
