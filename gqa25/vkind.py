import sys, re, numpy as np, collections
sys.path.insert(0, "/home/paul/memprobe/gqa")
import factgen as F, factgen2 as F2
r = np.random.default_rng(101); N = 30000
PA, PN = set(F2.PADJ), set(F2.PNOUN)
def kind(v):
    w = v.split()
    if len(w) == 2 and w[0] in PA and w[1] in PN: return "word-pair"
    if re.fullmatch(r"[A-Z]{2}-\d+", v): return "code"
    if v.isdigit(): return "number"
    return "single word"
J = collections.Counter(); K = collections.Counter()
for _ in range(N):
    st, cu, v, e, m = F.simple_meta(r)
    k = kind(v); K[k] += 1
    f = "question" if m["question"] else ("anchored" if m["anchored"] else "REWORDED")
    J[(f, k)] += 1
print("value kind:  " + "  ".join("%s %.1f%%" % (k, 100*K[k]/N) for k in ("single word","number","word-pair","code")))
print("battery cells -> training share:  code(anch x num) %.1f%%  paraphrase(rew x num) %.1f%%  phrase(anch x pair) %.1f%%  pairpara(rew x pair) %.1f%%"
      % tuple(100*J[x]/N for x in (("anchored","number"),("REWORDED","number"),("anchored","word-pair"),("REWORDED","word-pair"))))
print("distinct word-pair values possible: %d" % (len(PA)*len(PN)))
