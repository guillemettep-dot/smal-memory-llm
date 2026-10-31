import sys, numpy as np
sys.path.insert(0, "/home/paul/memprobe/gqa")
import factgen as F, factgen2 as F2
r = np.random.default_rng(5)
N = 8000; nov = 0; unlinked = 0; cur = 0; cur_unl = 0
for _ in range(N):
    st, cu, v, e, m = F.simple_meta(r)
    R = m["R"]; noun = R["nouns"][0]
    # does SOME relation-identifying token appear in both the fact and the cue?
    toks = set(noun.split()) | set(R["vp"].split()) - {"is", "a", "the", "of", "has", "was", "in", "as", "to"}
    inst = [t for t in toks if t in st.split() or (t + ".") in st.split() or (t + ":") in st]
    incue = [t for t in toks if t in cu.replace("?", " ").replace(":", " ").split()]
    linked = bool(set(inst) & set(incue))
    if m["novel"]:
        nov += 1; unlinked += int(not linked)
    else:
        cur += 1; cur_unl += int(not linked)
print("INVENTED relations: %d examples, fact and cue share NO relation word: %d (%.1f%%)" % (nov, unlinked, 100*unlinked/max(nov,1)))
print("CURATED  relations: %d examples, no shared word: %d (%.1f%%)  <- learnable: these relations recur, so 'lives in'~'city' can be learned"
      % (cur, cur_unl, 100*cur_unl/max(cur,1)))
print("=> share of ALL simple facts whose relation cannot be recovered from the cue: %.1f%%" % (100*unlinked/N))
