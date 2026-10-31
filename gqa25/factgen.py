"""Coherent diverse-fact generator: simple facts + multi-hop + procedural inference + paraphrased
queries + optional real-text context. Fresh-paired per example (non-memorizable), queried by
description among distractors. Returns gpt2 tokens + answer mask (1 on the query answer tokens)."""
import os
import numpy as np, tiktoken
enc = tiktoken.get_encoding("gpt2")
OBJECTS = ("table chair desk shelf couch stool bench cabinet bed lamp clock mirror vase pillow blanket rug towel "
           "broom candle kettle pot pan plate bowl cup mug bottle jar hammer wrench drill saw shovel rake ladder axe "
           "box crate basket bucket barrel car truck van bike scooter boat cart wagon guitar drum flute violin trumpet "
           "hat coat scarf glove boot shirt jacket belt phone radio camera speaker ball racket helmet kite anchor "
           "lantern spoon fork knife tent rope chain wheel engine").split()
NAMES = ("Maria James Aisha Noah Priya Omar Lena Tariq Sofia Diego Yuki Kwame Ingrid Raj Elena Mateo Fatima Hugo Nadia "
         "Sven Amara Bjorn Chiara Dmitri Esme Felix Greta Hassan Iris Jamal Kira Liam Mira Nikolai Olga Pedro Quinn Rosa "
         "Sami Tessa Umar Vera Wesley Xena Yara Zane Ana Boris Cleo Dario Enzo Faye Gino Hana Ivo Juno").split()
COLORS = "red blue green yellow black white orange purple brown grey pink teal maroon beige gold silver".split()
CITIES = "Paris Tokyo Lagos Lima Cairo Oslo Delhi Rome Seoul Quito Accra Riga Hanoi Dublin Vienna Manila Tunis Bergen Perth Osaka".split()
JOBS = "baker pilot nurse welder chemist teacher farmer lawyer painter sailor dancer plumber tailor guard".split()
MATERIALS = "wood steel glass copper leather bamboo marble plastic ceramic bronze".split()
ANIMALS = "cat dog rabbit parrot turtle hamster ferret goat pony finch gecko".split()
def _n(r,a,b): return str(int(r.integers(a,b)))
# Facts and cues come from factgen2, a COMPOSITIONAL generator, not a template list.
# WHY: the old SIMPLE table held 8 relations x a handful of fixed query strings, and every one of
# those queries was a QUESTION ending in "?".  The model was therefore never trained on a cue that
# stops MID-SENTENCE and expects the answer as the continuation -- the form the recall battery
# measures, where it scores 0.0% at every checkpoint while scoring 22-92% on verbatim repetition.
# A fixed list is also memorisable: 16 strings can be learned without learning the operation.
# factgen2 composes determiner x relation-noun x possessive-vs-of-phrase x copula x prefix clause
# (4,262 distinct cues per 8,000 draws), holds ~20% of (relation, frame, frame) combinations out of
# training for honest evaluation, and reports whether each cue shares an ANCHOR with its fact --
# the cue's last token equalling the fact's pre-answer token, which is the only thing that works
# today.  ANCHOR_P is the fraction given that shortcut.
import factgen2 as _F2
ANCHOR_P = float(os.environ.get("FACTGEN_ANCHOR_P", "0.5"))
# fraction of cues that are QUESTIONS rather than mid-sentence.  The old bank was 100% questions;
# the first rewrite over-corrected to 100% mid-sentence.  Both forms must train.
Q_FRAC = float(os.environ.get("FACTGEN_Q_FRAC", "0.25"))
SPLIT = os.environ.get("FACTGEN_SPLIT", "train")

def _simple(r):
    stmt, cue, v, ents, _m = _F2.sample(r, SPLIT, ANCHOR_P, q_frac=Q_FRAC)
    return stmt, cue, v, ents

def simple_meta(r):
    """As _simple but also returns meta (relation key, anchoring, cue form).  factgen_task uses it
    to build HARD NEGATIVES -- distractors that share the target's ENTITY (different relation) or
    its RELATION (different entity).  Without those, every distractor is unrelated and matching the
    relation word alone is enough; nothing forces the model to bind entity AND relation together."""
    return _F2.sample(r, SPLIT, ANCHOR_P, q_frac=Q_FRAC)

def sibling_fact(r, share, meta):
    """One distractor statement sharing EXACTLY ONE thing with a target (never queried).

    share="entity"    the SAME entity, written exactly as the target writes it ("the lantern", not
                      "lantern"), under a DIFFERENT relation of the same entity pool.  The first
                      version passed the bare entity and any relation, which produced "Records show
                      lantern works with Nadia." and could re-draw the target's own relation --
                      planting a contradictory fact about the queried pair.
    share="relation"  the SAME relation object (curated or invented) about a DIFFERENT entity."""
    if share == "entity":
        st, _c, _v, _e, _m = _F2.sample(r, SPLIT, ANCHOR_P, q_frac=Q_FRAC, pool=meta["pool"],
                                        stmt_entity=meta["E"], exclude_key=meta["rel"])
    else:
        st, _c, _v, _e, _m = _F2.sample(r, SPLIT, ANCHOR_P, q_frac=Q_FRAC, R_force=meta["R"],
                                        exclude_entity=meta["ent"])
    return st

def _multihop(r):
    # "A neighbor is B."  +  a fact about B  ->  asked about "A neighbor" (two hops).
    a, b = r.choice(_F2.NAMES, size=2, replace=False)
    stmt, cue, v, _e, _m = _F2.sample(r, SPLIT, ANCHOR_P, pool="name", q_frac=Q_FRAC,
                                      stmt_entity=str(b), cue_entity="%s neighbor" % a)
    return "%s neighbor is %s. %s" % (a, b, stmt), cue, v, (str(a), str(b))

def _inference(r):
    kind=int(r.integers(0,3))
    if kind==0:  # comparison: who is older is never stated
        a,b=r.choice(NAMES,size=2,replace=False); x,y=int(r.integers(5,95)),int(r.integers(5,95))
        while y==x: y=int(r.integers(5,95))
        s="%s is %d years old. %s is %d years old."%(a,x,b,y); older=a if x>y else b
        return s,"Who is older, %s or %s?"%(a,b),older,(a,b)
    if kind==1:  # remainder not stated
        o=str(r.choice(OBJECTS)); h=int(r.integers(4,40)); k=int(r.integers(1,h))
        s="The crate holds %d %s items. Someone removes %d."%(h,o,k)
        return s,"How many %s items remain?"%o,str(h-k),(o,)
    o=str(r.choice(OBJECTS)); p=int(r.integers(100,5000)); n=int(r.integers(2,6))  # total cost not stated
    s="Each %s costs %d coins. You buy %d."%(o,p,n)
    return s,"What is the total cost?",str(p*n),(o,)
def make_example(rng, nfacts=8, filler=None):
    stmts=[_simple(rng)[0] for _ in range(nfacts-1)]   # distractors
    gp=rng.random(); pick=_simple if gp<0.6 else _multihop if gp<0.8 else _inference
    ts,tq,ans,_=pick(rng); stmts.append(ts); rng.shuffle(stmts)
    body=" ".join(stmts)+" "+tq
    if filler is not None and len(filler)>0:
        body=enc.decode([int(t) for t in filler[:40]])+" "+body   # real-text context prefix
    tt=enc.encode(body); at=enc.encode(" "+ans)
    toks=np.array(tt+at,dtype=np.int64); mask=np.zeros(len(toks),dtype=np.float32); mask[len(tt):]=1.0
    return toks,mask,(tq,ans)
if __name__=="__main__":
    r=np.random.default_rng(0)
    print("=== category samples ===")
    for fn,nm in ((_simple,"simple"),(_multihop,"multihop"),(_inference,"inference")):
        for _ in range(3):
            s,q,a,_=fn(r); print("  [%-9s] %-62s Q: %-30s -> %s"%(nm,s[:62],q,a))
    print("=== full examples (distractors + one target, description query) ===")
    for _ in range(4):
        t,m,(q,a)=make_example(r,8); print("  Q: %-34s -> %-8s (%d tok, %d ans tok)"%(q,a,len(t),int(m.sum())))
