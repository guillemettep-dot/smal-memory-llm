"""task_lib-compatible natural-fact task ("factgen"): plant K coherent, fresh-paired facts (60% simple,
20% two-hop, 20% procedural inference; see factgen.py) into the real-text window, then after a gap
ask Q of them BY DESCRIPTION as  Q: <question> A: <answer>.  with only the answer tokens masked.
Same splice geometry as the other task_lib generators ([facts][filler ~gap][queries at the end])."""
import numpy as np
import factgen
from task_lib import _splice, _gap, _ret

_enc = factgen.enc


def _render_queries(items, ask_idx):
    qt, qm = [], []
    for j, i in enumerate(ask_idx):
        _, q, a = items[i]
        # A QUESTION gets the Q:/A: frame.  A MID-SENTENCE cue must NOT: wrapping it produced
        # "Q: The colour of the rug is A: gold.", which does not stop mid-sentence at all and
        # makes "A:" the anchor for every single trial -- the same degenerate shortcut the
        # mid-sentence cues exist to remove.  Verified by decoding the assembled window.
        # the first question of a block starts on its own line: it used to be glued to the filler
        # ("stored in theHugo's stint is"), tokenising as merged words, and the recall battery puts
        # a newline before every cue.
        if q.rstrip().endswith("?"):
            qtoks = _enc.encode((" Q: " if j else "\nQ: ") + q + " A:")
        else:
            qtoks = _enc.encode((" " if j else "\n") + q)
        atoks = _enc.encode(" " + a)
        dot = _enc.encode(".")
        qt += qtoks + atoks + dot
        qm += [0.0] * len(qtoks) + [1.0] * len(atoks) + [0.0] * len(dot)
    return np.array(qt, dtype=np.int64), np.array(qm, dtype=np.float32)


def _segmented(w, items, hard, rng, q_frac, S):
    """S consecutive [facts][gap][queries] segments, one per chunk of the window.

    WHY: with ONE block of K=24 facts, the block alone is several hundred tokens, so even at gap 16
    almost every queried fact sat 200+ tokens back -- factgen had ~0% of its answers within 64 tokens.
    Shrinking K fixed the distances but cut the answer tokens by 45-64%, and those answers are the
    reworded-cue signal.  Splitting the SAME facts into S short segments keeps every answer and makes
    each fact->question distance short.  A hard negative stays in its target's segment, so it is in
    context when that target is asked.  Returns None if a segment cannot fit; the caller then tries
    fewer segments."""
    n = len(w)
    idx = rng.permutation(len(items))
    groups = [list(int(x) for x in g) for g in np.array_split(idx, S)]
    where = {i: gi for gi, g in enumerate(groups) for i in g}
    gh = [[] for _ in range(S)]
    for ti, st in hard:
        gh[where[ti]].append(st)
    bounds = [n * k // S for k in range(S + 1)]
    outs, ms = [], []
    for gi in range(S):
        chunk = w[bounds[gi]:bounds[gi + 1]]
        nc = len(chunk)
        g, hs = list(groups[gi]), list(gh[gi])
        while True:
            stmts = [items[i][0] for i in g] + hs
            perm = rng.permutation(len(stmts))
            facts = np.array(_enc.encode("\n" + " ".join(stmts[k] for k in perm) + "\n"), dtype=np.int64)
            ask = [g[k] for k in rng.permutation(len(g))][:max(1, int(len(g) * q_frac))]
            queries, qmask = _render_queries(items, ask)
            if nc >= len(facts) + len(queries) + 48:
                break
            if len(g) <= 1:
                return None
            g = g[:-1]
        gap = _gap(rng, 16, nc - len(facts) - len(queries) - 32)
        o, m = _splice(chunk, facts, queries, gap, np.empty(0, dtype=np.int64), qmask)
        if o is None:
            return None
        outs.append(o); ms.append(m)
    out, m = np.concatenate(outs), np.concatenate(ms)
    assert len(out) == n and len(m) == n, "segmented layout changed the window length"
    return out, m


S_MAX = 6
P_SINGLE = 0.70     # probability of ONE long block; otherwise S is log-uniform in [2, S_MAX]


def factgen_task(w, T, rng, cue, want_mask=False, tok_per_fact=40, q_frac=0.5):
    n = len(w)
    K = int(np.clip(n // tok_per_fact, 3, 24))
    items = []
    hard = []          # (target item index, statement) -- distractors only, never queried
    for _ in range(K):
        r = rng.random()
        # _inference is GONE from the training mix (kept defined in factgen.py so the older
        # by-description eval scripts still import).  Measured 2026-09-29 on 42 trials per model:
        # saw 14.9% vs control 14.9%, delta EXACTLY +0.0, for BOTH gpt2 and our step-9000 arm.
        # It cannot move: "Who is older, A or B?" prints the answer inside the question, and the
        # other two kinds need arithmetic no memory can supply.  Its share goes to _simple.
        if r < 0.8:
            s, q, a, ents, meta = factgen.simple_meta(rng)
            # HARD NEGATIVES: with p=0.5 also plant a fact sharing this target's ENTITY (a different
            # relation about the same thing) or its RELATION (the same attribute of something else).
            # Otherwise matching the relation word alone suffices and nothing forces the model to
            # bind entity AND relation.
            if rng.random() < 0.5:
                share = "entity" if rng.random() < 0.5 else "relation"
                hard.append((len(items), factgen.sibling_fact(rng, share, meta)))
        else:
            s, q, a, _ = factgen._multihop(rng)
        items.append((s, q, a))

    # number of segments: ONE long block with probability P_SINGLE, else log-uniform in [2, S_MAX].
    # Tuned by measurement so fact->answer distances cover every octave from 32 to 1024 about
    # equally -- too many segments pushes training short and starves the long range instead.
    smax = min(S_MAX, K)
    if smax < 2 or rng.random() < P_SINGLE:
        S = 1
    else:
        S = int(np.clip(np.floor(np.exp(rng.uniform(np.log(2), np.log(smax + 1)))), 2, smax))
    while S > 1:
        res = _segmented(w, items, hard, rng, q_frac, S)
        if res is not None:
            return _ret(res[0], res[1], want_mask)
        S -= 1

    # S == 1: the original single-block layout
    hard1 = [st for _, st in hard]
    order = rng.permutation(K)
    ask = rng.permutation(K)
    # shrink (fewer facts / queries) until the block leaves >= 48 tokens of real text around it
    while True:
        _stmts = [items[i][0] for i in order] + hard1
        _perm = rng.permutation(len(_stmts))
        facts = np.array(_enc.encode("\n" + " ".join(_stmts[k] for k in _perm) + "\n"), dtype=np.int64)
        Q = max(1, int(len(order) * q_frac))
        queries, qmask = _render_queries(items, [i for i in ask if i in set(order.tolist())][:Q])
        if n >= len(facts) + len(queries) + 48:
            break
        if len(order) <= 2:
            return _ret(w, None, want_mask)      # cannot fit even 2 facts: leave the window unchanged
        order = order[:-1]
    gap = _gap(rng, 16, n - len(facts) - len(queries) - 32)
    out, m = _splice(w, facts, queries, gap, np.empty(0, dtype=np.int64), qmask)
    return _ret(w, None, want_mask) if out is None else _ret(out, m, want_mask)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    for T in (128, 256, 512, 1024):
        w = np.array(_enc.encode("the quick brown fox " * 400)[:T + 1], dtype=np.int64)
        out, m = factgen_task(w, T, rng, None, want_mask=True)
        assert len(out) == T + 1 and (m is None or len(m) == T + 1)
        if m is None:
            print("T=%d: did not fit" % T); continue
        ans = out[m > 0]
        print("T=%d: %d answer tokens; tail: %r" % (T, int(m.sum()), _enc.decode([int(t) for t in out[-60:]])))
    print("FACTGEN_TASK_OK")
