"""Diversified retrieval tasks for the memory LM. Every generator takes a token window
w (length n = T+1, int64) and returns a window of the SAME length with a task planted
in it, exactly like Shards._mqar / _repeat_passage. All facts are queried at the END
of the window (the answer tokens are ordinary next-token targets); the cue arrays are
pre-encoded token sequences supplied by the caller.

Distance control: every generator samples the fact->query gap FIRST (75% from the
upper half of the feasible range, like _repeat_passage) so batteries at 64..T are
all exercised by training.

ANSWER MASKS (want_mask=True): each generator also returns a float32 mask over its
own window marking the tokens that must be RECALLED. This exists because recovering
the answer span by string-matching a cue -- what the trainer did before -- is both
incomplete and imprecise: it misses every exercise that plants no cue (mqar, and the
half of _repeat_passage drawn with no cue), and inside a cued exercise it weights the
whole query region, including the given query keys, which are fresh random symbols
that are unpredictable BY CONSTRUCTION. Weighting those amplifies pure noise. The
generator knows exactly which positions are answers, so it says so.
Default want_mask=False returns the bare window, byte-identical to before, so the
recall probe and test_task_lib keep working unchanged.
"""
import numpy as np

_VOCAB = 50257           # gpt2; ids >= this are padding ids and never appear in text


def _ret(w, m, want_mask):
    return (w, m) if want_mask else w


def _qmask(nrec, stride, ans_idx):
    """Query region = `nrec` records of `stride` tokens; `ans_idx` are the answer
    offsets inside a record (the others are the given key and are left unweighted)."""
    p = np.zeros(stride, dtype=np.float32)
    p[list(ans_idx)] = 1.0
    return np.tile(p, nrec)


P_TOP = 0.30        # share of gap draws from the upper half; tuned 2026-09-30 for ~flat octaves


def _gap(rng, lo, hi):
    """Sample a gap in [lo, hi] LOG-UNIFORMLY: every octave (16-32, 32-64, ... 512-1024) gets about
    equal mass.  It used to draw 75% from the upper half, and measured through compose at T=1024 that
    left 0.9% of training with the answer within 64 tokens of its fact -- so recall at gap 64 was the
    weakest cell of every battery row, and "recall rises with distance" was partly the training mix."""
    if hi <= lo:
        return lo
    if rng.random() < P_TOP:
        # a share of draws from the upper half: the pure log-uniform undersamples the long end,
        # because most tasks' `hi` is well under the window length, so 512-1024 came out at ~5-11%
        return int(rng.integers((lo + hi) // 2, hi + 1))
    return int(np.clip(round(np.exp(rng.uniform(np.log(lo), np.log(hi + 1)))), lo, hi))


def _rand_tokens(rng, k, avoid=(), w=None):
    """k distinct random tokens absent from `avoid` AND from the window `w` (fresh
    symbols with no textual prior and no accidental occurrence in the filler)."""
    out = []
    bad = set(int(a) for a in avoid)
    if w is not None:
        bad.update(int(a) for a in np.unique(w))
    while len(out) < k:
        t = int(rng.integers(1000, _VOCAB))
        if t not in bad:
            bad.add(t); out.append(t)
    return np.array(out, dtype=np.int64)


def _splice(w, facts, queries, gap, cue, qmask=None):
    """[facts][filler of `gap` tokens][cue][queries], trimmed to len(w). Returns
    (window, answer mask) or (None, None) if it does not fit. `qmask` has one entry
    per query token; None means every query token is an answer."""
    n = len(w)
    fill = n - len(facts) - len(cue) - len(queries)
    if fill < max(32, gap):
        return None, None
    # place facts so that the fact->query distance is ~gap: facts sit at fill-gap
    pos = fill - gap
    body = np.concatenate([w[:pos], facts, w[pos:fill], cue, queries])
    m = np.zeros(n, dtype=np.float32)
    m[n - len(queries):] = 1.0 if qmask is None else qmask   # queries sit at the very end
    return body[:n], m


def keyed_span_recall(w, T, rng, cue, npairs=6, span_len=4, want_mask=False):
    n = len(w)
    L = span_len
    need = npairs * (1 + L) * 2 + len(cue) + 64
    if n < need:
        return _ret(w, None, want_mask)
    # keys: fresh symbols; values: real text spans copied from the window (natural payloads)
    keys = _rand_tokens(rng, npairs, w=w)
    starts = rng.choice(n - L - 1, size=npairs, replace=False)
    vals = [w[s:s + L].copy() for s in starts]
    facts = np.concatenate([np.concatenate([[k], v]) for k, v in zip(keys, vals)])
    order = rng.permutation(npairs)
    queries = np.concatenate([np.concatenate([[keys[i]], vals[i]]) for i in order])
    gap = _gap(rng, 16, n - len(facts) - len(cue) - len(queries) - 32)
    out, m = _splice(w, facts, queries, gap, cue, _qmask(npairs, 1 + L, range(1, 1 + L)))
    return _ret(w, None, want_mask) if out is None else _ret(out, m, want_mask)


def updated_fact(w, T, rng, cue, npairs=6, p_update=0.5, want_mask=False):
    n = len(w)
    keys = _rand_tokens(rng, npairs, w=w)
    v1 = _rand_tokens(rng, npairs, avoid=keys, w=w)
    upd = rng.random(npairs) < p_update
    v2 = np.where(upd, _rand_tokens(rng, npairs, avoid=np.concatenate([keys, v1]), w=w), v1)
    facts1 = np.stack([keys, v1], 1).reshape(-1)
    facts2 = np.stack([keys[upd], v2[upd]], 1).reshape(-1)
    order = rng.permutation(npairs)
    queries = np.stack([keys[order], v2[order]], 1).reshape(-1)   # LATEST binding wins
    need = len(facts1) + len(facts2) + len(cue) + len(queries) + 96
    if n < need:
        return _ret(w, None, want_mask)
    # facts1 ... gap1 ... facts2 ... gap2 ... cue queries
    g2 = _gap(rng, 16, (n - need) // 2 + 16)
    g1 = _gap(rng, 16, (n - need) // 2 + 16)
    fill = n - len(facts1) - len(facts2) - len(cue) - len(queries)
    p2 = max(0, fill - g2)
    p1 = max(0, p2 - g1)
    body = np.concatenate([w[:p1], facts1, w[p1:p2], facts2, w[p2:fill], cue, queries])
    m = np.zeros(n, dtype=np.float32)
    m[n - len(queries):] = _qmask(npairs, 2, [1])
    return _ret(body[:n], m, want_mask)


def tagged_pairs(w, T, rng, cue, npairs=6, want_mask=False):
    """Two-token keys [tag, shared] -> value. All pairs share `shared`; only the tag
    disambiguates -- the address must resolve a near-duplicate context."""
    n = len(w)
    shared = _rand_tokens(rng, 1, w=w)
    tags = _rand_tokens(rng, npairs, avoid=shared, w=w)
    vals = _rand_tokens(rng, npairs, avoid=np.concatenate([shared, tags]), w=w)
    facts = np.concatenate([[t, shared[0], v] for t, v in zip(tags, vals)])
    order = rng.permutation(npairs)
    queries = np.concatenate([[tags[i], shared[0], vals[i]] for i in order])
    if n < len(facts) + len(queries) + len(cue) + 64:
        return _ret(w, None, want_mask)
    gap = _gap(rng, 16, n - len(facts) - len(cue) - len(queries) - 32)
    out, m = _splice(w, facts, queries, gap, cue, _qmask(npairs, 3, [2]))
    return _ret(w, None, want_mask) if out is None else _ret(out, m, want_mask)


def list_recall(w, T, rng, cue, nitems=8, min_id=5000, want_mask=False):
    """Recall tokens that occurred exactly once in the window (rare-ish: id >= min_id),
    in order of appearance. Uses the natural text; nothing is planted except the cue."""
    n = len(w)
    body_len = n - len(cue) - nitems
    if body_len < 64:
        return _ret(w, None, want_mask)
    body = w[:body_len]
    vals, counts = np.unique(body, return_counts=True)
    cand = vals[(counts == 1) & (vals >= min_id) & (vals < _VOCAB)]
    if len(cand) < nitems:
        return _ret(w, None, want_mask)
    pick = rng.choice(cand, size=nitems, replace=False)
    first = {int(t): int(np.flatnonzero(body == t)[0]) for t in pick}
    ordered = np.array(sorted(pick, key=lambda t: first[int(t)]), dtype=np.int64)
    m = np.zeros(n, dtype=np.float32)
    m[n - nitems:] = 1.0                      # every recalled item is an answer
    return _ret(np.concatenate([body, cue, ordered])[:n], m, want_mask)


def multi_hop(w, T, rng, cue, npairs=5, want_mask=False):
    n = len(w)
    k1 = _rand_tokens(rng, npairs, w=w)
    v1 = _rand_tokens(rng, npairs, avoid=k1, w=w)
    v2 = _rand_tokens(rng, npairs, avoid=np.concatenate([k1, v1]), w=w)
    facts = np.concatenate([np.stack([k1, v1], 1).reshape(-1), np.stack([v1, v2], 1).reshape(-1)])
    rng.shuffle(facts.reshape(-1, 2))          # interleave the two hop tables
    order = rng.permutation(npairs)
    queries = np.stack([k1[order], v2[order]], 1).reshape(-1)   # k1 -> v2 through v1
    if n < len(facts) + len(queries) + len(cue) + 64:
        return _ret(w, None, want_mask)
    gap = _gap(rng, 16, n - len(facts) - len(cue) - len(queries) - 32)
    out, m = _splice(w, facts, queries, gap, cue, _qmask(npairs, 2, [1]))
    return _ret(w, None, want_mask) if out is None else _ret(out, m, want_mask)


def reverse_assoc(w, T, rng, cue, npairs=6, want_mask=False):
    n = len(w)
    keys = _rand_tokens(rng, npairs, w=w)
    vals = _rand_tokens(rng, npairs, avoid=keys, w=w)
    facts = np.stack([keys, vals], 1).reshape(-1)
    order = rng.permutation(npairs)
    queries = np.stack([vals[order], keys[order]], 1).reshape(-1)   # value -> key
    if n < len(facts) + len(queries) + len(cue) + 64:
        return _ret(w, None, want_mask)
    gap = _gap(rng, 16, n - len(facts) - len(cue) - len(queries) - 32)
    out, m = _splice(w, facts, queries, gap, cue, _qmask(npairs, 2, [1]))
    return _ret(w, None, want_mask) if out is None else _ret(out, m, want_mask)


def predecessor(w, T, rng, cue, nq=6, min_id=5000, want_mask=False):
    """Query a token that occurred once in the natural text; answer = its predecessor."""
    n = len(w)
    body_len = n - len(cue) - 2 * nq
    if body_len < 64:
        return _ret(w, None, want_mask)
    body = w[:body_len]
    vals, counts = np.unique(body[1:], return_counts=True)
    cand = vals[(counts == 1) & (vals >= min_id) & (vals < _VOCAB)]
    if len(cand) < nq:
        return _ret(w, None, want_mask)
    pick = rng.choice(cand, size=nq, replace=False)
    qa = []
    for t in pick:
        i = int(np.flatnonzero(body == t)[0])
        qa += [int(t), int(body[i - 1])]
    m = np.zeros(n, dtype=np.float32)
    m[n - 2 * nq:] = _qmask(nq, 2, [1])
    return _ret(np.concatenate([body, cue, np.array(qa, dtype=np.int64)])[:n], m, want_mask)


def factgen_recall(w, T, rng, cue, want_mask=False):
    """Natural facts (coherent, fresh-paired; simple / two-hop / inference) queried BY DESCRIPTION
    at the end of the window -- factgen_task.py (imported lazily: it uses this module's splicer)."""
    from factgen_task import factgen_task
    return factgen_task(w, T, rng, cue, want_mask=want_mask)


GENERATORS = {
    "keyed_span": keyed_span_recall, "updated": updated_fact, "tagged": tagged_pairs,
    "list": list_recall, "multihop": multi_hop, "reverse": reverse_assoc, "pred": predecessor,
    "factgen": factgen_recall,
}


def compose(w, T, rng, cue, names, weights=None, want_mask=False):
    """Pick one task by weight; for T >= 1024 with prob 0.4 plant TWO tasks (the second
    is planted into the first's filler region so both coexist in the store)."""
    names = list(names)
    p = None if weights is None else np.array(weights, dtype=float) / np.sum(weights)
    a, ma = GENERATORS[rng.choice(names, p=p)](w, T, rng, cue, want_mask=True)
    if ma is None:
        ma = np.zeros(len(a), dtype=np.float32)
    if T >= 1024 and rng.random() < 0.4:
        b = GENERATORS[rng.choice(names, p=p)]
        half = len(a) // 2
        inner, mi = b(a[:half], half - 1, rng, cue, want_mask=True)
        if len(inner) == half:
            a = np.concatenate([inner, a[half:]])
            ma = np.concatenate([np.zeros(half, dtype=np.float32) if mi is None else mi,
                                 ma[half:]])
    return _ret(a, ma, want_mask)
