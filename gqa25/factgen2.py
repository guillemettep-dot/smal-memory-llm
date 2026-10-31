# -*- coding: utf-8 -*-
"""Compositional fact/query generator (v2, expanded).

WHY IT IS BUILT THIS WAY
  A fixed template list is memorisable: the old bank held 8 relations x a handful of cue strings,
  and the recall battery shows the consequence -- 22-92% when the cue repeats the stored wording,
  0.0% when it does not, at every checkpoint and both payload types.  So cues are COMPOSED here,
  and three properties are controlled rather than hoped for:

  ANCHORING.  One rule predicted all 14 probe designs measured 2026-09-29: retrieval succeeds when
  the cue's last token equals the token preceding the value inside the fact.  `anchor_p` is the
  fraction of trials handed that shortcut, so it can be ramped down instead of given away forever.
  It is computed from the REALISED strings, never assumed from the template.

  CUE FORM.  Both mid-sentence cues ("The colour of the rug is") and questions ("What colour is the
  rug?") are generated.  Training on only one form swaps one blind spot for another -- the previous
  bank was 100% questions, and the first version of this file over-corrected to 100% mid-sentence.

  VALUE TYPE.  Single words, integers, MULTI-WORD PHRASES and alphanumeric codes.  Multi-word values
  were entirely absent before, yet two of the four battery cells score exactly that.
  The phrase vocabulary is DISJOINT from the battery's, so the eval stays a generalisation test.

  A held-out split reserves ~20% of (relation, stmt frame, query frame) combinations, so an eval cue
  is provably one the model never trained on.
"""
import hashlib
import numpy as np

OBJECTS = ("table chair desk shelf couch stool bench cabinet bed lamp clock mirror vase pillow blanket rug towel "
           "kettle basket crate barrel ladder bucket drawer cushion curtain carpet satchel canteen lantern").split()
NAMES = ("Maria James Aisha Noah Priya Omar Lena Tariq Sofia Diego Yuki Kwame Ingrid Raj Elena Mateo Fatima Hugo "
         "Nadia Kira Wesley Anders Rosa Linh Bilal Marta Soren Ayana Felix Noor").split()
COLORS = "red blue green yellow black white orange purple brown grey pink teal maroon beige gold silver".split()
CITIES = "Paris Tokyo Lagos Lima Cairo Oslo Delhi Rome Seoul Quito Accra Riga Hanoi Dublin Vienna Manila".split()
JOBS = "baker pilot nurse welder chemist teacher farmer lawyer painter sailor dancer plumber tailor guard".split()
MATERIALS = "wood steel glass copper leather bamboo marble plastic ceramic bronze".split()
ANIMALS = "cat dog rabbit parrot turtle hamster ferret goat pony finch gecko".split()
LANGS = "Portuguese Finnish Swahili Tamil Danish Korean Hungarian Quechua Berber Latvian".split()
PLACES = "vault annexe stairwell cellar loft pantry workshop gallery".split()
# DISJOINT from the recall battery's phrase vocabulary, so the eval's words are unseen in training.
# Word-pair vocabulary, DISJOINT from the recall battery's (silver crimson hollow northern quiet amber
# distant velvet / falcon lantern harbor meadow cipher anchor thistle beacon), so the battery's
# phrase cells stay a generalisation test.  Was 10 x 10 = 100 pairs -- small enough to memorise as
# fixed strings instead of learning to copy two arbitrary words; now 20 x 20 = 400.
PADJ = ("copper restless pale eastern solemn jade remote woven hidden patient "
        "brittle crooked sunken frosted molten gilded faded tangled drifting rustic").split()
PNOUN = ("heron compass quarry orchard sigil bastion bramble ember lattice harrow "
         "spire cairn glacier citadel willow furnace pylon garnet tundra rampart").split()
_LET = "KRTVMJQZBD"

_SYL_A = "ka to mi ra len sor vi nu bel dar ith os pel gar un ver".split()
_SYL_B = "don mer tal sin quo vash ren pil mos tay gund rix pha nel".split()

def _fresh_name(r):
    return (_SYL_A[int(r.integers(0, len(_SYL_A)))] + _SYL_B[int(r.integers(0, len(_SYL_B)))]).capitalize()
def _fresh_object(r):
    return _SYL_A[int(r.integers(0, len(_SYL_A)))] + _SYL_B[int(r.integers(0, len(_SYL_B)))]
def _pick(r, xs): return xs[int(r.integers(0, len(xs)))]
def _num(r, a, b): return str(int(r.integers(a, b)))
def _phrase(r): return "%s %s" % (_pick(r, PADJ), _pick(r, PNOUN))
def _code(r): return "%s%s-%d" % (_pick(r, list(_LET)), _pick(r, list(_LET)), int(r.integers(100, 9999)))

# ---------------------------------------------------------------------------------------------
# PROCEDURAL RELATIONS.  The curated table below holds 20 relations; measured, that is the binding
# constraint on generalisation.  Surface diversity is already ample (28k distinct cue strings per
# 60k draws, none above 0.05%), but the recall battery asks about a relation the model has never
# seen ("access code", "password"), so it must have induced "relation" as an ABSTRACT SLOT rather
# than learning 20 separate lookup behaviours -- and with each combo seen ~9,000 times over the run,
# memorising 20 special cases is the cheaper solution for a weak model.
# So most relations are INVENTED per example from these pools, including nonsense pairings: a
# relation the model cannot know the meaning of can only be handled structurally, which is the
# capability wanted.  p_novel controls the mix against the curated, sensible ones.
ATTR_NOUNS = ("grade tier batch finish hue span origin rating index bracket quota margin tally "
              "seal token stamp stripe crest emblem sigil charge bearing register lot band class "
              "cadre facet gauge girth heft measure notch parcel phase quarter reading sector "
              "segment shade slot spool stint strand tranche verge warrant yield zone allotment "
              "bearing caliper docket ferrule gantry hallmark ingot jetty keystone lintel mantle "
              "nodule ordinal pennant quiver rivet spindle transom vellum wicket").split()
_INV_A = "vor pel quen mar tis loq ren dal fen gib hux jorn kel".split()
_INV_B = "age ing ment ance ure ity oid ost ark elm ide ony".split()
def _invented_noun(r):
    return _pick(r, _INV_A) + _pick(r, _INV_B)

# An INVENTED relation's verb phrase is built FROM its noun, so the noun appears in every statement
# AND every cue.  The first version paired a random noun with an unrelated random verb phrase
# ("kelance" with "carries"); statement frames use one and cue frames the other, so in 43% of
# invented-relation examples the fact and the cue shared no relation word at all -- for a relation
# made up inside this one window that is unrecoverable, and with an entity-sharing distractor
# present it is simply label noise.  Measured: 26% of ALL simple facts.
INV_VP = ["has a {n} of", "carries a {n} of", "was assigned a {n} of", "shows a {n} of",
          "has its {n} set to", "records a {n} of", "was given a {n} of", "bears a {n} of",
          "lists a {n} of", "keeps a {n} of", "was logged with a {n} of", "holds a {n} of"]

def _value_kind(r):
    """(generator, suffix). Mixed types so no single answer shape can be assumed."""
    k = r.random()
    # Shares rebalanced 2026-10-02: word-pair answers were 17.6% of training against 27% for numbers,
    # and the two word-pair battery cells (phrase, pairpara) are the weakest and had stopped
    # improving; single words were 41.8% though no battery cell measures them.
    if k < 0.18: return (lambda rr: _pick(rr, COLORS + MATERIALS + ANIMALS + LANGS + PLACES + JOBS)), ""
    if k < 0.40: return (lambda rr: _num(rr, 2, 9999)), _pick(r, ["", " units", " grams", " coins", " points"])
    if k < 0.78: return _phrase, ""
    if k < 0.92: return _code, ""
    return (lambda rr: _pick(rr, CITIES + NAMES)), ""

def _novel_relation(r):
    noun = _invented_noun(r) if r.random() < 0.35 else _pick(r, ATTR_NOUNS)
    gen, suf = _value_kind(r)
    return dict(key="~" + noun, pool=_pick(r, ["obj", "name"]), nouns=[noun],
                vp=_pick(r, INV_VP).format(n=noun), suf=suf, gen=gen)

# vp   : verb phrase introducing the value in plain statement form
# suf  : text appended to the value INSIDE the statement only (the answer stays just the value)
RELATIONS = [
    dict(key="colour",   pool="obj",  nouns=["colour", "shade", "tint"],            vp="is",            suf="",             gen=lambda r: _pick(r, COLORS)),
    dict(key="material", pool="obj",  nouns=["material", "make", "fabric"],         vp="is made of",    suf="",             gen=lambda r: _pick(r, MATERIALS)),
    dict(key="weight",   pool="obj",  nouns=["weight", "mass"],                     vp="weighs",        suf=" grams",       gen=lambda r: _num(r, 50, 9950)),
    dict(key="price",    pool="obj",  nouns=["price", "cost", "value"],             vp="costs",         suf=" coins",       gen=lambda r: _num(r, 10, 9999)),
    dict(key="height",   pool="obj",  nouns=["height"],                             vp="stands",        suf=" cm tall",     gen=lambda r: _num(r, 10, 400)),
    dict(key="capacity", pool="obj",  nouns=["capacity", "volume"],                 vp="holds",         suf=" litres",      gen=lambda r: _num(r, 2, 900)),
    dict(key="serial",   pool="obj",  nouns=["serial", "reference", "catalogue number"], vp="is stamped", suf="",           gen=_code),
    dict(key="label",    pool="obj",  nouns=["label", "marking", "call sign"],      vp="is labelled",   suf="",             gen=_phrase),
    dict(key="store",    pool="obj",  nouns=["store", "cupboard", "storage place"], vp="is kept in the", suf="",            gen=lambda r: _pick(r, PLACES)),
    dict(key="maker",    pool="obj",  nouns=["maker", "builder"],                   vp="was made by",   suf="",             gen=lambda r: _pick(r, NAMES)),
    dict(key="city",     pool="name", nouns=["city", "town", "home"],               vp="lives in",      suf="",             gen=lambda r: _pick(r, CITIES)),
    dict(key="job",      pool="name", nouns=["job", "trade", "occupation"],         vp="works as a",    suf="",             gen=lambda r: _pick(r, JOBS)),
    dict(key="age",      pool="name", nouns=["age"],                                vp="is",            suf=" years old",   gen=lambda r: _num(r, 5, 95)),
    dict(key="pet",      pool="name", nouns=["pet", "animal"],                      vp="has a pet",     suf="",             gen=lambda r: _pick(r, ANIMALS)),
    dict(key="lang",     pool="name", nouns=["language", "tongue"],                 vp="speaks",        suf="",             gen=lambda r: _pick(r, LANGS)),
    dict(key="motto",    pool="name", nouns=["motto", "watchword", "byword"],       vp="always says",   suf="",             gen=_phrase),
    dict(key="badge",    pool="name", nouns=["badge", "staff number"],              vp="carries badge", suf="",             gen=_code),
    dict(key="room",     pool="name", nouns=["room", "office"],                     vp="sits in room",  suf="",             gen=lambda r: _num(r, 100, 899)),
    dict(key="partner",  pool="name", nouns=["partner", "colleague"],               vp="works with",    suf="",             gen=lambda r: _pick(r, NAMES)),
    dict(key="birth",    pool="name", nouns=["birth year", "year of birth"],        vp="was born in",   suf="",             gen=lambda r: _num(r, 1900, 2011)),
    dict(key="nickname", pool="name", nouns=["nickname", "moniker"],                vp="is nicknamed",  suf="",             gen=_phrase),
    dict(key="codename", pool="obj",  nouns=["codename", "project name"],           vp="goes by the codename", suf="",      gen=_phrase),
]

STMT_FRAMES = [
    "{E} {VP} {v}.",
    "According to the record, {E} {VP} {v}.",
    "The {N} of {E} is {v}.",
    "{E}, whose {N} is {v}, was listed last.",
    "It was noted that {E} {VP} {v}.",
    "The ledger gives the {N} of {E} as {v}.",
    "{E} {VP} {v}, the clerk confirmed.",
    "Records show {E} {VP} {v}.",
    "{E} -- {N}: {v}.",
    '"{E} {VP} {v}," the clerk wrote.',
    "For {E} the {N} was set at {v}.",
    "Nobody disputed that {E} {VP} {v}.",
]
# Mid-sentence cues: the answer is the CONTINUATION.  factgen_task must not wrap these in Q:/A:.
CUE_MID = [
    "{E} {VP}",
    "The {N} of {E} is",
    "{E}'s {N} is",
    "The {N} {E} has is",
    "Asked about {E}, the clerk said the {N} was",
    "In the record the {N} of {E} comes to",
    "{E} -- {N}:",
    "Looking it up, the {N} of {E} turned out to be",
    "The register lists the {N} of {E} as",
]
# Question cues: factgen_task wraps these as  Q: <cue> A: <answer>
CUE_Q = [
    "What is the {N} of {E}?",
    "What {N} does {E} have?",
    "{E} has what {N}?",
    "Can you give the {N} of {E}?",
]
QUERY_FRAMES = CUE_MID + CUE_Q
_N_MID = len(CUE_MID)

_SALT = "memlm-2026-09-29"

def _is_eval(rel_key, si, qi):
    h = hashlib.sha1(("%s|%d|%d|%s" % (rel_key, si, qi, _SALT)).encode()).digest()
    return h[0] < 51        # ~20%

def _fill(frame, E, VP, N, v=None):
    s = frame.replace("{E}", E).replace("{VP}", VP).replace("{N}", N)
    return s.replace("{v}", v) if v is not None else s

def _cap(s):
    # capitalise the first LETTER, not the first character: quoted frames begin with '"', which left
    # '"the ostay goes by the codename ..."' starting lowercase
    for i, ch in enumerate(s):
        if ch.isalpha():
            return s[:i] + ch.upper() + s[i + 1:]
    return s

def is_question(qi):
    return qi >= _N_MID

def _degenerate(si, qi):
    """Cue is merely a PREFIX of the statement -> literal copying, teaches nothing."""
    left = _fill(STMT_FRAMES[si].split("{v}")[0], "the widget", "is", "colour").strip()
    q = _fill(QUERY_FRAMES[qi], "the widget", "is", "colour").strip()
    return left == q

_COMBOS = {"train": [], "eval": []}
for _ri, _rel in enumerate(RELATIONS):
    for _si in range(len(STMT_FRAMES)):
        for _qi in range(len(QUERY_FRAMES)):
            if _degenerate(_si, _qi):
                continue
            _COMBOS["eval" if _is_eval(_rel["key"], _si, _qi) else "train"].append((_ri, _si, _qi))
assert _COMBOS["train"] and _COMBOS["eval"], "empty split"
# INVENTED relations are fresh per example, so they cannot be split by relation key; split their
# (stmt frame, cue frame) PAIRS instead, under a fixed pseudo-key.
_NCOMBOS = {"train": [], "eval": []}
for _si in range(len(STMT_FRAMES)):
    for _qi in range(len(QUERY_FRAMES)):
        if not _degenerate(_si, _qi):
            _NCOMBOS["eval" if _is_eval("~invented", _si, _qi) else "train"].append((None, _si, _qi))
assert _NCOMBOS["train"] and _NCOMBOS["eval"], "empty invented split"

def _entity(r, rel, p_fresh=0.25):
    if rel["pool"] == "obj":
        e = _fresh_object(r) if r.random() < p_fresh else _pick(r, OBJECTS)
        return "the " + e, e
    e = _fresh_name(r) if r.random() < p_fresh else _pick(r, NAMES)
    return e, e

# Anchoring is frame-determined, so resolve it in O(1) instead of realising strings per candidate.
# The last word before the value (and the last word of the cue) is always one of: the verb phrase's
# last word, the relation noun, or a literal.  _TAIL classifies which, once, at import.
_VPM, _NM = "\x01", "\x02"

def _tail_kind(frag):
    t = _fill(frag, "X", _VPM, _NM).strip()
    if not t:
        return ("LIT", "")
    w = t.split()[-1]
    if _VPM in w: return ("VP", w.replace(_VPM, ""))
    if _NM in w:  return ("N", w.replace(_NM, ""))
    return ("LIT", w)

_STAIL = [_tail_kind(f.split("{v}")[0]) for f in STMT_FRAMES]
_QTAIL = [_tail_kind(f) for f in QUERY_FRAMES]

def _resolve(kind_suffix, VP, N):
    kind, extra = kind_suffix
    if kind == "VP": return VP.split()[-1] + extra
    if kind == "N":  return N.split()[-1] + extra
    return extra

def _anchor_of(si, qi, VP, N):
    return _resolve(_STAIL[si], VP, N) == _resolve(_QTAIL[qi], VP, N)

def sample(r, split="train", anchor_p=0.5, tries=8, pool=None, q_frac=0.25,
           stmt_entity=None, cue_entity=None, relation=None, p_novel=0.6,
           R_force=None, exclude_key=None, exclude_entity=None):
    """-> (statement, cue, answer, (entity,), meta).

    anchor_p  target fraction of cues sharing an ANCHOR with their fact (cue's last token equals the
              fact's pre-value token).  Enforced by FILTERING the candidate frames, not by rejection
              sampling: earlier loops delivered 0.74 when 1.00 was asked, because whether a frame
              pair can anchor at all is fixed by the frames and no number of retries changes it.
    p_novel   fraction using an INVENTED relation (noun x verb phrase x value kind).  The curated
              table has 20 relations; the eval asks about one the model has never seen, so relation
              breadth -- not surface breadth -- is what generalisation depends on.
    relation  force a curated relation key (used for hard negatives that share it)."""
    want_a = r.random() < anchor_p
    want_q = r.random() < q_frac

    # R_force reuses an EXISTING relation object, curated or invented.  Hard negatives that must
    # share the target's relation need this: looking an invented relation up by key among the 20
    # curated ones found nothing and silently substituted a random one.
    if R_force is not None:
        R = R_force
        novel = str(R["key"]).startswith("~")
    else:
        novel = (relation is None) and (r.random() < p_novel)
        R = None
        if novel:
            for _t in range(8):
                R = _novel_relation(r)
                if exclude_key is None or R["key"] != exclude_key:
                    break
            if pool is not None:
                R["pool"] = pool
        if R is None:
            pick = [x for x in range(len(RELATIONS))
                    if (pool is None or RELATIONS[x]["pool"] == pool)
                    and (relation is None or RELATIONS[x]["key"] == relation)
                    and (exclude_key is None or RELATIONS[x]["key"] != exclude_key)]
            R = RELATIONS[_pick(r, pick)] if pick else RELATIONS[int(r.integers(0, len(RELATIONS)))]
    VP = R["vp"]

    # The frame pair must come from THIS relation's own split.  The previous version picked the
    # relation first and then drew frames from the whole split list without matching the relation,
    # so a combination reserved as "eval for relation A" was trained on under relation B --
    # 398 curated combinations appeared in both splits.
    if novel:
        pool_c = _NCOMBOS[split]
    else:
        _rix = next(i for i, x in enumerate(RELATIONS) if x is R)
        pool_c = [c for c in _COMBOS[split] if c[0] == _rix]
    base = [c for c in pool_c if is_question(c[2]) == want_q] or pool_c
    # (combo, noun) pairs that DELIVER the requested anchoring
    good = [(c, n) for c in base for n in R["nouns"] if _anchor_of(c[1], c[2], VP, n) == want_a]
    if not good:
        good = [(c, n) for c in base for n in R["nouns"]]
    (_ri_unused, si, qi), N = good[int(r.integers(0, len(good)))]

    E, ent = _entity(r, R)
    for _t in range(16):                      # a different entity when asked (hard negatives)
        if exclude_entity is None or ent != exclude_entity:
            break
        E, ent = _entity(r, R)
    Es = stmt_entity if stmt_entity is not None else E
    Ec = cue_entity if cue_entity is not None else E
    v = str(R["gen"](r))
    stmt = _cap(_fill(STMT_FRAMES[si], Es, VP, N, v + R["suf"]))
    cue = _cap(_fill(QUERY_FRAMES[qi], Ec, VP, N))
    a = _anchor_of(si, qi, VP, N)
    meta = dict(rel=R["key"], si=si, qi=qi, anchored=a, split=split,
                question=is_question(qi), novel=novel,
                E=Es, ent=ent if stmt_entity is None else None, R=R, pool=R["pool"])
    return stmt, cue, v, (ent,), meta
