# -*- coding: utf-8 -*-
"""BATTERY v3 -- the repaired recall battery.  Four tasks, paired control, any gaps.

Measures one thing: a fact is placed in context, then queried after `gap` tokens of unrelated text.
CONTROL = the identical trial with a DIFFERENT fact in the slot, so the cue and the answer tokens are
byte-identical and any advantage is retrieval of the fact actually shown.

WHAT v2 GOT WRONG, ALL FOUND BY VALIDATING AGAINST GPT-2 RATHER THAN TRUSTING THE SCORES
-----------------------------------------------------------------------------------------
1. THE FILLER WAS AN ENUMERATED LIST.  v2 filled the gap by repeating one frame,
   " The <name> was moved near the <place> that morning.", with only the nouns swapped.  Six
   near-identical sentences read as a numbered list, so GPT-2 CONTINUED THE SERIES: on `code` it
   answered the stored number PLUS ONE on 42 of 96 trials -- every single miss, not most of them --
   and scored 56.2% where the original battery had 96.9%.  Varied filler alone: 94.8%, off-by-1
   42 -> 5.  A preamble before the fact also "helped" (56.2 -> 71.9) but only by diluting the
   template; with a proper filler it HURTS (94.8 -> 92.7), so it is not used here.  The lesson is
   that the filler is not inert padding -- its STRUCTURE is part of the prompt.
2. THE FILLER LEAKED THE TARGET NAME.  It drew names from the same pool as the fact, so the queried
   name appeared inside the gap on 25% / 70% / 99.2% of trials at gaps 64 / 256 / 1024.  Saw and
   control share the filler so the delta survived, but interference then rose WITH DISTANCE -- the
   one axis the benchmark exists to vary.  Filler names here come from a disjoint pool.
3. GAP 1024 SILENTLY MEASURED NOTHING.  fact + 1024 filler + cue overran GPT-2\x27s 1024-token window,
   every trial was skipped and the row printed 0.0% at n=0.  Here the filler is clamped to what the
   window leaves and the row REPORTS THE GAP IT ACTUALLY SERVED.
4. A FIX VALIDATED AT ONE GAP IS NOT VALIDATED.  The varied filler above was a fixed pool of ten
   sentences: fine at gap 64 (six needed) but at gap 1024 it needs ~95 and each repeats ~9 times,
   which rebuilds the very list that caused defect 1.  So the filler here is COMBINATORIAL -- frames
   x slots -- and _selfcheck() asserts at the largest gap requested that no frame dominates and the
   target name never appears.  Run it at every gap you intend to report.

ANCHORS -- WHY THERE IS NO NARRATIVE `prose` TASK.  Fourteen designs were measured against GPT-2.
One rule predicts every one of them: retrieval succeeds exactly when the CUE'S LAST TOKEN equals the
token that precedes the answer inside the FACT, because what does the work is an induction head
matching the answer's local left-context.

    fact "...for X is silver falcon."   cue "...X's password is"        anchor is=is        92.7%
    fact "...wrote the passphrase A B inside..."  cue "...wrote the passphrase"  match  100.0%
    fact "...wrote the passphrase A B inside..."  cue "The passphrase X wrote is"  MISMATCH  0.0%
    fact "...settled on the passphrase A B."      cue "...settled on is"          MISMATCH  0.0%

A factorial separated the two things every failing design changed at once: a reworded cue ALONE is
fine (92.7%), a mid-sentence answer ALONE is fine (100.0%), and both together give 0.0%.  It is an
interaction, not a main effect.  So an anchor-free "by description" task is one GPT-2 itself cannot
do, and building one would produce a column where every model reads zero -- which is precisely how
the factgen by-description eval came to be saturated and uninformative.  Note what this costs
`paraphrase`: it is doable only because its reworded cue still ends in `is`, the same anchor.

VALIDATION GATE: run with GPT2=gpt2 first.  A full-attention model keeping the whole context must
score high on every task at every gap.  Any cell it cannot do is a broken task, not a finding.
"""
import os, sys, math, json, inspect, importlib
import numpy as np, torch, tiktoken
sys.path.insert(0, "/home/paul/memprobe/gqa")
os.environ.setdefault("MEMPROBE_ROOT", "/home/paul/memprobe")
enc = tiktoken.get_encoding("gpt2")
torch.set_num_threads(int(os.environ.get("THREADS", "4")))

# ---- vocab.  FNAME/FPLACE/FTHING are used ONLY by the filler and are disjoint from the fact vocab,
# ---- so filler text can never contain the queried name (defect 2).
NAMES = ["Dmitri","Chiara","Raj","Hana","Yara","Ana","Zane","Aisha","Pedro","Maria","Xena","Greta",
         "Nadia","Omar","Lena","Ivan","Sofia","Tomas","Ines","Kofi"]
ADJ   = ["silver","crimson","hollow","northern","quiet","amber","distant","velvet"]
NOUN  = ["falcon","lantern","harbor","meadow","cipher","anchor","thistle","beacon"]
PLACE = ["shelf","drawer","cabinet","locker","alcove","crate"]

FNAME = ["Bertrand","Sylvie","Halvor","Margit","Cosimo","Adaeze","Ruaridh","Mirela","Teodor",
         "Anneke","Feodor","Brigid","Lysander","Ottilie","Casimir","Perpetua"]
FPLACE= ["annexe","stairwell","loading bay","reading room","courtyard","basement","east wing",
         "print room","canteen","archive"]
FTHING= ["kettle","skylight","catalogue","fire door","photocopier","noticeboard","radiator",
         "window latch","filing cabinet","coffee urn","clock","carpet"]
FRAMES = [
 " The {thing} in the {place} had been awkward since the winter and nobody reported it.",
 " Rain came through the {place} again, so everything was stacked against the far wall.",
 " A courier arrived before eight and left the parcels with whoever answered the door.",
 " {name} preferred the quieter end of the {place}, away from the {thing}.",
 " Someone had propped the door of the {place} open, which the warden disliked.",
 " Lunch ran late because the lift was being serviced and the stairs were crowded.",
 " The notice about the car park stayed up long after the works had finished.",
 " Two of the lamps flickered whenever the heating came on in the early morning.",
 " {name} counted the inventory twice and the totals still did not agree.",
 " The {thing} was replaced in spring, though nobody could say who had ordered it.",
 " Deliveries to the {place} were rerouted while the resurfacing went on.",
 " {name} left a note on the {thing} asking for the spare key to be returned.",
 " Nobody had used the {place} since the reorganisation, and the light stayed off.",
 " The {thing} needed a new fuse, which took three weeks to arrive.",
]

def _sentence(rng):
    k = int(rng.integers(0, len(FRAMES)))
    return FRAMES[k].format(name=FNAME[int(rng.integers(0, len(FNAME)))],
                            place=FPLACE[int(rng.integers(0, len(FPLACE)))],
                            thing=FTHING[int(rng.integers(0, len(FTHING)))]), k

def filler(rng, n, frames_out=None):
    """`n` tokens of varied, non-repeating text.  Combinatorial, so it does not degenerate into a
    list at long gaps (defect 1/4).  Returns exactly n tokens.  `frames_out`, if given, collects the
    frame index of each sentence so _selfcheck can measure diversity EXACTLY instead of matching
    frame prefixes -- several frames begin " The", so a prefix count would be meaningless."""
    out = []; tot = 0
    while tot < n:
        txt, k = _sentence(rng)
        s = enc.encode(txt); out.append(s); tot += len(s)
        if frames_out is not None: frames_out.append(k)
    return np.array([t for s in out for t in s][:n], dtype=np.int64)

E = lambda s: np.array(enc.encode(s), dtype=np.int64)

# prose is NOT validated yet.  Every design tried so far scores <=25% for GPT-2 -- a model that
# holds the fact verbatim -- so a prose column would measure the task, not the memory.  It stays
# opt-in behind an explicit PROSE=<design> and is refused otherwise, rather than defaulting to a
# design that quietly produces a meaningless number.
PROSE = os.environ.get("PROSE", "")
_PROSE_VALIDATED = set()          # add a design here only once GPT-2 clears ~79% on it

def trial(t, task):
    """(fact, other_fact, cue, answer, rng).  Depends only on (SEED, t, task), so trial t is
    byte-identical across checkpoints AND the short-gap context is a strict prefix of the long one."""
    rng = np.random.default_rng([SEED, t])
    i = int(rng.integers(0, len(NAMES)))
    j = (i + 1 + int(rng.integers(0, len(NAMES) - 1))) % len(NAMES)
    n, m = NAMES[i], NAMES[j]
    if task in ("code", "paraphrase"):
        a, b = int(rng.integers(1000, 9999)), int(rng.integers(1000, 9999))
        f  = "\nThe access code for %s is %d.\n" % (n, a)
        of = "\nThe access code for %s is %d.\n" % (m, b)
        cue = ("\nThe access code for %s is" % n) if task == "code" else ("\n%s\x27s access code is" % n)
        ans = " %d." % a
    elif task in ("phrase", "pairpara"):
        # `pairpara` is `phrase` with the cue REWORDED, exactly as `paraphrase` is `code` reworded.
        # Together the four tasks are a 2x2: {verbatim, reworded} cue x {number, word-pair} answer.
        # It replaces the v2 `prose` column, which could not be reconstructed: every narrative design
        # tried scored 0% for GPT-2, and the factorial showed why -- see the note on ANCHORS below.
        p = (ADJ[int(rng.integers(0,len(ADJ)))], NOUN[int(rng.integers(0,len(NOUN)))])
        q = (ADJ[int(rng.integers(0,len(ADJ)))], NOUN[int(rng.integers(0,len(NOUN)))])
        f  = "\nThe password for %s is %s %s.\n" % (n, p[0], p[1])
        of = "\nThe password for %s is %s %s.\n" % (m, q[0], q[1])
        cue = ("\nThe password for %s is" % n) if task == "phrase" else ("\n%s\x27s password is" % n)
        ans = " %s %s." % (p[0], p[1])
    else:   # prose: fact in ordinary narrative, answer MID-sentence, query reworded
        a1, n1 = ADJ[int(rng.integers(0,len(ADJ)))], NOUN[int(rng.integers(0,len(NOUN)))]
        a2, n2 = ADJ[int(rng.integers(0,len(ADJ)))], NOUN[int(rng.integers(0,len(NOUN)))]
        pl = PLACE[int(rng.integers(0,len(PLACE)))]
        f  = "\n%s keeps a notebook marked %s %s in the %s.\n" % (n, a1, n1, pl)
        of = "\n%s keeps a notebook marked %s %s in the %s.\n" % (m, a2, n2, pl)
        cue = "\nThe marking on the notebook %s keeps is" % n
        ans = " %s %s." % (a1, n1)
    return E(f), E(of), E(cue), E(ans), rng

TASKS = os.environ.get("TASKS", "code,phrase,paraphrase,pairpara").split(",")
if "prose" in TASKS:
    assert PROSE in _PROSE_VALIDATED, (
        "prose design %r is not validated: GPT-2 must clear ~79%% exact on it first (see rtest.py). "
        "Validated designs: %s" % (PROSE, sorted(_PROSE_VALIDATED) or "none yet"))
DISTS = [int(x) for x in os.environ.get("DISTS", "64,256,1024").split(",")]
N     = int(os.environ.get("N", "96"))
SEED  = int(os.environ.get("SEED", "20260823"))

# ---- model
G = os.environ.get("GPT2", "")
if G:
    from transformers import GPT2LMHeadModel
    _m = GPT2LMHeadModel.from_pretrained(G).eval(); _m.requires_grad_(False)
    fwd = lambda x: _m(x).logits; tag = G; CTX = _m.config.n_positions
else:
    M = importlib.import_module(os.environ.get("MEMLM_MOD", "memlm_kv"))
    import cpu_shim; cpu_shim.install(M)
    st = torch.load(os.environ["CKPT"], map_location="cpu", weights_only=False)
    ok = set(inspect.signature(M.Cfg.__init__).parameters) - {"self"}
    _m = M.MemLM(M.Cfg(**{k: v for k, v in st["cfg"].items() if k in ok}))
    miss, unex = _m.load_state_dict(st["model"], strict=False)
    assert not miss and not unex, "checkpoint/model mismatch: %d missing, %d unexpected" % (len(miss), len(unex))
    _m = _m.float().eval(); _m.requires_grad_(False)
    try: _m.set_moe_loop(False)
    except Exception: pass
    fwd = lambda x: _m(x); tag = "step %s" % st["step"]; CTX = 10**9

def _selfcheck():
    """Assert the filler is still varied AND name-clean at the LARGEST gap actually requested --
    defect 4 was a fix that held at gap 64 and quietly failed at 1024."""
    d = max(DISTS); worst_frac = 0.0; worst_ratio = 0.0; leaks = 0; nsent = 0
    # A FIXED share threshold cannot work here: at gap 64 only ~5 sentences fit, so one frame being
    # 3 of 5 (60%) is ordinary sampling, while at gap 1024 (~65 sentences) 60% would be real
    # degeneration.  Calibrate against what uniform draws of the SAME size actually produce.
    _r0 = np.random.default_rng(12345); _exp = {}
    def _exp_max(n):
        if n not in _exp:
            ms = [np.bincount(_r0.integers(0, len(FRAMES), size=n), minlength=len(FRAMES)).max() / n
                  for _ in range(400)]
            _exp[n] = float(np.quantile(ms, 0.99))
        return _exp[n]
    for t in range(min(N, 48)):
        for task in TASKS:
            _f, _of, _c, _a, rng = trial(t, task)
            idx = []
            txt = enc.decode(filler(rng, d, idx).tolist())
            counts = np.bincount(np.array(idx, dtype=np.int64), minlength=len(FRAMES))
            n_ = max(len(idx), 1); frac = counts.max() / n_
            worst_frac = max(worst_frac, frac); nsent = max(nsent, n_)
            worst_ratio = max(worst_ratio, frac / max(_exp_max(n_), 1e-9))
            # check EVERY target name, not only this one: FNAME is meant to be disjoint from NAMES,
            # so a single hit anywhere means the two pools overlap.
            for nm in NAMES:
                leaks += int((" %s " % nm) in txt or ("\n%s " % nm) in txt or (" %s." % nm) in txt)
    assert leaks == 0, "FILLER LEAKS THE TARGET NAME in %d samples (defect 2 regressed)" % leaks
    # THREE independent asserts, because the ratio alone is blind to the defect it exists to catch:
    # _exp_max is computed from len(FRAMES), so a one-frame pool makes uniform draws 100% too and
    # the ratio is 1.00 by construction.  Verified: with FRAMES collapsed to one entry the ratio
    # check printed "100% of filler (1.00x uniform) OK".
    assert len(FRAMES) >= 12, ("FILLER POOL TOO SMALL: %d frames.  This is the v2 defect itself -- "
        "one repeated frame made GPT-2 continue the series and answer stored+1." % len(FRAMES))
    assert nsent < 20 or worst_frac <= 0.35, ("FILLER DEGENERATE at gap %d: one frame is %.0f%% of "
        "%d sentences -- that is a list, whatever the pool size (defect 1/4)" % (d, 100*worst_frac, nsent))
    assert worst_ratio <= 1.5, ("FILLER SAMPLING NON-UNIFORM at gap %d: most common frame is %.0f%% "
        "of %d sentences, %.2fx the 99th percentile of uniform draws of that size"
        % (d, 100 * worst_frac, nsent, worst_ratio))
    print("  selfcheck: gap %d -- %d filler sentences, target-name leaks 0, most common frame "
          "%.0f%% of filler (%.2fx uniform)  OK" % (d, nsent, 100 * worst_frac, worst_ratio), flush=True)

print("BATTERY v3 | model: %s | tasks %s | gaps %s | %d paired trials | seed %d | prose=%s"
      % (tag, ",".join(TASKS), DISTS, N, SEED, PROSE), flush=True)
_selfcheck()

@torch.no_grad()
def score(ctx, ans):
    seq = np.concatenate([ctx, ans])
    assert len(seq) - 1 <= CTX, "context overrun %d > %d -- the room clamp should have prevented this" % (len(seq)-1, CTX)
    lg = fwd(torch.tensor(seq[None])[:, :-1]).float()
    k = len(ans)
    g = lg[0, -k:].argmax(-1).tolist()
    return float(np.mean([x == y for x, y in zip(g, ans.tolist())])), g == ans.tolist()

res = {}
print("  %-11s %-16s %16s %18s %5s" % ("task", "gap (realised)", "tok saw (ctl)", "EXACT saw (ctl)", "n"))
for task in TASKS:
    for d in DISTS:
        ts = tc = es = ec = n = 0; gsum = 0
        # per-trial exact-match flags.  Trials are byte-identical across checkpoints (same seed), so
        # two runs can be compared with an EXACT paired McNemar test instead of two binomial CIs --
        # aggregates alone forced a "3.5-5.2 sigma" range where a p-value was available for free.
        sv = []; cv = []
        for t in range(N):
            f, of, cue, ans, rng = trial(t, task)
            # identical filler length for saw and control so the pair stays matched, clamped to the
            # model window so a long gap cannot silently empty the cell (defect 3)
            room = CTX + 1 - max(len(f), len(of)) - len(cue) - len(ans)
            de = min(d, room)
            if de < 0: continue
            mid = filler(rng, de)
            a = score(np.concatenate([f,  mid, cue]), ans)
            b = score(np.concatenate([of, mid, cue]), ans)
            ts += a[0]; es += a[1]; tc += b[0]; ec += b[1]; n += 1; gsum += de
            sv.append(int(a[1])); cv.append(int(b[1]))
        gm = int(round(gsum / max(n, 1)))
        res["%s|%d" % (task, d)] = dict(tok=100*ts/max(n,1), tok_ctl=100*tc/max(n,1),
                                        exact=100*es/max(n,1), exact_ctl=100*ec/max(n,1),
                                        n=n, gap_req=d, gap_real=gm,
                                        saw_trials=sv, ctl_trials=cv)
        r = res["%s|%d" % (task, d)]
        print("  %-11s %5d (real %5d)   %5.1f%% (%4.1f%%)   %6.1f%% (%4.1f%%) %5d"
              % (task, d, gm, r["tok"], r["tok_ctl"], r["exact"], r["exact_ctl"], n), flush=True)

out = os.environ.get("OUT", "")
if out:
    json.dump(res, open(out, "w"), indent=1); print("  -> %s" % out)
