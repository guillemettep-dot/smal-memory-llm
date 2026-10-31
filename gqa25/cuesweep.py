# -*- coding: utf-8 -*-
"""Cue sweep for the battery's `prose` task.

WHY. battery2's prose cue -- "The colour of the notebook Dmitri keeps is" -- gives GPT-2 124M only
24.0% exact at gap 64, where the SAME model gets 89.6-95.8% on phrase and paraphrase, and where the
lost original battery recorded 79.2% on prose.  A model holding the fact VERBATIM in its context
cannot be failing for lack of the fact, so the cue's English is the defect, not the retrieval.

WHAT THIS DECIDES AND WHAT IT DOES NOT.  It picks, among grammatical phrasings of the SAME question,
one that a model which HAS the fact can actually read.  That is instrument calibration: every model
then faces the identical cue.  It is not tuning toward GPT-2 -- the criterion is answerability given
the fact.  gap 8 is the decisive column: there the fact is essentially adjacent to the cue, so a low
score there is phrasing alone and nothing else.
"""
import os, numpy as np, torch, tiktoken
from transformers import GPT2LMHeadModel
enc = tiktoken.get_encoding("gpt2")
torch.set_num_threads(int(os.environ.get("THREADS", "3")))

NAMES = ["Dmitri","Chiara","Raj","Hana","Yara","Ana","Zane","Aisha","Pedro","Maria","Xena","Greta",
         "Nadia","Omar","Lena","Ivan","Sofia","Tomas","Ines","Kofi"]
COLOR = ["blue","green","black","amber","scarlet","ivory","violet","olive"]
PLACE = ["shelf","drawer","cabinet","locker","alcove","crate"]

# candidate cues, all asking the same question of the same stored sentence
CUES = {
 "A_current":  "\nThe colour of the notebook %s keeps is",
 "B_possess":  "\nThe colour of %s's notebook is",
 "C_bare":     "\n%s's notebook is",
 "D_relative": "\nThe notebook that %s keeps is",
 "E_qa":       "\nQ: What colour is %s's notebook? A:",
}

def filler(rng, n):
    out = []
    while sum(len(x) for x in out) < n:
        i, j = int(rng.integers(0, len(NAMES))), int(rng.integers(0, len(PLACE)))
        out.append(enc.encode(" The %s was moved near the %s that morning." % (NAMES[i], PLACE[j])))
    return np.array([t for s in out for t in s][:n], dtype=np.int64)

SEED = int(os.environ.get("SEED", "20260823")); N = int(os.environ.get("N", "96"))
m = GPT2LMHeadModel.from_pretrained("gpt2").eval(); m.requires_grad_(False)

@torch.no_grad()
def score(ctx, ans):
    seq = np.concatenate([ctx, ans]); k = len(ans)
    g = m(torch.tensor(seq[None])[:, :-1]).logits[0, -k:].argmax(-1).tolist()
    return float(np.mean([x == y for x, y in zip(g, ans.tolist())])), g == ans.tolist()

print("PROSE CUE SWEEP | gpt2 | %d paired trials | seed %d" % (N, SEED), flush=True)
print("  %-11s %5s %14s %16s" % ("cue", "gap", "tok saw (ctl)", "EXACT saw (ctl)"))
for name, pat in CUES.items():
    for d in (8, 64):
        ts = tc = es = ec = 0
        for t in range(N):
            # identical draw ORDER to battery2.trial(), so the facts are the same trials
            rng = np.random.default_rng([SEED, t])
            i = int(rng.integers(0, len(NAMES))); j = (i + 1 + int(rng.integers(0, len(NAMES)-1))) % len(NAMES)
            n_, m_ = NAMES[i], NAMES[j]
            c  = COLOR[int(rng.integers(0, len(COLOR)))]; pl = PLACE[int(rng.integers(0, len(PLACE)))]
            c2 = COLOR[(COLOR.index(c) + 1 + int(rng.integers(0, len(COLOR)-1))) % len(COLOR)]
            E = lambda s: np.array(enc.encode(s), dtype=np.int64)
            f  = E("\n%s keeps a %s notebook in the %s.\n" % (n_, c,  pl))
            of = E("\n%s keeps a %s notebook in the %s.\n" % (m_, c2, pl))
            cue, ans = E(pat % n_), E(" %s." % c)
            mid = filler(rng, d)
            a = score(np.concatenate([f,  mid, cue]), ans)
            b = score(np.concatenate([of, mid, cue]), ans)
            ts += a[0]; es += a[1]; tc += b[0]; ec += b[1]
        print("  %-11s %5d  %5.1f%% (%4.1f%%)   %6.1f%% (%4.1f%%)"
              % (name, d, 100*ts/N, 100*tc/N, 100*es/N, 100*ec/N), flush=True)
