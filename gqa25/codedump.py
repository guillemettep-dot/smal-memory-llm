# -*- coding: utf-8 -*-
"""GPT-2 fails 44% of a VERBATIM copy on the rebuilt `code` task (56.2% exact at gap 64) where the
original battery recorded 96.9%.  A full-attention model cannot be failing to SEE the fact, so the
defect is in the task text or the scoring.  Look at the failures."""
import os, numpy as np, torch, tiktoken
from transformers import GPT2LMHeadModel
enc = tiktoken.get_encoding("gpt2"); torch.set_num_threads(1)
NAMES = ["Dmitri","Chiara","Raj","Hana","Yara","Ana","Zane","Aisha","Pedro","Maria","Xena","Greta",
         "Nadia","Omar","Lena","Ivan","Sofia","Tomas","Ines","Kofi"]
PLACE = ["shelf","drawer","cabinet","locker","alcove","crate"]
def filler(rng, n):
    out = []
    while sum(len(x) for x in out) < n:
        i, j = int(rng.integers(0, len(NAMES))), int(rng.integers(0, len(PLACE)))
        out.append(enc.encode(" The %s was moved near the %s that morning." % (NAMES[i], PLACE[j])))
    return np.array([t for s in out for t in s][:n], dtype=np.int64)
m = GPT2LMHeadModel.from_pretrained("gpt2").eval(); m.requires_grad_(False)
SEED = 20260823
bad = 0
for t in range(20):
    rng = np.random.default_rng([SEED, t])
    i = int(rng.integers(0, len(NAMES))); j = (i + 1 + int(rng.integers(0, len(NAMES)-1))) % len(NAMES)
    n_ = NAMES[i]
    a, b = int(rng.integers(1000, 9999)), int(rng.integers(1000, 9999))
    f   = "\nThe access code for %s is %d.\n" % (n_, a)
    cue = "\nThe access code for %s is" % n_
    ans = " %d." % a
    mid = filler(rng, 64)
    seq = np.concatenate([np.array(enc.encode(f)), mid, np.array(enc.encode(cue))])
    at = enc.encode(ans)
    x = torch.tensor(seq[None])
    with torch.no_grad():
        out = []
        for _s in range(len(at) + 2):
            nxt = int(m(x).logits[0, -1].argmax()); out.append(nxt)
            x = torch.cat([x, torch.tensor([[nxt]])], 1)
    got = enc.decode(out[:len(at)])
    ok = out[:len(at)] == at
    bad += (not ok)
    print("  %-3s want %-8r (%d tok %s)  got %-10r  full %r" % ("OK" if ok else "MISS", ans, len(at), at, got, enc.decode(out)))
print("  -> %d/20 miss" % bad)
