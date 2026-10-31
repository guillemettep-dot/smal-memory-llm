# -*- coding: utf-8 -*-
"""The rebuilt `code` task makes GPT-2 INCREMENT: 14/20 misses are the stored code + 1.  Cause
hypothesis: the fact is the document\x27s opening line and the cue repeats it VERBATIM, so the pair
reads as an enumerated list and the model continues the series arithmetically.  That is why the
supposedly harder `paraphrase` (reworded cue, same fact) scores 95.8% while `code` scores 56.2%.

Three variants, same trials, same seed, one question each:
  V0 current            -- reproduce the trap
  V1 preamble           -- neutral sentences BEFORE the fact, so it is not the opening line
  V2 alphanumeric code  -- a code arithmetic cannot continue
  V3 preamble + alnum   -- both
"""
import os, numpy as np, torch, tiktoken
from transformers import GPT2LMHeadModel
enc = tiktoken.get_encoding("gpt2"); torch.set_num_threads(int(os.environ.get("THREADS","2")))
NAMES = ["Dmitri","Chiara","Raj","Hana","Yara","Ana","Zane","Aisha","Pedro","Maria","Xena","Greta",
         "Nadia","Omar","Lena","Ivan","Sofia","Tomas","Ines","Kofi"]
PLACE = ["shelf","drawer","cabinet","locker","alcove","crate"]
LET   = "KRTVMJQZ"
def filler(rng, n):
    out = []
    while sum(len(x) for x in out) < n:
        i, j = int(rng.integers(0, len(NAMES))), int(rng.integers(0, len(PLACE)))
        out.append(enc.encode(" The %s was moved near the %s that morning." % (NAMES[i], PLACE[j])))
    return np.array([t for s in out for t in s][:n], dtype=np.int64)
m = GPT2LMHeadModel.from_pretrained("gpt2").eval(); m.requires_grad_(False)
SEED = int(os.environ.get("SEED","20260823")); N = int(os.environ.get("N","96")); D = 64
E = lambda s: np.array(enc.encode(s), dtype=np.int64)

@torch.no_grad()
def sc(ctx, ans):
    seq = np.concatenate([ctx, ans]); k = len(ans)
    g = m(torch.tensor(seq[None])[:, :-1]).logits[0, -k:].argmax(-1).tolist()
    return float(np.mean([x==y for x,y in zip(g, ans.tolist())])), g == ans.tolist()

print("CODE TRAP TEST | gpt2 | %d paired trials | gap %d" % (N, D), flush=True)
print("  %-18s %14s %16s %8s" % ("variant","tok saw (ctl)","EXACT saw (ctl)","off-by-1"))
for var in ("V0_current","V1_preamble","V2_alnum","V3_both"):
    ts=tc=es=ec=0; plus1=0
    for t in range(N):
        rng = np.random.default_rng([SEED, t])
        i = int(rng.integers(0,len(NAMES))); j = (i+1+int(rng.integers(0,len(NAMES)-1))) % len(NAMES)
        n_, m_ = NAMES[i], NAMES[j]
        a, b = int(rng.integers(1000,9999)), int(rng.integers(1000,9999))
        if var in ("V2_alnum","V3_both"):
            ca = "%s%d" % (LET[a % len(LET)], a); cb = "%s%d" % (LET[b % len(LET)], b)
        else:
            ca, cb = "%d" % a, "%d" % b
        f  = "\nThe access code for %s is %s.\n" % (n_, ca)
        of = "\nThe access code for %s is %s.\n" % (m_, cb)
        cue = "\nThe access code for %s is" % n_; ans = " %s." % ca
        pre = E("\nThe office moved to the east wing last spring and the records were reboxed.\n") \
              if var in ("V1_preamble","V3_both") else np.array([], dtype=np.int64)
        mid = filler(rng, D)
        A = sc(np.concatenate([pre, E(f),  mid, E(cue)]), E(ans))
        B = sc(np.concatenate([pre, E(of), mid, E(cue)]), E(ans))
        ts+=A[0]; es+=A[1]; tc+=B[0]; ec+=B[1]
        if not A[1] and var in ("V0_current","V1_preamble"):
            seq = np.concatenate([pre, E(f), mid, E(cue)])
            x = torch.tensor(seq[None]); o=[]
            with torch.no_grad():
                for _ in range(len(E(ans))):
                    nx = int(m(x).logits[0,-1].argmax()); o.append(nx); x = torch.cat([x, torch.tensor([[nx]])],1)
            plus1 += int(enc.decode(o).strip().rstrip(".") == str(a+1))
    print("  %-18s  %5.1f%% (%4.1f%%)   %6.1f%% (%4.1f%%)  %5d" %
          (var, 100*ts/N, 100*tc/N, 100*es/N, 100*ec/N, plus1), flush=True)
