# -*- coding: utf-8 -*-
"""prose v2 candidates.

The rebuilt prose task answers with a COLOUR: eight options and a crushing prior, which GPT-2 falls
back on instead of reading the context (dumped: it answers " blue." for amber, violet and scarlet
alike).  Its control is 49% because the answer is two tokens and the period is free.

The ORIGINAL battery\x27s prose had GPT-2 at 79.2/83.3/80.2% exact with a control of 36.1/31.8/28.0.
A ~33% control is the signature of a THREE-token answer with no usable prior -- the same signature as
`code` (34.0%).  So the original answer was a multi-token low-prior item, not a colour.

Candidates keep what prose is FOR -- a fact stated in ordinary prose, queried in different words,
with the answer sitting mid-sentence rather than at the end -- and only raise the answer entropy.
"""
import os, numpy as np, torch, tiktoken
from transformers import GPT2LMHeadModel
enc = tiktoken.get_encoding("gpt2"); torch.set_num_threads(int(os.environ.get("THREADS","2")))
NAMES = ["Dmitri","Chiara","Raj","Hana","Yara","Ana","Zane","Aisha","Pedro","Maria","Xena","Greta",
         "Nadia","Omar","Lena","Ivan","Sofia","Tomas","Ines","Kofi"]
ADJ   = ["silver","crimson","hollow","northern","quiet","amber","distant","velvet"]
NOUN  = ["falcon","lantern","harbor","meadow","cipher","anchor","thistle","beacon"]
COLOR = ["blue","green","black","amber","scarlet","ivory","violet","olive"]
PLACE = ["shelf","drawer","cabinet","locker","alcove","crate"]
def filler(rng, n):
    out = []
    while sum(len(x) for x in out) < n:
        i, j = int(rng.integers(0,len(NAMES))), int(rng.integers(0,len(PLACE)))
        out.append(enc.encode(" The %s was moved near the %s that morning." % (NAMES[i], PLACE[j])))
    return np.array([t for s in out for t in s][:n], dtype=np.int64)
m = GPT2LMHeadModel.from_pretrained("gpt2").eval(); m.requires_grad_(False)
SEED=int(os.environ.get("SEED","20260823")); N=int(os.environ.get("N","96"))
E = lambda s: np.array(enc.encode(s), dtype=np.int64)
@torch.no_grad()
def sc(ctx, ans):
    seq=np.concatenate([ctx,ans]); k=len(ans)
    g=m(torch.tensor(seq[None])[:,:-1]).logits[0,-k:].argmax(-1).tolist()
    return float(np.mean([x==y for x,y in zip(g,ans.tolist())])), g==ans.tolist()

def build(var, rng):
    i=int(rng.integers(0,len(NAMES))); j=(i+1+int(rng.integers(0,len(NAMES)-1)))%len(NAMES)
    n_,m_=NAMES[i],NAMES[j]
    a1=ADJ[int(rng.integers(0,len(ADJ)))];  n1=NOUN[int(rng.integers(0,len(NOUN)))]
    a2=ADJ[int(rng.integers(0,len(ADJ)))];  n2=NOUN[int(rng.integers(0,len(NOUN)))]
    pl=PLACE[int(rng.integers(0,len(PLACE)))]
    c =COLOR[int(rng.integers(0,len(COLOR)))]
    c2=COLOR[(COLOR.index(c)+1+int(rng.integers(0,len(COLOR)-1)))%len(COLOR)]
    if var=="P0_colour":
        f="\n%s keeps a %s notebook in the %s.\n"%(n_,c,pl); of="\n%s keeps a %s notebook in the %s.\n"%(m_,c2,pl)
        cue="\nThe colour of the notebook %s keeps is"%n_; ans=" %s."%c
    elif var=="P1_marking":
        f="\n%s keeps a notebook marked %s %s in the %s.\n"%(n_,a1,n1,pl)
        of="\n%s keeps a notebook marked %s %s in the %s.\n"%(m_,a2,n2,pl)
        cue="\nThe marking on the notebook %s keeps is"%n_; ans=" %s %s."%(a1,n1)
    elif var=="P2_codeword":
        f="\n%s keeps a notebook under the codeword %s %s in the %s.\n"%(n_,a1,n1,pl)
        of="\n%s keeps a notebook under the codeword %s %s in the %s.\n"%(m_,a2,n2,pl)
        cue="\nThe codeword on %s\x27s notebook is"%n_; ans=" %s %s."%(a1,n1)
    else:  # P3_place -- answer is the PLACE, mid-sentence fact, reworded query
        pl2=PLACE[(PLACE.index(pl)+1+int(rng.integers(0,len(PLACE)-1)))%len(PLACE)]
        f="\n%s keeps a %s notebook in the %s.\n"%(n_,c,pl); of="\n%s keeps a %s notebook in the %s.\n"%(m_,c2,pl2)
        cue="\nThe place where %s keeps a notebook is the"%n_; ans=" %s."%pl
    return f,of,cue,ans

print("PROSE v2 | gpt2 | %d paired trials | gap 64 | target: GPT-2 exact ~79%%, control ~33%%"%N, flush=True)
print("  %-13s %6s %14s %16s"%("variant","ansTok","tok saw (ctl)","EXACT saw (ctl)"))
for var in ("P0_colour","P1_marking","P2_codeword","P3_place"):
    ts=tc=es=ec=0; ntok=0
    for t in range(N):
        rng=np.random.default_rng([SEED,t])
        f,of,cue,ans=build(var,rng); mid=filler(rng,64); ntok+=len(E(ans))
        A=sc(np.concatenate([E(f), mid,E(cue)]),E(ans)); B=sc(np.concatenate([E(of),mid,E(cue)]),E(ans))
        ts+=A[0]; es+=A[1]; tc+=B[0]; ec+=B[1]
    print("  %-13s %6.2f  %5.1f%% (%4.1f%%)   %6.1f%% (%4.1f%%)"%(var,ntok/N,100*ts/N,100*tc/N,100*es/N,100*ec/N), flush=True)
