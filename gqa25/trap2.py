# -*- coding: utf-8 -*-
"""Second trap hypothesis: the FILLER is a repeated template.

V1 (a preamble before the fact) cut the increment from 42/96 to 27/96 but every remaining miss is
still stored+1, so document position is not the driver.  The context between fact and cue is 64
tokens of " The <name> was moved near the <place> that morning." repeated with only the nouns
swapped -- a run of near-identical sentences IS an enumerated list, and would prime the model to
continue a series no matter where the fact sits.

  W0  templated filler (current)            -- baseline
  W1  varied natural filler                 -- same length, no repeated frame
  W2  varied filler + preamble
  W3  varied filler + preamble + alnum code
"""
import os, numpy as np, torch, tiktoken
from transformers import GPT2LMHeadModel
enc = tiktoken.get_encoding("gpt2"); torch.set_num_threads(int(os.environ.get("THREADS","2")))
NAMES=["Dmitri","Chiara","Raj","Hana","Yara","Ana","Zane","Aisha","Pedro","Maria","Xena","Greta",
       "Nadia","Omar","Lena","Ivan","Sofia","Tomas","Ines","Kofi"]
PLACE=["shelf","drawer","cabinet","locker","alcove","crate"]
LET="KRTVMJQZ"
SENT=[" The kettle in the staff room had been broken since the winter and nobody reported it.",
      " Rain came through the skylight again, so the boxes were stacked against the far wall.",
      " A courier arrived before eight and left the parcels with whoever answered the door.",
      " The old catalogue was still useful for anything filed before the move to the annexe.",
      " Someone had propped the fire door open with a chair, which the warden disliked.",
      " Lunch was late because the lift was being serviced and the stairs were crowded.",
      " The notice about the car park stayed up long after the works had finished.",
      " Two of the lamps flickered whenever the heating came on in the early morning.",
      " He preferred the quieter end of the corridor, away from the photocopier.",
      " The inventory had been counted twice and the totals still did not agree."]
def filler_tmpl(rng,n):
    out=[]
    while sum(len(x) for x in out)<n:
        i,j=int(rng.integers(0,len(NAMES))),int(rng.integers(0,len(PLACE)))
        out.append(enc.encode(" The %s was moved near the %s that morning."%(NAMES[i],PLACE[j])))
    return np.array([t for s in out for t in s][:n],dtype=np.int64)
def filler_var(rng,n):
    out=[]; order=list(range(len(SENT))); rng.shuffle(order); k=0
    while sum(len(x) for x in out)<n:
        out.append(enc.encode(SENT[order[k%len(order)]])); k+=1
    return np.array([t for s in out for t in s][:n],dtype=np.int64)
m=GPT2LMHeadModel.from_pretrained("gpt2").eval(); m.requires_grad_(False)
SEED=int(os.environ.get("SEED","20260823")); N=int(os.environ.get("N","96")); D=64
E=lambda s: np.array(enc.encode(s),dtype=np.int64)
@torch.no_grad()
def sc(ctx,ans):
    seq=np.concatenate([ctx,ans]); k=len(ans)
    g=m(torch.tensor(seq[None])[:,:-1]).logits[0,-k:].argmax(-1).tolist()
    return float(np.mean([x==y for x,y in zip(g,ans.tolist())])), g==ans.tolist()
print("TRAP TEST 2 -- is the FILLER the list? | gpt2 | %d trials | gap %d"%(N,D),flush=True)
print("  %-22s %14s %16s %8s"%("variant","tok saw (ctl)","EXACT saw (ctl)","off-by-1"))
for var in ("W0_tmpl","W1_varied","W2_varied_pre","W3_varied_pre_alnum"):
    ts=tc=es=ec=0; p1=0
    for t in range(N):
        rng=np.random.default_rng([SEED,t])
        i=int(rng.integers(0,len(NAMES))); j=(i+1+int(rng.integers(0,len(NAMES)-1)))%len(NAMES)
        n_,m_=NAMES[i],NAMES[j]
        a,b=int(rng.integers(1000,9999)),int(rng.integers(1000,9999))
        if var=="W3_varied_pre_alnum": ca,cb="%s%d"%(LET[a%len(LET)],a),"%s%d"%(LET[b%len(LET)],b)
        else: ca,cb="%d"%a,"%d"%b
        f="\nThe access code for %s is %s.\n"%(n_,ca); of="\nThe access code for %s is %s.\n"%(m_,cb)
        cue="\nThe access code for %s is"%n_; ans=" %s."%ca
        pre=E("\nThe office moved to the east wing last spring and the records were reboxed.\n") \
            if var in ("W2_varied_pre","W3_varied_pre_alnum") else np.array([],dtype=np.int64)
        mid=(filler_tmpl if var=="W0_tmpl" else filler_var)(rng,D)
        A=sc(np.concatenate([pre,E(f),mid,E(cue)]),E(ans)); B=sc(np.concatenate([pre,E(of),mid,E(cue)]),E(ans))
        ts+=A[0]; es+=A[1]; tc+=B[0]; ec+=B[1]
        if not A[1]:
            x=torch.tensor(np.concatenate([pre,E(f),mid,E(cue)])[None]); o=[]
            with torch.no_grad():
                for _ in range(len(E(ans))):
                    nx=int(m(x).logits[0,-1].argmax()); o.append(nx); x=torch.cat([x,torch.tensor([[nx]])],1)
            g=enc.decode(o).strip().rstrip(".").lstrip(LET)
            p1+=int(g==str(a+1))
    print("  %-22s  %5.1f%% (%4.1f%%)   %6.1f%% (%4.1f%%)  %5d"%(var,100*ts/N,100*tc/N,100*es/N,100*ec/N,p1),flush=True)
