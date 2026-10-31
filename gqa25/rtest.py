# -*- coding: utf-8 -*-
"""prose, factorial.  Six prose designs scored 0-25% for GPT-2 while phrase (89.6%) and paraphrase
(95.8%) are fine.  Every one of those six changed TWO things at once against the working tasks:

    phrase      VERBATIM cue + word-pair answer at the END of the fact   89.6%
    paraphrase  REWORDED cue + NUMBER     answer at the END of the fact   95.8%
    prose (all) REWORDED cue + word-pair answer MID-sentence               0.0%

so "reworded cue" and "mid-sentence answer" are confounded and neither is convicted.  This crosses
them.  It also fixes a real defect in the mid-sentence designs: they scored the answer WITH a
trailing period, but mid-sentence the phrase is followed by " inside a notebook", so a model copying
perfectly still failed exact match.  Mid-sentence answers here carry no period.

  R1  reworded cue,  pair at END            -- isolates the reworded cue
  R2  verbatim cue,  pair MID (no period)   -- isolates the position
  R3  reworded cue,  pair MID (no period)   -- both, i.e. the design that kept failing
  R4  reworded cue,  pair at END, NARRATIVE fact -- the candidate, if R1 clears
"""
import os, numpy as np, torch, tiktoken
from transformers import GPT2LMHeadModel
enc = tiktoken.get_encoding("gpt2"); torch.set_num_threads(int(os.environ.get("THREADS","2")))
NAMES=["Dmitri","Chiara","Raj","Hana","Yara","Ana","Zane","Aisha","Pedro","Maria","Xena","Greta",
       "Nadia","Omar","Lena","Ivan","Sofia","Tomas","Ines","Kofi"]
ADJ=["silver","crimson","hollow","northern","quiet","amber","distant","velvet"]
NOUN=["falcon","lantern","harbor","meadow","cipher","anchor","thistle","beacon"]
PLACE=["shelf","drawer","cabinet","locker","alcove","crate"]
FNAME=["Bertrand","Sylvie","Halvor","Margit","Cosimo","Adaeze","Ruaridh","Mirela","Teodor",
       "Anneke","Feodor","Brigid","Lysander","Ottilie","Casimir","Perpetua"]
FPLACE=["annexe","stairwell","loading bay","reading room","courtyard","basement","east wing",
        "print room","canteen","archive"]
FTHING=["kettle","skylight","catalogue","fire door","photocopier","noticeboard","radiator",
        "window latch","filing cabinet","coffee urn","clock","carpet"]
FRAMES=[" The {thing} in the {place} had been awkward since the winter and nobody reported it.",
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
 " The {thing} needed a new fuse, which took three weeks to arrive."]
def filler(rng,n):
    out=[];tot=0
    while tot<n:
        fr=FRAMES[int(rng.integers(0,len(FRAMES)))]
        s=enc.encode(fr.format(name=FNAME[int(rng.integers(0,len(FNAME)))],
                               place=FPLACE[int(rng.integers(0,len(FPLACE)))],
                               thing=FTHING[int(rng.integers(0,len(FTHING)))]))
        out.append(s); tot+=len(s)
    return np.array([t for s in out for t in s][:n],dtype=np.int64)
E=lambda s: np.array(enc.encode(s),dtype=np.int64)
m=GPT2LMHeadModel.from_pretrained("gpt2").eval(); m.requires_grad_(False)
SEED=int(os.environ.get("SEED","20260823")); N=int(os.environ.get("N","96")); D=64

def build(var, n_, m_, a1, n1, a2, n2, pl):
    if var == "R1_reworded_END":
        f  = "\nThe passphrase for %s is %s %s.\n" % (n_, a1, n1)
        of = "\nThe passphrase for %s is %s %s.\n" % (m_, a2, n2)
        return f, of, "\n%s's passphrase is" % n_, " %s %s." % (a1, n1)
    if var == "R2_verbatim_MID":
        f  = "\n%s wrote the passphrase %s %s inside a notebook in the %s.\n" % (n_, a1, n1, pl)
        of = "\n%s wrote the passphrase %s %s inside a notebook in the %s.\n" % (m_, a2, n2, pl)
        return f, of, "\n%s wrote the passphrase" % n_, " %s %s" % (a1, n1)
    if var == "R3_reworded_MID":
        f  = "\n%s wrote the passphrase %s %s inside a notebook in the %s.\n" % (n_, a1, n1, pl)
        of = "\n%s wrote the passphrase %s %s inside a notebook in the %s.\n" % (m_, a2, n2, pl)
        return f, of, "\nThe passphrase %s wrote is" % n_, " %s %s" % (a1, n1)
    f  = "\nBefore the move, %s settled on the passphrase %s %s.\n" % (n_, a1, n1)
    of = "\nBefore the move, %s settled on the passphrase %s %s.\n" % (m_, a2, n2)
    return f, of, "\nThe passphrase %s settled on is" % n_, " %s %s." % (a1, n1)

@torch.no_grad()
def sc(ctx,ans):
    seq=np.concatenate([ctx,ans]); k=len(ans)
    g=m(torch.tensor(seq[None])[:,:-1]).logits[0,-k:].argmax(-1).tolist()
    return float(np.mean([x==y for x,y in zip(g,ans.tolist())])), g==ans.tolist()

print("PROSE FACTORIAL | gpt2 | %d paired trials | gap %d"%(N,D),flush=True)
print("  %-18s %6s %14s %16s"%("variant","ansTok","tok saw (ctl)","EXACT saw (ctl)"))
for var in ("R1_reworded_END","R2_verbatim_MID","R3_reworded_MID","R4_narrative_END"):
    ts=tc=es=ec=0; nt=0
    for t in range(N):
        rng=np.random.default_rng([SEED,t])
        i=int(rng.integers(0,len(NAMES))); j=(i+1+int(rng.integers(0,len(NAMES)-1)))%len(NAMES)
        n_,m_=NAMES[i],NAMES[j]
        a1=ADJ[int(rng.integers(0,len(ADJ)))]; n1=NOUN[int(rng.integers(0,len(NOUN)))]
        a2=ADJ[int(rng.integers(0,len(ADJ)))]; n2=NOUN[int(rng.integers(0,len(NOUN)))]
        pl=PLACE[int(rng.integers(0,len(PLACE)))]
        f,of,cue,ans=build(var,n_,m_,a1,n1,a2,n2,pl); nt+=len(E(ans))
        mid=filler(rng,D)
        A=sc(np.concatenate([E(f),mid,E(cue)]),E(ans)); B=sc(np.concatenate([E(of),mid,E(cue)]),E(ans))
        ts+=A[0]; es+=A[1]; tc+=B[0]; ec+=B[1]
    print("  %-18s %6.2f  %5.1f%% (%4.1f%%)   %6.1f%% (%4.1f%%)"%(var,nt/N,100*ts/N,100*tc/N,100*es/N,100*ec/N),flush=True)
