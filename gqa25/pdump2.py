# -*- coding: utf-8 -*-
"""prose, take 3.  GPT-2 answers "a red circle" / "with a picture of a" to a cue built on `marked`
or `labelled`, because those words ask for a DESCRIPTION.  `password` works (phrase = 89.6%) because
it signals that an arbitrary token follows and must be copied.  So keep a slot-noun, but put the
answer MID-sentence in narrative and reword the cue -- which is what separates prose from phrase.

Scored, not just dumped: each variant gets the paired control, so a high number cannot come from a
prior.  ansTok must land near 3 and the control near 33% to match the original prose signature.
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

# (fact template(name,adj,noun,place), cue template(name))
VAR={
 "Q1_wrote":   ("\n%s wrote the passphrase %s %s inside a notebook kept in the %s.\n",
                "\nThe passphrase %s wrote is"),
 "Q2_chose":   ("\n%s chose %s %s as the passphrase for the %s.\n",
                "\nThe passphrase %s chose is"),
 "Q3_wasgiven":("\n%s was given the passphrase %s %s before leaving the %s.\n",
                "\nThe passphrase given to %s is"),
 "Q4_uses":    ("\nThe passphrase %s uses is %s %s, noted in a book in the %s.\n",
                "\n%s uses the passphrase"),
}
@torch.no_grad()
def sc(ctx,ans):
    seq=np.concatenate([ctx,ans]); k=len(ans)
    g=m(torch.tensor(seq[None])[:,:-1]).logits[0,-k:].argmax(-1).tolist()
    return float(np.mean([x==y for x,y in zip(g,ans.tolist())])), g==ans.tolist()
print("PROSE take 3 | gpt2 | %d paired trials | gap %d | target exact ~79%%, ctl ~33%%"%(N,D),flush=True)
print("  %-13s %6s %14s %16s"%("variant","ansTok","tok saw (ctl)","EXACT saw (ctl)"))
for name,(ff,cc) in VAR.items():
    ts=tc=es=ec=0; nt=0
    for t in range(N):
        rng=np.random.default_rng([SEED,t])
        i=int(rng.integers(0,len(NAMES))); j=(i+1+int(rng.integers(0,len(NAMES)-1)))%len(NAMES)
        n_,m_=NAMES[i],NAMES[j]
        a1=ADJ[int(rng.integers(0,len(ADJ)))]; n1=NOUN[int(rng.integers(0,len(NOUN)))]
        a2=ADJ[int(rng.integers(0,len(ADJ)))]; n2=NOUN[int(rng.integers(0,len(NOUN)))]
        pl=PLACE[int(rng.integers(0,len(PLACE)))]
        if name=="Q4_uses":
            f=ff%(n_,a1,n1,pl); of=ff%(m_,a2,n2,pl)
        else:
            f=ff%(n_,a1,n1,pl); of=ff%(m_,a2,n2,pl)
        cue=cc%n_; ans=" %s %s."%(a1,n1); nt+=len(E(ans))
        mid=filler(rng,D)
        A=sc(np.concatenate([E(f),mid,E(cue)]),E(ans)); B=sc(np.concatenate([E(of),mid,E(cue)]),E(ans))
        ts+=A[0]; es+=A[1]; tc+=B[0]; ec+=B[1]
    print("  %-13s %6.2f  %5.1f%% (%4.1f%%)   %6.1f%% (%4.1f%%)"%(name,nt/N,100*ts/N,100*tc/N,100*es/N,100*ec/N),flush=True)
