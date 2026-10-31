# -*- coding: utf-8 -*-
"""P1_marking scores 0.0% for GPT-2 though its answer has exactly the 3-token, low-prior shape the
original prose task had.  Look at what the model emits instead of guessing again."""
import numpy as np, torch, tiktoken
from transformers import GPT2LMHeadModel
enc = tiktoken.get_encoding("gpt2"); torch.set_num_threads(1)
NAMES=["Dmitri","Chiara","Raj","Hana","Yara","Ana","Zane","Aisha","Pedro","Maria","Xena","Greta",
       "Nadia","Omar","Lena","Ivan","Sofia","Tomas","Ines","Kofi"]
ADJ=["silver","crimson","hollow","northern","quiet","amber","distant","velvet"]
NOUN=["falcon","lantern","harbor","meadow","cipher","anchor","thistle","beacon"]
PLACE=["shelf","drawer","cabinet","locker","alcove","crate"]
m=GPT2LMHeadModel.from_pretrained("gpt2").eval(); m.requires_grad_(False)
SEED=20260823
VAR={
 "P1_marking":  ("\n%s keeps a notebook marked %s %s in the %s.\n", "\nThe marking on the notebook %s keeps is"),
 "P4_ends_on":  ("\n%s keeps a notebook marked %s %s in the %s.\n", "\nThe notebook that %s keeps is marked"),
 "P5_labelled": ("\n%s keeps a notebook labelled %s %s in the %s.\n", "\nThe label on %s's notebook reads"),
 "P6_passcode": ("\n%s wrote the passphrase %s %s inside a notebook.\n", "\nThe passphrase %s wrote is"),
}
for name,(ff,cc) in VAR.items():
    print("=== %s ==="%name)
    for t in range(6):
        rng=np.random.default_rng([SEED,t])
        i=int(rng.integers(0,len(NAMES))); _=(i+1+int(rng.integers(0,len(NAMES)-1)))%len(NAMES)
        n_=NAMES[i]
        a1=ADJ[int(rng.integers(0,len(ADJ)))]; n1=NOUN[int(rng.integers(0,len(NOUN)))]
        _a2=ADJ[int(rng.integers(0,len(ADJ)))]; _n2=NOUN[int(rng.integers(0,len(NOUN)))]
        pl=PLACE[int(rng.integers(0,len(PLACE)))]
        ctx=(ff%(n_,a1,n1,pl))+(cc%n_)
        x=torch.tensor([enc.encode(ctx)]); o=[]
        with torch.no_grad():
            for _s in range(5):
                nx=int(m(x).logits[0,-1].argmax()); o.append(nx); x=torch.cat([x,torch.tensor([[nx]])],1)
        print("   want %-20r got %r"%(" %s %s."%(a1,n1), enc.decode(o)))
