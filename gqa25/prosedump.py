# -*- coding: utf-8 -*-
"""What does GPT-2 actually emit on the prose cue?  The sweep says the cue is unanswerable even at
gap 8; before redesigning it, look at the failures instead of theorising about them."""
import os, numpy as np, torch, tiktoken
from transformers import GPT2LMHeadModel
enc = tiktoken.get_encoding("gpt2"); torch.set_num_threads(1)
NAMES = ["Dmitri","Chiara","Raj","Hana","Yara","Ana","Zane","Aisha","Pedro","Maria","Xena","Greta",
         "Nadia","Omar","Lena","Ivan","Sofia","Tomas","Ines","Kofi"]
COLOR = ["blue","green","black","amber","scarlet","ivory","violet","olive"]
PLACE = ["shelf","drawer","cabinet","locker","alcove","crate"]
CUES = {
 "A_current": "\nThe colour of the notebook %s keeps is",
 "E_qa":      "\nQ: What colour is %s\x27s notebook? A:",
 "F_whose":   "\n%s keeps a notebook whose colour is",
 "I_restate": "\n%s keeps a notebook. Its colour is",
 "K_amer":    "\nThe color of %s\x27s notebook is",
}
m = GPT2LMHeadModel.from_pretrained("gpt2").eval(); m.requires_grad_(False)
SEED = 20260823
for name, pat in CUES.items():
    print("=== %s ===" % name)
    for t in range(6):
        rng = np.random.default_rng([SEED, t])
        i = int(rng.integers(0, len(NAMES))); _ = (i + 1 + int(rng.integers(0, len(NAMES)-1))) % len(NAMES)
        n_ = NAMES[i]
        c  = COLOR[int(rng.integers(0, len(COLOR)))]; pl = PLACE[int(rng.integers(0, len(PLACE)))]
        ctx = "\n%s keeps a %s notebook in the %s.\n" % (n_, c, pl) + (pat % n_)
        x = torch.tensor([enc.encode(ctx)])
        with torch.no_grad():
            out = []
            for _s in range(5):
                nxt = int(m(x).logits[0, -1].argmax()); out.append(nxt)
                x = torch.cat([x, torch.tensor([[nxt]])], 1)
        print("   want %-9r got %r" % (" " + c, enc.decode(out)))
