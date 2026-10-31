# -*- coding: utf-8 -*-
"""GPT-2's `code` cell reads 92.7 / 65.6 / 95.8 at gaps 64 / 256 / 1024 -- a dip only at 256, which
reproduced at n=16 (56.2%) as well as n=96, so it is a property of the cell, not sampling.  It is in
the BASELINE column every memory number is compared against, so identify it rather than discard it.
Reuses battery3's own trial() and filler() so this is the identical task, not a lookalike."""
import os, sys, importlib.util, numpy as np, torch, tiktoken
os.environ["GPT2"] = "gpt2"; os.environ["TASKS"] = "code"; os.environ["DISTS"] = "256"
os.environ["N"] = "0"                      # build the module without running the scoring loop
sys.path.insert(0, "/home/paul/memprobe/gqa"); os.chdir("/home/paul/memprobe/gqa")
spec = importlib.util.spec_from_file_location("b3", "/home/paul/memprobe/gqa/battery3.py")
b3 = importlib.util.module_from_spec(spec); spec.loader.exec_module(b3)
enc = b3.enc
print("\n--- GPT-2 failures on code @ gap 256, battery3's own trials ---")
plus1 = other = exact = 0
for t in range(48):
    f, of, cue, ans, rng = b3.trial(t, "code")
    mid = b3.filler(rng, 256)
    seq = np.concatenate([f, mid, cue])
    x = torch.tensor(seq[None]); o = []
    with torch.no_grad():
        for _ in range(len(ans)):
            nx = int(b3.fwd(x)[0, -1].argmax()); o.append(nx); x = torch.cat([x, torch.tensor([[nx]])], 1)
    want = enc.decode(ans.tolist()); got = enc.decode(o)
    if o == ans.tolist(): exact += 1; continue
    try:
        w = int(want.strip().rstrip(".")); g = int(got.strip().rstrip("."))
        if g == w + 1: plus1 += 1; continue
    except Exception: pass
    other += 1
    if other <= 8: print("   want %-9r got %r" % (want, got))
print("   exact %d/48 | stored+1 %d | other %d" % (exact, plus1, other))
