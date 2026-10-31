# -*- coding: utf-8 -*-
"""Does OUR model fall into the same increment trap as GPT-2?

GPT-2 answers stored_code+1 on 42 of 96 `code` trials, so its 56.2% is not a retrieval score.  Our
gqa2k scores 83.3% on the same task.  If its 16 misses are also +1, our number is trapped too and
the comparison is meaningless in both directions; if they are unstructured, only GPT-2 was trapped
-- and then 83.3% vs 56.2% must NOT be reported as our model beating GPT-2.

Prints the miss taxonomy, not just a rate.
"""
import os, sys, inspect, importlib, numpy as np, torch, tiktoken
sys.path.insert(0, "/home/paul/memprobe/gqa")
os.environ.setdefault("MEMPROBE_ROOT", "/home/paul/memprobe")
enc = tiktoken.get_encoding("gpt2"); torch.set_num_threads(int(os.environ.get("THREADS","4")))
NAMES=["Dmitri","Chiara","Raj","Hana","Yara","Ana","Zane","Aisha","Pedro","Maria","Xena","Greta",
       "Nadia","Omar","Lena","Ivan","Sofia","Tomas","Ines","Kofi"]
PLACE=["shelf","drawer","cabinet","locker","alcove","crate"]
def filler(rng,n):
    out=[]
    while sum(len(x) for x in out)<n:
        i,j=int(rng.integers(0,len(NAMES))),int(rng.integers(0,len(PLACE)))
        out.append(enc.encode(" The %s was moved near the %s that morning."%(NAMES[i],PLACE[j])))
    return np.array([t for s in out for t in s][:n],dtype=np.int64)
M=importlib.import_module(os.environ.get("MEMLM_MOD","memlm_kv"))
import cpu_shim; cpu_shim.install(M)
st=torch.load(os.environ["CKPT"],map_location="cpu",weights_only=False)
ok=set(inspect.signature(M.Cfg.__init__).parameters)-{"self"}
mdl=M.MemLM(M.Cfg(**{k:v for k,v in st["cfg"].items() if k in ok}))
miss,unex=mdl.load_state_dict(st["model"],strict=False)
assert not miss and not unex, "checkpoint/model mismatch"
mdl=mdl.float().eval(); mdl.requires_grad_(False)
try: mdl.set_moe_loop(False)
except Exception: pass
SEED=int(os.environ.get("SEED","20260823")); N=int(os.environ.get("N","96")); D=int(os.environ.get("D","64"))
E=lambda s: np.array(enc.encode(s),dtype=np.int64)
print("OUR-MODEL INCREMENT CHECK | step %s | %d trials | gap %d"%(st["step"],N,D),flush=True)
exact=0; plus1=0; minus1=0; other=[]
for t in range(N):
    rng=np.random.default_rng([SEED,t])
    i=int(rng.integers(0,len(NAMES))); _=(i+1+int(rng.integers(0,len(NAMES)-1)))%len(NAMES)
    n_=NAMES[i]
    a,b=int(rng.integers(1000,9999)),int(rng.integers(1000,9999))
    f="\nThe access code for %s is %d.\n"%(n_,a); cue="\nThe access code for %s is"%n_; ans=" %d."%a
    mid=filler(rng,D); at=E(ans)
    x=torch.tensor(np.concatenate([E(f),mid,E(cue)])[None]); o=[]
    with torch.no_grad():
        for _s in range(len(at)):
            nx=int(mdl(x)[0,-1].argmax()); o.append(nx); x=torch.cat([x,torch.tensor([[nx]])],1)
    got=enc.decode(o)
    if o==at.tolist(): exact+=1; continue
    g=got.strip().rstrip(".")
    if g==str(a+1): plus1+=1
    elif g==str(a-1): minus1+=1
    else: other.append((a,got))
print("  exact      %3d/%d = %.1f%%"%(exact,N,100*exact/N))
print("  stored+1   %3d   <- the GPT-2 trap"%plus1)
print("  stored-1   %3d"%minus1)
print("  other      %3d"%len(other))
for a,g in other[:12]: print("     want %-6d got %r"%(a,g))
