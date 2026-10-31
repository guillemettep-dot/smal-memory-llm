# -*- coding: utf-8 -*-
"""BATTERY v2 -- rebuild of the lost battery.py. Four tasks, paired control, three gaps.

WHY IT IS BUILT THIS WAY. The original battery died with the pod. The factgen by-description eval
that replaced it turned out to be useless: GPT-2 124M scores 18.6% on it and gpt2-large 20.6%, both
with ~0% exact match, because the task gives NO answer convention -- a bare question mark, then the
answer is expected as the next tokens. A base LM continues with prose instead.

So every task here ends its cue MID-SENTENCE, so the answer is the only sensible continuation:
    stored : "The access code for Dmitri is 4821."
    code   : "The access code for Dmitri is"      -> 4821      (verbatim cue)
    para   : "Dmitri's access code is"            -> 4821      (SAME fact, reworded cue)
That is the only difference between code and paraphrase: whether the cue matches the stored wording.
It is the cleanest form of the question this whole project is about.

CONTROL: the same trial with a DIFFERENT fact in the slot -- identical cue, identical answer tokens,
so any gap above the control is retrieval of the fact actually shown.

VALIDATION GATE: run with GPT2=gpt2 first. A full-attention model that keeps the whole context must
score high on every task. Any task GPT-2 cannot do is a broken task, not a finding -- that check is
what was missing before.
"""
import os, sys, math, torch, numpy as np, importlib, inspect, tiktoken, json
sys.path.insert(0, "."); os.environ.setdefault("MEMPROBE_ROOT", os.path.expanduser("~/memprobe"))
enc = tiktoken.get_encoding("gpt2")
torch.set_num_threads(int(os.environ.get("THREADS", "5")))

NAMES = ["Dmitri","Chiara","Raj","Hana","Yara","Ana","Zane","Aisha","Pedro","Maria","Xena","Greta",
         "Nadia","Omar","Lena","Ivan","Sofia","Tomas","Ines","Kofi"]
ADJ   = ["silver","crimson","hollow","northern","quiet","amber","distant","velvet"]
NOUN  = ["falcon","lantern","harbor","meadow","cipher","anchor","thistle","beacon"]
COLOR = ["blue","green","black","amber","scarlet","ivory","violet","olive"]
PLACE = ["shelf","drawer","cabinet","locker","alcove","crate"]

def filler(rng, n):
    """Neutral text so the gap is real tokens, not padding."""
    out = []
    while sum(len(x) for x in out) < n:
        i, j = int(rng.integers(0, len(NAMES))), int(rng.integers(0, len(PLACE)))
        out.append(enc.encode(" The %s was moved near the %s that morning." % (NAMES[i], PLACE[j])))
    return np.array([t for s in out for t in s][:n], dtype=np.int64)

def trial(t, task):
    """Returns (fact, other_fact, cue, answer) for trial t. Depends only on (SEED, t, task)."""
    rng = np.random.default_rng([SEED, t])
    i = int(rng.integers(0, len(NAMES))); j = (i + 1 + int(rng.integers(0, len(NAMES) - 1))) % len(NAMES)
    n, m = NAMES[i], NAMES[j]
    if task in ("code", "paraphrase"):
        a, b = int(rng.integers(1000, 9999)), int(rng.integers(1000, 9999))
        f  = "\nThe access code for %s is %d.\n" % (n, a)
        of = "\nThe access code for %s is %d.\n" % (m, b)
        cue = ("\nThe access code for %s is" % n) if task == "code" else ("\n%s's access code is" % n)
        ans = " %d." % a
    elif task == "phrase":
        p = (ADJ[int(rng.integers(0,len(ADJ)))], NOUN[int(rng.integers(0,len(NOUN)))])
        q = (ADJ[int(rng.integers(0,len(ADJ)))], NOUN[int(rng.integers(0,len(NOUN)))])
        f  = "\nThe password for %s is %s %s.\n" % (n, p[0], p[1])
        of = "\nThe password for %s is %s %s.\n" % (m, q[0], q[1])
        cue = "\nThe password for %s is" % n
        ans = " %s %s." % (p[0], p[1])
    else:  # prose: a fact in ordinary text, asked in different words
        c, pl = COLOR[int(rng.integers(0,len(COLOR)))], PLACE[int(rng.integers(0,len(PLACE)))]
        c2    = COLOR[(COLOR.index(c) + 1 + int(rng.integers(0,len(COLOR)-1))) % len(COLOR)]
        f  = "\n%s keeps a %s notebook in the %s.\n" % (n, c, pl)
        of = "\n%s keeps a %s notebook in the %s.\n" % (m, c2, pl)
        cue = "\nThe colour of the notebook %s keeps is" % n
        ans = " %s." % c
    E = lambda s: np.array(enc.encode(s), dtype=np.int64)
    return E(f), E(of), E(cue), E(ans), rng

TASKS = os.environ.get("TASKS", "code,phrase,paraphrase,prose").split(",")
DISTS = [int(x) for x in os.environ.get("DISTS", "64,256,1024").split(",")]
N     = int(os.environ.get("N", "96"))
SEED  = int(os.environ.get("SEED", "20260823"))

G = os.environ.get("GPT2", "")
if G:
    from transformers import GPT2LMHeadModel
    _m = GPT2LMHeadModel.from_pretrained(G).eval(); _m.requires_grad_(False)
    fwd = lambda x: _m(x).logits; tag = G; CTX = _m.config.n_positions
else:
    M = importlib.import_module(os.environ.get("MEMLM_MOD", "memlm_kv"))
    import cpu_shim; cpu_shim.install(M)
    st = torch.load(os.environ["CKPT"], map_location="cpu", weights_only=False)
    ok = set(inspect.signature(M.Cfg.__init__).parameters) - {"self"}
    _m = M.MemLM(M.Cfg(**{k: v for k, v in st["cfg"].items() if k in ok}))
    miss, unex = _m.load_state_dict(st["model"], strict=False)
    assert not miss and not unex, "checkpoint/model mismatch: %d missing, %d unexpected" % (len(miss), len(unex))
    _m = _m.float().eval(); _m.requires_grad_(False)
    try: _m.set_moe_loop(False)
    except Exception: pass
    fwd = lambda x: _m(x); tag = "step %s" % st["step"]; CTX = 10**9
print("BATTERY v2 | model: %s | tasks %s | gaps %s | %d paired trials | seed %d"
      % (tag, ",".join(TASKS), DISTS, N, SEED), flush=True)

@torch.no_grad()
def score(ctx, ans):
    seq = np.concatenate([ctx, ans])
    assert len(seq) - 1 <= CTX, "context overrun %d > %d -- the room() clamp should have prevented this" % (len(seq) - 1, CTX)
    lg = fwd(torch.tensor(seq[None])[:, :-1]).float()
    k = len(ans)
    g = lg[0, -k:].argmax(-1).tolist()
    return float(np.mean([x == y for x, y in zip(g, ans.tolist())])), g == ans.tolist()

res = {}
for task in TASKS:
    for d in DISTS:
        ts = tc = es = ec = n = 0; gsum = 0
        for t in range(N):
            f, of, cue, ans, rng = trial(t, task)
            # FIX: at gap 1024 the trial overran GPT-2's 1024-token window, score() returned None
            # and EVERY trial was skipped -- the row printed 0.0% at n=0.  Clamp the filler to what
            # the window leaves and REPORT the gap actually realised; a probe that silently serves a
            # shorter context than the number it prints is a failure this project has paid for once.
            # max(len(f), len(of)) keeps the filler length IDENTICAL for saw and control, so the
            # pair stays matched, and the short-gap context stays a strict prefix of the long one.
            room = CTX + 1 - max(len(f), len(of)) - len(cue) - len(ans)
            de = min(d, room)
            if de < 0: continue
            mid = filler(rng, de)
            a = score(np.concatenate([f,  mid, cue]), ans)
            b = score(np.concatenate([of, mid, cue]), ans)
            ts += a[0]; es += a[1]; tc += b[0]; ec += b[1]; n += 1; gsum += de
        gm = int(round(gsum / max(n, 1)))
        res[(task, d)] = (100*ts/max(n,1), 100*tc/max(n,1), 100*es/max(n,1), 100*ec/max(n,1), n, gm)
        print("  %-11s gap %-5d(real %4d) tok %5.1f%% (ctl %5.1f%%)  exact %5.1f%% (ctl %5.1f%%)  n=%d"
              % (task, d, gm, *res[(task, d)][:5]), flush=True)

print()
print("  %-11s %22s %22s" % ("task", "token acc  saw / ctl", "exact  saw / ctl"))
for task in TASKS:
    v = [res[(task, d)] for d in DISTS]
    print("  %-11s %10.1f%% / %-8.1f%% %10.1f%% / %-8.1f%%"
          % (task, np.mean([x[0] for x in v]), np.mean([x[1] for x in v]),
             np.mean([x[2] for x in v]), np.mean([x[3] for x in v])))
out = os.environ.get("OUT", "")
if out:
    json.dump({"%s|%d" % k: v for k, v in res.items()}, open(out, "w"), indent=1)
    print("  -> %s" % out)
