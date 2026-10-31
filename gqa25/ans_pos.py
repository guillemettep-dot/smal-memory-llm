"""
needle_hi.py -- high-power needle benchmark.  Rebuild of the script lost with the old pod.

Same estimand and the same strict control as needle.py:
    an ARBITRARY fact is placed in context, then queried after a gap of `dist` tokens.
    CONTROL = the SAME trial with a DIFFERENT needle in the slot, so any advantage is
    specifically retrieval of the fact that was actually seen, not a prior over 4-digit
    numbers and not dilution by easy tokens.

What this adds over needle.py (which is TRIALS=32, DISTS<=1024, means only):

  1. TRIALS=256 and DISTS out to 3072.
  2. FULLY PAIRED IN BOTH AXES.
     Trial content is drawn from default_rng([SEED, trial]) -- a per-trial stream that
     does NOT depend on the DISTS list, on TRIALS, or on evaluation order.  Therefore:
       * across checkpoints : trial t is byte-identical, so checkpoint deltas are paired
                              (the project's measured paired floor is ~1.5pp, NOT the
                              ~6pp unpaired binomial bar).
       * across distances   : trial t uses ONE filler stream and `mid = stream[:dist]`,
                              so the short-gap context is a strict PREFIX of the long-gap
                              one.  The gap length is the only thing that varies.
     needle.py instead draws from one sequential generator, so changing DISTS or TRIALS
     silently re-rolls every later distance.  This does not.
  3. Real statistics, no scipy:
       * Wilson intervals on the exact-match rates
       * exact McNemar (binomial) on saw-vs-control exact match, which is the paired test
       * paired bootstrap CI and p-value on the per-trial loss lift
       * Bonferroni alpha = 0.05/len(DISTS), printed and applied
  4. Per-trial results written to JSON so checkpoint-vs-checkpoint paired comparison can
     be done later WITHOUT re-running (see needle_cmp.py).
  5. Cfg is rebuilt by introspecting Cfg.__init__ and passing every key the checkpoint
     carries.  needle.py hardcodes a key list and omits heads/dv/gbias/qpaths, so it
     silently builds the WRONG model for moe500w.  Weights load strict=True; a mismatch
     aborts loudly instead of benchmarking a half-initialised model.

env: CKPT MEMPROBE_ROOT TRIALS DISTS SEED OUT BOOT ALLOW_MISSING TRAIN_MAXLEN
"""
import os, sys, glob, json, math, inspect
import numpy as np
import torch, torch.nn.functional as F
import tiktoken
import importlib as _il
_mm = _il.import_module(os.environ.get("MEMLM_MOD", "memlm"))   # MEMLM_MOD=memlm_ks for kslot checkpoints
Cfg, MemLM = _mm.Cfg, _mm.MemLM

DEV = os.environ.get("DEV", "cuda")
if DEV == "cpu":
    import cpu_shim; cpu_shim.install(_mm); torch.set_num_threads(int(os.environ.get("THREADS", "12")))
    print("CPU mode: Triton kernels replaced by torch fallbacks (cpu_shim), fp32 model", flush=True)
ROOT = os.environ.get("MEMPROBE_ROOT", os.path.expanduser("~/memprobe"))
DATA = os.path.join(ROOT, "data")
CK = os.path.expanduser(os.environ.get("CKPT", os.path.join(ROOT, "moe500w_ckpt.pt")))
SEED = int(os.environ.get("SEED", "20260823"))
TRIALS = int(os.environ.get("TRIALS", "256"))
DISTS = [int(x) for x in os.environ.get(
    "DISTS", "64,128,256,512,800,1024,1536,2048,3072").split(",")]
BOOT = int(os.environ.get("BOOT", "10000"))
TRAIN_MAXLEN = int(os.environ.get("TRAIN_MAXLEN", "2048"))   # last --stages value
ALPHA = 0.05 / len(DISTS)                                    # Bonferroni, project rule
MAXDIST = max(DISTS)

enc = tiktoken.get_encoding("gpt2")

# ---------------------------------------------------------------- model
st = torch.load(CK, map_location="cpu", weights_only=False)
_c = dict(st["cfg"])
accepted = set(inspect.signature(Cfg.__init__).parameters) - {"self"}
kw = {k: v for k, v in _c.items() if k in accepted}
dropped = sorted(set(_c) - set(kw))
model = MemLM(Cfg(**kw))
missing, unexpected = model.load_state_dict(st["model"], strict=False)
missing = [k for k in missing]
unexpected = [k for k in unexpected]
if missing or unexpected:
    ok = os.environ.get("ALLOW_MISSING", "")
    allow = set(x for x in ok.split(",") if x)
    bad_m = [k for k in missing if k.rsplit(".", 1)[-1] not in allow]
    bad_u = [k for k in unexpected if k.rsplit(".", 1)[-1] not in allow]
    if bad_m or bad_u:
        print("FATAL: checkpoint/model mismatch -- refusing to benchmark a wrong model.")
        print(f"  cfg keys used     : {sorted(kw)}")
        print(f"  cfg keys IGNORED  : {dropped}   <-- Cfg does not accept these")
        print(f"  missing  ({len(bad_m)}): {bad_m[:12]}")
        print(f"  unexpected ({len(bad_u)}): {bad_u[:12]}")
        print("  If a key is a benign buffer, re-run with ALLOW_MISSING=bbias")
        sys.exit(1)
    print(f"note: tolerated missing={missing} unexpected={unexpected}", flush=True)
model = (model.to(DEV).to(torch.bfloat16) if DEV == "cuda" else model.to(DEV).float()).eval()
model.requires_grad_(False)
try:
    model.set_moe_loop(False)          # MANDATORY: else the per-expert python loop hangs
except Exception:
    pass
STEP = int(st["step"])
del st

def _g(k, d=None):
    return _c.get(k, d)

print(f"checkpoint {CK}")
print(f"step {STEP}  heads={_g('heads',8)} dk={_g('dk')} dv={_g('dv',96)} "
      f"qpaths={_g('qpaths',1)} nlk={_g('nlk',0)} moe={_g('moe',0)} "
      f"state/layer={_g('heads',8)*_g('dk')*_g('dv',96)}")
if dropped:
    print(f"note: cfg keys not accepted by this Cfg (ignored): {dropped}")
print(f"trials={TRIALS}  seed={SEED}  dists={DISTS}")
print(f"Bonferroni alpha = 0.05/{len(DISTS)} = {ALPHA:.4f}\n", flush=True)

# ---------------------------------------------------------------- data
idx = np.load(os.path.join(DATA, "docindex.npz"))
d_shard, d_start, d_len = idx["shard"], idx["start"], idx["length"]
held = (d_shard == 0) & (d_start < 10_000_000)          # held-out slice, same as needle.py
val = np.flatnonzero(held & (d_len >= 1200))
assert len(val) > 0, "no held-out docs found"
files = sorted(glob.glob(os.path.join(DATA, "shard_*.npy")))
mm = [np.load(f, mmap_mode="r") for f in files]
print(f"held-out pool: {len(val)} docs across {len(files)} shards", flush=True)

NAMES = ["Zarnak", "Quilvex", "Brommel", "Tavish", "Yondor", "Pelmoth", "Grivex", "Sordane"]


def filler(rng, n):
    """EXACTLY n held-out tokens, concatenating docs when one is too short.

    needle.py does mm[sh][s:s+n] on a pool filtered to len>=1200, which silently
    returns FEWER than n tokens once n>1200 -- that would corrupt the distance axis
    at 1536/2048/3072, the very range this script exists to measure.
    """
    out, got, tries = [], 0, 0
    while got < n:
        tries += 1
        if tries > 64 + 8 * n // 1200:
            raise RuntimeError(f"filler stuck at {got}/{n}: held-out pool yields empty "
                               f"segments (check docindex.npz against the shard files)")
        d = int(val[rng.integers(0, len(val))])
        s, L = int(d_start[d]), int(d_len[d])
        seg = np.asarray(mm[int(d_shard[d])][s:s + min(n - got, L)]).astype(np.int64)
        if seg.size == 0:
            continue
        out.append(seg)
        got += seg.size
    a = np.concatenate(out)
    assert a.size == n, (a.size, n)
    return a


def trial_content(t):
    """Everything trial t needs, from a stream that depends ONLY on (SEED, t)."""
    rng = np.random.default_rng([SEED, t])
    i = int(rng.integers(0, len(NAMES)))
    j = (i + 1 + int(rng.integers(0, len(NAMES) - 1))) % len(NAMES)
    code = int(rng.integers(1000, 9999))
    ocode = int(rng.integers(1000, 9999))
    fact = np.array(enc.encode_ordinary(
        f"\nThe access code for {NAMES[i]} is {code}.\n"), dtype=np.int64)
    ofact = np.array(enc.encode_ordinary(
        f"\nThe access code for {NAMES[j]} is {ocode}.\n"), dtype=np.int64)
    q = np.array(enc.encode_ordinary(
        f"\nThe access code for {NAMES[i]} is"), dtype=np.int64)
    ans = np.array(enc.encode_ordinary(f" {code}."), dtype=np.int64)
    pre = filler(rng, 32)
    stream = filler(rng, MAXDIST)      # mid at dist d is stream[:d] -> nested prefixes
    return fact, ofact, q, ans, pre, stream


@torch.no_grad()
def ans_loss(ctx, ans):
    seq = torch.tensor(np.concatenate([ctx, ans]), dtype=torch.long, device=DEV)[None]
    lg = model(seq[:, :-1])
    n = len(ans)
    per = F.cross_entropy(lg[0, -n:].float(), seq[0, -n:], reduction="none")
    greedy = lg[0, -n:].argmax(-1).tolist()
    del lg, seq
    return per.mean().item(), greedy


# ---------------------------------------------------------------- statistics
Z = 1.959963984540054


def wilson(k, n, z=Z):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1.0 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def mcnemar_exact(b, c):
    """Two-sided exact McNemar on discordant pairs. b=saw-only hits, c=control-only."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2.0 ** n)
    return min(1.0, 2.0 * tail)


def boot(x, reps=BOOT, seed=SEED + 1):
    """Paired bootstrap over trials: CI and two-sided p for mean(x) != 0."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    rs = np.random.default_rng(seed)
    m = x[rs.integers(0, n, size=(reps, n))].mean(axis=1)
    lo, hi = np.quantile(m, [0.025, 0.975])
    p = 2.0 * min(float((m <= 0).mean()), float((m >= 0).mean()))
    return float(lo), float(hi), min(1.0, p)


def pf(p, reps=BOOT):
    return f"<{2.0/reps:.0e}" if p < 2.0 / reps else f"{p:.2g}"


# ---- per-position answer accuracy (appended to needle_live.py lines 1-215 by mk_pos.sh)
import collections
NPOS = int(os.environ.get("NPOS_TRIALS", "64"))
res = {}
for dist in DISTS:
    cs, co, n3 = [0, 0, 0], [0, 0, 0], 0
    wrong_dot = collections.Counter()
    tot_s = tot_o = 0.0
    for t in range(NPOS):
        fact, ofact, q, ans, pre, stream = trial_content(t)
        mid = stream[:dist]
        _, g1 = ans_loss(np.concatenate([pre, fact, mid, q]), ans)
        _, g2 = ans_loss(np.concatenate([pre, ofact, mid, q]), ans)
        a = ans.tolist()
        tot_s += np.mean([x == y for x, y in zip(g1, a)]); tot_o += np.mean([x == y for x, y in zip(g2, a)])
        if len(a) != 3:
            continue
        n3 += 1
        for p in range(3):
            cs[p] += int(g1[p] == a[p]); co[p] += int(g2[p] == a[p])
        if g2[2] != a[2]:
            wrong_dot[enc.decode([g2[2]])] += 1
    res[dist] = dict(n3=n3, saw=[c / n3 for c in cs], ctl=[c / n3 for c in co],
                     tok_saw=tot_s / NPOS, tok_ctl=tot_o / NPOS, ctl_dot_wrong=wrong_dot.most_common(5))
    print(f"dist {dist:5d} n3={n3}  saw pos0/1/2(.) " + " ".join(f"{x:.1%}" for x in res[dist]['saw'])
          + "  ctl " + " ".join(f"{x:.1%}" for x in res[dist]['ctl'])
          + f"  | tok saw {tot_s/NPOS:.1%} ctl {tot_o/NPOS:.1%} | ctl '.' misses {wrong_dot.most_common(4)}", flush=True)
json.dump(dict(ckpt=CK, step=STEP, ntrials=NPOS, res=res), open(os.environ["OUT"], "w"))
