# By-description retrieval, PAIRED. Derived from kchunk_dev/eval_bydesc.py with three fixes:
#
#  1. MODEL MODULE IS CONFIGURABLE. The original hardcoded memlm_m5; our arm is memlm_kv, and with
#     strict=False that loads almost nothing and still prints a plausible number.
#  2. STRICT STATE-DICT CHECK. Refuse to benchmark on any missing/unexpected key, the way
#     needle_cpu.py does. A silently-wrong model is worse than no number.
#  3. A REAL CONTROL. The original drew a SECOND INDEPENDENT EXAMPLE and scored that -- the answer
#     is present there too, so it measured repeatability, not a floor (its own comment says
#     "approx: just measure token acc on a shuffled-answer control"). Here the control keeps the
#     SAME distractors, SAME query and SAME answer tokens, and only swaps the TARGET statement for
#     another distractor, so the queried fact is genuinely absent. Paired per trial.
import torch, sys, os, importlib, inspect, numpy as np
sys.path.insert(0, ".")
os.environ.setdefault("MEMPROBE_ROOT", os.path.expanduser("~/memprobe"))
import factgen as F
MOD = os.environ.get("MEMLM_MOD", "memlm_kv")
M = importlib.import_module(MOD)
import cpu_shim; cpu_shim.install(M)
torch.set_num_threads(int(os.environ.get("THREADS", "5")))

ck = os.path.expanduser(os.environ["CKPT"])
st = torch.load(ck, map_location="cpu", weights_only=False)
ok = set(inspect.signature(M.Cfg.__init__).parameters) - {"self"}
cfg = M.Cfg(**{k: v for k, v in st["cfg"].items() if k in ok})
m = M.MemLM(cfg)
miss, unex = m.load_state_dict(st["model"], strict=False)
if miss or unex:
    print("FATAL: checkpoint/model mismatch -- refusing to benchmark a wrong model.")
    print("  module=%s  missing=%d %s" % (MOD, len(miss), list(miss)[:8]))
    print("  unexpected=%d %s" % (len(unex), list(unex)[:8]))
    sys.exit(1)
m = m.float().eval(); m.requires_grad_(False)
try: m.set_moe_loop(False)
except Exception: pass
c = st["cfg"]
print("ckpt %s  step %s  | module %s | d_model %s heads %s n_slots %s tk_nq %s n_exp %s"
      % (os.path.basename(ck), st.get("step"), MOD, c["d_model"], c["heads"], c["n_slots"], c["tk_nq"], c["n_exp"]), flush=True)
print("state_dict: EXACT MATCH (0 missing, 0 unexpected)", flush=True)

N  = int(os.environ.get("N", "256"))
NF = int(os.environ.get("NF", "8"))
SEED = int(os.environ.get("SEED", "20260823"))

def paired(rng, nfacts):
    """SAW and CONTROL differ ONLY in whether the queried fact is in the context."""
    stmts = [F._simple(rng)[0] for _ in range(nfacts - 1)]        # distractors, shared by both
    gp = rng.random()
    pick = F._simple if gp < 0.6 else F._multihop if gp < 0.8 else F._inference
    ts, tq, ans, _ = pick(rng)                                     # the target fact + its query
    repl = F._simple(rng)[0]                                       # stands in for the target
    out = []
    for body_stmts in (stmts + [ts], stmts + [repl]):
        bs = list(body_stmts); rng.shuffle(bs)
        body = " ".join(bs) + " " + tq
        tt = F.enc.encode(body); at = F.enc.encode(" " + ans)
        toks = np.array(tt + at, dtype=np.int64)
        mask = np.zeros(len(toks), dtype=np.float32); mask[len(tt):] = 1.0
        out.append((toks, mask))
    return out[0], out[1], (tq, ans)

def score(toks, mask):
    x = torch.tensor(toks[None]); ans_pos = np.nonzero(mask)[0]
    lg = m(x[:, :-1]).float(); pred = lg[0].argmax(-1)
    hit = tot = 0
    for p in ans_pos:
        if p == 0: continue
        tot += 1; hit += int(pred[p - 1]) == int(toks[p])
    return hit, tot, (tot > 0 and hit == tot)

ts_h = ts_t = ts_e = 0; tc_h = tc_t = tc_e = 0
with torch.no_grad():
    for i in range(N):
        rng = np.random.default_rng([SEED, i])       # per-trial stream: trial i is reproducible
        (sa, sm), (ca, cm), _ = paired(rng, NF)
        h, t, e = score(sa, sm); ts_h += h; ts_t += t; ts_e += e
        h, t, e = score(ca, cm); tc_h += h; tc_t += t; tc_e += e
        if (i + 1) % 32 == 0:
            print("  %d/%d  saw tok %.1f%% exact %.1f%% | ctl tok %.1f%% exact %.1f%%"
                  % (i + 1, N, 100*ts_h/max(ts_t,1), 100*ts_e/(i+1), 100*tc_h/max(tc_t,1), 100*tc_e/(i+1)), flush=True)

print()
print("BY-DESCRIPTION RETRIEVAL  (%d facts/context, %d paired trials, seed %d)" % (NF, N, SEED))
print("  %-22s %9s %9s %9s" % ("", "saw", "control", "delta"))
print("  %-22s %8.1f%% %8.1f%% %+8.1f" % ("token accuracy", 100*ts_h/max(ts_t,1), 100*tc_h/max(tc_t,1),
                                          100*ts_h/max(ts_t,1) - 100*tc_h/max(tc_t,1)))
print("  %-22s %8.1f%% %8.1f%% %+8.1f" % ("exact match", 100*ts_e/N, 100*tc_e/N, 100*ts_e/N - 100*tc_e/N))
print()
print("  The control holds the distractors, the query and the answer tokens FIXED and removes only")
print("  the target statement, so any gap above is retrieval of the fact actually shown.")
