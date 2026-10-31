# -*- coding: utf-8 -*-
"""GPT-2 on the SAME by-description task, as a reference for whether the task is solvable at all.
Identical trials to eval_bydesc_kv.py: same factgen generator, same per-trial seed [SEED, i], same
paired control (target statement removed, distractors/query/answer held fixed), same scoring.
factgen tokenises with tiktoken "gpt2", so the ids are GPT-2's own -- no re-tokenisation.
GPT-2 keeps the whole context in its cache and compresses nothing; that is the point of the column.
"""
import os, sys, torch, numpy as np
sys.path.insert(0, ".")
os.environ.setdefault("MEMPROBE_ROOT", os.path.expanduser("~/memprobe"))
import factgen as F
from transformers import GPT2LMHeadModel

NAME = os.environ.get("GPT2", "gpt2")
torch.set_num_threads(int(os.environ.get("THREADS", "5")))
m = GPT2LMHeadModel.from_pretrained(NAME).eval()
m.requires_grad_(False)
CTX = m.config.n_positions
print("%s | %d params | context %d" % (NAME, sum(p.numel() for p in m.parameters()), CTX), flush=True)

N  = int(os.environ.get("N", "256"))
NF = int(os.environ.get("NF", "8"))
SEED = int(os.environ.get("SEED", "20260823"))

def paired(rng, nfacts):
    """Byte-identical to eval_bydesc_kv.paired()."""
    stmts = [F._simple(rng)[0] for _ in range(nfacts - 1)]
    gp = rng.random()
    pick = F._simple if gp < 0.6 else F._multihop if gp < 0.8 else F._inference
    ts, tq, ans, _ = pick(rng)
    repl = F._simple(rng)[0]
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
    if len(toks) > CTX:            # GPT-2 cannot see past its context
        return 0, 0, False, True
    x = torch.tensor(toks[None]); ans_pos = np.nonzero(mask)[0]
    lg = m(x[:, :-1]).logits.float()
    pred = lg[0].argmax(-1)
    hit = tot = 0
    for p in ans_pos:
        if p == 0: continue
        tot += 1; hit += int(pred[p - 1]) == int(toks[p])
    return hit, tot, (tot > 0 and hit == tot), False

sh = st = se = 0; ch = ct = ce = 0; over = 0
with torch.no_grad():
    for i in range(N):
        rng = np.random.default_rng([SEED, i])
        (sa, sm), (ca, cm), _ = paired(rng, NF)
        h, t, e, o = score(sa, sm); sh += h; st += t; se += e; over += o
        h, t, e, _ = score(ca, cm); ch += h; ct += t; ce += e
        if (i + 1) % 32 == 0:
            print("  %d/%d  saw tok %.1f%% exact %.1f%% | ctl tok %.1f%% exact %.1f%%"
                  % (i + 1, N, 100*sh/max(st,1), 100*se/(i+1), 100*ch/max(ct,1), 100*ce/(i+1)), flush=True)

print()
print("BY-DESCRIPTION RETRIEVAL -- %s  (%d facts/context, %d paired trials, seed %d)" % (NAME, NF, N, SEED))
print("  %-22s %9s %9s %9s" % ("", "saw", "control", "delta"))
print("  %-22s %8.1f%% %8.1f%% %+8.1f" % ("token accuracy", 100*sh/max(st,1), 100*ch/max(ct,1),
                                          100*sh/max(st,1) - 100*ch/max(ct,1)))
print("  %-22s %8.1f%% %8.1f%% %+8.1f" % ("exact match", 100*se/N, 100*ce/N, 100*se/N - 100*ce/N))
print("  trials over GPT-2's %d-token context (scored 0): %d of %d" % (CTX, over, N))
