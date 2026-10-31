"""Draw REAL training batches through train_sh's own Shards.batch with sh1's exact settings."""
import os, sys, collections
os.environ.update(MEMPROBE_ROOT="/home/paul/memprobe", MEMLM_MOD="memlm_sh", CUDA_VISIBLE_DEVICES="")
sys.path.insert(0, "/home/paul/memprobe/gqa")
import numpy as np, torch
import train_sh as TS
import task_lib as TL, recall_tasks2 as RT2
TS.DEV = "cpu"
names = "keyed_span:1,updated:1,tagged:1,list:1,multihop:1,reverse:1,pred:1,factgen:4,chat:2,record:2,codeconst:2,schedule:1,inventory:1,glossary:1,moves:1,quote:1"
tn = [x.split(":")[0] for x in names.split(",")]; tw = [float(x.split(":")[1]) for x in names.split(",")]
hits = collections.Counter(); planted = collections.Counter()
for n, g in list(TL.GENERATORS.items()):
    def wrap(w, T, rng, cue, want_mask=False, _g=g, _n=n):
        hits[_n] += 1
        out = _g(w, T, rng, cue, want_mask=want_mask)
        if want_mask and out[1] is not None and out[1].sum() > 0: planted[_n] += 1
        return out
    TL.GENERATORS[n] = wrap
sh = TS.Shards(doc_aware=True)
rng = np.random.default_rng(0)
T, bs, nb = 1024, 2, 300
syn = 0; ans_tok = 0; tot_tok = 0; wmax = 0
for i in range(nb):
    x, y, wt = sh.batch(bs, T, rng, p_synth=0.30, ans_w=5.0, p_mqar=0.2, mqar_pairs=32, p_broad=0.10,
                        long_frac=0.50, p_tasks=0.6, task_names=tn, task_weights=tw)
    wt = wt.numpy(); syn += int((wt > 1).any(1).sum()); ans_tok += int((wt > 1).sum()); tot_tok += wt.size
    wmax = max(wmax, float(wt.max()))
print("sequences with a weighted answer: %.1f%% of %d (target ~30%%; mqar/repeat count too)" % (100 * syn / (nb * bs), nb * bs))
print("answer tokens: %.2f%% of all tokens, weight %.1f (want 5.0)" % (100 * ans_tok / tot_tok, wmax))
print("effective loss share of answer tokens: %.1f%%" % (100 * 5 * ans_tok / (5 * ans_tok + tot_tok - ans_tok)))
tot = sum(hits.values())
print("task draws (incl. compose's second task):")
for n in tn:
    print("   %-11s %5.1f%% of draws   planted %d/%d" % (n, 100 * hits[n] / max(tot, 1), planted[n], hits[n]))
