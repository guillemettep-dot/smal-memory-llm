"""Does the memory's CONTENT help ordinary text?  Held-out plain text (no recall tasks), T=1024, CPU.
Each sequence is scored with its OWN memory, then with the memory written by ANOTHER sequence of the batch
(rolled by one along the batch).  A swapped memory is a real, in-distribution memory -- only its content is
irrelevant -- so CE(swapped) - CE(own) is what this text's own memory is worth, not the shock of an empty read.
Conditions: own | main swapped | enc swapped | both swapped.  Reported per position band, because the window
(16 tokens) covers everything local and the memory can only matter beyond it.
usage: memswap.py CKPT [NSEQ=16] [B=4]"""
import os, sys, inspect, time
os.environ.update(MEMPROBE_ROOT="/home/paul/memprobe", MEMLM_MOD="memlm_sh", KV_DETACH="1", KV_WQ="0", SLOT_ROPE="1", SLOT_ROPE_FRAC="0.5", CUDA_VISIBLE_DEVICES="")
sys.path.insert(0, "/home/paul/memprobe/gqa")
import numpy as np, torch, torch.nn.functional as F
import memlm_sh as S
torch.set_num_threads(int(os.environ.get("THREADS", "12")))
NSEQ = int(sys.argv[2]) if len(sys.argv) > 2 else 16
B = int(sys.argv[3]) if len(sys.argv) > 3 else 4
assert NSEQ % B == 0 and B >= 2
st = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
ok = set(inspect.signature(S.Cfg.__init__).parameters) - {"self"}
cfg = S.Cfg(**{k: v for k, v in st["cfg"].items() if k in ok})
m = S.MemLM(cfg)
miss, unexp = m.load_state_dict(st["model"], strict=False)
assert not miss and not unexp, (miss, unexp)
m.float().eval()
print("ckpt step %s | slot_hard %s gate_floor %s sh_wq %s | %d seqs x 1024, batch %d"
      % (st["step"], cfg.slot_hard, getattr(cfg, "sh_gate_floor", None), getattr(cfg, "sh_wq", 0), NSEQ, B), flush=True)
del st

SWAP = set()
_orig = S.SharedMixer._read
def _read(self, kbuf, buf, kinv, valid, x):
    if self.mid in SWAP:
        kbuf, buf, kinv, valid = (t.roll(1, 0) for t in (kbuf, buf, kinv, valid))
    return _orig(self, kbuf, buf, kinv, valid, x)
S.SharedMixer._read = _read
mids = sorted({b.mix.mid for b in m.blocks if b.mix.mid is not None})
print("memories in this model:", mids, flush=True)

import train_sh as TS
TS.DEV = "cpu"
sh = TS.Shards(doc_aware=True)
rng = np.random.default_rng(20261008)                 # fixed: every checkpoint sees the same text
W = np.stack([sh._window(1024, rng, True) for _ in range(NSEQ)]).astype(np.int64)
assert W.shape == (NSEQ, 1025), W.shape
X, Y = torch.from_numpy(W[:, :1024]), torch.from_numpy(W[:, 1:1025])
assert not (X == TS.EOT).any(), "a window crosses a document boundary"

conds = [("own", set()), ("main swapped", {"main"}), ("enc swapped", {"enc"}), ("both swapped", {"main", "enc"})]
conds = [c for c in conds if c[1] <= set(mids)]
ce = {}
t0 = time.time()
with torch.no_grad():
    for name, sw in conds:
        SWAP.clear(); SWAP.update(sw)
        per = []
        for i in range(0, NSEQ, B):
            lg = m(X[i:i + B])
            per.append(F.cross_entropy(lg.float().transpose(1, 2), Y[i:i + B], reduction="none"))
        ce[name] = torch.cat(per)                       # (NSEQ, 1024)
        print("  %-13s done  %.0fs" % (name, time.time() - t0), flush=True)
SWAP.clear()

# self-check: with B=1 the roll is the identity, so a swapped run must equal own exactly
SWAP.update(set(mids))
with torch.no_grad():
    a = F.cross_entropy(m(X[:1]).float().transpose(1, 2), Y[:1], reduction="none")
SWAP.clear()
d0 = (a - ce["own"][:1]).abs().max().item()
print("selfcheck: B=1 swap vs own max |dCE| = %.2e (must be ~0)  %s" % (d0, "OK" if d0 < 1e-3 else "FAIL"), flush=True)
assert d0 < 1e-3

bands = [(0, 16), (16, 64), (64, 256), (256, 1024), (0, 1024)]
print("\n  CE by position (nats); swapped - own = what this text's own memory is worth")
print("  %-13s" % "condition" + "".join("%12s" % ("%d-%d" % b) for b in bands))
for name, _ in conds:
    print("  %-13s" % name + "".join("%12.4f" % ce[name][:, a:b].mean().item() for a, b in bands))
for name, _ in conds[1:]:
    d = ce[name] - ce["own"]
    # per-sequence spread on the full band, so the reader can see if the gain is broad or one sequence
    s = d.mean(1)
    print("  %-13s" % ("d " + name.split()[0]) + "".join("%+12.4f" % d[:, a:b].mean().item() for a, b in bands)
          + "   per-seq mean %+.4f sd %.4f, %d/%d seqs helped" % (s.mean(), s.std(), int((s > 0).sum()), len(s)))
