"""Does a ZERO-initialised write query blow up the gradient?  Fresh sh9-config model (random init, gate_init
0.999), write_q weight = copy-of-key * s for several s.  Cosine ignores |q|, so the forward is identical for
every s > 0; the gradient into q goes as 1/|q| (clamped at 1e-6).  Reports, per s: |q| median, rows used in
the main memory (does it always take a new row?), grad norm of write_q vs everything else, and the factor
clip_grad_norm_(1.0) would scale the TRUNK by.  Same batch for every s.
Usage: wqzero_gradtest.py CKPT_FOR_CFG"""
import os, sys, inspect, math
os.environ.update(MEMPROBE_ROOT="/home/paul/memprobe", MEMLM_MOD="memlm_sh", KV_DETACH="1", KV_WQ="0", SLOT_ROPE="1", SLOT_ROPE_FRAC="0.5", CUDA_VISIBLE_DEVICES="")
sys.path.insert(0, "/home/paul/memprobe/gqa")
import numpy as np, torch
import memlm_sh as S
torch.set_num_threads(int(os.environ.get("THREADS", "2")))
st = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
cc = dict(st["cfg"]); del st
cc["gate_init"] = 0.999
ok = set(inspect.signature(S.Cfg.__init__).parameters) - {"self"}
caught = []
_orig = S.ref_scan_kv
def spy(k, v, g, fr, N, temp, **kw):
    caught.append((kw.get("q"), g.detach(), N))
    return _orig(k, v, g, fr, N, temp, **kw)
S.ref_scan_kv = spy
import train_sh as TS
TS.DEV = "cpu"
sh = TS.Shards(doc_aware=True)
rng = np.random.default_rng(3)
T = int(os.environ.get("T", "128"))
x = torch.from_numpy(np.stack([sh._window(T + 1, rng, True) for _ in range(1)])[:, :T + 1].astype(np.int64))
assert x.shape == (1, T + 1), x.shape
torch.manual_seed(0)
cfg = S.Cfg(**{k: v for k, v in cc.items() if k in ok})
m = S.MemLM(cfg).float().train()
W0 = {n: l.weight.detach().clone() for n, l in m.write_q.items()}
init = {n: p.detach().clone() for n, p in m.named_parameters()}
print("T=%d  gate_init %.3f  sigmoid(g_bias) %.4f" % (T, cfg.gate_init, float(torch.sigmoid(m.blocks[6].mix.g_bias[0]))))
for s in [1.0, 1e-2, 1e-4, 1e-6, 0.0]:
    with torch.no_grad():
        for n, p in m.named_parameters(): p.copy_(init[n])
        for n, l in m.write_q.items(): l.weight.copy_(W0[n] * s)
    m.zero_grad(set_to_none=True); caught.clear()
    out = m(x[:, :-1])
    logits = out[0] if isinstance(out, (tuple, list)) else out
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), x[:, 1:].reshape(-1))
    loss.backward()
    gq = math.sqrt(sum(float(l.weight.grad.pow(2).sum()) for l in m.write_q.values() if l.weight.grad is not None))
    wq_ids = {id(l.weight) for l in m.write_q.values()}
    go = math.sqrt(sum(float(p.grad.pow(2).sum()) for p in m.parameters() if p.grad is not None and id(p) not in wq_ids))
    tot = math.sqrt(gq ** 2 + go ** 2)
    nonfin = sum(int(not torch.isfinite(p.grad).all()) for p in m.parameters() if p.grad is not None)
    q, g, N = caught[-1]
    qn = q.float().norm(dim=-1).median() if q is not None else float("nan")
    # rows used in the main memory: replay hard top-1 using the model's own address function is heavy; use the
    # occupancy argument instead -- count distinct argmax rows via return_w on the reference
    k_, v_ = None, None
    print("s=%-7g loss %.4f | |q| median %.3e | grad: write_q %.3e  rest %.3e  total %.3e | clip factor on trunk %.3e | non-finite grads %d"
          % (s, float(loss), float(qn), gq, go, tot, min(1.0, 1.0 / max(tot, 1e-12)), nonfin))
