"""Which write-query INIT makes a fresh model take a new row on (nearly) every write?  Fresh sh9-config model
(random init, gate_init 0.999 unless GATE set), main memory (layer 6) and encoder (layer 2).  Variants:
  copy        write_q = key projection (sh9's init)
  copy*0.01   same direction, 1/100 the length (cosine ignores length -> must equal 'copy')
  rand 0.01   independent N(0, 0.01) matrix
For each: rows used per head (distinct hard top-1 rows among writes that land, g>0), % writes into an
already-used row, |q| median, and grad norms (write_q vs rest; clip factor at 1.0).  Same batch for every variant.
Usage: wqinit_rowtest.py CKPT_FOR_CFG"""
import os, sys, inspect, math
os.environ.update(MEMPROBE_ROOT="/home/paul/memprobe", MEMLM_MOD="memlm_sh", KV_DETACH="1", KV_WQ="0", SLOT_ROPE="1", SLOT_ROPE_FRAC="0.5", CUDA_VISIBLE_DEVICES="")
sys.path.insert(0, "/home/paul/memprobe/gqa")
import numpy as np, torch
import memlm_sh as S, slotkern_kv as SK
torch.set_num_threads(int(os.environ.get("THREADS", "2")))
st = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
cc = dict(st["cfg"]); del st
cc["gate_init"] = float(os.environ.get("GATE", "0.999"))
ok = set(inspect.signature(S.Cfg.__init__).parameters) - {"self"}
caught = []
_orig = S.ref_scan_kv
def spy(k, v, g, fr, N, temp, **kw):
    caught.append(dict(k=k.detach().float(), q=kw["q"].detach().float(), g=g.detach().float(), fr=fr.detach(),
                       temp=temp.detach(), N=N, kw={kk: kw[kk] for kk in ("hard", "hard_gate", "detach_addr")}))
    return _orig(k, v, g, fr, N, temp, **kw)
S.ref_scan_kv = spy
import train_sh as TS
TS.DEV = "cpu"
sh = TS.Shards(doc_aware=True)
rng = np.random.default_rng(3)
T = int(os.environ.get("T", "256"))
x = torch.from_numpy(np.stack([sh._window(T + 1, rng, True) for _ in range(1)])[:, :T + 1].astype(np.int64))
assert x.shape == (1, T + 1), x.shape
torch.manual_seed(0)
cfg = S.Cfg(**{k: v for k, v in cc.items() if k in ok})
m = S.MemLM(cfg).float().train()
assert hasattr(m, "write_q") and len(m.write_q) == 2, "expected 2 write queries"
init = {n: p.detach().clone() for n, p in m.named_parameters()}
W0 = {n: l.weight.detach().clone() for n, l in m.write_q.items()}
gen = torch.Generator().manual_seed(1)
RAND = {n: torch.randn(w.shape, generator=gen) * 0.01 for n, w in W0.items()}
print("T=%d  gate_init %.3f  sigmoid(g_bias) %.4f  hard %s  floor %.3f" % (T, cfg.gate_init, float(torch.sigmoid(m.blocks[6].mix.g_bias[0])), cfg.slot_hard, cfg.sh_gate_floor))
for name, W in (("copy", W0), ("copy*0.01", {n: w * 0.01 for n, w in W0.items()}), ("rand 0.01", RAND)):
    with torch.no_grad():
        for n, p in m.named_parameters(): p.copy_(init[n])
        for n, l in m.write_q.items(): l.weight.copy_(W[n])
    m.zero_grad(set_to_none=True); caught.clear()
    out = m(x[:, :-1])
    logits = out[0] if isinstance(out, (tuple, list)) else out
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), x[:, 1:].reshape(-1))
    loss.backward()
    wq_ids = {id(l.weight) for l in m.write_q.values()}
    gq = math.sqrt(sum(float(l.weight.grad.pow(2).sum()) for l in m.write_q.values()))
    go = math.sqrt(sum(float(p.grad.pow(2).sum()) for p in m.parameters() if p.grad is not None and id(p) not in wq_ids))
    tot = math.sqrt(gq ** 2 + go ** 2)
    nonfin = sum(int(not torch.isfinite(p.grad).all()) for p in m.parameters() if p.grad is not None)
    print("\n== %-9s loss %.4f | grad write_q %.3e rest %.3e total %.3e | clip factor %.3e | non-finite %d"
          % (name, float(loss), gq, go, tot, min(1.0, 1 / max(tot, 1e-12)), nonfin))
    assert len(caught) == 2
    for lab, c in zip(("ENC", "MAIN"), caught):
        with torch.no_grad():
            _, _, a = SK._ref_scan_kv_impl(c["k"], torch.zeros_like(c["k"]), c["g"], c["fr"], c["N"], c["temp"], 1,
                                           c["kw"]["hard"], c["kw"]["hard_gate"], 1e-4, True, c["kw"]["detach_addr"], c["q"], pin=False)
        r = a.argmax(-1)[0]                                   # (T,H) chosen row per token
        wr = (c["g"][0] > 0)                                  # (T,H) writes that land
        H = r.shape[1]; used = []; over = 0; nw = 0
        for h in range(H):
            seen = set()
            for t in range(r.shape[0]):
                if not wr[t, h]: continue
                j = int(r[t, h]); nw += 1
                if j in seen: over += 1
                seen.add(j)
            used.append(len(seen))
        print("   %-4s N=%d temp %.2f fr %.2f | |q| med %.3e | writes landing %.1f%% | rows used per head %s | into a used row %.1f%%"
              % (lab, c["N"], float(c["temp"]), float(c["fr"]), float(c["q"].norm(dim=-1).median()),
                 100 * float(wr.float().mean()), used, 100 * over / max(nw, 1)))
