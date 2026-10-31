"""Is a memory's read flat because the QUERIES ignore the token, or because the SLOT KEYS are all alike?
Plain held-out text, 4 seqs x 1024, CPU.  Per reading layer: mean cosine between the read queries of two
different tokens (t, and t+97 -- far enough apart to be different content); per memory: mean cosine between
two different used slots' keys at the last snapshot.  usage: qkdiv.py CKPT"""
import os, sys, inspect
os.environ.update(MEMPROBE_ROOT="/home/paul/memprobe", MEMLM_MOD="memlm_sh", KV_DETACH="1", KV_WQ="0", SLOT_ROPE="1", SLOT_ROPE_FRAC="0.5", CUDA_VISIBLE_DEVICES="")
sys.path.insert(0, "/home/paul/memprobe/gqa")
import numpy as np, torch
import memlm_sh as S
torch.set_num_threads(12)
st = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
ok = set(inspect.signature(S.Cfg.__init__).parameters) - {"self"}
m = S.MemLM(S.Cfg(**{k: v for k, v in st["cfg"].items() if k in ok}))
miss, unexp = m.load_state_dict(st["model"], strict=False); assert not miss and not unexp
m.float().eval(); print("step", st["step"]); del st
for i, b in enumerate(m.blocks): b.mix._li = i
Q, MEM = {}, {}
_o = S.SharedMixer._read
def _read(self, kbuf, buf, kinv, valid, x):
    B, T, _ = x.shape
    q = self.tk_q(x).reshape(B, T, self.Hs, self.NQ, self.dk).float()
    Q[self._li] = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    MEM[self.mid] = (kbuf, valid)
    return _o(self, kbuf, buf, kinv, valid, x)
S.SharedMixer._read = _read
import train_sh as TS
TS.DEV = "cpu"
sh = TS.Shards(doc_aware=True)
rng = np.random.default_rng(20261009)
X = torch.from_numpy(np.stack([sh._window(1024, rng, True) for _ in range(4)])[:, :1024].astype(np.int64))
with torch.no_grad(): m(X)
print("\nQUERIES: cosine between two tokens' read queries, same head, same query index (1 = identical, ~0 = unrelated)")
print("  layer  memory   tokens 97 apart   adjacent tokens   query-norm sanity")
for L in sorted(Q):
    q = Q[L]                                                      # (B,T,Hs,NQ,dk), unit
    far = (q[:, 64:-97] * q[:, 64 + 97:]).sum(-1).mean().item()
    adj = (q[:, 64:-1] * q[:, 65:]).sum(-1).mean().item()
    print("  %5d  %-6s  %16.3f  %16.3f  %8.3f" % (L, m.blocks[L].mix.mid, far, adj, q.norm(dim=-1).mean().item()))
print("\nSLOT KEYS at the last snapshot: cosine between two DIFFERENT used slots of the same head")
for mid, (kb, valid) in MEM.items():
    k = kb[:, -1].float(); v = valid[:, -1]                        # (B,Hs,N,dk), (B,Hs,N)
    kn = k / k.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    c = torch.einsum("bhnd,bhmd->bhnm", kn, kn)
    mask = (v.unsqueeze(-1) & v.unsqueeze(-2)) & ~torch.eye(k.shape[2], dtype=torch.bool)
    print("  %-5s used slots/head %.1f | mean cosine between different slots %.3f | 90th pct %.3f"
          % (mid, v.float().sum(-1).mean().item(), c[mask].mean().item(), c[mask].quantile(0.9).item() if c[mask].numel() < 16_000_000 else float("nan")))
