"""How does the shared memory WRITE?  Real held-out text, T=1024, CPU.  For each memory (encoder@2, main@6):
gate g per (token, head): does every token write a little, or does it choose?  Write distribution a (pre-gate):
how concentrated is each write (top-row share)?  Rows: how many filled, how distinct (cosine between rows)."""
import os, sys, inspect
os.environ.update(MEMPROBE_ROOT="/home/paul/memprobe", MEMLM_MOD="memlm_sh", KV_DETACH="1", KV_WQ="0", SLOT_ROPE="1", SLOT_ROPE_FRAC="0.5", CUDA_VISIBLE_DEVICES="")
sys.path.insert(0, "/home/paul/memprobe/gqa")
import numpy as np, torch
import memlm_sh as S, slotkern_kv as SK
torch.set_num_threads(12)
CK = sys.argv[1]
st = torch.load(CK, map_location="cpu", weights_only=False)
ok = set(inspect.signature(S.Cfg.__init__).parameters) - {"self"}
m = S.MemLM(S.Cfg(**{k: v for k, v in st["cfg"].items() if k in ok})); m.load_state_dict(st["model"], strict=False); m.float().eval()
caught = []
orig = S.ref_scan_kv
def spy(k, v, g, fr, N, temp, **kw):
    kb, vb, a = SK._ref_scan_kv_impl(k.float(), v.float(), g.float(), fr, N, temp, 1, kw.get("hard", False), kw.get("hard_gate", False),
                                     1e-4, True, kw.get("detach_addr", False), None, pin=kw.get("pin", True))
    caught.append(dict(g=g.detach(), a=a.detach(), K=kb[:, -1].detach(), N=N))
    G = kw.get("G", 1)
    return kb[:, ::G], vb[:, ::G]
S.ref_scan_kv = spy
# real text: a held-out shard window through the trainer's own loader
import train_sh as TS
TS.DEV = "cpu"
sh = TS.Shards(doc_aware=True)
rng = np.random.default_rng(3)
x = torch.from_numpy(np.stack([sh._window(1024, rng, True) for _ in range(2)])[:, :1024].astype(np.int64))
with torch.no_grad():
    m(x)
for name, c in zip(("encoder memory (layer 2)", "main memory (layer 6)"), caught):
    g, a, K = c["g"], c["a"], c["K"]            # g (B,T,H), a (B,T,H,N), K (B,H,N,dk) final state
    top = a.max(-1).values                       # share of the write landing on its single best row
    Kn = K / K.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    filled = (K.norm(dim=-1) > 1e-6)
    cos = torch.einsum("bhnd,bhmd->bhnm", Kn, Kn)
    offd = cos[filled.unsqueeze(-1) & filled.unsqueeze(-2) & ~torch.eye(c["N"], dtype=torch.bool)]
    q = torch.quantile(g.flatten(), torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95]))
    print("== %s, %d slots/head" % (name, c["N"]))
    print("   write gate g: mean %.3f | quantiles 5/25/50/75/95%%: %s" % (g.mean(), " ".join("%.3f" % v for v in q)))
    print("   tokens writing > 0.5: %.1f%% | > 0.9: %.1f%% | < 0.05: %.1f%% | < 0.01: %.1f%%"
          % (100 * (g > 0.5).float().mean(), 100 * (g > 0.9).float().mean(), 100 * (g < 0.05).float().mean(), 100 * (g < 0.01).float().mean()))
    print("   write concentration (share on the top row): median %.2f | mean %.2f | writes >0.9 on one row: %.1f%%"
          % (top.median(), top.mean(), 100 * (top > 0.9).float().mean()))
    print("   rows filled at T=1024: %.1f of %d per head | cosine between distinct filled rows: mean %.2f, 90th pct %.2f"
          % (filled.float().sum(-1).mean(), c["N"], offd.mean(), torch.quantile(offd, 0.9)))
