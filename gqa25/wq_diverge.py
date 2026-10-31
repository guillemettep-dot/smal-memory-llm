"""Has sh8's write query diverged from the stored key?  (1) weights: write_q.<L> vs the key rows of w_in;
(2) on real text: per-token cosine between the write query and the key it stores; (3) whether the slot
chosen by the write query differs from the slot the key alone would have chosen."""
import os, sys, inspect
os.environ.update(MEMPROBE_ROOT="/home/paul/memprobe", MEMLM_MOD="memlm_sh", KV_DETACH="1", KV_WQ="0", SLOT_ROPE="1", SLOT_ROPE_FRAC="0.5", CUDA_VISIBLE_DEVICES="")
sys.path.insert(0, "/home/paul/memprobe/gqa")
import numpy as np, torch
import memlm_sh as S, slotkern_kv as SK
torch.set_num_threads(12)
st = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
print("checkpoint step", st["step"])
m = st["model"]
for L in (2, 6):
    W = m["write_q.%d.weight" % L].float(); K = m["blocks.%d.mix.w_in.weight" % L][2304:3072].float()
    rel = ((W - K).norm() / K.norm()).item()
    cos = torch.nn.functional.cosine_similarity(W.flatten(), K.flatten(), dim=0).item()
    rowcos = torch.nn.functional.cosine_similarity(W, K, dim=1)
    print("layer %d weights: |Wq - Wk| / |Wk| = %.4f | cosine of the whole matrices %.5f | per-row cosine min %.4f median %.4f"
          % (L, rel, cos, rowcos.min(), rowcos.median()))
ok = set(inspect.signature(S.Cfg.__init__).parameters) - {"self"}
model = S.MemLM(S.Cfg(**{k: v for k, v in st["cfg"].items() if k in ok})); model.load_state_dict(m, strict=True); model.float().eval()
caught = []
def spy(k, v, g, fr, N, temp, **kw):
    q = kw.get("q")
    out = []
    for qq in (q, None):     # address chosen by the write query, and by the key alone (same state otherwise)
        kb, vb, a = SK._ref_scan_kv_impl(k.float(), v.float(), g.float(), fr, N, temp, 1, kw.get("hard", False), False,
                                         1e-4, True, True, None if qq is None else qq.float(), pin=False)
        out.append(a.argmax(-1))
    caught.append(dict(k=k.detach(), q=q.detach(), g=g.detach(), slot_q=out[0], slot_k=out[1]))
    kb, vb = SK._ref_scan_kv_impl(k.float(), v.float(), g.float(), fr, N, temp, kw.get("G", 1), kw.get("hard", False), False,
                                  1e-4, False, True, None if q is None else q.float(), pin=False)
    return kb, vb
S.ref_scan_kv = spy
import train_sh as TS
TS.DEV = "cpu"
sh = TS.Shards(doc_aware=True)
rng = np.random.default_rng(3)
x = torch.from_numpy(np.stack([sh._window(1024, rng, True) for _ in range(2)])[:, :1024].astype(np.int64))
with torch.no_grad():
    model(x)
for name, c in zip(("encoder memory (layer 2)", "main memory (layer 6)"), caught):
    cs = torch.nn.functional.cosine_similarity(c["q"], c["k"], dim=-1)          # (B,T,H)
    w = c["g"] > 0
    diff = (c["slot_q"] != c["slot_k"]) & w
    print("%s: cosine(write query, stored key) per token: median %.4f, 5th pct %.4f | writing tokens whose chosen slot differs: %.1f%%"
          % (name, cs.median(), torch.quantile(cs.flatten(), 0.05), 100 * diff.sum().item() / max(w.sum().item(), 1)))
