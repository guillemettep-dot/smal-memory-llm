import os, sys
os.environ.update(KV_DETACH="1", KV_WQ="0", SLOT_ROPE="1", SLOT_ROPE_FRAC="0.5")
sys.path.insert(0, "/home/paul/memprobe/gqa")
import torch, memlm_sh as S
from slotkern_kv import ref_scan_kv, _ref_scan_kv_impl
import inspect
rel = lambda a, b: ((a.double() - b.double()).abs().max() / b.double().abs().max().clamp_min(1e-30)).item()
cap = {}
real = S.kv_scan_g
def spy(*a, **k):
    cap["a"], cap["k"] = [x.detach().clone() if torch.is_tensor(x) else x for x in a], dict(k); return real(*a, **k)
S.kv_scan_g = spy
torch.manual_seed(0)
kw = dict(vocab=997, d_model=128, layers=6, heads=2, dv=16, lanes=1, d_ff=128, moe=0, n_exp=2, top_k=2, pre=1,
          kv_window=16, kv_dim=32, kv_heads=2, comp_off=1, tksel=1, slot_hard=0, slot_hard_gate=0, krep=1.0,
          sh_write=2, sh_heads=12, sh_slots=192, sh_kdim=64, sh_dv=64, sh_nq=2, sh_stride=16)
c = S.Cfg(**kw); print("free_bias", c.free_bias, "gate_init", c.gate_init)
mm = S.MemLM(c).cuda().float().train()
xx = torch.randint(0, 997, (2, 300), device="cuda")
with torch.no_grad(): mm(xx)
k, v, g, fr, N, tp = cap["a"]; kw2 = cap["k"]
print("kwargs", kw2, "| k norm mean %.3f  g mean %.3f  fr %.3f tp %.3f" % (k.norm(dim=-1).mean(), g.mean(), fr, tp))
a1, b1 = real(k, v, g, fr, N, tp, **kw2)
a2, b2 = ref_scan_kv(k, v, g, fr, N, tp, **kw2)
print("captured inputs: triton vs ref  kbuf %.1e vbuf %.1e" % (rel(a1, a2), rel(b1, b2)))
d64 = [t.double() if torch.is_tensor(t) else t for t in (k, v, g, fr, N, tp)]
a3, b3 = _ref_scan_kv_impl(*d64, **kw2) if "G" in inspect.signature(_ref_scan_kv_impl).parameters else (None, None)
if a3 is not None:
    print("vs fp64 ref:  triton kbuf %.1e vbuf %.1e | fp32 ref kbuf %.1e vbuf %.1e" % (rel(a1, a3), rel(b1, b3), rel(a2, a3), rel(b2, b3)))
print("per-block triton-vs-ref vbuf:", " ".join("%.0e" % (b1[:, j] - b2[:, j]).abs().max().item() for j in range(b1.shape[1])))
print("ref_scan_kv src:\n" + inspect.getsource(ref_scan_kv))
