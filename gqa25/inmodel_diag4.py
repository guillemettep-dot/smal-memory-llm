import os, sys
os.environ.update(KV_DETACH="1", KV_WQ="0", SLOT_ROPE="1", SLOT_ROPE_FRAC="0.5")
sys.path.insert(0, "/home/paul/memprobe/gqa")
import torch, memlm_sh as S
from slotkern_kv import ref_scan_kv
from slotkern_kvt import kv_scan
import slotkern_kvg as KG
rel = lambda a, b: ((a.double() - b.double()).abs().max() / b.double().abs().max().clamp_min(1e-30)).item()
cap = {}
real = S.kv_scan_g
def spy(*a, **k):
    cap["a"] = [x.detach().clone() if torch.is_tensor(x) else x for x in a]; return real(*a, **k)
S.kv_scan_g = spy
torch.manual_seed(0)
kw = dict(vocab=997, d_model=128, layers=6, heads=2, dv=16, lanes=1, d_ff=128, moe=0, n_exp=2, top_k=2, pre=1,
          kv_window=16, kv_dim=32, kv_heads=2, comp_off=1, tksel=1, slot_hard=0, slot_hard_gate=0, krep=1.0,
          sh_write=2, sh_heads=12, sh_slots=192, sh_kdim=64, sh_dv=64, sh_nq=2, sh_stride=16)
mm = S.MemLM(S.Cfg(**kw)).cuda().float().train()
xx = torch.randint(0, 997, (2, 300), device="cuda")
with torch.no_grad(): mm(xx)
k, v, g, fr, N, tp = cap["a"]
torch.save(dict(k=k.cpu(), v=v.cpu(), g=g.cpu(), fr=fr.cpu(), tp=tp.cpu()), "/home/paul/memprobe/gqa/_scan_inputs.pt")
print("contig", k.is_contiguous(), v.is_contiguous(), g.is_contiguous(), "strides k", k.stride())
with torch.no_grad():
    r1k, r1v = ref_scan_kv(k, v, g, fr, N, tp, G=1, detach_addr=True)
    o1k, o1v = kv_scan(k, v, g, fr, N, tp, detach_addr=True)
    print("OLD kernel (slotkern_kvt.kv_scan, G=1) vs ref: kbuf %.1e vbuf %.1e" % (rel(o1k, r1k), rel(o1v, r1v)))
    e = (o1v - r1v).abs().amax(dim=(0, 2, 3, 4)); bad = (e > 1e-3).nonzero()
    print("   first t with err>1e-3:", bad[0].item() if len(bad) else None)
    from slotkern_kvt import kv_scan_fwd; o16k, o16v = kv_scan_fwd(k, v, g, fr, N, tp, G=16)[:2]
    print("OLD kernel at G=16 vs ref[::16]: vbuf %.1e" % rel(o16v, r1v[:, ::16]))
    for nw in (4, 8, 16):
        try:
            gk, gv = KG.kv_scan_g(k, v, g, fr, N, tp, G=16, detach_addr=True, nw=nw)
            print("kv_scan_g nw=%d vs ref[::16]: vbuf %.1e" % (nw, rel(gv, r1v[:, ::16])))
        except Exception as ex:
            print("nw", nw, "error", ex)
    for B_ in (1,):
        gk, gv = KG.kv_scan_g(k[:1].contiguous(), v[:1].contiguous(), g[:1].contiguous(), fr, N, tp, G=16, detach_addr=True)
        print("kv_scan_g batch0 only: vbuf %.1e" % rel(gv, r1v[:1, ::16]))
