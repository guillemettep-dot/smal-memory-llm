import sys; sys.path.insert(0, "/home/paul/memprobe/gqa")
import torch
from slotkern_kv import _ref_scan_kv_impl
from slotkern_kvt import kv_scan
from slotkern_kvg import kv_scan_g
d = torch.load("/home/paul/memprobe/gqa/_scan_inputs.pt")
rel = lambda a, b: ((a.double() - b.double()).abs().max() / b.double().abs().max().clamp_min(1e-30)).item()
for detach in (True, False):
  for G in (1, 16):
    leaves = [d[n][:1].clone().double().requires_grad_() for n in ("k", "v", "g")] + \
             [d[n].clone().double().requires_grad_() for n in ("fr", "tp")]
    torch.manual_seed(3)
    kb, vb = _ref_scan_kv_impl(*leaves[:4], 192, leaves[4], G, False, False, 1e-4, False, detach, None, pin=False)
    wk, wv = torch.randn_like(kb), torch.randn_like(vb)
    (kb * wk + vb * wv).sum().backward()
    ref = [t.grad.clone() for t in leaves]
    lf = [t.detach().float().cuda().requires_grad_() for t in leaves]
    if G == 1:
        a, b = kv_scan(lf[0], lf[1], lf[2], lf[3], 192, lf[4], detach_addr=detach)
    else:
        a, b = kv_scan_g(lf[0], lf[1], lf[2], lf[3], 192, lf[4], G=16, detach_addr=detach)
    (a * wk.float().cuda() + b * wv.float().cuda()).sum().backward()
    print("%-22s detach=%d  fwd v %.1e | grads k %.1e v %.1e gate %.1e fr %.1e temp %.1e" % (
        "kv_scan (LIVE, G=1)" if G == 1 else "kv_scan_g (G=16)", detach, rel(b.cpu(), vb),
        *[rel(l.grad.cpu(), r) if l.grad is not None else float("nan") for l, r in zip(lf, ref)]), flush=True)
    del leaves, lf, kb, vb, a, b, ref, wk, wv; torch.cuda.empty_cache()
