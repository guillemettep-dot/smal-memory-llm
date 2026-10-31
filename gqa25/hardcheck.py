"""kv_scan_g with HARD (top-1 forward, straight-through softmax backward) vs the fp64 reference, at the
sh shape, stride 8, detach 1 -- forward and all five gradients.  The reference is the pinned-free one."""
import sys; sys.path.insert(0, "/home/paul/memprobe/gqa")
import torch
from slotkern_kv import _ref_scan_kv_impl
from slotkern_kvg import kv_scan_g
rel = lambda a, b: ((a.double() - b.double()).abs().max() / b.double().abs().max().clamp_min(1e-30)).item()
for (H, N, T, G, det, hg) in ((12, 192, 128, 8, True, False), (12, 192, 256, 8, True, False), (4, 96, 128, 8, True, False), (12, 192, 128, 8, False, False), (12, 192, 128, 1, True, False)):
    torch.manual_seed(0)
    k0 = torch.randn(1, T, H, 64) * 0.3; v0 = torch.randn(1, T, H, 64); g0 = torch.sigmoid(torch.randn(1, T, H) - 0.5)
    L = [t.double().clone().requires_grad_() for t in (k0, v0, g0, torch.tensor(4.48), torch.tensor(16.0))]
    kb, vb = _ref_scan_kv_impl(L[0], L[1], L[2], L[3], N, L[4], G, True, hg, 1e-4, False, det, None, pin=False)
    torch.manual_seed(1); wk, wv = torch.randn_like(kb), torch.randn_like(vb)
    (kb * wk + vb * wv).sum().backward(); R = [t.grad for t in L]
    lf = [t.detach().float().cuda().requires_grad_() for t in L]
    a, b = kv_scan_g(lf[0], lf[1], lf[2], lf[3], N, lf[4], G=G, hard=True, hard_gate=hg, detach_addr=det)
    (a * wk.float().cuda() + b * wv.float().cuda()).sum().backward()
    print("H%d N%d T%d G%d detach=%d hard=1: fwd k %.1e v %.1e | grad k %.1e v %.1e gate %.1e fr %.1e temp %.1e" % (
        H, N, T, G, det, rel(a.detach().cpu(), kb.detach()), rel(b.detach().cpu(), vb.detach()),
        *[rel(l.grad.cpu(), r) for l, r in zip(lf, R)]), flush=True)
