"""Warp-count dependence of the scan backward = a race (or a reduction bug).  Every nw against an fp64
CPU reference, T=1024, at the NEW shape (192 x 64+64, stride kernel) and the LIVE shape (96 x 64+64, G=1)."""
import sys, os; sys.path.insert(0, "/home/paul/memprobe/gqa")
import torch
from slotkern_kv import _ref_scan_kv_impl
import slotkern_kvt as KT
from slotkern_kvg import kv_scan_g
rel = lambda a, b: ((a.double() - b.double()).abs().max() / b.double().abs().max().clamp_min(1e-30)).item()
T = int(os.environ.get("T", "1024"))
for label, N, G in (("NEW 192 slots, stride 16", 192, 16), ("LIVE 96 slots, G=1", 96, 1)):
    torch.manual_seed(0)
    B, H, D = 1, 2, 64
    k0 = torch.randn(B, T, H, D) * 0.3; v0 = torch.randn(B, T, H, D); g0 = torch.sigmoid(torch.randn(B, T, H))
    fr0 = torch.tensor(4.48); tp0 = torch.tensor(16.0)
    L = [t.double().clone().requires_grad_() for t in (k0, v0, g0, fr0, tp0)]
    kb, vb = _ref_scan_kv_impl(L[0], L[1], L[2], L[3], N, L[4], G, False, False, 1e-4, False, True, None, pin=False)
    torch.manual_seed(1); wk, wv = torch.randn_like(kb), torch.randn_like(vb)
    (kb * wk + vb * wv).sum().backward()
    R = [t.grad for t in L]
    print("%s, T=%d" % (label, T), flush=True)
    for nw in (1, 2, 4, 8, 16):
        lf = [t.detach().float().cuda().requires_grad_() for t in L]
        try:
            if G == 1:
                a, b = KT._KVScan.apply(lf[0], lf[0], lf[1], lf[2], lf[3], lf[4], N, False, False, 1e-4, True) if hasattr(KT, "_KVScan") and False else (None, None)
                # the live wrapper picks nw itself; call it with nw forced through the module knob
                _orig = KT._nw; KT._nw = lambda *a_, _n=nw: _n
                try:
                    a, b = KT.kv_scan(lf[0], lf[1], lf[2], lf[3], N, lf[4], detach_addr=True)
                    (a * wk.float().cuda() + b * wv.float().cuda()).sum().backward()
                finally:
                    KT._nw = _orig
            else:
                a, b = kv_scan_g(lf[0], lf[1], lf[2], lf[3], N, lf[4], G=G, detach_addr=True, nw=nw)
                (a * wk.float().cuda() + b * wv.float().cuda()).sum().backward()
            print("   nw=%2d  fwd v %.1e | grad k %.1e v %.1e gate %.1e fr %.1e temp %.1e" % (
                nw, rel(b.detach().cpu(), vb.detach()), *[rel(l.grad.cpu(), r) for l, r in zip(lf, R)]), flush=True)
        except Exception as e:
            print("   nw=%2d failed: %s" % (nw, str(e)[:120]), flush=True)
