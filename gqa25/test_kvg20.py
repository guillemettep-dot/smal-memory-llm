import sys, torch
sys.path.insert(0, "/home/paul/memprobe/gqa20")
from slotkern_kv import _ref_scan_kv_impl
from slotkern_kvg import kv_scan_g
torch.manual_seed(0)
cases = [  # B, T, H, N, DK, DV, G, detach, separate_q
    (2, 50, 3, 12, 8, 8, 16, False, False),    # last segment shorter (50 = 3*16 + 2)
    (2, 50, 3, 12, 8, 8, 16, True, False),     # the training setting (KV_DETACH=1)
    (1, 64, 2, 24, 16, 16, 7, False, True),    # stride not dividing T, separate search query
    (1, 40, 2, 10, 8, 8, 1, True, False),      # G=1 must reproduce the old kernel's contract
    (1, 128, 12, 192, 64, 64, 16, True, False),  # the target shape
]
worst = 0.0
for B, T, H, N, DK, DV, G, det, sq in cases:
    k = torch.randn(B, T, H, DK, dtype=torch.float64, device="cuda")
    q = torch.randn(B, T, H, DK, dtype=torch.float64, device="cuda") if sq else None
    v = torch.randn(B, T, H, DV, dtype=torch.float64, device="cuda")
    g = torch.rand(B, T, H, dtype=torch.float64, device="cuda")
    fr = torch.tensor(2.0, dtype=torch.float64, device="cuda")
    tp = torch.tensor(16.0, dtype=torch.float64, device="cuda")
    leaves_r = [t.clone().requires_grad_() for t in (k, v, g, fr)] + ([q.clone().requires_grad_()] if sq else [])
    kr, vr, gr, frr = leaves_r[:4]; qr = leaves_r[4] if sq else None
    # the INNER impl: ref_scan_kv itself casts to fp32, so it is not a float64 reference
    kb_r, vb_r = _ref_scan_kv_impl(kr, vr, gr, frr.reshape(1), N, tp, G=G, detach_addr=det, q=qr)
    wk, wv = torch.randn_like(kb_r), torch.randn_like(vb_r)
    (kb_r * wk + vb_r * wv).sum().backward()
    leaves_t = [t.detach().float().requires_grad_() for t in (k, v, g, fr)] + ([q.detach().float().requires_grad_()] if sq else [])
    kt_, vt_, gt_, frt = leaves_t[:4]; qt_ = leaves_t[4] if sq else None
    kb_t, vb_t = kv_scan_g(kt_, vt_, gt_, frt.reshape(1), N, tp.float(), G=G, detach_addr=det, q=qt_)
    (kb_t * wk.float() + vb_t * wv.float()).sum().backward()
    rel = lambda a, b: ((a.double() - b.double()).abs().max() / b.double().abs().max().clamp_min(1e-12)).item()
    errs = {"kbuf": rel(kb_t, kb_r), "vbuf": rel(vb_t, vb_r)}
    names = ["dk", "dv", "dgate", "dfr"] + (["dq"] if sq else [])
    for nm, a, b in zip(names, leaves_t, leaves_r):
        errs[nm] = rel(a.grad, b.grad)
    m = max(errs.values()); worst = max(worst, m)
    print("B%d T%-4d H%-2d N%-3d D%d/%d G%-2d detach=%d q=%d | " % (B, T, H, N, DK, DV, G, det, sq)
          + "  ".join("%s %.1e" % kv for kv in errs.items()) + ("   OK" if m < 1e-4 else "   <-- CHECK"))
print("\nworst relative error over all cases: %.2e  ->  %s" % (worst, "PASS" if worst < 1e-4 else "FAIL"))
