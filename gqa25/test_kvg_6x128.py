"""gqa21 kv_scan_g vs the fp64 reference at the 6 x 128 shapes (main 192 rows, encoder 96), plus the old target shape.
Soft address (the reference gradient is exact there); pin=False because the kernel has no zero-row pin.
A HARD forward is also compared: an argmax near-tie can flip between fp32 and fp64, so it reports the share of
(b, t, h) where the chosen row differs instead of failing on it."""
import sys, torch
sys.path.insert(0, "/home/paul/memprobe/gqa21")
import slotkern_kvg as KG, slotkern_kv as KV
assert "gqa21" in KG.__file__ and "gqa21" in KV.__file__
torch.manual_seed(0)
cases = [  # B, T, H, N, DK, DV, G, detach, separate_q, nwb
    (1, 128, 12, 192, 64, 64, 8, True, True, None),     # sh21 main shape
    (1, 128, 6, 192, 128, 128, 8, True, True, None),    # 6x128 main
    (1, 128, 6, 192, 128, 128, 8, True, True, 16),      # 6x128 main, 16 backward warps
    (1, 128, 6, 96, 128, 128, 8, True, True, None),     # 6x128 encoder
    (2, 50, 6, 192, 128, 128, 8, False, False, None),   # address grads through K, short last segment
]
worst = 0.0
for B, T, H, N, DK, DV, G, det, sq, nwb in cases:
    d = dict(dtype=torch.float64, device="cuda")
    k = torch.randn(B, T, H, DK, **d); q = torch.randn(B, T, H, DK, **d) if sq else None
    v = torch.randn(B, T, H, DV, **d); g = torch.rand(B, T, H, **d)
    fr = torch.tensor(2.0, **d); tp = torch.tensor(16.0, **d)
    lr_ = [t.clone().requires_grad_() for t in (k, v, g, fr)] + ([q.clone().requires_grad_()] if sq else [])
    kr, vr, gr, frr = lr_[:4]; qr = lr_[4] if sq else None
    kb_r, vb_r = KV._ref_scan_kv_impl(kr, vr, gr, frr.reshape(1), N, tp, G=G, detach_addr=det, q=qr, pin=False, occ_bin=True)
    wk, wv = torch.randn_like(kb_r), torch.randn_like(vb_r)
    (kb_r * wk + vb_r * wv).sum().backward()
    lt = [t.detach().float().requires_grad_() for t in (k, v, g, fr)] + ([q.detach().float().requires_grad_()] if sq else [])
    kt_, vt_, gt_, frt = lt[:4]; qt_ = lt[4] if sq else None
    kb_t, vb_t = KG.kv_scan_g(kt_, vt_, gt_, frt.reshape(1), N, tp.float(), G=G, detach_addr=det, q=qt_, nwb=nwb, occ_bin=True)
    (kb_t * wk.float() + vb_t * wv.float()).sum().backward()
    rel = lambda a, b: ((a.double() - b.double()).abs().max() / b.double().abs().max().clamp_min(1e-12)).item()
    errs = {"kbuf": rel(kb_t, kb_r), "vbuf": rel(vb_t, vb_r)}
    for nm, a, b in zip(["dk", "dv", "dgate", "dfr"] + (["dq"] if sq else []), lt, lr_):
        errs[nm] = rel(a.grad, b.grad)
    m = max(errs.values()); worst = max(worst, m)
    print("B%d T%-4d H%-2d N%-3d D%d/%d G%d det=%d q=%d nwb=%s | " % (B, T, H, N, DK, DV, G, det, sq, nwb)
          + "  ".join("%s %.1e" % kv for kv in errs.items()) + ("   OK" if m < 1e-4 else "   <-- CHECK"), flush=True)
    # hard forward: kernel fp32 vs reference fp64 -- report the realised state difference
    with torch.no_grad():
        kh_r, vh_r = KV._ref_scan_kv_impl(k, v, g, fr.reshape(1), N, tp, G=G, hard=True, q=q, pin=False, occ_bin=True)
        kh_t, vh_t = KG.kv_scan_g(k.float(), v.float(), g.float(), fr.float().reshape(1), N, tp.float(), G=G, hard=True,
                                  q=None if q is None else q.float(), occ_bin=True)
        rowdiff = ((kh_t.double() - kh_r).abs().amax(-1) > 1e-3).double().mean().item()
        print("      hard fwd: share of (snapshot, head, row) differing > 1e-3: %.4f" % rowdiff, flush=True)
print("\nworst soft relative error: %.2e -> %s" % (worst, "PASS" if worst < 1e-4 else "FAIL"))
