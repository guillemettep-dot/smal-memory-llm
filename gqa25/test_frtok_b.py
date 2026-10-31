"""sh20 kernel test: per-token fr (B,T,H) in kv_scan_g vs the float64 torch reference, all gradients incl. dfr per token.
Also: a per-token fr that is CONSTANT must equal the scalar path (forward exactly, dfr summed == scalar dfr).
Keys are CLUSTERED so taken rows genuinely compete with the free row (otherwise fr would never matter)."""
import sys, torch
CODE = sys.argv[1] if len(sys.argv) > 1 else "/home/paul/memprobe/gqa20"
sys.path.insert(0, CODE)
torch.cuda.set_per_process_memory_fraction(0.15, 0)
from slotkern_kv import _ref_scan_kv_impl
from slotkern_kvg import kv_scan_g
torch.manual_seed(0)
dev = "cuda"

def clustered(B, T, H, D, ncl, noise):
    cen = torch.randn(B, ncl, H, D, dtype=torch.float64, device=dev)
    idx = torch.randint(0, ncl, (B, T), device=dev)
    x = torch.gather(cen, 1, idx[:, :, None, None].expand(B, T, H, D))
    return x + noise * torch.randn(B, T, H, D, dtype=torch.float64, device=dev)

rel = lambda a, b: ((a.double() - b.double()).abs().max() / b.double().abs().max().clamp_min(1e-12)).item()
cases = [
    (2, 50, 3, 12, 8, 8, 16, False, False, True),
    (2, 50, 3, 12, 8, 8, 16, False, True, False),
    (2, 50, 3, 12, 8, 8, 16, False, False, False),
    (1, 512, 12, 192, 64, 64, 8, True, True, True),
    (2, 1024, 12, 192, 64, 64, 8, True, True, True),
]
worst = 0.0
merge_tot = 0
for B, T, H, N, DK, DV, G, hard, occb, det in cases:
    k = clustered(B, T, H, DK, 6, 0.3)
    q = k + 0.2 * torch.randn_like(k)
    v = torch.randn(B, T, H, DV, dtype=torch.float64, device=dev)
    g = torch.rand(B, T, H, dtype=torch.float64, device=dev)
    tp = torch.tensor(16.0, dtype=torch.float64, device=dev)
    fr = tp * (0.2 + 1.3 * torch.rand(B, T, H, dtype=torch.float64, device=dev))   # fr/temp in [0.2, 1.5]
    L = [t.clone().requires_grad_() for t in (k, q, v, g, fr)]
    tr = tp.clone().requires_grad_()
    kb_r, vb_r = _ref_scan_kv_impl(*[L[0], L[2], L[3], L[4]], N, tr, G=G, hard=hard, detach_addr=det, q=L[1], pin=False, occ_bin=occb)
    wk, wv = torch.randn_like(kb_r), torch.randn_like(vb_r)
    (kb_r * wk + vb_r * wv).sum().backward()
    M = [t.detach().float().requires_grad_() for t in (k, q, v, g, fr)]
    tt = tp.detach().float().requires_grad_()
    kb_t, vb_t = kv_scan_g(M[0], M[2], M[3], M[4], N, tt, G=G, hard=hard, detach_addr=det, q=M[1], occ_bin=occb)
    (kb_t * wk.float() + vb_t * wv.float()).sum().backward()
    errs = {"kbuf": rel(kb_t, kb_r), "vbuf": rel(vb_t, vb_r)}
    for nm, a, b in zip(["dk", "dq", "dv", "dg", "dfr"], M, L):
        errs[nm] = rel(a.grad, b.grad)
    errs["dtemp"] = rel(tt.grad, tr.grad)
    assert M[4].grad.shape == (B, T, H)
    nz = (L[4].grad.abs() > 1e-9).float().mean().item()
    m = max(errs.values()); worst = max(worst, m)
    print("B%d T%-4d H%-2d N%-3d D%d/%d G%-2d hard=%d occb=%d det=%d | " % (B, T, H, N, DK, DV, G, hard, occb, det)
          + "  ".join("%s %.1e" % kv for kv in errs.items()) + "  | dfr nonzero %.0f%%" % (100 * nz) + ("   OK" if m < 1e-4 else "   <-- CHECK"))

# constant per-token fr == scalar fr
B, T, H, N, D = 2, 128, 4, 32, 16
k = clustered(B, T, H, D, 5, 0.3).float(); q = (k + 0.2 * torch.randn_like(k)); v = torch.randn(B, T, H, D, device=dev); g = torch.rand(B, T, H, device=dev)
for hard in (False, True):
    fr_s = torch.tensor([9.0], device=dev, requires_grad=True)
    fr_t = torch.full((B, T, H), 9.0, device=dev, requires_grad=True)
    t1 = torch.tensor(16.0, device=dev)
    a = kv_scan_g(k, v, g, fr_s, N, t1, G=8, hard=hard, detach_addr=True, q=q, occ_bin=True)
    b = kv_scan_g(k, v, g, fr_t, N, t1, G=8, hard=hard, detach_addr=True, q=q, occ_bin=True)
    same = all(torch.equal(x, y) for x, y in zip(a, b))
    w = [torch.randn_like(x) for x in a]
    sum(((x * wx).sum() for x, wx in zip(a, w))).backward(); sum(((x * wx).sum() for x, wx in zip(b, w))).backward()
    d = abs(fr_s.grad.item() - fr_t.grad.sum().item()) / max(abs(fr_s.grad.item()), 1e-9)
    print(f"constant per-token fr == scalar (hard={hard}): forward bit-identical {same}, dfr sum rel diff {d:.1e}")
    worst = max(worst, 0 if same else 1, d)
print("\nworst relative error: %.2e  ->  %s" % (worst, "PASS" if worst < 1e-4 else "FAIL"))
