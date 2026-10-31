"""Case 3 diagnosis: is a 2e-3 gradient error a logic bug in the stride-G backward, or fp32 rounding
amplified by the recurrence (detach=0 + separate query)?  Same inputs everywhere."""
import sys, torch
sys.path.insert(0, "/home/paul/memprobe/gqa")
from slotkern_kv import _ref_scan_kv_impl, ref_scan_kv
from slotkern_kvg import kv_scan_g
from slotkern_kvt import kv_scan as kv_scan_g1
torch.manual_seed(0)
B, T, H, N, DK, DV = 1, 64, 2, 24, 16, 16
for t_ in range(2): torch.randn(1)
k = torch.randn(B, T, H, DK, dtype=torch.float64, device="cuda"); q = torch.randn_like(k)
v = torch.randn(B, T, H, DV, dtype=torch.float64, device="cuda"); g = torch.rand(B, T, H, dtype=torch.float64, device="cuda")
fr = torch.tensor([2.0], dtype=torch.float64, device="cuda"); tp = torch.tensor(16.0, dtype=torch.float64, device="cuda")
def run(fn, dtype, G, det, **kw):
    L = [t.detach().to(dtype).requires_grad_() for t in (k, v, g, fr, q)]
    kb, vb = fn(L[0], L[1], L[2], L[3], N, tp.to(dtype), G=G, detach_addr=det, q=L[4], **kw) if fn is not kv_scan_g1 else \
             fn(L[0], L[1], L[2], L[3], N, tp.to(dtype), detach_addr=det, q=L[4])
    torch.manual_seed(5); wk = torch.randn(kb.shape, dtype=torch.float64, device="cuda"); wv = torch.randn(vb.shape, dtype=torch.float64, device="cuda")
    (kb * wk.to(kb.dtype) + vb * wv.to(vb.dtype)).sum().backward()
    return [kb.detach().double(), vb.detach().double()] + [x.grad.double() for x in L]
rel = lambda a, b: ((a - b).abs().max() / b.abs().max().clamp_min(1e-12)).item()
names = ["kbuf", "vbuf", "dk", "dv", "dgate", "dfr", "dq"]
def show(lbl, A, Bb):
    print("%-46s " % lbl + "  ".join("%s %.1e" % (n, rel(a, b)) for n, a, b in zip(names, A, Bb)))
for G in (7, 1):
    for det in (False, True):
        ref64 = run(_ref_scan_kv_impl, torch.float64, G, det)
        ref32 = run(_ref_scan_kv_impl, torch.float32, G, det)
        mine = run(kv_scan_g, torch.float32, G, det)
        print("--- G=%d detach=%d ---" % (G, det))
        show("(1) fp32 reference   vs fp64 reference", ref32, ref64)
        show("(3) stride-G kernel  vs fp32 reference", mine, ref32)
        show("    stride-G kernel  vs fp64 reference", mine, ref64)
        if G == 1:
            old = run(kv_scan_g1, torch.float32, 1, det)
            show("(2) OLD G=1 kernel   vs fp64 reference", old, ref64)
            show("    stride-G kernel  vs OLD G=1 kernel", mine, old)
