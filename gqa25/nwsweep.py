import sys, time; sys.path.insert(0, "/home/paul/memprobe/gqa")
import torch
from slotkern_kvg import kv_scan_g
B, T, H, N, D = 2, 1024, 12, 192, 64
torch.manual_seed(0)
k = (torch.randn(B, T, H, D, device="cuda") * 0.3).requires_grad_(); v = torch.randn(B, T, H, D, device="cuda").requires_grad_()
g = torch.sigmoid(torch.randn(B, T, H, device="cuda")).requires_grad_()
fr = torch.tensor(4.48, device="cuda", requires_grad=True); tp = torch.tensor(16.0, device="cuda", requires_grad=True)
ref = None
for nw in (4, 8, 16, 32):
    try:
        def go():
            a, b = kv_scan_g(k, v, g, fr, N, tp, G=16, detach_addr=True, nw=nw)
            return a, b
        a, b = go(); torch.cuda.synchronize()
        wa, wb = torch.randn_like(a), torch.randn_like(b)
        for _ in range(2):
            a, b = go(); (a * wa + b * wb).sum().backward()
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(5): a, b = go()
        torch.cuda.synchronize(); tf = (time.time() - t0) / 5 * 1000
        t0 = time.time()
        for _ in range(5):
            a, b = go(); (a * wa + b * wb).sum().backward()
        torch.cuda.synchronize(); tfb = (time.time() - t0) / 5 * 1000
        k.grad = None; a, b = go(); (a * wa + b * wb).sum().backward()
        gk = k.grad.clone()
        if ref is None: ref = gk
        print("nw=%2d  fwd %.1f ms  bwd %.1f ms  (grad k vs nw=4: %.1e)" % (nw, tf, tfb - tf, ((gk - ref).abs().max() / ref.abs().max()).item()), flush=True)
        k.grad = v.grad = g.grad = None
    except Exception as e:
        print("nw=%d failed: %s" % (nw, str(e)[:150]))
