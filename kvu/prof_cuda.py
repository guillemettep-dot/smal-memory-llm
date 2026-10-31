"""Time each of the four CUDA scan kernels separately (CUDA events), plus the Python-side overhead of the
autograd wrapper, at the training setting.  Wraps every compiled kernel's __call__ to record events around it."""
import sys, os, time, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, "/home/paul/memprobe/gqa21")
import slotkern_cuda as KC
dev, T, G = "cuda", 1024, 8
REC = {}
def wrap(ks):
    for name, kern in list(ks.items()):
        if getattr(kern, "_wrapped", False): continue
        call = kern.__call__
        class W:
            _wrapped = True
            def __init__(s, kern, name): s.kern, s.name, s.func = kern, name, kern.func
            def __call__(s, **kw):
                e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                e0.record(); s.kern(**kw); e1.record()
                REC.setdefault(s.name, []).append((e0, e1))
        ks[name] = W(kern, name)
for label, B, H, N, D in [("sh21 main", 2, 12, 192, 64), ("sh21 enc ", 2, 12, 96, 64), ("6x128 main", 2, 6, 192, 128),
                          ("6x128 enc ", 2, 6, 96, 128)]:
    torch.manual_seed(0)
    k, q, v = (torch.randn(B, T, H, D, device=dev).requires_grad_() for _ in range(3))
    g = torch.rand(B, T, H, device=dev).requires_grad_(); fr = (torch.rand(B, T, H, device=dev) * 2).requires_grad_()
    tp = torch.tensor(16.0, device=dev, requires_grad=True)
    ks, _ = KC._kernels(N, D, True, True, 1e-4); wrap(ks)
    for rep in range(4):
        REC.clear()
        torch.cuda.synchronize(); t0 = time.time()
        kb, vb = KC.kv_scan_c(k, v, g, fr, N, tp, G=G, hard=True, detach_addr=True, q=q, occ_bin=True)
        (kb.sum() + vb.sum()).backward()
        torch.cuda.synchronize(); wall = (time.time() - t0) * 1e3
    parts = {n: sum(a.elapsed_time(b) for a, b in v) for n, v in REC.items()}
    print(f"{label} B{B}: " + "  ".join(f"{n} {t:6.2f}" for n, t in parts.items())
          + f" | kernels {sum(parts.values()):6.2f} ms, wall fwd+bwd {wall:6.2f} ms", flush=True)
