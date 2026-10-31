"""Triton kv_scan_g vs CUDA kv_scan_c at the training setting, whole memory at once (grid = B*H as in the model).
Alternates the two per shape; on a shared GPU compare the columns, not absolute ms."""
import sys, os, time, torch
sys.path.insert(0, "/home/paul/memprobe/gqa21")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import slotkern_kvg as KG, slotkern_cuda as KC
dev, T, G = "cuda", 1024, 8
def timed(fn, B, H, N, D, reps=5):
    torch.manual_seed(0)
    k, q, v = (torch.randn(B, T, H, D, device=dev).requires_grad_() for _ in range(3))
    g = torch.rand(B, T, H, device=dev).requires_grad_(); fr = (torch.rand(B, T, H, device=dev) * 2).requires_grad_()
    tp = torch.tensor(16.0, device=dev, requires_grad=True)
    def once():
        torch.cuda.synchronize(); t0 = time.time()
        kb, vb = fn(k, v, g, fr, N, tp, G=G, hard=True, detach_addr=True, q=q, occ_bin=True)
        torch.cuda.synchronize(); t1 = time.time()
        (kb.sum() + vb.sum()).backward(); torch.cuda.synchronize()
        return t1 - t0, time.time() - t1
    once(); once()
    f = b = 0.0
    for _ in range(reps):
        a, c = once(); f += a / reps * 1e3; b += c / reps * 1e3
    return f, b
for label, B, H, N, D in [("sh21 main 12x192x64", 2, 12, 192, 64), ("sh21 enc  12x96x64 ", 2, 12, 96, 64),
                          ("sh21 main B4       ", 4, 12, 192, 64),
                          ("6x128 main         ", 2, 6, 192, 128), ("6x128 enc          ", 2, 6, 96, 128),
                          ("6x128 main B4      ", 4, 6, 192, 128), ("6x128 enc  B4      ", 4, 6, 96, 128)]:
    f0, b0 = timed(KG.kv_scan_g, B, H, N, D)
    f1, b1 = timed(KC.kv_scan_c, B, H, N, D)
    print(f"{label} B{B} | triton {f0:6.1f} + {b0:6.1f} = {f0+b0:6.1f} ms | cuda {f1:6.1f} + {b1:6.1f} = {f1+b1:6.1f} ms | x{(f0+b0)/(f1+b1):.2f}", flush=True)
    torch.cuda.empty_cache()
