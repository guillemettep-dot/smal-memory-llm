"""Time the replay backward (kv_scan_g) against the undo-log backward (kv_scan_u) at the training setting
(hard address, occupancy indicator, per-token fr, detached address, separate query, G=8, T=1024), then print
registers / spills of every compiled undo kernel.  Old and new alternate within each shape, so a busy GPU
inflates both alike; compare the two columns, not absolute ms."""
import sys, os, time, torch
sys.path.insert(0, os.environ.get("CODEDIR", "/home/paul/memprobe/gqa21"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import slotkern_kvg as KG, slotkern_kvu as KU
assert "gqa21" in KG.__file__
dev, T, G = "cuda", 1024, 8
def timed(fn, B, H, N, D, reps, nw=None, nwb=None):
    torch.manual_seed(0)
    k, q, v = (torch.randn(B, T, H, D, device=dev).requires_grad_() for _ in range(3))
    g = torch.rand(B, T, H, device=dev).requires_grad_(); fr = (torch.rand(B, T, H, device=dev) * 2).requires_grad_()
    tp = torch.tensor(16.0, device=dev)
    def once():
        torch.cuda.synchronize(); t0 = time.time()
        kb, vb = fn(k, v, g, fr, N, tp, G=G, hard=True, detach_addr=True, q=q, occ_bin=True, nw=nw, nwb=nwb)
        torch.cuda.synchronize(); t1 = time.time()
        (kb.sum() + vb.sum()).backward(); torch.cuda.synchronize()
        return t1 - t0, time.time() - t1
    once()
    f = b = 0.0
    for _ in range(reps):
        a, c = once(); f += a / reps * 1e3; b += c / reps * 1e3
    return f, b
for label, B, H, N, D in [("sh21 main  12x192x64 ", 2, 12, 192, 64), ("sh21 enc   12x96x64  ", 2, 12, 96, 64),
                          ("6x128 main 6x192x128 ", 2, 6, 192, 128), ("6x128 enc  6x96x128  ", 2, 6, 96, 128),
                          ("6x128 main B4        ", 4, 6, 192, 128)]:
    f0, b0 = timed(KG.kv_scan_g, B, H, N, D, 3)
    line = f"{label} B{B} | replay {f0:6.1f}+{b0:6.1f} = {f0+b0:6.1f} ms |"
    for nw in (8, 16):
        for nwb in (8, 16):
            f1, b1 = timed(KU.kv_scan_u, B, H, N, D, 3, nw=nw, nwb=nwb)
            line += f" u{nw}/{nwb} {f1:5.1f}+{b1:5.1f}={f1+b1:6.1f}"
    torch.cuda.empty_cache()
    print(line, flush=True)
print("\nundo kernels (warps / registers / spilled bytes):")
for fn in (KU._kvfwd_u, KU._kvbwd_u):
    for dv_, c in getattr(fn, "device_caches", {}).items():
        kc = c[0] if isinstance(c, tuple) else c
        for key, ck in kc.items():
            md = getattr(ck, "metadata", None)
            print("  %-9s warps %-3s regs %-4s spills %s" % (fn.fn.__name__, getattr(md, "num_warps", "?"),
                  getattr(ck, "n_regs", "?"), getattr(ck, "n_spills", "?")))
