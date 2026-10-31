"""Time the training scan (hard, occ_bin, per-token fr, detached address, G=8, T=1024, B=2) for the whole memory:
6 x 128 with the CUDA scan (sh22/sh23 today) vs 1 x 768 (the proposed shape; the CUDA scan only takes D 64/128,
so it falls back to the Triton scan).  Main 192 rows, encoder 96.  GPU memory capped beside the trainer."""
import sys, time, torch
sys.path.insert(0, "/home/paul/memprobe/gqa23")
torch.cuda.set_per_process_memory_fraction(0.2, 0)
import slotkern_kvg as KG, slotkern_cuda as KC
assert "gqa23" in KG.__file__ and "gqa23" in KC.__file__
dev, T, G, B = "cuda", 1024, 8, 2
def run(H, N, D, fn, reps=3):
    torch.manual_seed(0)
    k = torch.randn(B, T, H, D, device=dev).requires_grad_(); q = torch.randn(B, T, H, D, device=dev).requires_grad_()
    v = torch.randn(B, T, H, D, device=dev).requires_grad_(); g = torch.rand(B, T, H, device=dev).requires_grad_()
    fr = (torch.rand(B, T, H, device=dev) * 2).requires_grad_(); tp = torch.tensor(16.0, device=dev)
    def once():
        kb, vb = fn(k, v, g, fr, N, tp, G=G, hard=True, detach_addr=True, q=q, occ_bin=True)
        torch.cuda.synchronize(); t1 = time.time()
        (kb.sum() + vb.sum()).backward(); torch.cuda.synchronize()
        return t1
    once()
    tf = tb = 0.0
    for _ in range(reps):
        torch.cuda.synchronize(); t0 = time.time(); t1 = once(); t2 = time.time()
        tf += (t1 - t0) / reps * 1e3; tb += (t2 - t1) / reps * 1e3
    return tf, tb
for label, H, D, fn in [("6 x 128 CUDA  ", 6, 128, KC.kv_scan_c), ("6 x 128 Triton", 6, 128, KG.kv_scan_g), ("1 x 768 Triton", 1, 768, KG.kv_scan_g)]:
    for mem, N in (("main", 192), ("enc ", 96)):
        try:
            tf, tb = run(H, N, D, fn)
            print("%s %s N%-3d | fwd %8.1f ms  bwd %8.1f ms  total %8.1f ms" % (label, mem, N, tf, tb, tf + tb), flush=True)
        except Exception as e:
            print(label, mem, "FAILED", type(e).__name__, str(e)[:200], flush=True)
        torch.cuda.empty_cache()
print("\nTriton kernels (warps / regs / spills):")
for fn in (KG._kvfwd, KG._kvbwd_g):
    for dv_, c in getattr(fn, "device_caches", {}).items():
        kc = c[0] if isinstance(c, tuple) else c
        for key, ck in kc.items():
            md = getattr(ck, "metadata", None)
            print("  %-9s warps %-3s regs %-4s spills %-6s %s" % (fn.fn.__name__, getattr(md, "num_warps", "?"), getattr(ck, "n_regs", "?"), getattr(ck, "n_spills", "?"), str(key)[:90]))
