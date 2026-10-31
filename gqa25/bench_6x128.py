"""Time the REAL training scan (hard address, occ_bin, per-token fr, detached address, G=8, T=1024) for the
whole memory at once (grid = B*H programs, as in the model): sh21 (12 x 64) vs 6 x 128, batch 2 and 4,
main (192 rows) and encoder (96 rows); then report registers/spills of every compiled scan kernel."""
import sys, time, torch
sys.path.insert(0, "/home/paul/memprobe/gqa21")
import slotkern_kvg as KG
assert "gqa21" in KG.__file__
dev, T, G = "cuda", 1024, 8
def run(B, H, N, D, nw=None, nwb=None, reps=3):
    torch.manual_seed(0)
    k = torch.randn(B, T, H, D, device=dev).requires_grad_(); q = torch.randn(B, T, H, D, device=dev).requires_grad_()
    v = torch.randn(B, T, H, D, device=dev).requires_grad_(); g = torch.rand(B, T, H, device=dev).requires_grad_()
    fr = (torch.rand(B, T, H, device=dev) * 2).requires_grad_(); tp = torch.tensor(16.0, device=dev)
    def once():
        kb, vb = KG.kv_scan_g(k, v, g, fr, N, tp, G=G, hard=True, detach_addr=True, q=q, nw=nw, nwb=nwb, occ_bin=True)
        torch.cuda.synchronize(); t1 = time.time()
        (kb.sum() + vb.sum()).backward(); torch.cuda.synchronize()
        return t1
    once()                                            # compile
    tf = tb = 0.0
    for _ in range(reps):
        torch.cuda.synchronize(); t0 = time.time(); t1 = once(); t2 = time.time()
        tf += (t1 - t0) / reps * 1e3; tb += (t2 - t1) / reps * 1e3
    return tf, tb
rows = []
for label, H, D in [("12 x 64 ", 12, 64), ("6 x 128", 6, 128)]:
    for B in (2, 4):
        for mem, N in (("main", 192), ("enc ", 96)):
            nwbs = (None, 16) if (D == 128 and mem == "main") else (None,)
            for nwb in nwbs:
                tf, tb = run(B, H, N, D, nwb=nwb)
                print("%s  B%d  %s N%-3d nwb=%-4s | fwd %7.1f ms  bwd %7.1f ms  total %7.1f ms  per 1k tokens %6.1f ms"
                      % (label, B, mem, N, nwb, tf, tb, tf + tb, (tf + tb) / B), flush=True)
                torch.cuda.empty_cache()
print("\ncompiled scan kernels (registers / spilled bytes):")
for fn in (KG._kvfwd, KG._kvbwd_g):
    caches = getattr(fn, "device_caches", None)
    seen = 0
    if caches is not None:
        for dv_, c in caches.items():
            kc = c[0] if isinstance(c, tuple) else c
            for key, ck in kc.items():
                md = getattr(ck, "metadata", None)
                print("  %-9s warps %-3s regs %-4s spills %-6s  %s" % (fn.fn.__name__, getattr(md, "num_warps", "?"),
                      getattr(ck, "n_regs", "?"), getattr(ck, "n_spills", "?"), str(key)[:110]))
                seen += 1
    if not seen: print("  could not read the cache of", fn)
