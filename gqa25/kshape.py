"""Which per-head memory shapes can the key-matched scan run without spilling, on this GPU?
Compiles the REAL forward and backward kernels at each candidate shape and several warp counts,
reports registers/spills from the compiled binary, and times a T=1024 forward+backward of ONE head.
Timing runs beside the live training job, so absolute ms are inflated; compare shapes to each other."""
import sys, time, torch, triton
sys.path.insert(0, "/home/paul/memprobe/gqa")
import slotkern_kvt as S
print("triton", triton.__version__, "| torch", torch.__version__, "|", torch.cuda.get_device_name(0))
dev = "cuda"
T = 1024
shapes = [("current  96 x (64+64)",   96,  64,  64),
          ("192 x (64+64)",           192, 64,  64),
          ("256 x (64+64)",           256, 64,  64),
          ("128 x (128+128)",         128, 128, 128),
          ("asked 192 x (128+128)",   192, 128, 128)]
def info(ck):
    return getattr(ck, "n_regs", "?"), getattr(ck, "n_spills", "?")
for name, N, DK, DV in shapes:
    NP, DKP, DVP = S._p2(N), S._p2(DK), S._p2(DV)
    for nw in ((8,) if N == 96 else (8, 16, 32)):
        torch.manual_seed(0)
        B, H = 1, 1
        k = torch.randn(B, T, H, DK, device=dev); q = k.clone()
        v = torch.randn(B, T, H, DV, device=dev); g = torch.rand(B, T, H, device=dev)
        fr = torch.tensor([2.0], device=dev); temp = torch.tensor([16.0], device=dev)
        kbuf = torch.empty(B, T, H, N, DK, device=dev); vbuf = torch.empty(B, T, H, N, DV, device=dev)
        obuf = torch.empty(B, T, H, N, device=dev)
        try:
            torch.cuda.synchronize(); t0 = time.time()
            ckf = S._kvfwd[(B * H,)](k, q, v, g, fr, temp, kbuf, vbuf, obuf, T, H, 1, T, 1e-4,
                                     N=N, DK=DK, DV=DV, NP=NP, DKP=DKP, DVP=DVP, HARD=False, HARDG=False, num_warps=nw)
            torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(3):
                S._kvfwd[(B * H,)](k, q, v, g, fr, temp, kbuf, vbuf, obuf, T, H, 1, T, 1e-4,
                                   N=N, DK=DK, DV=DV, NP=NP, DKP=DKP, DVP=DVP, HARD=False, HARDG=False, num_warps=nw)
            torch.cuda.synchronize(); tf = (time.time() - t0) / 3 * 1000
            dkb = torch.randn_like(kbuf); dvb = torch.randn_like(vbuf)
            dk = torch.zeros_like(k); dq = torch.zeros_like(q); dv = torch.zeros_like(v); dg = torch.zeros_like(g)
            dfr = torch.zeros(1, device=dev); dtp = torch.zeros(1, device=dev)
            args = (k, q, v, g, fr, temp, kbuf, vbuf, obuf, dkb, dvb, dk, dq, dv, dg, dfr, dtp, T, H, 1e-4)
            kw = dict(N=N, DK=DK, DV=DV, NP=NP, DKP=DKP, DVP=DVP, HARD=False, HARDG=False, DETACH=False, num_warps=nw)
            ckb = S._kvbwd[(B * H,)](*args, **kw); torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(3): S._kvbwd[(B * H,)](*args, **kw)
            torch.cuda.synchronize(); tb = (time.time() - t0) / 3 * 1000
            rf, sf = info(ckf); rb, sb = info(ckb)
            print("%-24s nw=%2d | fwd regs %-4s spills %-6s %7.1f ms | bwd regs %-4s spills %-6s %7.1f ms"
                  % (name, nw, rf, sf, tf, rb, sb, tb), flush=True)
        except Exception as e:
            print("%-24s nw=%2d | FAILED: %s" % (name, nw, str(e).split("\n")[0][:90]), flush=True)
        del kbuf, vbuf, obuf
        torch.cuda.empty_cache()
