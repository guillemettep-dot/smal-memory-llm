import os, sys
sys.path.insert(0, "/home/paul/memprobe/gqa")
import torch
from slotkern_kv import ref_scan_kv
from slotkern_kvt import kv_scan, kv_scan_fwd
import slotkern_kvg as KG
d = torch.load("/home/paul/memprobe/gqa/_scan_inputs.pt")
k, v, g, fr, tp = (d[n].cuda() for n in ("k", "v", "g", "fr", "tp"))
k, v, g = k[:1].contiguous(), v[:1].contiguous(), g[:1].contiguous()      # one sequence: small
N = 192
def rel(a, b): return ((a - b).abs().max() / b.abs().max().clamp_min(1e-30)).item()
with torch.no_grad():
    rk, rv = ref_scan_kv(k, v, g, fr, N, tp, G=1, detach_addr=True)
    ok_, ov = kv_scan(k, v, g, fr, N, tp, detach_addr=True)
    print("OLD kernel G=1 (LIVE RUN'S KERNEL) vs ref: kbuf %.1e vbuf %.1e" % (rel(ok_, rk), rel(ov, rv)))
    e = (ov - rv).abs().amax(dim=(0, 2, 3, 4)); bad = (e > 1e-3).nonzero()
    print("   first t with err>1e-3:", bad[0].item() if len(bad) else None)
    del ok_, ov
    rk16, rv16 = rk[:, ::16].contiguous(), rv[:, ::16].contiguous(); del rk, rv
    a, b, _ = kv_scan_fwd(k, v, g, fr, N, tp, G=16)
    print("OLD fwd kernel G=16 vs ref[::16]: kbuf %.1e vbuf %.1e" % (rel(a, rk16), rel(b, rv16)))
    for nw in (None, 4, 8, 16):
        a, b = KG.kv_scan_g(k, v, g, fr, N, tp, G=16, detach_addr=True, nw=nw)
        print("kv_scan_g nw=%s: kbuf %.1e vbuf %.1e" % (nw, rel(a, rk16), rel(b, rv16)))
    print("nw chosen by _nw:", __import__("slotkern_kvt")._nw(256, 64, 64), " by _nw_g:", KG._nw_g(256, 64, 64))
