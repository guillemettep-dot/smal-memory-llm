# -*- coding: utf-8 -*-
"""Does scan time scale with the rack the model must walk per token, H*N*(DK+DV)?
If it does, the 2-head geometry costs exactly 2x h1's scan -- the state doubling is free in
COMPRESSION RATIO but not in TIME."""
import os, time, torch
os.environ.setdefault("KV_DETACH", "1")
from slotkern_kv import scan_kv
dev="cuda"; B,T=1,1024
def timed(fn, iters=6, warmup=2):
    for _ in range(warmup): fn()
    torch.cuda.synchronize(); t0=time.perf_counter()
    for _ in range(iters): fn()
    torch.cuda.synchronize(); return (time.perf_counter()-t0)/iters
fr=torch.rand(1,device=dev)*0.1; temp=torch.ones(1,device=dev)
print("  %-26s %10s %12s %10s"%("geometry","rack/token","fwd+bwd ms","ms per 1k elems"))
base=None
for label,H,N,DK,DV in (("h1      1 head  x 96x64",1,96,64,64),
                        ("THIS    2 heads x 96x64",2,96,64,64),
                        ("        2 heads x 48x64",2,48,64,64),
                        ("        1 head  x 96x32",1,96,32,32)):
    k=torch.randn(B,T,H,DK,device=dev,requires_grad=True)
    v=torch.randn(B,T,H,DV,device=dev,requires_grad=True)
    g=torch.rand(B,T,H,device=dev,requires_grad=True)
    def fb():
        kb,bf=scan_kv(k.float(),v.float(),g,fr,N,temp,detach_addr=True)
        (kb.sum()+bf.sum()).backward()
    t=timed(fb); rack=H*N*(DK+DV)
    if base is None: base=(t,rack)
    print("  %-26s %10d %12.2f %10.3f   %s"%(label,rack,t*1e3,t*1e3/(rack/1000),
          "" if base[1]==rack else "%.2fx h1 time, %.2fx h1 rack"%(t/base[0], rack/base[1])))
    del k,v,g; torch.cuda.empty_cache()
