import sys, torch
sys.path.insert(0, "/home/paul/memprobe/gqa20")
torch.cuda.set_per_process_memory_fraction(0.12,0)
from slotkern_kv import _ref_scan_kv_impl
from slotkern_kvg import kv_scan_g
torch.manual_seed(0); dev="cuda"
def clustered(B,T,H,D,ncl,noise):
    cen=torch.randn(B,ncl,H,D,dtype=torch.float64,device=dev); idx=torch.randint(0,ncl,(B,T),device=dev)
    return torch.gather(cen,1,idx[:,:,None,None].expand(B,T,H,D))+noise*torch.randn(B,T,H,D,dtype=torch.float64,device=dev)
rel=lambda a,b:((a.double()-b.double()).abs().max()/b.double().abs().max().clamp_min(1e-12)).item()
B,T,H,N,D=2,50,3,12,8
k=clustered(B,T,H,D,6,.3); q=k+.2*torch.randn_like(k); v=torch.randn(B,T,H,D,dtype=torch.float64,device=dev); g=torch.rand(B,T,H,dtype=torch.float64,device=dev)
tp=torch.tensor(16.,dtype=torch.float64,device=dev)
for occb in (False, True):
    frs=torch.tensor([12.0],dtype=torch.float64,device=dev)
    # fp64 ref, scalar
    L=[t.clone().requires_grad_() for t in (k,q,v,g,frs)]; tr=tp.clone().requires_grad_()
    kb,vb=_ref_scan_kv_impl(L[0],L[2],L[3],L[4],N,tr,G=16,detach_addr=True,q=L[1],pin=False,occ_bin=occb)
    wk,wv=torch.randn_like(kb),torch.randn_like(vb); (kb*wk+vb*wv).sum().backward()
    out={}
    for mode in ("scalar","pertok"):
        fr_in = frs.float() if mode=="scalar" else torch.full((B,T,H),12.0,device=dev)
        M=[t.detach().float().requires_grad_() for t in (k,q,v,g)]+[fr_in.clone().requires_grad_()]; tt=tp.float().requires_grad_()
        a,b=kv_scan_g(M[0],M[2],M[3],M[4],N,tt,G=16,detach_addr=True,q=M[1],occ_bin=occb)
        (a*wk.float()+b*wv.float()).sum().backward()
        out[mode]=[x.grad.clone() for x in M[:4]]+[M[4].grad.sum().reshape(1), tt.grad.clone()]
        print(f"occb={occb} {mode:6s} kernel vs fp64 ref: "+"  ".join(f"{n} {rel(x,y.grad):.1e}" for n,x,y in zip(["dk","dq","dv","dg","dfr","dtemp"], out[mode], L+[tr])))
    print(f"occb={occb} scalar vs pertok kernel: "+"  ".join(f"{n} {rel(x,y):.1e}" for n,x,y in zip(["dk","dq","dv","dg","dfr","dtemp"],out["pertok"],out["scalar"])))
