import sys, torch
sys.path.insert(0, "/home/paul/memprobe/gqa20")
from slotkern_kv import _ref_scan_kv_impl
torch.manual_seed(0)
def clustered(B,T,H,D,ncl,noise):
    cen=torch.randn(B,ncl,H,D,dtype=torch.float64); idx=torch.randint(0,ncl,(B,T))
    return torch.gather(cen,1,idx[:,:,None,None].expand(B,T,H,D))+noise*torch.randn(B,T,H,D,dtype=torch.float64)
B,T,H,N,D=2,50,3,12,8
k=clustered(B,T,H,D,6,.3); q=k+.2*torch.randn_like(k); v=torch.randn(B,T,H,D,dtype=torch.float64); g=torch.rand(B,T,H,dtype=torch.float64)
tp=torch.tensor(16.,dtype=torch.float64); fr=tp*(0.2+1.3*torch.rand(B,T,H,dtype=torch.float64))
def run(dt, per_tok, occb):
    L=[t.to(dt).clone().requires_grad_() for t in (k,q,v,g,(fr if per_tok else fr.mean().reshape(1)))]
    tr=tp.to(dt).clone().requires_grad_()
    kb,vb=_ref_scan_kv_impl(L[0],L[2],L[3],L[4],N,tr,G=16,detach_addr=True,q=L[1],pin=False,occ_bin=occb)
    torch.manual_seed(5); wk=torch.randn(kb.shape,dtype=torch.float64).to(dt); wv=torch.randn(vb.shape,dtype=torch.float64).to(dt)
    (kb*wk+vb*wv).sum().backward()
    return [x.grad.double() for x in L]+[tr.grad.double()]
rel=lambda a,b:((a-b).abs().max()/b.abs().max().clamp_min(1e-12)).item()
for per in (False,True):
  for occb in (False,True):
    a=run(torch.float64,per,occb); b=run(torch.float32,per,occb)
    print(f"per_token={per} occb={occb}: fp32-ref vs fp64-ref  "+"  ".join(f"{n} {rel(x,y):.1e}" for n,x,y in zip(["dk","dq","dv","dg","dfr","dtemp"],b,a)))
