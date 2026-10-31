import sys, torch
sys.path.insert(0, "/home/paul/memprobe/gqa20")
torch.cuda.set_per_process_memory_fraction(0.12,0)
from slotkern_kv import _ref_scan_kv_impl
from slotkern_kvg import kv_scan_g
dev="cuda"
rel=lambda a,b:((a.double()-b.double()).abs().max()/b.double().abs().max().clamp_min(1e-12)).item()
B,T,H,N,D=2,50,3,12,8
for seed in range(6):
  torch.manual_seed(seed)
  cen=torch.randn(B,6,H,D,dtype=torch.float64,device=dev); idx=torch.randint(0,6,(B,T),device=dev)
  k=torch.gather(cen,1,idx[:,:,None,None].expand(B,T,H,D))+.3*torch.randn(B,T,H,D,dtype=torch.float64,device=dev)
  q=k+.2*torch.randn_like(k); v=torch.randn(B,T,H,D,dtype=torch.float64,device=dev); g=torch.rand(B,T,H,dtype=torch.float64,device=dev)
  tp=torch.tensor(16.,dtype=torch.float64,device=dev); fr=tp*(0.2+1.3*torch.rand(B,T,H,dtype=torch.float64,device=dev))
  wk=wv=None; res={}
  for name,dt in (("ref64",torch.float64),("ref32",torch.float32),("kern",torch.float32)):
    L=[t.to(dt).clone().requires_grad_() for t in (k,q,v,g,fr)]; tr=tp.to(dt).clone().requires_grad_()
    if name=="kern": kb,vb=kv_scan_g(L[0],L[2],L[3],L[4],N,tr,G=16,detach_addr=True,q=L[1],occ_bin=False)
    else: kb,vb=_ref_scan_kv_impl(L[0],L[2],L[3],L[4],N,tr,G=16,detach_addr=True,q=L[1],pin=False,occ_bin=False)
    if wk is None: wk,wv=torch.randn(kb.shape,dtype=torch.float64,device=dev),torch.randn(vb.shape,dtype=torch.float64,device=dev)
    (kb*wk.to(dt)+vb*wv.to(dt)).sum().backward()
    res[name]=[x.grad for x in L]+[tr.grad]
  f=lambda a: "  ".join(f"{n} {rel(x,y):.0e}" for n,x,y in zip(["dk","dq","dv","dg","dfr","dtemp"],res[a],res["ref64"]))
  print(f"seed {seed} dtemp64 {res[chr(114)+chr(101)+chr(102)+chr(54)+chr(52)][5].item():+.3e} | ref32: {f(chr(114)+chr(101)+chr(102)+chr(51)+chr(50))} | kern: {f(chr(107)+chr(101)+chr(114)+chr(110))}")
