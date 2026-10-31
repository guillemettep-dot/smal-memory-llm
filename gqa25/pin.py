import sys, inspect
sys.path.insert(0, "/home/paul/memprobe/gqa")
import torch, slotkern_kv as SK
src = inspect.getsource(SK._ref_scan_kv_impl)
PIN = "        s = torch.where(_Ka.norm(dim=-1) < 1e-6, torch.zeros_like(s), s)\n"
assert src.count(PIN) == 1
ns = {}; exec(src.replace(PIN, "").replace("def _ref_scan_kv_impl", "def nopin"), SK.__dict__, ns); nopin = ns["nopin"]
d = torch.load("/home/paul/memprobe/gqa/_scan_inputs.pt")
k, v, g = (d[n][:1].double() for n in ("k", "v", "g")); fr, tp = d["fr"].double(), d["tp"].double()
def rel(a, b): return ((a - b).abs().max() / b.abs().max().clamp_min(1e-30)).item()
with torch.no_grad():
    pk, pv = SK._ref_scan_kv_impl(k, v, g, fr, 192, tp, 16, False, False, 1e-4, False, True)
    nk, nv = nopin(k, v, g, fr, 192, tp, 16, False, False, 1e-4, False, True)
torch.save(dict(pk=pk.float(), pv=pv.float(), nk=nk.float(), nv=nv.float()), "/home/paul/memprobe/gqa/_pin_cmp.pt")
print("fp64 pinned vs fp64 UNPINNED reference: vbuf rel %.1e" % rel(nv, pv))
