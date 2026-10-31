import sys; sys.path.insert(0, "/home/paul/memprobe/gqa")
import torch, importlib.util
from slotkern_kv import ref_scan_kv
spec = importlib.util.spec_from_file_location("old", "/home/paul/memprobe/gqa/_old_slotkern_kv.py")
old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
d = torch.load("/home/paul/memprobe/gqa/_scan_inputs.pt"); k, v, g = (d[n][:1] for n in ("k", "v", "g")); fr, tp = d["fr"], d["tp"]
a = ref_scan_kv(k, v, g, fr, 192, tp, G=16, detach_addr=True); b = old.ref_scan_kv(k, v, g, fr, 192, tp, G=16, detach_addr=True)
c = ref_scan_kv(k, v, g, fr, 192, tp, G=16, detach_addr=True, pin=False); p = torch.load("/home/paul/memprobe/gqa/_pin_cmp.pt")
print("default vs old file: bit-identical", torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]))
print("pin=False vs fp64 unpinned: %.1e" % ((c[1] - p["nv"]).abs().max() / p["nv"].abs().max()).item())
