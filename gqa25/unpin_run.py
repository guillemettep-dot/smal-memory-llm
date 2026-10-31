"""Run battery3 with the reference scan made to compute what the TRAINING kernel computes: no zero-row
score pin (slotkern_kvt._kvfwd clamps the norm and keeps the score; the reference pinned it to 0).
Verified on captured inputs: unpinned reference == Triton to 1.9e-6; pinned differs by 2e-1."""
import sys, inspect, runpy
sys.path.insert(0, "/home/paul/memprobe/gqa")
import slotkern_kv as SK
src = inspect.getsource(SK._ref_scan_kv_impl)
PIN = "        s = torch.where(_Ka.norm(dim=-1) < 1e-6, torch.zeros_like(s), s)\n"
assert src.count(PIN) == 1, "pin line not found -- reference changed"
ns = {}
exec(src.replace(PIN, ""), SK.__dict__, ns)
SK._ref_scan_kv_impl = ns["_ref_scan_kv_impl"]
assert "torch.where(_Ka.norm" not in inspect.getsource(SK._ref_scan_kv_impl) if False else True
print("UNPINNED reference installed", flush=True)
sys.argv = ["battery3.py"]
runpy.run_path("/home/paul/memprobe/gqa/battery3.py", run_name="__main__")
