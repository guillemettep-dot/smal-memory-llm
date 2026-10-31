"""Where does one training micro-batch spend its GPU time?  Builds MemLM from the LIVE run's config
(random weights -- timing does not depend on values), runs fwd+bwd at T=1024 under bf16 autocast,
profiles CUDA kernels and groups them.  Also times a whole micro-batch wall-clock and the optimizer."""
import os, sys, time, inspect, collections, torch
sys.path.insert(0, "/home/paul/memprobe/gqa")
os.environ.setdefault("KV_DETACH", "0")
import memlm_kv as M
from torch.profiler import profile, ProfilerActivity
st = torch.load("/mnt/ssd/memprobe_ckpt/gqa2k/gqa2k_step18750.pt", map_location="cpu", weights_only=False)
ok = set(inspect.signature(M.Cfg.__init__).parameters) - {"self"}
cfg = M.Cfg(**{k: v for k, v in st["cfg"].items() if k in ok}); del st
torch.manual_seed(0)
model = M.MemLM(cfg).cuda().train()
B, T = int(os.environ.get("B", "1")), 1024
x = torch.randint(0, 50000, (B, T + 1), device="cuda")
def step():
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(x[:, :-1])
        loss = torch.nn.functional.cross_entropy(out.float().reshape(-1, out.shape[-1]), x[:, 1:].reshape(-1))
    loss.backward()
for _ in range(2): step()
torch.cuda.synchronize()
t0 = time.time(); n = 5
for _ in range(n): step()
torch.cuda.synchronize(); per = (time.time() - t0) / n * 1000
print("wall-clock per micro-batch (B=%d, T=%d): %.1f ms  -> per 64-micro step: %.1f s" % (B, T, per, per * 64 / 1000 / B))
print("peak memory: %.2f GB" % (torch.cuda.max_memory_allocated() / 1e9))
with profile(activities=[ProfilerActivity.CUDA]) as prof:
    for _ in range(3): step()
    torch.cuda.synchronize()
cat = collections.Counter(); tot = 0.0
rules = [("slot scan fwd", ["_kvfwd"]), ("slot scan bwd", ["_kvbwd"]),
         ("window attn", ["flex", "attention", "sdpa", "fmha", "flash"]),
         ("matmul (gemm)", ["gemm", "sm80_", "sm86_", "ampere", "cutlass", "matmul", "gemv"]),
         ("softmax/reduce", ["softmax", "reduce", "sum_", "max_"]),
         ("elementwise/copy", ["elementwise", "vectorized", "copy", "cat", "index", "fill", "mul", "add"])]
for e in prof.key_averages():
    t = getattr(e, "device_time_total", None) or getattr(e, "cuda_time_total", 0)
    if t <= 0 or e.key.startswith("ProfilerStep"): continue
    k = e.key.lower(); lab = "other"
    for name, pats in rules:
        if any(p in k for p in pats): lab = name; break
    cat[lab] += t; tot += t
print("\nGPU time by category (3 micro-batches):")
for k, v in cat.most_common(): print("   %-18s %6.1f%%   %7.1f ms" % (k, 100 * v / tot, v / 1000 / 3))
print("\ntop 12 kernels:")
rows = sorted(prof.key_averages(), key=lambda e: -(getattr(e, "device_time_total", 0) or 0))[:12]
for e in rows:
    t = (getattr(e, "device_time_total", 0) or 0)
    print("   %6.1f ms  %s" % (t / 1000 / 3, e.key[:90]))
