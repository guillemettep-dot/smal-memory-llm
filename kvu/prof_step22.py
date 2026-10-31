"""Where does one sh22 training micro-batch spend its GPU time?  sh22's own weights and config, whole-model bf16,
blocks compiled exactly as train_sh.py does, CUDA scan on, B=4 x T=1024, forward + backward with the trainer's own
chunked CE.  Groups every CUDA kernel by kind and
prints the top kernels.  Needs the GPU to itself.   argv: checkpoint"""
import os, sys, time, inspect, collections
os.environ.update(MEMPROBE_ROOT="/home/paul/memprobe", MEMLM_MOD="memlm_sh", KV_DETACH="1", KV_WQ="0",
                  SLOT_ROPE="1", SLOT_ROPE_FRAC="0.5", CUDA_VISIBLE_DEVICES="0", KV_CUDA="1")
sys.path.insert(0, "/home/paul/memprobe/gqa22")
import torch, torch._dynamo as _dynamo
import memlm_sh as S
from torch.profiler import profile, ProfilerActivity
assert "gqa22" in S.__file__
st = torch.load(sys.argv[1], map_location="cpu", weights_only=False, mmap=True)
ok = set(inspect.signature(S.Cfg.__init__).parameters) - {"self"}
m = S.MemLM(S.Cfg(**{k: v for k, v in st["cfg"].items() if k in ok}))
m.load_state_dict(st["model"]); print("step", st["step"]); del st
m = m.cuda().to(torch.bfloat16).train()
torch.backends.cuda.matmul.allow_tf32 = True
for _k, _v in (("recompile_limit", 64), ("cache_size_limit", 64), ("accumulated_recompile_limit", 4096)):
    if hasattr(_dynamo.config, _k): setattr(_dynamo.config, _k, _v)
for _blk in m.blocks: _blk.forward = torch.compile(_blk.forward, dynamic=False)
m.nf.forward = torch.compile(m.nf.forward, dynamic=False)
for _pb in m.pre_blocks: _pb.forward = torch.compile(_pb.forward, dynamic=False)
B, T = 2, 1024
import numpy as np, glob
mm = np.load(sorted(glob.glob("/home/paul/memprobe/data/shard_*.npy"))[0], mmap_mode="r")
x = torch.tensor(np.asarray(mm[5_000_000:5_000_000 + B * (T + 1)]).astype(np.int64).reshape(B, T + 1), device="cuda")
import train_sh as TS                     # main() runs only under __main__; this imports the trainer's own loss
assert "gqa22" in TS.__file__
def step():
    hid = m(x[:, :-1], skip_head=True)    # the trainer's path (--ce_fused 1): chunked CE from hidden states
    loss, _, _ = TS.chunked_ce_h(hid, m.head.weight, x[:, 1:])
    loss.backward()
for _ in range(3): step()
torch.cuda.synchronize(); t0 = time.time(); n = 5
for _ in range(n): step()
torch.cuda.synchronize(); per = (time.time() - t0) / n * 1000
print(f"wall per micro-batch (B={B}, T={T}): {per:.1f} ms -> {B*T/per*1000:,.0f} tok/s (no optimizer) | peak {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
with profile(activities=[ProfilerActivity.CUDA]) as prof:
    for _ in range(3): step()
    torch.cuda.synchronize()
ev = [e for e in prof.key_averages() if e.device_type is not None and "cuda" in str(e.device_type).lower()]
def kind(n):
    s = n.lower()
    for k, keys in [("memory scan (CUDA)", ("addr_fwd", "val_fwd", "rows_bwd", "addr_bwd")),
                    ("memory read (kcol)", ("_rd_fwd", "_rd_bwd")),
                    ("block-sparse (bsparse)", ("_bs_",)),
                    ("matmul (cuBLAS/cutlass)", ("gemm", "cutlass", "xmma", "s16816", "s1688", "wgrad", "dgrad", "cublas")),
                    ("attention/softmax", ("softmax", "flash", "attention", "fmha", "sdpa")),
                    ("inductor fused (triton_)", ("triton_",)),
                    ("cross-entropy", ("nll", "cross_entropy", "log_softmax")),
                    ("copy/cast/fill", ("copy", "cast", "fill", "memcpy", "memset", "to_copy")),
                    ("index/scatter/gather (MoE routing)", ("index", "scatter", "gather", "sort", "topk", "cumsum", "nonzero")),
                    ("elementwise/reduce (aten)", ("elementwise", "reduce", "vectorized"))]:
        if any(x in s for x in keys): return k
    return "other"
tot = collections.Counter(); cnt = collections.Counter()
for e in ev:
    t = getattr(e, "self_device_time_total", None) or getattr(e, "self_cuda_time_total", 0)
    tot[kind(e.key)] += t; cnt[kind(e.key)] += e.count
allt = sum(tot.values())
print(f"\nGPU time per micro-batch: {allt/3/1000:.1f} ms (wall {per:.1f})")
for k, t in tot.most_common():
    print(f"  {k:36s} {t/3/1000:7.1f} ms  {t/allt:6.1%}  ({cnt[k]//3} launches)")
print("\ntop 30 kernels:")
rows = sorted(ev, key=lambda e: -(getattr(e, "self_device_time_total", None) or getattr(e, "self_cuda_time_total", 0)))[:30]
for e in rows:
    t = getattr(e, "self_device_time_total", None) or getattr(e, "self_cuda_time_total", 0)
    print(f"  {t/3/1000:7.2f} ms  x{e.count//3:<5d} {kind(e.key)[:14]:14s} {e.key[:110]}")
