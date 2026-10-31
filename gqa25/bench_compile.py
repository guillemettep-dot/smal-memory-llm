"""Shared-memory model at the launch shape, B=2 T=1024, training-matched (bf16 weights, chunked CE,
krep, KV_DETACH=1).  Eager vs torch.compile variants.  Same seed and input in every variant, so the
first-step loss must agree -- a compile that changes the maths is reported, not timed."""
import os, sys, time, inspect, gc
for k, v in dict(MEMLM_MOD="memlm_sh", KV_DETACH="1", KV_WQ="0", SLOT_ROPE="1", SLOT_ROPE_FRAC="0.5").items():
    os.environ[k] = v
sys.path.insert(0, "/home/paul/memprobe/gqa")
import torch, torch._dynamo
import train_memlm as TM
import memlm_sh as S
torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
torch._dynamo.config.cache_size_limit = 64
st = torch.load("/mnt/ssd/memprobe_ckpt/gqa2k/gqa2k_step18750.pt", map_location="cpu", weights_only=False)
args = st["cfg"]; del st
B, T = int(os.environ.get("B", "2")), 1024
def build():
    ok = set(inspect.signature(S.Cfg.__init__).parameters) - {"self"}
    d = {k: v for k, v in args.items() if k in ok}
    d.update(sh_write=int(os.environ.get("SHW", "4")), sh_heads=12, sh_slots=192, sh_kdim=64, sh_dv=64, sh_nq=2, sh_stride=int(os.environ.get("SHG", "16")),
             sh_enc_write=int(os.environ.get("SHE", "-1")), sh_enc_slots=96, sh_qbits=int(os.environ.get("SHQ", "0")))
    torch.manual_seed(0)
    m = S.MemLM(S.Cfg(**d)).cuda().to(torch.bfloat16).train(); m.set_moe_loop(False)
    return m
x = torch.randint(0, 50000, (B, T + 1), device="cuda", generator=torch.Generator("cuda").manual_seed(1))
def run(model, label, n=6):
    def step():
        h = model(x[:, :-1], skip_head=True)
        loss = TM.chunked_ce_h(h, model.head.weight, x[:, 1:])[0]
        if model.krep is not None: loss = loss + model.krep
        loss.backward()
        return loss.detach().float().item()
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    try:
        t0 = time.time(); l0 = step(); t1 = time.time() - t0
        for _ in range(2): step()
        torch.cuda.synchronize(); t0 = time.time()
        for _ in range(n): step()
        torch.cuda.synchronize(); ms = (time.time() - t0) / n * 1000
        if os.environ.get("PROF"):
            from torch.profiler import profile, ProfilerActivity
            import collections
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                step(); torch.cuda.synchronize()
            rows = sorted(((getattr(e, "device_time_total", 0) or 0, e.key) for e in prof.key_averages()), reverse=True)
            tot = sum(r[0] for r in rows)
            print("   profile: total %.0f ms device" % (tot / 1000))
            for t_, k_ in rows[:22]:
                print("   %7.1f ms  %s" % (t_ / 1000, k_[:110]))
        print("%-34s first-step loss %.5f | %6.1f ms/micro %6.0f tok/s | peak %.2f GB | 1st call %.0fs"
              % (label, l0, ms, B * T / ms * 1000, torch.cuda.max_memory_allocated() / 1e9, t1), flush=True)
    except Exception as e:
        print("%-34s FAILED: %s: %s" % (label, type(e).__name__, str(e).split("\n")[0][:200]), flush=True)
V = os.environ.get("V", "eager,blocks,ffn,mix").split(",")
for v in V:
    model = build()
    if v == "blocks":
        for b in model.blocks: b.forward = torch.compile(b.forward, dynamic=False)
    elif v == "ffn":
        for b in model.blocks:
            b.ffn.forward = torch.compile(b.ffn.forward, dynamic=False)
    elif v == "mix":
        for b in model.blocks:
            b.mix._read = torch.compile(b.mix._read, dynamic=False)
            b.ffn.forward = torch.compile(b.ffn.forward, dynamic=False)
    elif v == "all":
        for pb in model.pre_blocks: pb.forward = torch.compile(pb.forward, dynamic=False)
        for b in model.blocks: b.forward = torch.compile(b.forward, dynamic=False)
        model.nf.forward = torch.compile(model.nf.forward, dynamic=False)
    elif v == "pre":
        for pb in model.pre_blocks: pb.forward = torch.compile(pb.forward, dynamic=False)
        for b in model.blocks:
            b.ffn.forward = torch.compile(b.ffn.forward, dynamic=False)
            b.mix._read = torch.compile(b.mix._read, dynamic=False)
    run(model, "%s (B=%d)" % (v, B))
    del model
    torch._dynamo.reset()
