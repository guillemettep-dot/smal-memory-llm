"""Benchmark the current model against the shared-memory variants, set up EXACTLY as training runs:
whole-model bf16 (no autocast -- train_memlm --amp bf16), the trainer's own chunked_ce_h loss from the
hidden states, krep added, gradients accumulated across micro-batches, MoE grouped path, and the
launcher's environment (KV_DETACH=1 KV_WQ=0 SLOT_ROPE=1 SLOT_ROPE_FRAC=0.5).  No optimizer step:
that runs once per 64 micro-batches and is measured separately.

Also checks the Triton scan against the reference at the NEW per-head shape, forward and backward."""
import os, sys, time, inspect, collections, gc
for k, v in dict(MEMLM_MOD="memlm_kv", KV_DETACH="1", KV_WQ="0", SLOT_ROPE="1", SLOT_ROPE_FRAC="0.5").items():
    os.environ[k] = v
sys.path.insert(0, "/home/paul/memprobe/gqa")
import torch
import train_memlm as TM
import memlm_kv as K, memlm_sh as S
from slotkern_kv import ref_scan_kv
from slotkern_kvt import kv_scan
from torch.profiler import profile, ProfilerActivity

ONLY = os.environ.get("ONLY", "")
T = 1024
# ---------------------------------------------------------------- scan correctness at the new shape
if not ONLY or ONLY == "scan":
    torch.manual_seed(0)
    Bq, Tq, H, N, D = 1, 192, 12, 192, 64
    k = torch.randn(Bq, Tq, H, D, device="cuda", dtype=torch.float64, requires_grad=True)
    v = torch.randn(Bq, Tq, H, D, device="cuda", dtype=torch.float64, requires_grad=True)
    g = torch.rand(Bq, Tq, H, device="cuda", dtype=torch.float64, requires_grad=True)
    fr = torch.tensor(2.0, device="cuda", dtype=torch.float64, requires_grad=True)
    tp = torch.tensor(16.0, device="cuda", dtype=torch.float64)
    kb_r, vb_r = ref_scan_kv(k, v, g, fr, N, tp, detach_addr=True)
    wk, wv = torch.randn_like(kb_r), torch.randn_like(vb_r)
    (kb_r * wk + vb_r * wv).sum().backward()
    gr = [t.grad.clone() for t in (k, v, g)]
    for t in (k, v, g): t.grad = None
    kf, vf, gf = (t.detach().float().requires_grad_() for t in (k, v, g))
    kb_t, vb_t = kv_scan(kf, vf, gf, fr.detach().float(), N, tp.float(), detach_addr=True)
    (kb_t * wk.float() + vb_t * wv.float()).sum().backward()
    rel = lambda a, b: ((a.float() - b.float()).abs().max() / b.float().abs().max().clamp_min(1e-12)).item()
    print("SCAN CHECK at 12 heads x 192 slots x (64+64), T=%d, fp32 Triton vs fp64 reference:" % Tq)
    print("   forward  kbuf rel err %.2e   vbuf rel err %.2e" % (rel(kb_t, kb_r), rel(vb_t, vb_r)))
    print("   backward dk %.2e   dv %.2e   dgate %.2e" % (rel(kf.grad, gr[0]), rel(vf.grad, gr[1]), rel(gf.grad, gr[2])))
    del k, v, g, kb_r, vb_r, kb_t, vb_t, kf, vf, gf; torch.cuda.empty_cache()

# ---------------------------------------------------------------- in-model: Triton stride-G == reference
if not ONLY or ONLY == "scan":
    torch.manual_seed(0)
    kw = dict(vocab=997, d_model=128, layers=6, heads=2, dv=16, lanes=1, d_ff=128, moe=1, n_exp=2, top_k=2, pre=1,
              kv_window=16, kv_dim=32, kv_heads=2, comp_off=1, tksel=1, slot_hard=0, slot_hard_gate=0, krep=1.0,
              sh_write=2, sh_heads=12, sh_slots=192, sh_kdim=64, sh_dv=64, sh_nq=2, sh_stride=16)
    mm = S.MemLM(S.Cfg(**kw)).cuda().float().train(); mm.set_moe_loop(False)
    xx = torch.randint(0, 997, (2, 300), device="cuda")              # 300 = not a multiple of 16
    res = []
    for use_triton in (True, False):
        mm.zero_grad(set_to_none=True)
        S_kvg = S.kv_scan_g
        if not use_triton: S.kv_scan_g = None
        out = mm(xx); (out.float().pow(2).mean() + mm.krep).backward()
        S.kv_scan_g = S_kvg
        res.append((out.detach().clone(), {n: p.grad.detach().clone() for n, p in mm.named_parameters() if p.grad is not None}))
    (o1, g1), (o2, g2) = res
    worst = max(rel(g1[n], g2[n]) for n in g2)
    wn = max(g2, key=lambda n: rel(g1[n], g2[n]))
    print("IN-MODEL Triton stride-16 vs reference (fp32, T=300): logits rel %.2e, worst grad rel %.2e (%s), same grad set %s"
          % (rel(o1, o2), worst, wn, set(g1) == set(g2)), flush=True)
    del mm, res, o1, o2, g1, g2; torch.cuda.empty_cache()

# ---------------------------------------------------------------- whole-model timing
st = torch.load("/mnt/ssd/memprobe_ckpt/gqa2k/gqa2k_step18750.pt", map_location="cpu", weights_only=False)
args = st["cfg"]; del st
def build(mod, over):
    ok = set(inspect.signature(mod.Cfg.__init__).parameters) - {"self"}
    d = {k: v for k, v in args.items() if k in ok}; d.update(over)
    return mod.MemLM(mod.Cfg(**d)), mod.Cfg(**d)
SH = dict(sh_write=4, sh_heads=12, sh_slots=192, sh_kdim=64, sh_dv=64, sh_stride=16)
configs = [("current: 12 per-layer memories", K, {}),
           ("shared 12h x 192 x (64+64), nq 1, G16", S, dict(SH, sh_nq=1)),
           ("shared 12h x 192 x (64+64), nq 2, G16", S, dict(SH, sh_nq=2)),
           ("shared 12h x 192 x (64+64), nq 4, G16", S, dict(SH, sh_nq=4)),
           ("shared  6h x 192 x (128+128), nq 2, G16 (asked)", S, dict(SH, sh_heads=6, sh_kdim=128, sh_dv=128, sh_nq=2))]
if ONLY and ONLY != "scan":
    configs = [c for c in configs if ONLY in c[0]]
rules = [("slot scan", ["_kvfwd", "_kvbwd"]), ("matmul", ["gemm", "cutlass", "matmul", "gemv"]),
         ("softmax/reduce", ["softmax", "reduce", "norm"]), ("window attn", ["flex", "attention", "fmha"]),
         ("elementwise/copy", ["elementwise", "copy", "cat", "index", "fill"])]
B = int(os.environ.get("B", "1"))
print("\nWHOLE MODEL, B=%d T=%d, bf16 weights, chunked CE, KV_DETACH=1" % (B, T))
for name, mod, over in configs:
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    try:
        torch.manual_seed(0)
        model, cfg = build(mod, over)
        model = model.cuda().to(torch.bfloat16).train(); model.set_moe_loop(False)
        npar = sum(p.numel() for p in model.parameters())
        x = torch.randint(0, 50000, (B, T + 1), device="cuda")
        def step():
            h = model(x[:, :-1], skip_head=True)
            loss = TM.chunked_ce_h(h, model.head.weight, x[:, 1:])[0]
            if getattr(model, "krep", None) is not None:
                loss = loss + 1.0 * model.krep
            loss.backward()
        for _ in range(2): step()
        torch.cuda.synchronize(); t0 = time.time(); n = 6
        for _ in range(n): step()
        torch.cuda.synchronize(); ms = (time.time() - t0) / n * 1000
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(2): step()
            torch.cuda.synchronize()
        cat = collections.Counter(); tot = 0.0
        for e in prof.key_averages():
            t = getattr(e, "device_time_total", 0) or 0
            if t <= 0: continue
            lab = next((nm for nm, ps in rules if any(p in e.key.lower() for p in ps)), "other")
            cat[lab] += t; tot += t
        brk = "  ".join("%s %.0f" % (k, v / 1000 / 2) for k, v in cat.most_common())
        print("%-44s %6.1f ms/micro  %6.0f tok/s  peak %5.2f GB  %5.1fM params\n      [ms] %s"
              % (name, ms, B * T / ms * 1000, torch.cuda.max_memory_allocated() / 1e9, npar / 1e6, brk), flush=True)
        del model, x
    except torch.cuda.OutOfMemoryError:
        print("%-44s OUT OF MEMORY at B=%d" % (name, B), flush=True)
        model = x = None
