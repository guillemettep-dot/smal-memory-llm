import os, sys
os.environ.update(KV_DETACH="1", KV_WQ="0", SLOT_ROPE="1", SLOT_ROPE_FRAC="0.5")
sys.path.insert(0, "/home/paul/memprobe/gqa")
import torch, memlm_sh as S
rel = lambda a, b: ((a.double() - b.double()).abs().max() / b.double().abs().max().clamp_min(1e-30)).item()
print("tf32 matmul", torch.backends.cuda.matmul.allow_tf32, "cudnn", torch.backends.cudnn.allow_tf32, "prec", torch.get_float32_matmul_precision())
def run(mm, xx, use):
    mm.zero_grad(set_to_none=True); keep = S.kv_scan_g
    if not use: S.kv_scan_g = None
    out = mm(xx); (out.float().pow(2).mean() + mm.krep).backward(); S.kv_scan_g = keep
    return out.detach().clone(), {n: p.grad.detach().clone() for n, p in mm.named_parameters() if p.grad is not None}
def cmp(a, b):
    w = sorted(((rel(a[1][n], b[1][n]), n) for n in b[1]), reverse=True)[:2]
    return "logits %.1e  worst %s" % (rel(a[0], b[0]), ["%s %.1e" % (n.replace("blocks.", "b"), e) for e, n in w])
for tf32 in (True, False):
    torch.backends.cuda.matmul.allow_tf32 = tf32; torch.backends.cudnn.allow_tf32 = tf32
    torch.set_float32_matmul_precision("high" if tf32 else "highest")
    torch.manual_seed(0)
    kw = dict(vocab=997, d_model=128, layers=6, heads=2, dv=16, lanes=1, d_ff=128, moe=0, n_exp=2, top_k=2, pre=1,
              kv_window=16, kv_dim=32, kv_heads=2, comp_off=1, tksel=1, slot_hard=0, slot_hard_gate=0, krep=1.0,
              sh_write=2, sh_heads=12, sh_slots=192, sh_kdim=64, sh_dv=64, sh_nq=2, sh_stride=16)
    mm = S.MemLM(S.Cfg(**kw)).cuda().float().train()
    xx = torch.randint(0, 997, (2, 300), device="cuda")
    t1, t2, r1, r2 = run(mm, xx, True), run(mm, xx, True), run(mm, xx, False), run(mm, xx, False)
    print("tf32=%s  triton-vs-triton: %s" % (tf32, cmp(t1, t2)))
    print("tf32=%s  ref-vs-ref:       %s" % (tf32, cmp(r1, r2)))
    print("tf32=%s  triton-vs-ref:    %s" % (tf32, cmp(t1, r1)))
