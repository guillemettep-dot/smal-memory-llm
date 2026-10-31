import os, sys
os.environ.update(KV_DETACH="1", KV_WQ="0", SLOT_ROPE="1", SLOT_ROPE_FRAC="0.5")
sys.path.insert(0, "/home/paul/memprobe/gqa")
import torch, memlm_sh as S
from slotkern_kv import ref_scan_kv
from slotkern_kvg import kv_scan_g
rel = lambda a, b: ((a.double() - b.double()).abs().max() / b.double().abs().max().clamp_min(1e-30)).item()
torch.manual_seed(0)
# 1. scan alone, model-like inputs, T not a multiple of G
for T in (288, 300):
    B, H, N, D = 2, 12, 192, 64
    k = (torch.randn(B, T, H, D, device="cuda") * 0.3).requires_grad_()
    v = (torch.randn(B, T, H, D, device="cuda") * 0.3).requires_grad_()
    g = torch.sigmoid(torch.randn(B, T, H, device="cuda") - 0.3).requires_grad_()
    fr = torch.tensor(4.48 * 2.0, device="cuda", requires_grad=True); tp = torch.tensor(16.0, device="cuda", requires_grad=True)
    outs = []
    for f in (kv_scan_g, ref_scan_kv):
        for t in (k, v, g, fr, tp): t.grad = None
        kb, vb = f(k, v, g, fr, N, tp, G=16, detach_addr=True)
        torch.manual_seed(1); wk, wv = torch.randn_like(kb), torch.randn_like(vb)
        (kb * wk + vb * wv).sum().backward()
        outs.append((kb.detach(), vb.detach(), [None if t.grad is None else t.grad.clone() for t in (k, v, g, fr, tp)]))
    (a1, b1, g1), (a2, b2, g2) = outs
    print("T=%d scan: kbuf %.1e vbuf %.1e shapes %s %s | grads k %.1e v %.1e g %.1e fr %s tp %s" % (
        T, rel(a1, a2), rel(b1, b2), tuple(a1.shape), tuple(a2.shape), rel(g1[0], g2[0]), rel(g1[1], g2[1]), rel(g1[2], g2[2]),
        None if g1[3] is None or g2[3] is None else "%.1e" % rel(g1[3], g2[3]), None if g1[4] is None or g2[4] is None else "%.1e(%s vs %s)" % (rel(g1[4], g2[4]), g1[4].item(), g2[4].item())))
    # per-block error of the forward
    print("   per-block kbuf err:", " ".join("%.0e" % (a1[:, j] - a2[:, j]).abs().max().item() for j in range(a1.shape[1])))
# 2. in-model with dense FFN (no MoE routing flips)
for moe in (0, 1):
    torch.manual_seed(0)
    kw = dict(vocab=997, d_model=128, layers=6, heads=2, dv=16, lanes=1, d_ff=128, moe=moe, n_exp=2, top_k=2, pre=1,
              kv_window=16, kv_dim=32, kv_heads=2, comp_off=1, tksel=1, slot_hard=0, slot_hard_gate=0, krep=1.0,
              sh_write=2, sh_heads=12, sh_slots=192, sh_kdim=64, sh_dv=64, sh_nq=2, sh_stride=16)
    mm = S.MemLM(S.Cfg(**kw)).cuda().float().train()
    if moe: mm.set_moe_loop(False)
    xx = torch.randint(0, 997, (2, 300), device="cuda")
    res = []
    for use in (True, False):
        mm.zero_grad(set_to_none=True); keep = S.kv_scan_g
        if not use: S.kv_scan_g = None
        out = mm(xx); (out.float().pow(2).mean() + mm.krep).backward(); S.kv_scan_g = keep
        res.append((out.detach().clone(), {n: p.grad.detach().clone() for n, p in mm.named_parameters() if p.grad is not None}))
    (o1, g1), (o2, g2) = res
    worst = sorted(((rel(g1[n], g2[n]), n) for n in g2), reverse=True)[:4]
    print("in-model moe=%d: logits %.1e, worst grads %s" % (moe, rel(o1, o2), ["%s %.1e" % (n, e) for e, n in worst]))
