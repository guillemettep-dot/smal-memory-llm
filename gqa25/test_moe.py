import sys, copy; sys.path.insert(0, "/home/paul/memprobe/gqa")
import torch, memlm_m5 as M5, memlm_sh as S
class C: pass
c = C(); c.n_exp, c.top_k, c.moe_u, c.cap, c.d_model, c.d_ff = 4, 2, 1e-3, 2.0, 256, 192
rel = lambda a, b: ((a.double() - b.double()).abs().max() / b.double().abs().max().clamp_min(1e-30)).item()
fails = []
for dev in (["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"]):
    for dt in (torch.float32, torch.bfloat16):
        torch.manual_seed(0)
        ref = M5.MoE(c).to(dev, dt).train(); new = S.ExactMoE(c).to(dev, dt).train()
        new.load_state_dict(ref.state_dict())
        with torch.no_grad(): ref.ebias.copy_(torch.randn(4) * 0.05); new.ebias.copy_(ref.ebias)
        x = torch.randn(3, 333, 256, device=dev, dtype=dt)
        xa, xb = x.clone().requires_grad_(), x.clone().requires_grad_()
        oa, ob = ref(xa), new(xb)
        w = torch.randn_like(oa)
        (oa.float() * w.float()).sum().backward(); (ob.float() * w.float()).sum().backward()
        tol = 2e-5 if dt == torch.float32 else 2e-2
        res = dict(out=rel(ob, oa), dx=rel(xb.grad, xa.grad), **{n: rel(pb.grad, pa.grad) for (n, pa), (_, pb) in zip(ref.named_parameters(), new.named_parameters())})
        res["ebias"] = rel(new.ebias, ref.ebias); res["maxvio"] = abs(new.maxvio.item() - ref.maxvio.item())
        bad = {k: v for k, v in res.items() if v > tol}
        print("%s %-8s ref.drop=%.4f | %s %s" % (dev, str(dt).split(".")[-1], ref.drop.item(),
              " ".join("%s %.1e" % kv for kv in res.items()), "OK" if not bad else "FAIL %s" % bad))
        if bad: fails.append((dev, dt, bad))
print("ALL PASS" if not fails else "FAILURES")
