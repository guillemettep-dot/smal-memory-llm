"""bf16 p.grad accumulation over ACC micro-batches vs an fp32 sum of the SAME per-micro-batch grads."""
import os, sys, inspect
for k, v in dict(MEMLM_MOD="memlm_sh", KV_DETACH="1", KV_WQ="0", SLOT_ROPE="1", SLOT_ROPE_FRAC="0.5").items(): os.environ[k] = v
sys.path.insert(0, "/home/paul/memprobe/gqa")
import torch, train_memlm as TM, memlm_sh as S
st = torch.load("/mnt/ssd/memprobe_ckpt/gqa2k/gqa2k_step18750.pt", map_location="cpu", weights_only=False); args = st["cfg"]; del st
ok = set(inspect.signature(S.Cfg.__init__).parameters) - {"self"}
d = {k: v for k, v in args.items() if k in ok}; d.update(sh_write=4, sh_heads=12, sh_slots=192, sh_kdim=64, sh_dv=64, sh_nq=2, sh_stride=16)
torch.manual_seed(0)
m = S.MemLM(S.Cfg(**d)).cuda().to(torch.bfloat16).train(); m.set_moe_loop(False)
ACC = 32; B, T = 2, 1024
big = [p for p in m.parameters() if p.dim() >= 2]
g = torch.Generator("cuda").manual_seed(5)
for i in range(ACC):
    x = torch.randint(0, 50000, (B, T + 1), device="cuda", generator=g)
    h = m(x[:, :-1], skip_head=True)
    loss = TM.chunked_ce_h(h, m.head.weight, x[:, 1:])[0] / ACC
    if m.krep is not None: loss = loss + m.krep / ACC
    loss.backward()
    del h, loss
# exact fp32 reference needs per-micro grads uncontaminated: redo with grads zeroed each micro and summed in fp32
bf16_sum = [p.grad.float().cpu() for p in big]
m.zero_grad(set_to_none=True)
acc32 = [torch.zeros_like(p, dtype=torch.float32) for p in big]
g = torch.Generator("cuda").manual_seed(5)
for i in range(ACC):
    x = torch.randint(0, 50000, (B, T + 1), device="cuda", generator=g)
    h = m(x[:, :-1], skip_head=True)
    loss = TM.chunked_ce_h(h, m.head.weight, x[:, 1:])[0] / ACC
    if m.krep is not None: loss = loss + m.krep / ACC
    loss.backward()
    for a, p in zip(acc32, big): a += p.grad.float()
    m.zero_grad(set_to_none=True); del h, loss
names = [n for n, p in m.named_parameters() if p.dim() >= 2]
acc32 = [a.cpu() for a in acc32]
errs = sorted((((b - a).norm() / a.norm().clamp_min(1e-30)).item(), n) for a, b, n in zip(acc32, bf16_sum, names))
tot = (sum(((b - a).norm() ** 2) for a, b in zip(acc32, bf16_sum)) / sum(a.norm() ** 2 for a in acc32)).sqrt().item()
cos = (sum((a * b).sum() for a, b in zip(acc32, bf16_sum)) / (sum(a.norm()**2 for a in acc32).sqrt() * sum(b.norm()**2 for b in bf16_sum).sqrt())).item()
print("ACC=%d: whole-gradient relative error of bf16 accumulation %.4f, cosine to fp32 sum %.6f" % (ACC, tot, cos))
print("median tensor %.4f, worst 3:" % errs[len(errs) // 2][0], ["%s %.4f" % (n, e) for e, n in errs[-3:]])
# scale: how big is that against the gradient's own noise between two disjoint 32-micro-batch sums?
