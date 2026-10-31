"""Is bf16 p.grad accumulation lossy, or is the per-micro-batch gradient itself nondeterministic?
Three passes over the SAME 32 micro-batches: fp32-summed (A), fp32-summed again (B), bf16-accumulated (C).
A vs B = the nondeterminism floor.  A vs C = accumulation + floor."""
import os, sys, inspect
for k, v in dict(MEMLM_MOD="memlm_sh", KV_DETACH="1", KV_WQ="0", SLOT_ROPE="1", SLOT_ROPE_FRAC="0.5").items(): os.environ[k] = v
sys.path.insert(0, "/home/paul/memprobe/gqa")
import torch, train_memlm as TM, memlm_sh as S
st = torch.load("/mnt/ssd/memprobe_ckpt/gqa2k/gqa2k_step18750.pt", map_location="cpu", weights_only=False); args = st["cfg"]; del st
ok = set(inspect.signature(S.Cfg.__init__).parameters) - {"self"}
d = {k: v for k, v in args.items() if k in ok}; d.update(sh_write=4, sh_heads=12, sh_slots=192, sh_kdim=64, sh_dv=64, sh_nq=2, sh_stride=16)
torch.manual_seed(0)
m = S.MemLM(S.Cfg(**d)).cuda().to(torch.bfloat16).train(); m.set_moe_loop(False)
ACC = int(os.environ.get("ACC", "32")); B, T = 2, 1024
names = [n for n, p in m.named_parameters() if p.dim() >= 2]
big = [p for p in m.parameters() if p.dim() >= 2]
def micro(x):
    h = m(x[:, :-1], skip_head=True)
    loss = TM.chunked_ce_h(h, m.head.weight, x[:, 1:])[0] / ACC
    if m.krep is not None: loss = loss + m.krep / ACC
    loss.backward()
def fp32_pass():
    g = torch.Generator("cuda").manual_seed(5); acc = [torch.zeros_like(p, dtype=torch.float32) for p in big]
    for i in range(ACC):
        micro(torch.randint(0, 50000, (B, T + 1), device="cuda", generator=g))
        for a, p in zip(acc, big): a += p.grad.float()
        m.zero_grad(set_to_none=True)
    return [a.cpu() for a in acc]
def bf16_pass():
    g = torch.Generator("cuda").manual_seed(5); m.zero_grad(set_to_none=True)
    for i in range(ACC):
        micro(torch.randint(0, 50000, (B, T + 1), device="cuda", generator=g))
    out = [p.grad.float().cpu() for p in big]; m.zero_grad(set_to_none=True); return out
def cmp(a, b):
    tot = (sum(((y - x).norm() ** 2) for x, y in zip(a, b)) / sum(x.norm() ** 2 for x in a)).sqrt().item()
    per = sorted((((y - x).norm() / x.norm().clamp_min(1e-30)).item(), n) for x, y, n in zip(a, b, names))
    return "rel err %.4f | median tensor %.4f | worst %s %.4f" % (tot, per[len(per) // 2][0], per[-1][1], per[-1][0])
snap = {k: v.detach().clone() for k, v in m.state_dict().items()}
def restore():
    m.load_state_dict(snap)
bufs0 = {n: b.detach().clone() for n, b in m.named_buffers()}
restore(); A = fp32_pass()
moved = [n for n, b in m.named_buffers() if not torch.equal(b, bufs0[n])]
print("buffers changed by %d training forwards: %d of %d, e.g. %s" % (ACC, len(moved), len(bufs0), moved[:3]))
restore(); Bp = fp32_pass(); restore(); C = bf16_pass()
print("ACC=%d  fp32 vs fp32 (nondeterminism floor): %s" % (ACC, cmp(A, Bp)))
print("ACC=%d  fp32 vs bf16-accumulated:            %s" % (ACC, cmp(A, C)))
