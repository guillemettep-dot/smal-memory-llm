import os, sys, inspect
os.environ.update(KV_DETACH="1", CUDA_VISIBLE_DEVICES="")
sys.path.insert(0, "/home/paul/memprobe/gqa")
import torch, memlm_sh as S
sig = set(inspect.signature(S.Cfg.__init__).parameters) - {"self"}
assert "sh_wq" in sig
kw = dict(vocab=997, d_model=64, layers=6, heads=2, dv=16, lanes=1, d_ff=64, moe=1, n_exp=2, top_k=2, pre=1, kv_window=4, kv_dim=16, kv_heads=2,
          comp_off=1, tksel=1, slot_hard=1, slot_hard_gate=0, krep=1.0, sh_write=4, sh_heads=3, sh_slots=8, sh_kdim=8, sh_dv=8, sh_nq=2, sh_stride=4,
          sh_enc_write=1, sh_enc_slots=4, sh_gate_floor=0.05)
f = lambda **o: S.Cfg(**dict({k: v for k, v in kw.items() if k in sig}, **o))
torch.manual_seed(0); m0 = S.MemLM(f(sh_wq=0))
torch.manual_seed(0); m1 = S.MemLM(f(sh_wq=1))
miss = m1.load_state_dict(m0.state_dict(), strict=False)
assert set(miss.missing_keys) == {"write_q.1.weight", "write_q.4.weight"}, miss
# re-copy the key rows from the loaded weights (the converter does exactly this)
for i, lin in m1.write_q.items():
    mx = m1.blocks[int(i)].mix; k0 = sum(mx.sizes[:4])
    with torch.no_grad(): lin.weight.copy_(mx.w_in.weight[k0:k0 + mx.Hs * mx.dk])
names = [n for n, _ in m1.named_parameters()]
assert names[-2:] == ["write_q.1.weight", "write_q.4.weight"], names[-3:]
print("write_q params are the LAST two parameters:", names[-2:])
torch.manual_seed(1); x = torch.randint(0, 997, (2, 38))
for mode in ("eval", "train"):
    getattr(m0, mode)(); getattr(m1, mode)()
    a, b = m0(x), m1(x)
    assert torch.equal(a, b), (mode, (a - b).abs().max())
print("write query = key copy: outputs bit-identical to the one-key model (eval and train)")
m1.train(); out = m1(x); (out[:, -1].float().pow(2).mean() + m1.krep).backward()
for i, lin in m1.write_q.items():
    assert lin.weight.grad is not None and torch.isfinite(lin.weight.grad).all() and lin.weight.grad.abs().sum() > 0, i
print("both write queries receive finite nonzero gradient")
m1.eval()
with torch.no_grad():
    for lin in m1.write_q.values(): lin.weight.add_(0.05 * torch.randn_like(lin.weight))   # make the two keys differ
    for j in (4, 17, 33):
        a = m1(x); x2 = x.clone(); x2[:, j] = (x2[:, j] + 1) % 997; b = m1(x2)
        assert (a[:, :j] - b[:, :j]).abs().max() == 0, j
    c = m0.eval()(x)
assert (m1(x) - c).abs().max() > 0
print("causal with distinct write query; a different write query changes the output")
print("ALL PASS")
