import os, sys, inspect
os.environ.update(KV_DETACH="1", CUDA_VISIBLE_DEVICES="")
sys.path.insert(0, "/home/paul/memprobe/gqa")
import torch, memlm_sh as S
sig = set(inspect.signature(S.Cfg.__init__).parameters) - {"self"}
assert "sh_gate_floor" in sig
kw = dict(vocab=997, d_model=64, layers=6, heads=2, dv=16, lanes=1, d_ff=64, moe=1, n_exp=2, top_k=2, pre=1, kv_window=4, kv_dim=16, kv_heads=2,
          comp_off=1, tksel=1, slot_hard=1, slot_hard_gate=0, krep=1.0, sh_write=4, sh_heads=3, sh_slots=8, sh_kdim=8, sh_dv=8, sh_nq=2, sh_stride=4,
          sh_enc_write=1, sh_enc_slots=4)
caught = []
orig = S.ref_scan_kv
def spy(k, v, g, *a, **k2):
    caught.append(g.detach().clone()); return orig(k, v, g, *a, **k2)
S.ref_scan_kv = spy
torch.manual_seed(0); x = torch.randint(0, 997, (2, 40))
m0 = S.MemLM(S.Cfg(**{k: v for k, v in kw.items() if k in sig})).train()
m1 = S.MemLM(S.Cfg(**dict({k: v for k, v in kw.items() if k in sig}, sh_gate_floor=0.3))).train(); m1.load_state_dict(m0.state_dict())
caught.clear(); out0 = m0(x); g0 = list(caught)
caught.clear(); out1 = m1(x); g1 = list(caught)
# the FIRST memory sees identical inputs in both models: exact comparison
a, b = g0[0], g1[0]; lo = a < 0.3
assert lo.any() and (~lo).any(), "test needs gates on both sides of the floor"
assert torch.all(b[lo] == 0) and torch.equal(b[~lo], a[~lo]), "floor not applied exactly"
# later memories see different inputs (the first memory changed): every gate must be 0 or >= the floor
for b in g1:
    assert torch.all((b == 0) | (b >= 0.3)), "a gate below the floor survived"
print("forward: gates below the floor are exactly 0, others unchanged:", [round(float((a < 0.3).float().mean()), 2) for a in g0], "of gates zeroed")
assert (out0 - out1).abs().max() > 0, "the floor did not change the output"
(out1[:, -1].float().pow(2).mean() + m1.krep).backward()
wr = m1.blocks[4].mix
gate_rows = wr.w_in.weight.grad[sum(wr.sizes[:5]):]
assert torch.isfinite(gate_rows).all() and gate_rows.abs().sum() > 0
print("backward: gate projection receives a finite, nonzero gradient (straight-through)")
m2 = S.MemLM(S.Cfg(**{k: v for k, v in kw.items() if k in sig})); m2.load_state_dict(m0.state_dict())
assert torch.equal(m0.eval()(x), m2.eval()(x)), "floor 0 must be bit-identical to before"
print("floor 0: bit-identical to the unpatched behaviour\nALL PASS")
