import sys, inspect, torch
sys.path.insert(0, "/home/paul/memprobe/gqa")
import memlm_sh as S, memlm_kv as K
torch.manual_seed(0)
fails = []
def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond: fails.append(msg)

# 1. the config filter used by train_memlm and battery3 sees every parameter
sig = set(inspect.signature(S.Cfg.__init__).parameters) - {"self"}
base = set(inspect.signature(K.Cfg.__init__).parameters) - {"self"}
check(base <= sig, "Cfg signature contains all %d base parameters" % len(base))
check({"sh_write", "sh_heads", "sh_slots", "sh_kdim", "sh_dv", "sh_nq", "sh_stride", "sh_qbits", "sh_qgroup", "sh_enc_write", "sh_enc_slots"} <= sig, "Cfg signature contains the shared-memory parameters")
kw = dict(vocab=997, d_model=64, layers=6, heads=2, dv=16, lanes=1, d_ff=64, moe=1, n_exp=2, top_k=2, pre=1,
          kv_window=4, kv_dim=16, kv_heads=2, comp_off=1, tksel=1, slot_hard=0, slot_hard_gate=0, krep=1.0,
          sh_write=2, sh_heads=3, sh_slots=8, sh_kdim=8, sh_dv=8, sh_nq=2, sh_stride=4)
c = S.Cfg(**{k: v for k, v in kw.items() if k in sig})
check(c.sh_write == 2 and c.sh_slots == 8 and c.d_model == 64, "filtered dict builds the intended config: %r" % c)
m = S.MemLM(c).train()

# 2. memory parameters exist only where they should
for i, b in enumerate(m.blocks):
    w, r = b.mix.is_writer, b.mix.is_reader
    check(w == (i == 2) and r == (i >= 2) and hasattr(b.mix, "tk_q") == r and hasattr(b.mix, "g_bias") == w,
          "layer %d: writer=%s reader=%s, params match" % (i, w, r))

# 3. forward / backward, finite, gradients reach the writer THROUGH the readers
B, T = 2, 38          # NOT a multiple of the stride: exercises the padded last block
x = torch.randint(0, 997, (B, T))
out = m(x)
check(out.shape == (B, T, 997), "logits shape %s" % (tuple(out.shape),))
loss = torch.nn.functional.cross_entropy(out[:, -1].float(), x[:, 0]) + (m.krep if m.krep is not None else 0)
loss.backward()
check(all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None), "all gradients finite")
wr = m.blocks[2].mix
check(wr.w_in.weight.grad[sum(wr.sizes[:3]):].abs().sum() > 0, "writer's key/value/gate projections receive gradient")
check(wr.log_wsharp.grad is not None and wr.log_wsharp.grad.abs() > 0, "write sharpness receives gradient")
check(all(m.blocks[i].mix.tk_q.weight.grad.abs().sum() > 0 for i in range(2, 6)), "every reader's query receives gradient")

# 4. causality: changing token j must not change any output before j
m.eval()
with torch.no_grad():
    for j in (3, 4, 5, 8, 16, 17, 31, 36, 37):     # block starts, ends, interiors, last (partial) block
        a = m(x); x2 = x.clone(); x2[:, j] = (x2[:, j] + 1) % 997; b2 = m(x2)
        d_before = (a[:, :j] - b2[:, :j]).abs().max().item()
        d_at = (a[:, j:] - b2[:, j:]).abs().max().item()
        check(d_before == 0.0 and d_at > 0, "perturb token %d: max change before %.2e (must be 0), at/after %.2e" % (j, d_before, d_at))

# 5. memory actually carries information past the window (kv_window=4 here)
torch.manual_seed(1)
m2 = S.MemLM(c).eval()
with torch.no_grad():
    base_out = m2(x)
    xa = x.clone(); xa[:, 3] = (xa[:, 3] + 7) % 997          # 36 tokens before the last position, far outside the window
    far = (m2(xa)[:, -1] - base_out[:, -1]).abs().max().item()
check(far > 0, "a token 36 positions back (window=4) still changes the last output: %.2e (memory path is live)" % far)

# 5b. the strided read equals a naive per-token read of the state at the START of the token's block
from slotkern_kv import ref_scan_kv
import torch.nn.functional as F
mx = m2.blocks[2].mix
with torch.no_grad():
    h = torch.randn(B, T, 64)
    parts = mx.w_in(h).split(mx.sizes, -1)
    v = parts[3].reshape(B, T, 3, 8); kk = parts[4].reshape(B, T, 3, 8)
    g = torch.sigmoid(parts[5].float() + mx.g_bias.float())
    fr = torch.exp(mx.log_free.float()) * mx.free_bias; tp = torch.exp(mx.log_wsharp.float())
    k1, v1 = ref_scan_kv(kk.double(), v.double(), g.double(), fr.double(), 8, tp.double(), G=1)   # every position
    kG, vG = ref_scan_kv(kk.float(), v.float(), g, fr, 8, tp, G=4)
    check(kG.shape[1] == (T + 3) // 4, "stride-4 scan emits ceil(T/4)=%d copies (got %d)" % ((T + 3) // 4, kG.shape[1]))
    check(torch.allclose(kG.double(), k1[:, ::4].double(), atol=1e-5) and torch.allclose(vG.double(), v1[:, ::4].double(), atol=1e-5),
          "stride-4 copy j == per-position state at t=4j")
    nsq = kG.pow(2).sum(-1)
    rd_m = m2.blocks[3].mix
    fast = rd_m._read(kG, vG, torch.rsqrt(nsq.clamp_min(1e-12)), nsq > 1e-12, h)
    q = rd_m.tk_q(h).reshape(B, T, 3, 2, 8).double(); q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    slow = torch.zeros(B, T, 3, 2, 8, dtype=torch.float64)
    for t in range(T):
        K_ = k1[:, (t // 4) * 4].double(); V_ = v1[:, (t // 4) * 4].double()           # (B,3,8,8): state before write t0 = block start
        n = K_.norm(dim=-1); ok = n > 1e-6
        sc = torch.einsum("bhqd,bhnd->bhqn", q[:, t], K_ / n.clamp_min(1e-6).unsqueeze(-1)) * torch.exp(rd_m.log_rsel.double())
        a = torch.nan_to_num(torch.softmax(sc.masked_fill(~ok.unsqueeze(-2), float("-inf")), -1))
        slow[:, t] = torch.einsum("bhqn,bhnv->bhqv", a, V_)
    err = (fast.double() - slow).abs().max().item()
    check(err < 1e-4, "strided read == naive block-start read, max err %.2e (T=%d, partial last block)" % (err, T))
    check(slow[:, :4].abs().max().item() == 0.0 and fast[:, :4].abs().max().item() == 0.0,
          "block 0 reads an empty memory (exactly zero): nothing visible before any write")
try:
    S.Cfg(**dict({k: v for k, v in kw.items() if k in sig}, sh_stride=5)); check(False, "stride > window must be rejected")
except AssertionError as e:
    check("kv_window" in str(e), "stride 5 > window 4 rejected at config time")


# 7. 3-bit QAT: the quantiser and the model with it on
q3 = S._fq
torch.manual_seed(3)
xx = torch.randn(2, 5, 3, 7, 64)
xx[0, 0, 0, 0] = 0.0                                             # an exactly-zero (never written) row
qq = q3(xx, 3, 16)
g = qq.reshape(*qq.shape[:-1], 4, 16)
lv = [(gg / (gg.abs().amax() if gg.abs().amax() > 0 else 1)).unique().numel() for gg in g.reshape(-1, 16)[:50]]
check(max(lv) <= 8, "3-bit groups of 16 use at most 8 distinct values (max seen %d)" % max(lv))
check(qq[0, 0, 0, 0].abs().max().item() == 0.0, "an all-zero row stays exactly zero")
e3 = ((qq - xx).norm() / xx.norm()).item(); e4 = ((q3(xx, 4, 16) - xx).norm() / xx.norm()).item()
check(e3 < 0.2 and e4 < e3, "rel error 3-bit %.3f, 4-bit %.3f (more bits -> less error)" % (e3, e4))
xg = xx.clone().requires_grad_(); (q3(xg, 3, 16) * 2.0).sum().backward()
check(torch.allclose(xg.grad, torch.full_like(xg, 2.0)), "straight-through gradient is exactly the identity")
cq_ = S.Cfg(**dict({k: v for k, v in kw.items() if k in sig}, sh_qbits=3, sh_qgroup=8))
check("q3b/g8" in repr(cq_), "config shows the storage precision: %r" % cq_)
torch.manual_seed(0); mq = S.MemLM(cq_).train()
out = mq(x); (out[:, -1].float().pow(2).mean() + (mq.krep if mq.krep is not None else 0)).backward()
check(all(torch.isfinite(p.grad).all() for p in mq.parameters() if p.grad is not None), "QAT model: gradients finite")
check(mq.blocks[2].mix.w_in.weight.grad[sum(mq.blocks[2].mix.sizes[:3]):].abs().sum() > 0, "QAT model: writer receives gradient through the quantiser")
mq.eval()
with torch.no_grad():
    for j in (4, 17, 37):
        a = mq(x); x2 = x.clone(); x2[:, j] = (x2[:, j] + 1) % 997; b2 = mq(x2)
        check((a[:, :j] - b2[:, :j]).abs().max().item() == 0.0, "QAT model causal at token %d" % j)
    m_fp = S.MemLM(S.Cfg(**{k: v for k, v in kw.items() if k in sig})).eval(); m_fp.load_state_dict(mq.state_dict())
    d = (m_fp(x) - mq(x)).abs().max().item()
check(d > 0, "QAT actually changes the forward (fp vs 3-bit logits differ by %.2e)" % d)


# 8. ENCODER MEMORY: written at layer 1, read by layers 1..sh_write-1 (=1..3); main written at 4, read 4..5
ce = S.Cfg(**dict({k: v for k, v in kw.items() if k in sig}, sh_write=4, sh_enc_write=1, sh_enc_slots=4))
check("enc mem write@1 4 slots" in repr(ce), "config shows the encoder memory: %r" % ce)
torch.manual_seed(0); me = S.MemLM(ce).train()
roles = [(b.mix.mid, b.mix.is_writer, b.mix.N) for b in me.blocks]
check(roles == [(None, False, 8), ("enc", True, 4), ("enc", False, 4), ("enc", False, 4), ("main", True, 8), ("main", False, 8)],
      "layer roles (memory, writer, slots): %s" % roles)
check(hasattr(me.blocks[1].mix, "g_bias") and hasattr(me.blocks[4].mix, "g_bias") and not hasattr(me.blocks[2].mix, "g_bias"),
      "write parameters exist on both writers only")
out = me(x); (out[:, -1].float().pow(2).mean() + (me.krep if me.krep is not None else 0)).backward()
check(all(torch.isfinite(p.grad).all() for p in me.parameters() if p.grad is not None), "enc model: gradients finite")
for li in (1, 4):
    w_ = me.blocks[li].mix
    check(w_.w_in.weight.grad[sum(w_.sizes[:3]):].abs().sum() > 0, "writer at layer %d receives gradient" % li)
me.eval()
with torch.no_grad():
    for j in (4, 17, 37):
        a = me(x); x2 = x.clone(); x2[:, j] = (x2[:, j] + 1) % 997; b2 = me(x2)
        check((a[:, :j] - b2[:, :j]).abs().max().item() == 0.0, "enc model causal at token %d" % j)
    # the lower half now sees far back: perturb token 3 and look at layer 3's output (below the main writer)
    hs = {}
    hk = me.blocks[3].register_forward_hook(lambda m, i, o: hs.__setitem__("h", o[0]))
    me(x); h0 = hs["h"][:, -1].clone()
    xa = x.clone(); xa[:, 3] = (xa[:, 3] + 7) % 997; me(xa); h1 = hs["h"][:, -1].clone(); hk.remove()
    hs2 = {}
    m_off = S.MemLM(S.Cfg(**dict({k: v for k, v in kw.items() if k in sig}, sh_write=4))).eval()
    hk = m_off.blocks[3].register_forward_hook(lambda m, i, o: hs2.__setitem__("h", o[0]))
    m_off(x); g0 = hs2["h"][:, -1].clone(); m_off(xa); g1 = hs2["h"][:, -1].clone(); hk.remove()
check((h1 - h0).abs().max().item() > 0 and (g1 - g0).abs().max().item() == 0.0,
      "layer 3 sees a token 34 back ONLY with the encoder memory (with %.2e, without %.2e)"
      % ((h1 - h0).abs().max().item(), (g1 - g0).abs().max().item()))
torch.manual_seed(0); m_a = S.MemLM(S.Cfg(**dict({k: v for k, v in kw.items() if k in sig}, sh_enc_write=-1)))
check(sum(p.numel() for p in m_a.parameters()) == sum(p.numel() for p in m.parameters()), "enc off: same parameter count as before")

# 6. the target configuration: size
tgt = dict(vocab=50304, d_model=2048, layers=12, heads=2, dv=64, lanes=1, d_ff=1408, moe=1, n_exp=4, top_k=2, pre=2,
           kv_window=16, kv_dim=128, kv_heads=4, comp_off=1, tksel=1, slot_hard=0, slot_hard_gate=0, krep=1.0,
           sh_write=6, sh_heads=12, sh_slots=192, sh_kdim=64, sh_dv=64, sh_nq=2, sh_stride=16, sh_qbits=0, sh_qgroup=16, sh_enc_write=2, sh_enc_slots=96)
ct = S.Cfg(**tgt)
with torch.device("meta"):
    mt = S.MemLM(ct)
n_new = sum(p.numel() for p in mt.parameters())
print("  target: %r" % ct)
print("  target params: %.1fM  (current gqa2k: 617.6M)" % (n_new / 1e6))
print("\nALL PASS" if not fails else "\nFAILURES: %s" % fails)
