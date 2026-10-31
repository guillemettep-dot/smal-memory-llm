"""gqa23 sh_rsharp tests on sh22's real weights, CPU fp32.   argv: ref|new  ckpt  out_or_ref_logits
  ref (CODEDIR gqa22): write the reference logits of the unpatched code.
  new (CODEDIR gqa23): (1) sh_rsharp=0 logits == ref bitwise; (2) sh_rsharp=2 with zero read_s logits == ref bitwise;
      (3) read_s gets finite non-zero grads, every other grad == the rsharp=0 grads bitwise;
      (4) _read with RANDOM read_s == a naive per-token loop (T=37, not a multiple of G); (5) fp64 finite difference."""
import os, sys, inspect, math
os.environ.update(MEMPROBE_ROOT="/home/paul/memprobe", MEMLM_MOD="memlm_sh", KV_DETACH="1", KV_WQ="0",
                  SLOT_ROPE="1", SLOT_ROPE_FRAC="0.5", CUDA_VISIBLE_DEVICES="")
mode, CK, LF = sys.argv[1], sys.argv[2], sys.argv[3]
CODE = {"ref": "gqa22", "new": "gqa23"}[mode]
sys.path.insert(0, "/home/paul/memprobe/" + CODE)
import numpy as np, glob, torch, torch.nn.functional as F
import memlm_sh as S
assert os.path.basename(os.path.dirname(S.__file__)) == CODE, S.__file__
import cpu_shim; cpu_shim.install(S); torch.set_num_threads(12)
st = torch.load(CK, map_location="cpu", weights_only=False, mmap=True)
OK = set(inspect.signature(S.Cfg.__init__).parameters) - {"self"}
def build(**over):
    c = {k: v for k, v in st["cfg"].items() if k in OK}; c.update(over)
    torch.manual_seed(0)
    m = S.MemLM(S.Cfg(**c))
    mi = m.load_state_dict(st["model"], strict=False)
    return m.float(), mi
mm = np.load(sorted(glob.glob("/home/paul/memprobe/data/shard_*.npy"))[0], mmap_mode="r")
T = 256
x = torch.tensor(np.asarray(mm[7_000_000:7_000_000 + T + 1]).astype(np.int64))[None]
inp, tgt = x[:, :-1], x[:, 1:]
def logits(m):
    m.eval()
    with torch.no_grad():
        return m(inp).float()
def grads(m):
    m.train(); m.zero_grad(set_to_none=True); torch.manual_seed(123)
    out = m(inp)
    loss = F.cross_entropy(out.float().reshape(-1, out.shape[-1]), tgt.reshape(-1))
    loss.backward()
    # exact fingerprint of every grad (int64 sums over its raw fp32 bits, plain and position-weighted): any 1-bit
    # difference changes it, at 16 bytes per tensor instead of a 2.6 GB copy (the copy OOM-killed the first run)
    fp = {}
    for n, p in m.named_parameters():
        if p.grad is None: continue
        bi = p.grad.detach().float().contiguous().view(torch.int32).reshape(-1)
        s1 = s2 = 0
        for o in range(0, bi.numel(), 1 << 22):                         # 4M-element chunks: bounded RAM
            b = bi[o:o + (1 << 22)].long()
            w = (torch.arange(o + 1, o + 1 + b.numel(), dtype=torch.long) % 1000003)
            s1 += int(b.sum()); s2 += int((b * w).sum())
        fp[n] = (s1, s2, float(p.grad.norm()), bool(torch.isfinite(p.grad).all()))
    m.zero_grad(set_to_none=True)
    return float(loss), fp
print("ckpt step", st["step"], "| code", S.__file__, flush=True)
if mode == "ref":
    m, mi = build(); assert not mi.missing_keys and not mi.unexpected_keys, mi
    L0 = logits(m); torch.save(L0, LF); print("ref logits", tuple(L0.shape), "written", LF); sys.exit(0)

L0 = torch.load(LF)
fails = []
def check(name, ok, msg=""):
    print(("PASS " if ok else "FAIL ") + name + ("  " + msg if msg else ""), flush=True)
    if not ok: fails.append(name)
# (1)
m, mi = build(sh_rsharp=0.0); assert not mi.missing_keys and not mi.unexpected_keys, mi
L1 = logits(m); check("(1) rsharp=0 logits == gqa22 bitwise", torch.equal(L1, L0), f"max|d| {(L1-L0).abs().max():.3e}")
loss0, g0 = grads(m); del m
# (2)
m, mi = build(sh_rsharp=2.0)
new = sorted(mi.missing_keys)
check("(2a) only read_s.* missing", all(k.startswith("read_s.") for k in new) and len(new) == 10 and not mi.unexpected_keys,
      f"{len(new)} missing: {new[:3]}..")
check("(2b) read_s weights are zero", all(float(p.abs().max()) == 0 for p in m.read_s.parameters()))
names = [n for n, _ in m.named_parameters()]
dec = [n for n, p in m.named_parameters() if p.dim() >= 2]
check("(2c) read_s weights are the LAST 10 decay-group params", dec[-10:] == [f"read_s.{i}.weight" for i in range(2, 12)], str(dec[-10:][:2]))
L2 = logits(m); check("(2d) rsharp=2 zero-init logits == gqa22 bitwise", torch.equal(L2, L0), f"max|d| {(L2-L0).abs().max():.3e}")
# (3)
loss2, g2 = grads(m)
check("(3a) loss identical", loss2 == loss0, f"{loss2} vs {loss0}")
same = all(g2[n][:2] == g0[n][:2] for n in g0)
check("(3b) every existing grad bitwise identical", same and set(g0) <= set(g2))
rs = {n: g2[n] for n in g2 if n.startswith("read_s.")}
check("(3c) read_s grads finite and non-zero in all 10 readers",
      len(rs) == 10 and all(g[3] and g[2] > 0 for g in rs.values()),
      " ".join(f"{n.split('.')[1]}:{g[2]:.2e}" for n, g in rs.items()))
print("  rs_a after fwd (layer 6):", [float(t) for t in m.blocks[6].mix.rs_a], "(expect 0, 0)")
# (4) layout: _read vs a naive per-token loop, random read_s, T=37
mx = m.blocks[6].mix; torch.manual_seed(1)
Hs, NQ, dk, dv, G, N = mx.Hs, mx.NQ, mx.dk, mx.dv, mx.G, mx.N
Tq = 37; NB = -(-Tq // G); D = m.c.d_model if hasattr(m, "c") else st["cfg"]["d_model"]
with torch.no_grad():
    mx._rsq[0].weight.normal_(0, 0.05)
    xs = torch.randn(1, Tq, D)
    kbuf = torch.randn(1, NB, Hs, N, dk); buf = torch.randn(1, NB, Hs, N, dv)
    kinv = 1 / kbuf.norm(dim=-1); valid = torch.rand(1, NB, Hs, N) > 0.3
    rd = mx._read(kbuf, buf, kinv, valid, xs)                          # (1,Tq,Hs,NQ,dv)
    q = mx.tk_q(xs)
    if mx.pos:
        pc = S._pos_code(Tq, xs.device, *mx.pos_args)
        if mx.pos_scale != 1.0: pc = pc * mx.pos_scale
        q = q + mx.pos_q(pc)[None]
    q = q.reshape(1, Tq, Hs, NQ, dk); q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    s = mx._rsq[0](xs)[0]; s = 2.0 * torch.tanh(s / 2.0)               # (Tq, Hs*NQ)
    ref = torch.zeros_like(rd); rsel = math.exp(float(mx.log_rsel))
    for t in range(Tq):
        j = t // G
        for h in range(Hs):
            for n in range(NQ):
                sc = (kbuf[0, j, h] @ q[0, t, h, n]) * kinv[0, j, h] * rsel * math.exp(float(s[t, h * NQ + n]))
                sc = sc.masked_fill(~valid[0, j, h], float("-inf"))
                ref[0, t, h, n] = torch.softmax(sc, -1) @ buf[0, j, h]
    d4 = float((rd - ref).abs().max())
    print(f"  s range {float(s.min()):+.2f}..{float(s.max()):+.2f} (bound +-2)")
check("(4) _read == naive per-token loop (random read_s, T=37)", d4 < 1e-4, f"max|d| {d4:.2e}  |ref| {float(ref.abs().max()):.2f}")
# (4b) the multiplier MATTERS: zero vs random read_s must differ
with torch.no_grad():
    w = mx._rsq[0].weight.clone(); mx._rsq[0].weight.zero_(); rd0 = mx._read(kbuf, buf, kinv, valid, xs); mx._rsq[0].weight.copy_(w)
check("(4b) random read_s changes the read", float((rd - rd0).abs().max()) > 1e-2, f"{float((rd-rd0).abs().max()):.3f}")
# (5) fp64 directional finite difference w.r.t. read_s weight
mx.double(); xs, kbuf, buf, kinv = xs.double(), kbuf.double(), buf.double(), kinv.double()
W = mx._rsq[0].weight; V = torch.randn_like(W); P = torch.randn(1, Tq, Hs, NQ, dv, dtype=torch.float64)
W.grad = None
(mx._read(kbuf, buf, kinv, valid, xs) * P).sum().backward()
an = float((W.grad * V).sum())
with torch.no_grad():
    e = 1e-6; W.add_(e * V); fp = float((mx._read(kbuf, buf, kinv, valid, xs) * P).sum())
    W.sub_(2 * e * V); fm = float((mx._read(kbuf, buf, kinv, valid, xs) * P).sum()); W.add_(e * V)
fd = (fp - fm) / (2 * e)
check("(5) fp64 grad == finite difference", abs(an - fd) <= 1e-6 * max(1.0, abs(fd)), f"analytic {an:.8f} fd {fd:.8f}")
print("ALL PASS" if not fails else f"FAILED: {fails}")
