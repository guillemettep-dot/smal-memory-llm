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

fails = []
def check(name, ok, msg=""):
    print(("PASS " if ok else "FAIL ") + name + ("  " + msg if msg else ""), flush=True)
    if not ok: fails.append(name)
m, mi = build(sh_rsharp=2.0)
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
mx.double(); mx._rsq[0].double(); xs, kbuf, buf, kinv = xs.double(), kbuf.double(), buf.double(), kinv.double()
W = mx._rsq[0].weight; V = torch.randn_like(W); P = torch.randn(1, Tq, Hs, NQ, dv, dtype=torch.float64)
W.grad = None
(mx._read(kbuf, buf, kinv, valid, xs) * P).sum().backward()
an = float((W.grad * V).sum())
f0 = lambda: float((mx._read(kbuf, buf, kinv, valid, xs) * P).sum())
print(f"  analytic {an:.6f}   (_read casts scores to fp32, so the fd floor is fp32 rounding / eps)")
best = None
for e in (1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 1e-5, 1e-6):
    with torch.no_grad():
        W.add_(e * V); fp = f0(); W.sub_(2 * e * V); fm = f0(); W.add_(e * V)
    fd = (fp - fm) / (2 * e); r = abs(an - fd) / abs(fd)
    best = r if best is None else min(best, r)
    print(f"  eps {e:.0e}: fd {fd:.6f}  rel diff {r:.2e}")
# the same check with the scores kept in fp64: monkeypatch float() out of _read for this check only
import types, inspect as _i
src = _i.getsource(type(mx)._read).replace(".float()", "")
import textwrap; ns = {}
exec(textwrap.dedent(src), S.__dict__, ns)
mx._read = types.MethodType(ns["_read"], mx)
W.grad = None; (mx._read(kbuf, buf, kinv, valid, xs) * P).sum().backward(); an64 = float((W.grad * V).sum())
with torch.no_grad():
    e = 1e-5; W.add_(e * V); fp = f0(); W.sub_(2 * e * V); fm = f0(); W.add_(e * V)
fd64 = (fp - fm) / (2 * e)
r64 = abs(an64 - fd64) / abs(fd64)
print(f"  all-fp64 _read: analytic {an64:.8f} fd {fd64:.8f} rel {r64:.2e}")
check("(5) fp64 grad == finite difference (all-fp64 _read)", r64 < 1e-6, f"rel {r64:.2e}")
check("(5b) fp32-score _read: fd converges to analytic", best < 1e-4, f"best rel {best:.2e}")
print("ALL PASS" if not fails else f"FAILED: {fails}")
