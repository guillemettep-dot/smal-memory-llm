"""Whole-model TRAINING gradients with the CUDA scan vs the Triton scan, as the trainer runs it (bf16 weights, no
autocast), on one micro-batch of 2 x 1024 held-out tokens.  Baselines: Triton run twice (its backward has atomic
adds, so not necessarily bitwise) and Triton with ~1e-6 relative noise on its first scan output.
Reports, against the first Triton run: loss, the cosine of the full gradient vector, |g - g_ref| / |g_ref|, and
the same for the memory's own parameters (write_q*, write_qk*, *.mix.*).  Needs the GPU to itself.
argv: checkpoint"""
import os, sys, glob, inspect
os.environ.update(MEMPROBE_ROOT="/home/paul/memprobe", MEMLM_MOD="memlm_sh", KV_DETACH="1", KV_WQ="0",
                  SLOT_ROPE="1", SLOT_ROPE_FRAC="0.5", CUDA_VISIBLE_DEVICES="0")
sys.path.insert(0, "/home/paul/memprobe/gqa22")
import numpy as np, torch, torch.nn.functional as Fn
import memlm_sh as S
import slotkern_cuda as KC
assert "gqa22" in S.__file__ and "gqa22" in KC.__file__
DATA = "/home/paul/memprobe/data"
idx = np.load(os.path.join(DATA, "docindex.npz"))
d_shard, d_start, d_len = idx["shard"], idx["start"], idx["length"]
val = np.flatnonzero((d_shard == 0) & (d_start < 10_000_000) & (d_len >= 1100))
mm = np.load(sorted(glob.glob(os.path.join(DATA, "shard_*.npy")))[0], mmap_mode="r")
rng = np.random.default_rng(20261025)
seqs = np.stack([np.asarray(mm[int(d_start[d]):int(d_start[d]) + 1025]).astype(np.int64) for d in rng.choice(val, 2, replace=False)])
x = torch.tensor(seqs[:, :-1], device="cuda"); y = torch.tensor(seqs[:, 1:], device="cuda")

st = torch.load(sys.argv[1], map_location="cpu", weights_only=False, mmap=True)
ok = set(inspect.signature(S.Cfg.__init__).parameters) - {"self"}
m = S.MemLM(S.Cfg(**{k: v for k, v in st["cfg"].items() if k in ok}))
mi, un = m.load_state_dict(st["model"], strict=False); assert not mi and not un, (mi, un)
print("checkpoint step", st["step"]); del st
m = m.to("cuda").to(torch.bfloat16).train()
names = [n for n, p in m.named_parameters() if p.requires_grad]
memp = [n for n in names if n.startswith("write_q") or ".mix." in n]

_tri = S.kv_scan_g
NOISE = [0]
def noisy(k, v, g, fr, N, temp, **kw):
    kb, vb = _tri(k, v, g, fr, N, temp, **kw)
    if NOISE[0] == 0:
        gen = torch.Generator(device=kb.device).manual_seed(99)
        kb = kb * (1 + 1e-6 * torch.randn(kb.shape, device=kb.device, generator=gen))
        vb = vb * (1 + 1e-6 * torch.randn(vb.shape, device=vb.device, generator=gen))
    NOISE[0] += 1
    return kb, vb
CALLS = [0]
def counted(k, v, g, fr, N, temp, **kw):
    CALLS[0] += 1
    return KC.kv_scan_c(k, v, g, fr, N, temp, **kw)

def grads(fn):
    S.kv_scan_g = fn; NOISE[0] = 0
    m.zero_grad(set_to_none=True)
    torch.manual_seed(0)
    lg = m(x)
    lg = lg[0] if isinstance(lg, (tuple, list)) else lg
    loss = Fn.cross_entropy(lg.float().reshape(-1, lg.shape[-1]), y.reshape(-1))
    loss.backward()
    S.kv_scan_g = _tri
    out = {n: p.grad.detach().float().cpu() for n, p in m.named_parameters() if p.grad is not None}
    return loss.item(), out

def compare(ref, oth, sel):
    dot = nr = no = nd = 0.0
    for n in sel:
        a, b = ref.get(n), oth.get(n)
        if a is None and b is None: continue
        assert a is not None and b is not None, ("grad missing", n)
        a = a.double(); b = b.double()
        dot += (a * b).sum().item(); nr += (a * a).sum().item(); no += (b * b).sum().item(); nd += ((a - b) ** 2).sum().item()
    return dot / max((nr * no) ** 0.5, 1e-300), (nd / max(nr, 1e-300)) ** 0.5

l0, g0 = grads(_tri)
print(f"triton           loss {l0:.6f}  params with grad {len(g0)} (memory params {sum(n in g0 for n in memp)})", flush=True)
for label, fn in (("triton again", _tri), ("triton + 1e-6", noisy), ("cuda", counted)):
    l, g = grads(fn)
    ca, ra = compare(g0, g, names); cm, rm = compare(g0, g, memp)
    print(f"{label:16s} loss {l:.6f} (d {l - l0:+.2e}) | all params: cos {ca:.6f} rel diff {ra:.2e}"
          f" | memory params: cos {cm:.6f} rel diff {rm:.2e}", flush=True)
    del g
print("cuda scan calls in its run:", CALLS[0])
assert CALLS[0] > 0
