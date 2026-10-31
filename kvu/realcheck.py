"""The CUDA scan on the MODEL'S OWN inputs (sh21 bank, held-out documents, T=1024).
  1. whole-model logits and next-token loss with the Triton scan vs with the CUDA scan swapped in;
  2. every scan call of those forwards (main + encoder memory) captured and replayed through both kernels with
     gradients (random upstream), comparing snapshots (incl. the share of rows that differ = a different argmax)
     and every gradient.
argv: checkpoint   env: NDOC (4)  MEMFRAC (0.24)"""
import os, sys, glob, inspect
os.environ.update(MEMPROBE_ROOT="/home/paul/memprobe", MEMLM_MOD="memlm_sh", KV_DETACH="1", KV_WQ="0",
                  SLOT_ROPE="1", SLOT_ROPE_FRAC="0.5", CUDA_VISIBLE_DEVICES="0")
sys.path.insert(0, "/home/paul/memprobe/gqa21")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, torch.nn.functional as Fn
import memlm_sh as S
import slotkern_cuda as KC
assert "gqa21" in S.__file__
torch.cuda.set_per_process_memory_fraction(float(os.environ.get("MEMFRAC", "0.24")), 0)
ck = sys.argv[1]
NDOC = int(os.environ.get("NDOC", "4"))
DATA = "/home/paul/memprobe/data"
idx = np.load(os.path.join(DATA, "docindex.npz"))
d_shard, d_start, d_len = idx["shard"], idx["start"], idx["length"]
val = np.flatnonzero((d_shard == 0) & (d_start < 10_000_000) & (d_len >= 1100))
mm = np.load(sorted(glob.glob(os.path.join(DATA, "shard_*.npy")))[0], mmap_mode="r")
rng = np.random.default_rng(20261024)
docs = []
for d in rng.choice(val, NDOC, replace=False):
    s = int(d_start[d]); docs.append(np.asarray(mm[s:s + 1025]).astype(np.int64))

st = torch.load(ck, map_location="cpu", weights_only=False, mmap=True)
ok = set(inspect.signature(S.Cfg.__init__).parameters) - {"self"}
m = S.MemLM(S.Cfg(**{k: v for k, v in st["cfg"].items() if k in ok}))
mi, un = m.load_state_dict(st["model"], strict=False); assert not mi and not un, (mi, un)
print("checkpoint step", st["step"]); del st
m = m.to("cuda", torch.bfloat16).eval()

_tri = S.kv_scan_g
CAP = []
def spy(k, v, g, fr, N, temp, **kw):
    CAP.append(dict(k=k.detach().clone(), v=v.detach().clone(), g=g.detach().float().clone(),
                    fr=fr.detach().float().clone() if torch.is_tensor(fr) else fr, N=N,
                    temp=torch.as_tensor(temp).detach().float().clone(),
                    kw={kk: (vv.detach().float().clone() if torch.is_tensor(vv) else vv) for kk, vv in kw.items()}))
    return _tri(k, v, g, fr, N, temp, **kw)

def logits(x):
    out = m(x)
    out = out[0] if isinstance(out, (tuple, list)) else out
    return out.float()

NOISE = [0]
def noisy(k, v, g, fr, N, temp, **kw):
    # CONTROL: Triton itself, with ~1e-6 relative noise on the FIRST scan call's output only (about the size of
    # the CUDA-vs-Triton difference): measures how much the model amplifies a perturbation that small
    kb, vb = _tri(k, v, g, fr, N, temp, **kw)
    if NOISE[0] == 0:
        gen = torch.Generator(device=kb.device).manual_seed(99)
        kb = kb * (1 + 1e-6 * torch.randn(kb.shape, device=kb.device, generator=gen))
        vb = vb * (1 + 1e-6 * torch.randn(vb.shape, device=vb.device, generator=gen))
    NOISE[0] += 1
    return kb, vb
def stats(a, b, y):
    return ((a - b).abs().max().item(), Fn.cross_entropy(b[0], y).item(), (a[0].argmax(-1) == b[0].argmax(-1)).float().mean().item())
worst_logit = 0.0
with torch.no_grad():
    for i, d in enumerate(docs):
        x = torch.tensor(d[:-1], device="cuda")[None]; y = torch.tensor(d[1:], device="cuda")
        S.kv_scan_g = spy; lt = logits(x)
        S.kv_scan_g = _tri; lt2 = logits(x)
        NOISE[0] = 0; S.kv_scan_g = noisy; ln = logits(x)
        S.kv_scan_g = KC.kv_scan_c; lc = logits(x)
        S.kv_scan_g = _tri
        ce_t = Fn.cross_entropy(lt[0], y).item()
        r2, rn, rc = stats(lt, lt2, y), stats(lt, ln, y), stats(lt, lc, y)
        worst_logit = max(worst_logit, rc[0])
        print(f"doc {i}: loss triton {ce_t:.5f} | max|dlogit| / loss / top-1 agreement vs triton:"
              f"  triton again {r2[0]:.1e} {r2[1]:.5f} {r2[2]:.4f}"
              f" | triton+1e-6 noise {rn[0]:.2e} {rn[1]:.5f} {rn[2]:.4f}"
              f" | cuda {rc[0]:.2e} {rc[1]:.5f} {rc[2]:.4f}", flush=True)
del m; torch.cuda.empty_cache()
print(f"{len(CAP)} scan calls captured; kwargs of the first: " +
      str({k: (tuple(v.shape) if torch.is_tensor(v) else v) for k, v in CAP[0]['kw'].items()}), flush=True)

worst = {}
rows_bad = 0.0
for ci, c in enumerate(CAP):
    res = []
    gen = torch.Generator(device="cuda").manual_seed(ci)
    for fn in (_tri, KC.kv_scan_c):
        leaves = dict(k=c["k"].clone().requires_grad_(), v=c["v"].clone().requires_grad_(), g=c["g"].clone().requires_grad_(),
                      fr=c["fr"].clone().requires_grad_(), temp=c["temp"].clone().requires_grad_())
        kw = dict(c["kw"])
        if torch.is_tensor(kw.get("q")): kw["q"] = kw["q"].clone().requires_grad_(); leaves["q"] = kw["q"]
        kb, vb = fn(leaves["k"], leaves["v"], leaves["g"], leaves["fr"], c["N"], leaves["temp"], **kw)
        if not res:
            gk = torch.randn(kb.shape, device="cuda", generator=gen); gv = torch.randn(vb.shape, device="cuda", generator=gen)
        (kb * gk + vb * gv).sum().backward()
        res.append((kb.detach(), vb.detach(), {n: t.grad for n, t in leaves.items()}))
    (kt, vt, gt), (kc_, vc, gc) = res
    rd = (((kt - kc_).abs().amax(-1) > 1e-3) | ((vt - vc).abs().amax(-1) > 1e-3)).double().mean().item()
    rows_bad = max(rows_bad, rd)
    line = f"call {ci:2d} N{c['N']:3d}: snapshots max|d| {max((kt-kc_).abs().max().item(), (vt-vc).abs().max().item()):.1e} rows differing {rd:.4f} |"
    for n in gt:
        a, b = gt[n], gc[n]
        if a is None or b is None:
            line += f" {n} {'none' if a is None and b is None else 'MISSING'}"; continue
        r = ((a.double() - b.double()).abs().max() / a.double().abs().max().clamp_min(1e-30)).item()
        worst[n] = max(worst.get(n, 0.0), r); line += f" {n} {r:.1e}"
    print(line, flush=True)
    del res; torch.cuda.empty_cache()
print("\nworst over all calls: rows differing %.4f | " % rows_bad + "  ".join(f"{n} {v:.1e}" for n, v in worst.items())
      + f" | logits {worst_logit:.2e}")
