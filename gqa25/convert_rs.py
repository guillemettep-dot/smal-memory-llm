"""sh22 checkpoint -> sh23 checkpoint with --sh_rsharp C.  The model gains, per reader layer L, read_s.L.weight
(Hs*NQ x D), ZERO (multiplier exp(0) = 1: the same function, verified bitwise in kvu/test_rs.py).  They are registered
last, so the new weights are the LAST ids of group 0 (decay, 2-D); group-1 ids shift by k, every existing parameter
keeps its Adam moments (re-indexed, not re-initialised -- memory: resume into new params = optimizer shock), and the k
new ones start with no state.  Pattern: gqa21/convert_wqk.py.   argv: src dst C   (dst '-' = dry run: verify, write nothing)"""
import sys, os, inspect, torch
sys.path.insert(0, "/home/paul/memprobe/gqa23")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import memlm_sh as S
assert "gqa23" in S.__file__
src, dst, C = sys.argv[1], sys.argv[2], float(sys.argv[3])
assert C > 0
st = torch.load(src, map_location="cpu", weights_only=False)
m, o = st["model"], st["opt"]
assert not any(k.startswith("read_s.") for k in m), "already converted"
cfg = dict(st["cfg"]); assert float(cfg.get("sh_rsharp", 0.0)) == 0.0; cfg["sh_rsharp"] = C
ok = set(inspect.signature(S.Cfg.__init__).parameters) - {"self"}
with torch.device("meta"):
    mm = S.MemLM(S.Cfg(**{kk: v for kk, v in cfg.items() if kk in ok}))
readers = sorted(int(k) for k in mm.read_s.keys())
print("source", src, "step", st["step"], "| readers at layers", readers)
P = dict(mm.named_parameters())
dt = m["emb.weight"].dtype
for L in readers:
    m[f"read_s.{L}.weight"] = torch.zeros(tuple(P[f"read_s.{L}.weight"].shape), dtype=dt)
g0, g1 = o["param_groups"]
n0, n1, k = len(g0["params"]), len(g1["params"]), len(readers)
assert g0["params"] == list(range(n0)) and g1["params"] == list(range(n0, n0 + n1)), "unexpected id layout"
dec = [n for n, p in mm.named_parameters() if p.dim() >= 2]
nod = [n for n, p in mm.named_parameters() if p.dim() < 2]
assert len(dec) == n0 + k and len(nod) == n1, (len(dec), n0 + k, len(nod), n1)
assert dec[-k:] == [f"read_s.{L}.weight" for L in readers], dec[-k:]
# the OLD order must be the new order minus the new params (same names, same positions)
with torch.device("meta"):
    c0 = dict(cfg); c0["sh_rsharp"] = 0.0
    m0 = S.MemLM(S.Cfg(**{kk: v for kk, v in c0.items() if kk in ok}))
dec0 = [n for n, p in m0.named_parameters() if p.dim() >= 2]; nod0 = [n for n, p in m0.named_parameters() if p.dim() < 2]
assert dec0 == dec[:-k] and nod0 == nod, "old parameter order is not the new order minus read_s"
new_state = {(i if i < n0 else i + k): s for i, s in o["state"].items()}
g0["params"] = list(range(n0 + k))
g1["params"] = list(range(n0 + k, n0 + k + n1))
o["state"] = new_state
newnames = dec + nod
n_checked = 0
for i, s in new_state.items():
    pn = newnames[i]; numel = P[pn].numel(); found = False
    for d in (s, s.get("__bnb_optimizer_quant_state__", {})):
        for kk_, t in d.items():
            if torch.is_tensor(t) and kk_ in ("state1", "state2", "exp_avg", "exp_avg_sq"):
                assert t.numel() == numel, (i, pn, kk_, t.numel(), numel)
                found = True
    n_checked += found
assert n_checked == len(new_state), f"size-checked {n_checked} of {len(new_state)} state entries"
print(f"size check: every one of {n_checked} carried state entries matches its new parameter's size")
new_ids = [n0 + j for j in range(k)]
assert all(i not in new_state for i in new_ids) and all(newnames[i].startswith("read_s.") for i in new_ids)
missing = set(mm.state_dict().keys()) - set(m.keys()); extra = set(m.keys()) - set(mm.state_dict().keys())
assert not missing and not extra, (missing, extra)
st["cfg"] = cfg
print("verified: groups", len(g0["params"]), len(g1["params"]), "| state entries", len(new_state),
      "| new params without state:", [(i, newnames[i]) for i in new_ids][:3], "...")
if dst != "-":
    assert not os.path.exists(dst), "exists: " + dst
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    torch.save(st, dst + ".tmp"); os.replace(dst + ".tmp", dst)
    print("written", dst)
else:
    print("dry run: nothing written")
