"""sh20 checkpoint -> sh21 checkpoint with --sh_wqk 1.  The model gains, per writer layer L, write_qk.L.weight (Hs x D)
and write_qk.L.bias (Hs), both ZERO (a = tanh(0) = 0: the same function, verified bitwise in wqk_ident.py).  They are
registered last, so the two new weights are the LAST ids of group 0 (decay, 2-D) and the two new biases the LAST ids
of group 1 (no decay, 1-D): group-1 ids shift by 2, every existing parameter keeps its Adam moments (re-indexed, not
re-initialised -- see memory: resume into new params = optimizer shock), and the 4 new ones start with no state.
Pattern: convert_wq.py (sh7 -> sh8).   argv: src dst   (dst '-' = dry run: verify, write nothing)"""
import sys, os, inspect, torch
sys.path.insert(0, "/home/paul/memprobe/gqa21")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import memlm_sh as S
assert "gqa21" in S.__file__
src, dst = sys.argv[1], sys.argv[2]
st = torch.load(src, map_location="cpu", weights_only=False)
m, o = st["model"], st["opt"]
assert not any(k.startswith("write_qk.") for k in m), "already converted"
writers = sorted({int(k.split(".")[1]) for k in m if k.endswith("mix.g_bias")})
print("source", src, "step", st["step"], "| writers at layers", writers)
cfg = dict(st["cfg"]); cfg["sh_wqk"] = 1
ok = set(inspect.signature(S.Cfg.__init__).parameters) - {"self"}
with torch.device("meta"):
    mm = S.MemLM(S.Cfg(**{kk: v for kk, v in cfg.items() if kk in ok}))
P = dict(mm.named_parameters())
for L in writers:
    wq = m[f"write_q.{L}.weight"]
    m[f"write_qk.{L}.weight"] = torch.zeros(tuple(P[f"write_qk.{L}.weight"].shape), dtype=wq.dtype)
    m[f"write_qk.{L}.bias"] = torch.zeros(tuple(P[f"write_qk.{L}.bias"].shape), dtype=wq.dtype)
g0, g1 = o["param_groups"]
n0, n1, k = len(g0["params"]), len(g1["params"]), len(writers)
assert g0["params"] == list(range(n0)) and g1["params"] == list(range(n0, n0 + n1)), "unexpected id layout"
# the new model's parameter order, split exactly as the trainer splits it (dim >= 2 -> group 0)
dec = [n for n, p in mm.named_parameters() if p.dim() >= 2]
nod = [n for n, p in mm.named_parameters() if p.dim() < 2]
assert len(dec) == n0 + k and len(nod) == n1 + k, (len(dec), n0 + k, len(nod), n1 + k)
assert dec[-k:] == [f"write_qk.{L}.weight" for L in writers], dec[-k:]
assert nod[-k:] == [f"write_qk.{L}.bias" for L in writers], nod[-k:]
old_state = o["state"]
new_state = {}
for i, s in old_state.items():
    new_state[i if i < n0 else i + k] = s
g0["params"] = list(range(n0 + k))
g1["params"] = list(range(n0 + k, n0 + k + n1 + k))
o["state"] = new_state
# every carried state must sit on a parameter of the SAME size in the new layout (catches any off-by-one in the shift)
newnames = dec + nod
# bitsandbytes keeps the moments inside a nested '__bnb_optimizer_quant_state__' dict (8-bit) or at the top level
# (32-bit, small params); torch AdamW at the top level.  A check that only looks at the top level checks NOTHING here
# (found on the first dry run), so count what was checked and require it to cover every entry that has a moment.
n_checked = n_with_moment = 0
for i, s in new_state.items():
    pn = newnames[i]; numel = P[pn].numel()
    found = False
    for d in (s, s.get("__bnb_optimizer_quant_state__", {})):
        for kk_, t in d.items():
            if torch.is_tensor(t) and kk_ in ("state1", "state2", "exp_avg", "exp_avg_sq"):
                assert t.numel() == numel, (i, pn, kk_, t.numel(), numel)
                found = True
    n_with_moment += found; n_checked += found
assert n_checked == len(new_state), f"size-checked {n_checked} of {len(new_state)} state entries"
print(f"size check: every one of {n_checked} carried state entries matches its new parameter's size")
new_ids = [n0 + j for j in range(k)] + [n0 + k + n1 + j for j in range(k)]
assert all(i not in new_state for i in new_ids) and all(newnames[i].startswith("write_qk") for i in new_ids)
missing = set(mm.state_dict().keys()) - set(m.keys()); extra = set(m.keys()) - set(mm.state_dict().keys())
assert not missing and not extra, (missing, extra)
st["cfg"] = cfg
print("verified: groups", len(g0["params"]), len(g1["params"]), "| state entries", len(new_state),
      "| new params without state:", [(i, newnames[i]) for i in new_ids])
if dst != "-":
    assert not os.path.exists(dst), "exists: " + dst
    torch.save(st, dst + ".tmp"); os.replace(dst + ".tmp", dst)
    print("written", dst)
else:
    print("dry run: nothing written")
