"""sh23 checkpoint -> sh25 checkpoint with --kv_nq Q.  The model gains, per window layer L, win_q2.L.weight
(KVH*(Q-1)*kvd x D) and win_o2.L.weight (D x KVH*(Q-1)*kvd).  win_o2 is ZERO (the same function: kvu/test_kvnq.py
checks it bitwise); win_q2 is random normal with, per layer, the std of that layer's TRAINED window-query rows, so the
new queries' scores start at the trained scale.  Registered LAST (after read_s): the new weights are the last ids of
group 0 (decay, 2-D), group-1 ids shift by k, every existing parameter keeps its Adam moments (re-indexed, not
re-initialised -- memory: resume into new params = optimizer shock), and the k new ones start with no state.
Pattern: gqa23/convert_rs.py.   argv: src dst Q   (dst '-' = dry run: verify, write nothing)"""
import sys, os, inspect, torch
sys.path.insert(0, "/home/paul/memprobe/gqa25")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import memlm_sh as S
assert "gqa25" in S.__file__, S.__file__
src, dst, Q = sys.argv[1], sys.argv[2], int(sys.argv[3])
assert Q > 1
st = torch.load(src, map_location="cpu", weights_only=False)
m, o = st["model"], st["opt"]
assert not any(k.startswith(("win_q2.", "win_o2.")) for k in m), "already converted"
cfg = dict(st["cfg"]); assert int(cfg.get("kv_nq", 1)) == 1; cfg["kv_nq"] = Q
ok = set(inspect.signature(S.Cfg.__init__).parameters) - {"self"}
with torch.device("meta"):
    mm = S.MemLM(S.Cfg(**{kk: v for kk, v in cfg.items() if kk in ok}))
wl = sorted(int(k) for k in mm.win_q2.keys())
assert wl == sorted(int(k) for k in mm.win_o2.keys())
print("source", src, "step", st["step"], "| window layers", wl)
P = dict(mm.named_parameters())
dt = m["emb.weight"].dtype
gen = torch.Generator().manual_seed(20261028)
for L in wl:
    mx = mm.blocks[L].mix
    W = m[f"blocks.{L}.mix.w_in.weight"].float()
    q0, q1 = mx.sizes[0], mx.sizes[0] + mx.sizes[1]                      # [window k | window q | window v | ...]
    assert mx.sizes[0] == mx.sizes[1] == mx.KVH * mx.kvd
    sd = float(W[q0:q1].std())
    shp = tuple(P[f"win_q2.{L}.weight"].shape)
    m[f"win_q2.{L}.weight"] = (torch.randn(shp, generator=gen) * sd).to(dt)
    m[f"win_o2.{L}.weight"] = torch.zeros(tuple(P[f"win_o2.{L}.weight"].shape), dtype=dt)
    print(f"  layer {L}: trained window-q std {sd:.4f} -> win_q2 {shp}, win_o2 zero {tuple(P[f'win_o2.{L}.weight'].shape)}")
g0, g1 = o["param_groups"]
n0, n1, k = len(g0["params"]), len(g1["params"]), 2 * len(wl)
assert g0["params"] == list(range(n0)) and g1["params"] == list(range(n0, n0 + n1)), "unexpected id layout"
dec = [n for n, p in mm.named_parameters() if p.dim() >= 2]
nod = [n for n, p in mm.named_parameters() if p.dim() < 2]
assert len(dec) == n0 + k and len(nod) == n1, (len(dec), n0 + k, len(nod), n1)
assert dec[-k:] == [f"win_q2.{L}.weight" for L in wl] + [f"win_o2.{L}.weight" for L in wl], dec[-k:]
# the OLD order must be the new order minus the new params (same names, same positions)
with torch.device("meta"):
    c0 = dict(cfg); c0["kv_nq"] = 1
    m0 = S.MemLM(S.Cfg(**{kk: v for kk, v in c0.items() if kk in ok}))
dec0 = [n for n, p in m0.named_parameters() if p.dim() >= 2]; nod0 = [n for n, p in m0.named_parameters() if p.dim() < 2]
assert dec0 == dec[:-k] and nod0 == nod, "old parameter order is not the new order minus win_q2/win_o2"
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
new_ids = list(range(n0, n0 + k))
assert all(i not in new_state for i in new_ids) and all(newnames[i].startswith(("win_q2.", "win_o2.")) for i in new_ids)
missing = set(mm.state_dict().keys()) - set(m.keys()); extra = set(m.keys()) - set(mm.state_dict().keys())
assert not missing and not extra, (missing, extra)
for kk_, v in mm.state_dict().items():
    assert tuple(m[kk_].shape) == tuple(v.shape), (kk_, tuple(m[kk_].shape), tuple(v.shape))
st["cfg"] = cfg
print("verified: groups", len(g0["params"]), len(g1["params"]), "| state entries", len(new_state),
      "| new params without state:", k, "| params", sum(p.numel() for p in mm.parameters()))
if dst != "-":
    assert not os.path.exists(dst), "exists: " + dst
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    torch.save(st, dst + ".tmp"); os.replace(dst + ".tmp", dst)
    print("written", dst)
else:
    print("dry run: nothing written")
