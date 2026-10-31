"""sh7 checkpoint -> sh8 checkpoint with the write query: model gains write_q.<L>.weight = the key rows of each
writer's fused w_in (same function); the optimizer state is remapped -- the two new 2-D params are the LAST two
of group 0 (registered last), so group-1 ids shift by 2, and every existing parameter keeps its Adam moments."""
import sys, os, inspect, torch
sys.path.insert(0, "/home/paul/memprobe/gqa")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
src, dst = sys.argv[1], sys.argv[2]
st = torch.load(src, map_location="cpu", weights_only=False)
m, o = st["model"], st["opt"]
writers = sorted({int(k.split(".")[1]) for k in m if k.endswith("mix.g_bias")})
print("source step", st["step"], "| writers at layers", writers)
for L in writers:
    w = m["blocks.%d.mix.w_in.weight" % L]
    assert w.shape[0] == 3084, w.shape
    m["write_q.%d.weight" % L] = w[2304:3072].clone()           # key rows: after 3 x 512 window rows and 768 value rows
g0, g1 = o["param_groups"]
n0, n1, k = len(g0["params"]), len(g1["params"]), len(writers)
assert g0["params"] == list(range(n0)) and g1["params"] == list(range(n0, n0 + n1)), "unexpected id layout"
g0["params"] = list(range(n0 + k))
g1["params"] = list(range(n0 + k, n0 + k + n1))
o["state"] = {(i if i < n0 else i + k): s for i, s in o["state"].items()}
st["cfg"]["sh_wq"] = 1
# verify: the converted model loads STRICTLY into an sh_wq=1 model with the same param order
import memlm_sh as S
ok = set(inspect.signature(S.Cfg.__init__).parameters) - {"self"}
cfg = S.Cfg(**{kk: v for kk, v in st["cfg"].items() if kk in ok})
with torch.device("meta"):
    mm = S.MemLM(cfg)
names = [n for n, _ in mm.named_parameters()]
decay = [n for n, p in mm.named_parameters() if p.dim() >= 2]
assert decay[-k:] == ["write_q.%d.weight" % L for L in writers], decay[-k:]
assert len(decay) == n0 + k and len(names) - len(decay) == n1, (len(decay), n0 + k, len(names) - len(decay), n1)
missing = set(mm.state_dict().keys()) - set(m.keys()); extra = set(m.keys()) - set(mm.state_dict().keys())
assert not missing and not extra, (missing, extra)
torch.save(st, dst + ".tmp"); os.replace(dst + ".tmp", dst)
print("written", dst, "| groups", len(g0["params"]), len(g1["params"]), "| state entries", len(o["state"]), "| new params without state:", [n0 + i for i in range(k)])
