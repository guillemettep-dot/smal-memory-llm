"""Build sh21 and the 6x128 variant on the meta device: compare parameter counts and memory state size."""
import sys, os, inspect, torch
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, "/home/paul/memprobe/gqa21")
import memlm_sh as S
assert "gqa21" in S.__file__
st = torch.load("/mnt/ssd/memprobe_ckpt/sh21/sh21_ckpt.pt", map_location="cpu", mmap=True, weights_only=False)
cfg = dict(st["cfg"]); print("step", st["step"])
ok = set(inspect.signature(S.Cfg.__init__).parameters) - {"self"}
def build(over):
    c = dict(cfg); c.update(over)
    with torch.device("meta"):
        m = S.MemLM(S.Cfg(**{k: v for k, v in c.items() if k in ok}))
    return m, m.cfg if hasattr(m, "cfg") else None
for name, over in [("sh21", {}), ("6x128", dict(sh_heads=6, sh_kdim=128, sh_dv=128))]:
    m, _ = build(over)
    n = sum(p.numel() for p in m.parameters())
    mem = {k: tuple(p.shape) for k, p in m.named_parameters() if ("write" in k or "mix" in k) and ".6." in k}
    print(f"{name}: params {n:,}")
    for k, s in sorted(mem.items()): print("   ", k, s)
