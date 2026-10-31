"""Write-gate distribution of the shared memories: what fraction of tokens actually write (g >= floor),
per head, plus g_bias and gate_init.  Same spy + data as addrprobe.py (2 x 1024 real windows, rng 3).
Usage: gateprobe.py CKPT [CKPT ...]"""
import os, sys, inspect
os.environ.update(MEMPROBE_ROOT="/home/paul/memprobe", MEMLM_MOD="memlm_sh", KV_DETACH="1", KV_WQ="0", SLOT_ROPE="1", SLOT_ROPE_FRAC="0.5", CUDA_VISIBLE_DEVICES="")
sys.path.insert(0, "/home/paul/memprobe/gqa")
import numpy as np, torch
import memlm_sh as S, slotkern_kv as SK
torch.set_num_threads(int(os.environ.get("THREADS", "4")))
caught = []
_orig = S.ref_scan_kv
def spy(k, v, g, fr, N, temp, **kw):
    caught.append(g.detach().float().clone())
    return _orig(k, v, g, fr, N, temp, **kw)
S.ref_scan_kv = spy
import train_sh as TS
TS.DEV = "cpu"
sh = TS.Shards(doc_aware=True)
rng = np.random.default_rng(3)
x = torch.from_numpy(np.stack([sh._window(1024, rng, True) for _ in range(2)])[:, :1024].astype(np.int64))
assert x.shape == (2, 1024), x.shape
ok = set(inspect.signature(S.Cfg.__init__).parameters) - {"self"}
for path in sys.argv[1:]:
    st = torch.load(path, map_location="cpu", weights_only=False)
    cc = st["cfg"]
    cfg = S.Cfg(**{k: v for k, v in cc.items() if k in ok})
    m = S.MemLM(cfg); m.load_state_dict(st["model"], strict=False); m.float().eval()
    caught.clear()
    with torch.no_grad():
        m(x)
    floor = float(cc.get("sh_gate_floor", 0.0))
    print("\n##### %s  step %s  gate_init %s  floor %.3f  slot_hard %s  sh_wq %s  | %d memories"
          % (os.path.basename(path), st["step"], cc.get("gate_init"), floor, cc.get("slot_hard"), cc.get("sh_wq"), len(caught)))
    gb = {n: p.detach().float() for n, p in m.named_parameters() if n.endswith("g_bias")}
    for name, g in zip(("ENC (layer 2)", "MAIN (layer 6)"), caught):
        # g is the gate AFTER the floor (forward value): exact 0 where the raw gate was below the floor
        g = g.reshape(g.shape[0], g.shape[1], -1)               # (B,T,H)
        B, T, H = g.shape
        thr = 0.5 if int(cc.get("slot_hard_gate", 0)) else 0.0   # hard gate: only g > 0.5 writes (fully)
        live = (g > thr)
        print("  %s  H=%d  mean g %.4f  median %.4f  p90 %.4f  max %.4f  | writing tokens (g>%.1f) %.1f%%  | sum g per seq per head %.1f"
              % (name, H, g.mean(), g.median(), float(np.percentile(g.numpy(), 90)), g.max(), thr, 100 * live.float().mean(), g.sum(1).mean()))
        ph = live.float().mean((0, 1)); pm = g.mean((0, 1))
        print("    per head %% writing: " + " ".join("%5.1f" % (100 * v) for v in ph))
        print("    per head mean g   : " + " ".join("%5.3f" % v for v in pm))
    for n, p in gb.items():
        print("  %s sigmoid(g_bias) per head: %s" % (n, " ".join("%.3f" % v for v in torch.sigmoid(p))))
    del m, st
