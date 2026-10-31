"""kvnq_run.py CODE CKPT OUT MODE  -- one CPU fp32 forward+backward on fixed held-out tokens, saved for kvnq_cmp.py.
MODE: plain  = load CKPT as is (strict) and run
      o2rand = (kv_nq ckpt) set every win_o2 to seeded random (std 0.02) first, so the extra queries contribute
      as16   = (kv_nq ckpt) o2rand, then rebuild the SAME function as a kv_nq=1 model with kv_heads = KVH*Q: the K/V
               rows duplicated (extra head j uses K/V head j // (Q-1)), q rows [orig | q2], wo_kv = [wo_kv | o2].
               Run through the existing 16-head window path, so it independently checks the GQA mapping + rope.
Saves logits, loss, and grads of every parameter (by name; for as16 the window weights are mapped back)."""
import os, sys, inspect
CODE, CK, OUT, MODE = sys.argv[1:5]
os.environ.update(MEMPROBE_ROOT="/home/paul/memprobe", MEMLM_MOD="memlm_sh", KV_DETACH="1", KV_WQ="0",
                  SLOT_ROPE="1", SLOT_ROPE_FRAC="0.5", CUDA_VISIBLE_DEVICES="")
sys.path.insert(0, "/home/paul/memprobe/" + CODE)
import numpy as np, torch, torch.nn.functional as F
import memlm_sh as S
assert ("/" + CODE + "/") in S.__file__, S.__file__
import cpu_shim; cpu_shim.install(S)
torch.set_num_threads(int(os.environ.get("THREADS", "8")))
assert not os.path.exists(OUT), "exists " + OUT
TL = int(os.environ.get("TL", "300"))
data = np.load("/home/paul/memprobe/data/shard_000000.npy", mmap_mode="r") if os.path.exists("/home/paul/memprobe/data/shard_000000.npy") else None
if data is None:
    import glob; data = np.load(sorted(glob.glob("/home/paul/memprobe/data/shard_*.npy"))[0], mmap_mode="r")
rng = np.random.default_rng(11)
idx = torch.from_numpy(np.stack([np.asarray(data[s:s + TL + 1]) for s in rng.integers(0, 5_000_000, 2)]).astype(np.int64))
st = torch.load(CK, map_location="cpu", weights_only=False)
st.pop("opt", None)   # not needed here; frees ~1.5 GB
cfg = dict(st["cfg"]); sd = {k: v.float() for k, v in st["model"].items()}
ok = set(inspect.signature(S.Cfg.__init__).parameters) - {"self"}
if MODE in ("o2rand", "as16"):
    g = torch.Generator().manual_seed(5)
    for k in sorted(sd):
        if k.startswith("win_o2."):
            sd[k] = torch.randn(sd[k].shape, generator=g) * 0.02
back = {}
if MODE == "as16":
    Q = int(cfg["kv_nq"]); KVH = int(cfg["kv_heads"]); kvd = int(cfg["kv_dim"]); r = Q - 1
    c16 = dict(cfg); c16["kv_nq"] = 1; c16["kv_heads"] = KVH * Q
    with torch.device("meta"):
        mref = S.MemLM(S.Cfg(**{k: v for k, v in cfg.items() if k in ok}))
    for L in sorted(int(k.split(".")[1]) for k in sd if k.startswith("win_q2.")):
        mx = mref.blocks[L].mix
        W = sd.pop(f"blocks.{L}.mix.w_in.weight"); a = KVH * kvd
        wk, wq, wv, rest = W[:a], W[a:2 * a], W[2 * a:3 * a], W[3 * a:]
        heads = lambda w: w.reshape(KVH, kvd, -1)
        dup = lambda w: torch.cat([w, heads(w).repeat_interleave(r, 0).reshape(KVH * r * kvd, -1)], 0)
        q2 = sd.pop(f"win_q2.{L}.weight")
        sd[f"blocks.{L}.mix.w_in.weight"] = torch.cat([dup(wk), torch.cat([wq, q2], 0), dup(wv), rest], 0)
        o2 = sd.pop(f"win_o2.{L}.weight")
        sd[f"blocks.{L}.mix.wo_kv.weight"] = torch.cat([sd[f"blocks.{L}.mix.wo_kv.weight"], o2], 1)
        back[L] = (a, KVH * r * kvd, W.shape[0])
    cfg = c16
model = S.MemLM(S.Cfg(**{k: v for k, v in cfg.items() if k in ok}))
mis, unx = model.load_state_dict(sd, strict=False)
assert not mis and not unx, (mis[:4], unx[:4])
scfg = dict(st["cfg"]); del st, sd
model.float().train()
logits = model(idx[:, :-1])
loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), idx[:, 1:].reshape(-1))
loss.backward()
grads = {n: p.grad for n, p in model.named_parameters()}
if MODE == "as16":
    # map the 16-head window grads back to the kv_nq layout (orig rows, and for k/v the SUM over duplicates)
    for L, (a, e, nrow) in back.items():
        G = grads.pop(f"blocks.{L}.mix.w_in.weight")
        KVH_, r_ = int(cfg["kv_heads"]) // int(scfg["kv_nq"]), int(scfg["kv_nq"]) - 1
        A = a + e
        gk, gq, gv, grest = G[:A], G[A:2 * A], G[2 * A:3 * A], G[3 * A:]
        fold = lambda w: w[:a] + w[a:].reshape(KVH_, r_, a // KVH_, -1).sum(1).reshape(a, -1)
        grads[f"blocks.{L}.mix.w_in.weight"] = torch.cat([fold(gk), gq[:a], fold(gv), grest], 0)
        grads[f"win_q2.{L}.weight"] = gq[a:]
        Go = grads.pop(f"blocks.{L}.mix.wo_kv.weight")
        grads[f"blocks.{L}.mix.wo_kv.weight"] = Go[:, :a]; grads[f"win_o2.{L}.weight"] = Go[:, a:]
torch.save(dict(logits=logits.detach(), loss=loss.detach(), grads=grads, mode=MODE, code=CODE, T=TL), OUT)
print(MODE, CODE, "T", TL, "loss %.6f" % float(loss), "params", len(grads), "->", OUT)
