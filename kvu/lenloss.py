"""lenloss.py CODE CKPT OUT.npz  -- plain-text loss at every context position, on held-out documents.
Each sample is the first L+1 tokens of ONE held-out document (shard 0, wholly inside the first 10M tokens, the
training hold-out), so position t really has t tokens of the same document behind it. Same docs for every
checkpoint (fixed seed, ids saved) so checkpoints can be compared doc-by-doc. CPU fp32, batch 1, no facts spliced in.
env: L (2048) N (192) SEED (20260828) THREADS (14)
     OFF (0): start each sample OFF tokens into the document; POOL_L (L): the doc pool/selection length, so
     OFF=1024 L=1024 POOL_L=2048 scores the SAME docs' tokens 1024..2047 with the context restarted at 1024."""
import os, sys, inspect, hashlib, time
CODE, CK, OUT = sys.argv[1:4]
assert not os.path.exists(OUT), "exists " + OUT
os.environ.update(MEMPROBE_ROOT="/home/paul/memprobe", MEMLM_MOD="memlm_sh", KV_DETACH="1", KV_WQ="0",
                  SLOT_ROPE="1", SLOT_ROPE_FRAC="0.5", CUDA_VISIBLE_DEVICES="")
sys.path.insert(0, "/home/paul/memprobe/" + CODE)
import numpy as np, torch, torch.nn.functional as F
import memlm_sh as S
assert ("/" + CODE + "/") in S.__file__, S.__file__
import cpu_shim; cpu_shim.install(S)
torch.set_num_threads(int(os.environ.get("THREADS", "14")))
L, N, SEED = int(os.environ.get("L", "2048")), int(os.environ.get("N", "192")), int(os.environ.get("SEED", "20260828"))
OFF = int(os.environ.get("OFF", "0")); PL = int(os.environ.get("POOL_L", str(L))); assert OFF + L <= PL, (OFF, L, PL)
D = "/home/paul/memprobe/data"
idx = np.load(os.path.join(D, "docindex.npz")); dsh, dst, dln = idx["shard"], idx["start"], idx["length"]
pool = np.flatnonzero((dsh == 0) & (dln >= PL + 1) & (dst + dln <= 10_000_000))
assert len(pool) >= N, (len(pool), N)
docs = np.sort(np.random.default_rng(SEED).choice(pool, N, replace=False))
import glob
f0 = sorted(glob.glob(os.path.join(D, "shard_*.npy")))[0]; assert f0.endswith("shard_0000.npy"), f0   # = Shards.mm[0], the one with the hold-out
mm = np.load(f0, mmap_mode="r")
toks = np.stack([np.asarray(mm[int(dst[d]) + OFF:int(dst[d]) + OFF + L + 1]).astype(np.int64) for d in docs])
assert toks.shape == (N, L + 1)
dig = hashlib.sha1(toks.tobytes()).hexdigest()[:12]
st = torch.load(CK, map_location="cpu", weights_only=False); st.pop("opt", None)
cfg = dict(st["cfg"]); ok = set(inspect.signature(S.Cfg.__init__).parameters) - {"self"}
model = S.MemLM(S.Cfg(**{k: v for k, v in cfg.items() if k in ok}))
mis, unx = model.load_state_dict(st["model"], strict=False)
assert not mis and not unx, (mis[:4], unx[:4])
step = int(st["step"]); del st
model = model.float().eval(); model.requires_grad_(False)
try: model.set_moe_loop(False)
except Exception: pass
print(f"{CK} step {step} code {CODE} | {N} docs x {L} tokens at offset {OFF} from a pool of {len(pool)} (len>={PL+1}), tokens sha {dig}", flush=True)
ce = np.zeros((N, L), np.float32); t0 = time.time()
with torch.no_grad():
    for i in range(N):
        x = torch.from_numpy(toks[i:i + 1])
        lg = model(x[:, :-1])
        assert lg.shape == (1, L, lg.shape[-1]), lg.shape
        c = F.cross_entropy(lg[0].float(), x[0, 1:], reduction="none")
        assert torch.isfinite(c).all(), i
        ce[i] = c.numpy()
        if (i + 1) % 16 == 0 or i == 0:
            print(f"  {i+1}/{N} docs  {time.time()-t0:.0f}s  mean ce so far {ce[:i+1].mean():.4f}"
                  f"  [1-64] {ce[:i+1,:64].mean():.3f} [512-1024] {ce[:i+1,511:1024].mean():.3f} [1024-{L}] {ce[:i+1,1023:].mean():.3f}", flush=True)
np.savez(OUT, ce=ce, docs=docs, sha=dig, step=step, ckpt=CK, code=CODE, L=L, OFF=OFF)
print("saved", OUT, "mean", ce.mean(), flush=True)
