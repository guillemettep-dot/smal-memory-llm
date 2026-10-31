"""Build a document-boundary index over the token shards (one pass, then sampling is O(log n))."""
import os, glob, time
import numpy as np

DATA = os.path.join(os.environ.get("MEMPROBE_ROOT", os.path.expanduser("~/memprobe")), "data")
EOT = 50256

files = sorted(glob.glob(os.path.join(DATA, "shard_*.npy")))
print(f"{len(files)} shards", flush=True)

shard_id, starts, lens = [], [], []
t0 = time.time()
for si, f in enumerate(files):
    a = np.load(f, mmap_mode="r")
    eot = np.flatnonzero(np.asarray(a) == EOT)
    if eot.size < 2:
        continue
    s = eot[:-1] + 1            # doc starts just after an EOT
    e = eot[1:]                 # doc ends at the next EOT
    L = (e - s).astype(np.int32)
    keep = L > 8
    shard_id.append(np.full(keep.sum(), si, dtype=np.int32))
    starts.append(s[keep].astype(np.int64))
    lens.append(L[keep])
    if si % 10 == 0:
        print(f"  shard {si}  {time.time()-t0:.0f}s", flush=True)

shard_id = np.concatenate(shard_id)
starts = np.concatenate(starts)
lens = np.concatenate(lens)

out = os.path.join(DATA, "docindex.npz")
np.savez(out, shard=shard_id, start=starts, length=lens, nshards=len(files))
print(f"docs {len(lens):,}  mean len {lens.mean():.0f}  median {np.median(lens):.0f}", flush=True)
for T in (128, 256, 512, 1024):
    n = int((lens >= T + 1).sum())
    tok = int((lens[lens >= T + 1] - T).sum())
    print(f"  docs >= {T+1:5d} tokens: {n:9,} ({100*n/len(lens):5.1f}%)  "
          f"valid windows {tok/1e6:8.1f}M", flush=True)
print(f"saved {out}  ({os.path.getsize(out)/2**20:.0f} MB)  in {time.time()-t0:.0f}s", flush=True)
