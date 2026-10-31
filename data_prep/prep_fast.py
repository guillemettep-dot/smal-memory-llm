"""
Parallel FineWeb-Edu tokeniser. Shards the stream across N worker processes.

The single-threaded version managed ~3M tok/s and took ~25 minutes for 5.5B tokens --
expensive when it runs on a rented GPU that is sitting idle. With 32 workers this is
network/CPU bound at roughly 30M+ tok/s.

Each worker writes its own shard files (shard_w<worker>_<n>.npy) so there is no
coordination and no locking; build_index.py globs them all afterwards.
"""
import os, sys, time
import numpy as np
import tiktoken
from multiprocessing import Process

OUT = os.path.join(os.environ.get("MEMPROBE_ROOT", os.path.expanduser("~/memprobe")), "data")
SHARD_TOKENS = int(os.environ.get("SHARD_TOKENS", 50_000_000))
WORKERS = int(os.environ.get("TOK_WORKERS", "32"))
TARGET = int(float(sys.argv[1]) if len(sys.argv) > 1 else 5.5e9)


def worker(wid, target_tokens):
    from datasets import load_dataset
    enc = tiktoken.get_encoding("gpt2")
    EOT = enc.eot_token
    # MUST be sample-100BT: sample-10BT has only ~14 files, so ds.shard(num_shards=32)
    # leaves most workers with nothing and the run silently produces a fraction of
    # the requested tokens.
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-100BT",
                      split="train", streaming=True)
    ds = ds.shard(num_shards=WORKERS, index=wid)

    buf = np.empty(SHARD_TOKENS, dtype=np.uint16)
    fill = 0
    n_shard = 0
    done = 0
    t0 = time.time()
    for doc in ds:
        ids = enc.encode_ordinary(doc["text"])
        ids.append(EOT)
        pos = 0
        while pos < len(ids):
            take = min(len(ids) - pos, SHARD_TOKENS - fill)
            buf[fill:fill + take] = np.array(ids[pos:pos + take], dtype=np.uint16)
            fill += take
            pos += take
            if fill == SHARD_TOKENS:
                p = os.path.join(OUT, f"shard_w{wid:02d}_{n_shard:04d}.npy")
                np.save(p + ".tmp.npy", buf)
                os.replace(p + ".tmp.npy", p)
                n_shard += 1
                done += SHARD_TOKENS
                fill = 0
                if wid == 0:
                    el = time.time() - t0
                    print(f"[w0] {done/1e6:.0f}M tokens  {done/max(el,1):,.0f} tok/s/worker "
                          f"(~{WORKERS*done/max(el,1)/1e6:.0f}M tok/s total)", flush=True)
                if done >= target_tokens:
                    return
    # flush the tail so short workers still contribute
    if fill > 1_000_000:
        p = os.path.join(OUT, f"shard_w{wid:02d}_{n_shard:04d}.npy")
        np.save(p + ".tmp.npy", buf[:fill])
        os.replace(p + ".tmp.npy", p)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    per = TARGET // WORKERS
    print(f"tokenising {TARGET/1e9:.1f}B tokens with {WORKERS} workers "
          f"({per/1e6:.0f}M each)", flush=True)
    t0 = time.time()
    ps = [Process(target=worker, args=(w, per)) for w in range(WORKERS)]
    for p in ps:
        p.start()
    for p in ps:
        p.join()
    files = [f for f in os.listdir(OUT) if f.startswith("shard_") and f.endswith(".npy")]
    total = sum(os.path.getsize(os.path.join(OUT, f)) for f in files) // 2
    el = time.time() - t0
    print(f"DONE {len(files)} shards  {total/1e9:.2f}B tokens in {el:.0f}s "
          f"({total/max(el,1)/1e6:.1f}M tok/s)", flush=True)
