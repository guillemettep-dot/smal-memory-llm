"""
Stream FineWeb-Edu -> GPT-2 BPE -> uint16 shards on disk.

Runs continuously in the background; the trainer consumes shards as they appear.
Training eats ~20k tok/s (~40 KB/s of uint16), so this only has to sustain a trickle
to stay far ahead.
"""
import os, sys, time
import numpy as np
import tiktoken
from datasets import load_dataset

OUT = os.path.expanduser("~/memprobe/data")
SHARD_TOKENS = 100_000_000          # 100M tokens per shard (~200 MB)
TARGET_TOKENS = int(float(sys.argv[1]) if len(sys.argv) > 1 else 5.5e9)

os.makedirs(OUT, exist_ok=True)
enc = tiktoken.get_encoding("gpt2")
EOT = enc.eot_token

existing = sorted(f for f in os.listdir(OUT) if f.startswith("shard_") and f.endswith(".npy"))
shard_idx = len(existing)
done_tokens = shard_idx * SHARD_TOKENS
print(f"resuming: {shard_idx} shards on disk ({done_tokens/1e9:.2f}B tokens)", flush=True)

ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT",
                  split="train", streaming=True)
# skip what we already wrote (approximate: docs average ~1k tokens)
if shard_idx:
    ds = ds.skip(int(done_tokens / 1000))

buf = np.empty(SHARD_TOKENS, dtype=np.uint16)
fill = 0
t0 = time.time()
ndoc = 0

for doc in ds:
    ids = enc.encode_ordinary(doc["text"])
    ids.append(EOT)
    ndoc += 1
    pos = 0
    while pos < len(ids):
        take = min(len(ids) - pos, SHARD_TOKENS - fill)
        buf[fill:fill + take] = np.array(ids[pos:pos + take], dtype=np.uint16)
        fill += take
        pos += take
        if fill == SHARD_TOKENS:
            path = os.path.join(OUT, f"shard_{shard_idx:04d}.npy")
            np.save(path + ".tmp.npy", buf)
            os.replace(path + ".tmp.npy", path)      # atomic: trainer never sees a partial shard
            shard_idx += 1
            done_tokens += SHARD_TOKENS
            el = max(time.time() - t0, 1e-9)
            rate = SHARD_TOKENS / el
            print(f"shard {shard_idx:04d}  {done_tokens/1e9:.2f}B tokens  "
                  f"{rate:,.0f} tok/s  {ndoc:,} docs", flush=True)
            t0 = time.time()
            fill = 0
            if done_tokens >= TARGET_TOKENS:
                print("TARGET REACHED", flush=True)
                sys.exit(0)
print("stream exhausted", flush=True)
