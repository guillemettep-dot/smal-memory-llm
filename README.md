# smal-memory-llm

A language model whose long-range memory is a **small, fixed-size learned slot store** instead of a key/value
cache that grows with the context. The model writes into a few hundred slots as it reads, and later tokens read
facts back out of those slots, so the memory costs the same at 1,000 tokens as at 1,000,000.

This repo is a verbatim snapshot of the code of a live research run (**sh25**), trained on one RTX 3060 (12 GB).
It is research code, not a library: the comments record what was tried, what was measured, and what failed.

## The model (sh25)

| | |
|---|---|
| Parameters | 734.5M (output head tied to the embedding); ~527M active per token |
| Trunk | 12 layers, d_model 2048, mixture-of-experts FFN (4 experts, top-2) + 2 dense layers before the stack |
| Shared memory | written at layer 6: 6 heads x 192 slots x (128-dim key + 128-dim value), read by every later layer; plus an encoder memory written at layer 2 (96 slots); 442,368 numbers in total |
| Local attention | a 16-token rolling window in every layer, 4 K/V heads x 128, each read by 4 queries |
| Write | hard (argmax) slot address with a straight-through / detached address gradient, a learned per-token write gate and free-row preference |
| Read | cosine attention over the slots with a learned per-token read sharpness |
| Tokens | GPT-2 BPE, FineWeb-Edu (sample-10BT), 65,536 tokens per optimizer step, length curriculum 128 -> 1024 |
| Kernel | a custom CUDA slot-scan kernel (`gqa25/slotkern_cuda.py`, compiled at runtime), ~5-12x faster than the Triton version it replaced |

Lineage: sh22 (from scratch) -> sh23 (adds per-token read sharpness, branched at step 2,985) -> sh25 (adds 4 queries
per window head, branched at step 9,486). Each branch is bitwise-identical to its parent at the switch; the
conversion scripts (`convert_*.py`) re-index the optimizer state onto the new parameters.

## Layout

```
gqa25/            the model, trainer, kernels and tests for the current run
  memlm_sh.py       shared slot memory (write, read, storage quantisation)
  memlm_kv.py       slot-memory language model (built on memlm_kv_base.py / memlm_m5.py)
  train_sh.py       trainer (curriculum, retrieval-task mix, WSD schedule, checkpointing)
  slotkern_*.py     slot-scan kernels: torch reference, Triton, CUDA
  run_sh25.sh       the exact command line of the current run; run_shN.sh are the earlier arms, kept as history
  guard_sh25.sh     respawn + stall watchdog that resumes from the rolling checkpoint
  needle_pl.py      the main recall probe (fact planted N tokens back, scored against a different-fact control)
  test_*.py         kernel/model correctness tests against float64 references
kvu/              probe and analysis scripts (digit-only needle scoring with a paired bootstrap, CUDA tests, profiling)
data_prep/        FineWeb-Edu -> GPT-2 BPE uint16 shards, and the document-boundary index
```

## Running it

Requirements as used: Python 3.12, PyTorch 2.13 (CUDA 13.0), Triton 3.7, numpy, tiktoken, datasets
(`fla` is only needed for the older delta-rule mixer in `memlm_m5.py`).

1. `python data_prep/prep_data.py` (or `prep_fast.py`) to stream and tokenise the data, then `python data_prep/build_index.py`.
2. `bash gqa25/run_sh25.sh`.

Paths are hard-coded to the machine the run lives on (`/home/paul/memprobe` for code and data, `/mnt/ssd/...` for
checkpoints); `MEMPROBE_ROOT` overrides the root in the scripts that read it. `run_sh25.sh` resumes from a checkpoint;
to train from scratch, start from `run_sh22.sh`, the last from-scratch arm.

## License

No license has been chosen yet, so default copyright applies.
