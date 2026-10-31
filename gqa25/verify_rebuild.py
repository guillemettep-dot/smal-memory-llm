# -*- coding: utf-8 -*-
"""Two checks before any training time is spent on the reconstructed source.

A. FAITHFULNESS -- the GQA patch claims tk_nq defaults to 1 and then builds exactly the tensors
   the pre-GQA model built. Compare every parameter NAME and SHAPE against memlm_kv_base.
B. SIZING       -- build the target geometry and report the parameter count and the state budget.
"""
import os, sys, torch
os.environ.setdefault("KV_DETACH", "1"); os.environ.setdefault("KV_WQ", "0")
os.environ.setdefault("SLOT_ROPE", "1"); os.environ.setdefault("SLOT_ROPE_FRAC", "0.5")
import memlm_kv, memlm_kv_base

SMALL = dict(vocab=1024, d_model=256, layers=2, heads=2, dk=32, dv=32, kdim=32, d_ff=256, lanes=1,
             n_slots=16, kv_window=16, kv_dim=32, kv_heads=2, moe=1, n_exp=4, top_k=2,
             tksel=1, tksel_dk=32, comp_off=1, xbus=0, slot_hard=0, slot_hard_gate=0, krep=1.0, kdetach=1)

def shapes(mod, **kw):
    cfg = mod.Cfg(**kw)
    m = mod.MemLM(cfg)
    return {n: tuple(p.shape) for n, p in m.named_parameters()}

print("=== A. faithfulness: patched@tk_nq=1  vs  pre-GQA base ===")
a = shapes(memlm_kv_base, **SMALL)
b = shapes(memlm_kv, tk_nq=1, **SMALL)
only_a, only_b = set(a) - set(b), set(b) - set(a)
diff = {k: (a[k], b[k]) for k in set(a) & set(b) if a[k] != b[k]}
print("  base params %d | patched params %d" % (len(a), len(b)))
print("  only in base:    %s" % (sorted(only_a) or "none"))
print("  only in patched: %s" % (sorted(only_b) or "none"))
print("  differing shapes: %s" % (diff or "none"))
ok = not only_a and not only_b and not diff
print("  ->", "IDENTICAL - the patch is transparent at tk_nq=1" if ok else "*** DIVERGED ***")
if not ok: sys.exit(1)

print("\n=== A2. tk_nq=4 changes only the query/out projections ===")
c = shapes(memlm_kv, tk_nq=4, **SMALL)
ch = {k: (b[k], c[k]) for k in set(b) & set(c) if b[k] != c[k]}
for k, (x, y) in sorted(ch.items()): print("    %-42s %s -> %s" % (k, x, y))
print("  changed: %d of %d | added/removed: %s" % (len(ch), len(b), sorted(set(c) ^ set(b)) or "none"))

print("\n=== B. target geometry: d_model 2048, 2 heads of 64, 4 queries each ===")
TARGET = dict(vocab=50304, d_model=2048, layers=12, heads=2, dk=64, dv=64, kdim=64, d_ff=1408, lanes=1,
              n_slots=96, kv_window=16, kv_dim=128, kv_heads=4, tk_nq=4, moe=1, n_exp=12, top_k=2,
              tksel=1, tksel_dk=32, comp_off=1, xbus=0, slot_hard=0, slot_hard_gate=0,
              gate_init=0.42, krep=1.0, krep_tau=0.85, kdetach=1, pre=2, scan_chunk=256,
              buf_stride=1, comp_nb=5, sel_heads=0, cap=2.0)
m = memlm_kv.MemLM(memlm_kv.Cfg(**TARGET))
n = sum(p.numel() for p in m.parameters())
H, R, K, V, NQ, L = TARGET["heads"], TARGET["n_slots"], TARGET["kdim"], TARGET["dv"], TARGET["tk_nq"], TARGET["layers"]
KVH, KVD, W = TARGET["kv_heads"], TARGET["kv_dim"], TARGET["kv_window"]
slot, win = H*R*(K+V), KVH*W*KVD*2
print("  parameters          %.0fM" % (n/1e6))
print("  slot state / layer  %d   (h1 was 12,288 -> x%.1f)" % (slot, slot/12288))
print("  window / layer      %d   (was 8,192 -> x%.1f)" % (win, win/8192))
print("  numbers per fact    %d   (h1 128 -> x%.1f)  [heads x (k+v)]" % (H*(K+V), H*(K+V)/128))
print("  whole state         %d numbers = %.0f KiB at 3 bits" % ((slot+win)*L, (slot+win)*L*3/8/1024))
print("  fp16 KV cache/token %d B -> whole memory = %.2f tokens of cache" %
      (2*L*TARGET["d_model"]*2, ((slot+win)*L*3/8) / (2*L*TARGET["d_model"]*2)))
