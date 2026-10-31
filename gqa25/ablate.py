# -*- coding: utf-8 -*-
"""WHICH parameters caused recall to fall from step 8500 (code@64 62.5%) to step 11000 (22.9%)?

Norms alone cannot answer it -- wsharp drifted only 8-11%, and at 8500 the deep layers were ALREADY
at 27-30 with healthy recall, so the absolute value is not the story.  So intervene: take the 11000
weights and revert ONE GROUP at a time back to 8500, changing nothing else, and re-measure.

  wsharp    log_wsharp only          -- the write-sharpness hypothesis
  mix       the whole memory module  -- is it the memory at all?
  trunk     everything EXCEPT .mix.  -- this project's own precedent says the trunk, not the memory
  ebias10   one tensor               -- the only parameter qualitatively out of family (0.015->0.964)

Each variant is scored on code@64 and code@256, where the 8500-vs-11000 gap is largest and
unambiguous (62.5/83.3 vs 22.9/22.9).  Recovery toward the 8500 numbers convicts that group.
"""
import os, sys, json, subprocess, torch

CK = "/mnt/ssd/memprobe_ckpt/gqa2k"
OLD, NEW = "%s/gqa2k_step8500.pt" % CK, "%s/gqa2k_step11000.pt" % CK
TMP = "%s/_ablate_tmp.pt" % CK

def group(k, g):
    if g == "wsharp":  return "log_wsharp" in k
    if g == "mix":     return ".mix." in k
    if g == "trunk":   return ".mix." not in k
    if g == "ebias10": return k == "blocks.10.ffn.ebias"
    raise ValueError(g)

a = torch.load(OLD, map_location="cpu", weights_only=False)
b = torch.load(NEW, map_location="cpu", weights_only=False)
print("baseline: step %s (donor) -> step %s (host)" % (a["step"], b["step"]), flush=True)
print("%-9s %7s | %-28s" % ("variant", "swapped", "code@64 / code@256 exact"), flush=True)
print("-" * 58, flush=True)

results = {}
for g in ("wsharp", "mix", "trunk", "ebias10"):
    st = {k: (a["model"][k].clone() if group(k, g) else v.clone()) for k, v in b["model"].items()}
    n = sum(1 for k in b["model"] if group(k, g))
    assert n > 0, "group %s matched nothing" % g
    torch.save({"model": st, "cfg": b["cfg"], "step": b["step"]}, TMP)
    env = dict(os.environ,
               MEMPROBE_ROOT="/home/paul/memprobe", MEMLM_MOD="memlm_kv", CKPT=TMP,
               TASKS="code", DISTS="64,256", N="96", THREADS="4",
               OUT="/home/paul/memprobe/ablate_%s.json" % g)
    subprocess.run(["nice", "-n", "15", "/home/paul/memprobe/.venv/bin/python3", "-u",
                    "/home/paul/memprobe/gqa/battery3.py"], env=env,
                   cwd="/home/paul/memprobe/gqa",
                   stdout=open("/home/paul/memprobe/ablate_%s.log" % g, "w"),
                   stderr=subprocess.STDOUT, check=True)
    r = json.load(open("/home/paul/memprobe/ablate_%s.json" % g))
    e64, e256 = r["code|64"]["exact"], r["code|256"]["exact"]
    results[g] = (e64, e256)
    print("%-9s %7d | %5.1f%% / %5.1f%%" % (g, n, e64, e256), flush=True)

os.path.exists(TMP) and os.remove(TMP)
print("\nreference  step 8500: 62.5%% / 83.3%%   step 11000: 22.9%% / 22.9%%")
json.dump(results, open("/home/paul/memprobe/ablate_summary.json", "w"), indent=1)
