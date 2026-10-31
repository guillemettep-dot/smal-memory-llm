import io, ast
p = "/home/paul/memprobe/gqa/train_sh.py"; s = io.open(p, encoding="utf-8").read()
a = '''        print(f"resumed from step {step0}", flush=True)
'''
b = '''        print(f"resumed from step {step0}", flush=True)
        # the checkpoint was loaded onto the GPU (map_location=DEV): model + optimizer, ~2.5 GB for this model.
        # Its tensors have been copied into the model and the optimizer, so the loaded dict is dead weight --
        # and it was never freed.  Measured 2026-10-07: sh6 resumed at 11.0 GiB and OOMed twice in the first
        # steps, where an un-resumed run peaks near 8 GB.  Drop it.
        del st, _rs
        torch.cuda.empty_cache()
'''
assert s.count(a) == 1; s = s.replace(a, b)
ast.parse(s); io.open(p, "w", encoding="utf-8").write(s); print("resume memory patch applied")
