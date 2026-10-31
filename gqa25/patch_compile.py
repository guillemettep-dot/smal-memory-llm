import io, ast
p = "/home/paul/memprobe/gqa/train_sh.py"; s = io.open(p, encoding="utf-8").read()
a = '''        if getattr(model, "shadow", None) is not None:
            model.shadow._shadow_inputs = torch.compile(model.shadow._shadow_inputs, dynamic=False)
        for _blk in model.blocks:'''
b = '''        if getattr(model, "shadow", None) is not None:
            model.shadow._shadow_inputs = torch.compile(model.shadow._shadow_inputs, dynamic=False)
        if _mm.__name__ == "memlm_sh":
            # whole blocks (norms + residual adds fuse too; measured 515 vs 528 ms for mixer-only), the
            # final norm, the pre-blocks.  ExactMoE opts itself out (data-dependent slice sizes).
            for _blk in model.blocks:
                _blk.forward = torch.compile(_blk.forward, dynamic=False)
            model.nf.forward = torch.compile(model.nf.forward, dynamic=False)
        for _blk in (model.blocks if _mm.__name__ != "memlm_sh" else []):'''
assert s.count(a) == 1; s = s.replace(a, b)
ast.parse(s); io.open(p, "w", encoding="utf-8").write(s); print("compile patch applied")
