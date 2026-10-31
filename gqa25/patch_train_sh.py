import io, ast
p = "/home/paul/memprobe/gqa/train_sh.py"; s = io.open(p, encoding="utf-8").read()
def rep(a, b):
    global s
    assert s.count(a) == 1, (s.count(a), a[:60]); s = s.replace(a, b)
rep('''            _nm, _wt = _it.split(":"); _task_names.append(_nm.strip()); _task_weights.append(float(_wt))
''', '''            _nm, _wt = _it.split(":"); _task_names.append(_nm.strip()); _task_weights.append(float(_wt))
        # the natural-format families (chat, record, codeconst, ...) register themselves into
        # task_lib.GENERATORS on import; an unknown name must fail HERE, not at its first draw hours in
        import task_lib as _TLc, recall_tasks2 as _RT2
        _bad = [n for n in _task_names if n not in _TLc.GENERATORS]
        assert not _bad, "unknown task(s) in --task_mix: %s (known: %s)" % (_bad, sorted(_TLc.GENERATORS))
''')
ast.parse(s); io.open(p, "w", encoding="utf-8").write(s); print("train_sh patched")
