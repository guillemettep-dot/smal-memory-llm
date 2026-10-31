import io, ast
p = "/home/paul/memprobe/gqa/train_sh.py"; s = io.open(p, encoding="utf-8").read()
def rep(a, b):
    global s
    assert s.count(a) == 1, (s.count(a), a[:60]); s = s.replace(a, b)
rep('''    ap.add_argument("--warmup", type=int, default=2000)''',
    '''    ap.add_argument("--warmup", type=int, default=2000)
    # wsd = warmup, then CONSTANT lr, then a cosine decay over the last decay_frac of the run.  In the
    # constant phase lr does not depend on total_steps, so the token budget can be changed by resuming
    # with a different --tokens at any point before the decay starts -- no restart, no discontinuity.
    ap.add_argument("--sched", type=str, default="cos", choices=["cos", "wsd"])
    ap.add_argument("--decay_frac", type=float, default=0.2)''')
rep('''    def lr_at(s):
        if s < a.warmup:
            return a.lr * (s + 1) / a.warmup
        p = (s - a.warmup) / max(1, total_steps - a.warmup)''',
'''    def lr_at(s):
        if s < a.warmup:
            return a.lr * (s + 1) / a.warmup
        if a.sched == "wsd":
            d0 = max(a.warmup, int(total_steps * (1.0 - a.decay_frac)))
            if s < d0:
                return a.lr
            p = (s - d0) / max(1, total_steps - d0)
            return a.lr * (a.min_lr_frac + (1 - a.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * min(p, 1.0))))
        p = (s - a.warmup) / max(1, total_steps - a.warmup)''')
ast.parse(s); io.open(p, "w", encoding="utf-8").write(s); print("wsd patch applied")
