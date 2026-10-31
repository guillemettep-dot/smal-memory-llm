import io, ast
src = "/home/paul/memprobe/gqa/train_memlm.py"; dst = "/home/paul/memprobe/gqa/train_sh.py"
s = io.open(src, encoding="utf-8").read()
def rep(a, b):
    global s
    assert s.count(a) == 1, ("anchor count %d" % s.count(a), a[:70]); s = s.replace(a, b)
rep('    ap.add_argument("--ans_w", type=float, default=1.0)',
    '''    # shared memory (MEMLM_MOD=memlm_sh): ONE memory written at layer sh_write, read by every layer >= it
    ap.add_argument("--sh_write", type=int, default=4)
    ap.add_argument("--sh_heads", type=int, default=12)
    ap.add_argument("--sh_slots", type=int, default=192)
    ap.add_argument("--sh_kdim", type=int, default=64)
    ap.add_argument("--sh_dv", type=int, default=64)
    ap.add_argument("--sh_nq", type=int, default=2)
    ap.add_argument("--sh_stride", type=int, default=16)   # must be <= kv_window (Cfg asserts it)
    ap.add_argument("--ans_w", type=float, default=1.0)''')
rep('hsel_lam=a.hsel_lam, hsel_tau=a.hsel_tau, hsel_prev=a.hsel_prev)',
    'hsel_lam=a.hsel_lam, hsel_tau=a.hsel_tau, hsel_prev=a.hsel_prev,\n'
    '              sh_write=a.sh_write, sh_heads=a.sh_heads, sh_slots=a.sh_slots, sh_kdim=a.sh_kdim,\n'
    '              sh_dv=a.sh_dv, sh_nq=a.sh_nq, sh_stride=a.sh_stride)')
ast.parse(s)
io.open(dst, "w", encoding="utf-8").write(s)
print("train_sh.py written")
