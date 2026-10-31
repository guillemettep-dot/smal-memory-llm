import io, ast
p = "/home/paul/memprobe/gqa/train_sh.py"; s = io.open(p, encoding="utf-8").read()
def rep(a, b):
    global s
    assert s.count(a) == 1, (s.count(a), a[:60]); s = s.replace(a, b)
rep('''    ap.add_argument("--sh_qgroup", type=int, default=16)   # channels per quantisation scale
''', '''    ap.add_argument("--sh_qgroup", type=int, default=16)   # channels per quantisation scale
    # QAT SCHEDULE (DeepSeek-V4.1 pretrains its global KV in full precision and adds FP4 QAT only in
    # post-training): memory stays full precision until step qat_start_frac * total_steps, then trains at
    # qat_bits.  Overrides --sh_qbits once it fires; the checkpoint cfg records the precision in force.
    ap.add_argument("--qat_bits", type=int, default=0)
    ap.add_argument("--qat_start_frac", type=float, default=0.8)
    ap.add_argument("--sh_enc_write", type=int, default=-1)   # encoder memory write layer (-1 = off)
    ap.add_argument("--sh_enc_slots", type=int, default=96)
''')
rep('''              sh_qbits=a.sh_qbits, sh_qgroup=a.sh_qgroup)''',
    '''              sh_qbits=a.sh_qbits, sh_qgroup=a.sh_qgroup,
              sh_enc_write=a.sh_enc_write, sh_enc_slots=a.sh_enc_slots)''')
rep('''    for step in range(step0, total_steps):
        lr = lr_at(step)''', '''    _qat_now = None
    for step in range(step0, total_steps):
        lr = lr_at(step)
        if a.qat_bits > 0:
            _want = a.qat_bits if step >= int(total_steps * a.qat_start_frac) else 0
            if _want != _qat_now:
                for _b in model.blocks:
                    _b.mix.qbits = _want
                a.sh_qbits = _want                 # saved in the checkpoint cfg -> batteries score what trained
                _qat_now = _want
                print(f"QAT step {step}: shared-memory storage {'%d-bit' % _want if _want else 'full precision'}", flush=True)''')
ast.parse(s); io.open(p, "w", encoding="utf-8").write(s); print("trainer: QAT schedule + encoder-memory args")
