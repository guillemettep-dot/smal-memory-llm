import io, ast
p = "/home/paul/memprobe/gqa/train_sh.py"; s = io.open(p, encoding="utf-8").read()
def rep(a, b):
    global s
    assert s.count(a) == 1, (s.count(a), a[:60]); s = s.replace(a, b)
rep('''    ap.add_argument("--ckpt_every", type=int, default=1000)''',
    '''    ap.add_argument("--ckpt_every", type=int, default=1000)
    # BANKED copies: model + cfg + step only (what a battery needs, ~half a full checkpoint), written by
    # the trainer itself at every multiple of bank_every.  The old external collector had to copy the
    # rolling file before the next save overwrote it, and lost a checkpoint when a forced save landed
    # between its polls.
    ap.add_argument("--bank_every", type=int, default=0)
    ap.add_argument("--bank_dir", type=str, default="")''')
rep('''                    torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                                "step": step, "cfg": vars(a), "rng": _rng_state()}, ck + ".tmp")
                    os.replace(ck + ".tmp", ck)
''', '''                    torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                                "step": step, "cfg": vars(a), "rng": _rng_state()}, ck + ".tmp")
                    os.replace(ck + ".tmp", ck)
                    if a.bank_every > 0 and a.bank_dir and step % a.bank_every == 0:
                        os.makedirs(a.bank_dir, exist_ok=True)
                        _bk = os.path.join(a.bank_dir, f"{a.tag}_step{step}.pt")
                        torch.save({"model": model.state_dict(), "step": step, "cfg": vars(a)}, _bk + ".tmp")
                        os.replace(_bk + ".tmp", _bk)
''')
ast.parse(s); io.open(p, "w", encoding="utf-8").write(s); print("bank patch applied")
