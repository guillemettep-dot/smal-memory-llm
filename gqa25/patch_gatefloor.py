import io, ast, shutil
p = "/home/paul/memprobe/gqa/memlm_sh.py"
shutil.copy(p, p + ".bak_pregatefloor")
s = io.open(p, encoding="utf-8").read()
def rep(a, b):
    global s
    assert s.count(a) == 1, (s.count(a), a[:70]); s = s.replace(a, b)
rep("sh_qbits=0, sh_qgroup=16, sh_enc_write=-1, sh_enc_slots=96)", "sh_qbits=0, sh_qgroup=16, sh_enc_write=-1, sh_enc_slots=96,\n                    sh_gate_floor=0.0)")
rep("sh_qbits=0, sh_qgroup=16, sh_enc_write=-1, sh_enc_slots=96, **kw):", "sh_qbits=0, sh_qgroup=16, sh_enc_write=-1, sh_enc_slots=96,\n                 sh_gate_floor=0.0, **kw):")
rep('''        assert self.sh_enc_write < self.sh_write, "the encoder memory must be written below the main writer"
''', '''        assert self.sh_enc_write < self.sh_write, "the encoder memory must be written below the main writer"
        # GATE FLOOR: write-gate values below this are zeroed in the FORWARD (the token writes nothing) while the
        # BACKWARD passes the gradient straight through, so a closed token can still learn to open.  With top-1
        # writes (slot_hard) a weak write lands entirely on one slot and overwrites part of a stored fact;
        # measured on sh6 @7000, 82% of tokens wrote more than 0.01.  0 = off.
        self.sh_gate_floor = float(sh_gate_floor)
''')
rep('''        self.qbits, self.qgroup = c.sh_qbits, c.sh_qgroup
''', '''        self.qbits, self.qgroup = c.sh_qbits, c.sh_qgroup
        self.gfloor = float(getattr(c, "sh_gate_floor", 0.0))
''')
rep('''            self.wrate = (g > 0.5).float().mean().detach()
''', '''            self.wrate = (g > 0.5).float().mean().detach()
            if self.gfloor > 0:
                # forward: exactly 0 below the floor; backward: identity (straight-through)
                g = g + (g * (g >= self.gfloor).to(g.dtype) - g).detach()
''')
ast.parse(s); io.open(p, "w", encoding="utf-8").write(s); print("memlm_sh: gate floor added")
p2 = "/home/paul/memprobe/gqa/train_sh.py"; t = io.open(p2, encoding="utf-8").read()
a = '''    ap.add_argument("--sh_enc_slots", type=int, default=96)
'''
b = a + '''    ap.add_argument("--sh_gate_floor", type=float, default=0.0)   # write-gate values below this write nothing (straight-through)
'''
c = "sh_enc_write=a.sh_enc_write, sh_enc_slots=a.sh_enc_slots)"
d = "sh_enc_write=a.sh_enc_write, sh_enc_slots=a.sh_enc_slots,\n              sh_gate_floor=a.sh_gate_floor)"
assert t.count(a) == 1 and t.count(c) == 1
t = t.replace(a, b).replace(c, d); ast.parse(t); io.open(p2, "w", encoding="utf-8").write(t); print("train_sh: --sh_gate_floor added")
