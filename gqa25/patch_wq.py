import io, ast, shutil
p = "/home/paul/memprobe/gqa/memlm_sh.py"
shutil.copy(p, p + ".bak_prewq")
s = io.open(p, encoding="utf-8").read()
def rep(a, b):
    global s
    assert s.count(a) == 1, (s.count(a), a[:70]); s = s.replace(a, b)
rep("                    sh_gate_floor=0.0)", "                    sh_gate_floor=0.0, sh_wq=0)")
rep("                 sh_gate_floor=0.0, **kw):", "                 sh_gate_floor=0.0, sh_wq=0, **kw):")
rep('''        self.sh_gate_floor = float(sh_gate_floor)
''', '''        self.sh_gate_floor = float(sh_gate_floor)
        # WRITE QUERY (two keys at write time): sh_wq=1 gives each writer its OWN projection used only to
        # CHOOSE the slot (cosine against the stored keys); the stored key stays the key projection.  With one
        # vector doing both jobs, "where should this go" and "what will match this later" fight each other --
        # measured on sh7 @8000 (top-1): 13.5 of 192 slots used, 96.6% of writes overwrite, median life 6 tokens.
        # The write query is initialised as an exact COPY of the key projection, so switching it on does not
        # change the function; it then learns its own job.  Registered LAST in the model (see MemLM) so a resumed
        # optimizer state can be remapped exactly.
        self.sh_wq = int(sh_wq)
''')
rep('''            if kv_scan_g is not None and kk.is_cuda:
                kbuf, buf = kv_scan_g(kk.float(), v.float(), g, fr, self.N, temp, G=self.G,
                                      hard=self.hard, hard_gate=self.hard_gate, detach_addr=bool(self.kdetach))''',
'''            qq = self._wq[0](x).reshape(B, T, self.Hs, self.dk).float() if self._wq else None   # the address-only key
            if kv_scan_g is not None and kk.is_cuda:
                kbuf, buf = kv_scan_g(kk.float(), v.float(), g, fr, self.N, temp, G=self.G,
                                      hard=self.hard, hard_gate=self.hard_gate, detach_addr=bool(self.kdetach), q=qq)''')
rep('''                kbuf, buf = ref_scan_kv(kk.float(), v.float(), g, fr, self.N, temp, G=self.G,
                                        hard=self.hard, hard_gate=self.hard_gate, detach_addr=bool(self.kdetach),
                                        pin=False)''', '''                kbuf, buf = ref_scan_kv(kk.float(), v.float(), g, fr, self.N, temp, G=self.G,
                                        hard=self.hard, hard_gate=self.hard_gate, detach_addr=bool(self.kdetach),
                                        pin=False, q=qq)''')
rep('''        self.krep_loss = self.wbud_loss = self.osel_loss = self.xs_loss = None
''', '''        self.krep_loss = self.wbud_loss = self.osel_loss = self.xs_loss = None
        self._wq = []          # the write-query Linear lives in MemLM.write_q (registered last); held here unregistered
''')
rep('''        self.apply(self._init)
''', '''        self.apply(self._init)
        # write queries: registered AFTER everything else so they come last in model.parameters(); each starts as
        # an exact copy of its writer's key projection (same function at the switch)
        if int(getattr(c, "sh_wq", 0)):
            self.write_q = nn.ModuleDict()
            for i, b in enumerate(self.blocks):
                mx = b.mix
                if mx.is_writer:
                    lin = nn.Linear(c.d_model, mx.Hs * mx.dk, bias=False)
                    k0 = sum(mx.sizes[:4]); k1 = k0 + mx.Hs * mx.dk                # the key rows of the fused w_in
                    with torch.no_grad():
                        lin.weight.copy_(mx.w_in.weight[k0:k1])
                    self.write_q[str(i)] = lin
                    mx._wq = [lin]
''')
ast.parse(s); io.open(p, "w", encoding="utf-8").write(s); print("memlm_sh: write query added")
p2 = "/home/paul/memprobe/gqa/train_sh.py"; t = io.open(p2, encoding="utf-8").read()
a = '''    ap.add_argument("--sh_gate_floor", type=float, default=0.0)'''
assert t.count(a) == 1
t = t.replace(a, a + '''\n    ap.add_argument("--sh_wq", type=int, default=0)   # 1 = a separate write-query key chooses the slot''')
c = "              sh_gate_floor=a.sh_gate_floor)"
assert t.count(c) == 1
t = t.replace(c, "              sh_gate_floor=a.sh_gate_floor, sh_wq=a.sh_wq)")
ast.parse(t); io.open(p2, "w", encoding="utf-8").write(t); print("train_sh: --sh_wq added")
