"""sh20 patch (asked 2026-10-20): PER-TOKEN free-row preference, learned (--sh_frtok c0 > 0).
Default (sh_frtok 0) must stay bit-identical: every change is behind FRT / frtok.
Run from inside the code dir.  Exact-anchor replacements, each asserted to match the expected count."""
import sys

def rep(path, old, new, count=1):
    s = open(path).read()
    n = s.count(old)
    assert n == count, f"{path}: anchor found {n}x, expected {count}:\n{old}"
    s = s.replace(old, new)
    open(path, "w").write(s)

# ---------------- forward kernel (slotkern_kvt._kvfwd) ----------------
P = "slotkern_kvt.py"
rep(P, """           HARD: tl.constexpr, HARDG: tl.constexpr, OCCB: tl.constexpr = False):
    pid = tl.program_id(0)""",
"""           HARD: tl.constexpr, HARDG: tl.constexpr, OCCB: tl.constexpr = False, FRT: tl.constexpr = False):
    # FRT (sh_frtok): fr_ptr holds one free-row preference PER TOKEN, laid out (B, T, H) like the gate.
    pid = tl.program_id(0)""")
rep(P, """        g = tl.load(gbase + t * H)
        if HARDG:
            g = (g > 0.5).to(tl.float32)""",
"""        g = tl.load(gbase + t * H)
        if FRT:
            fr = tl.load(fr_ptr + b * T * H + t * H + h)
        if HARDG:
            g = (g > 0.5).to(tl.float32)""")

# ---------------- backward kernel + autograd (slotkern_kvg) ----------------
P = "slotkern_kvg.py"
rep(P, """             HARD: tl.constexpr, HARDG: tl.constexpr, DETACH: tl.constexpr, OCCB: tl.constexpr = False):
    pid = tl.program_id(0)""",
"""             HARD: tl.constexpr, HARDG: tl.constexpr, DETACH: tl.constexpr, OCCB: tl.constexpr = False,
             FRT: tl.constexpr = False):
    # FRT (sh_frtok): fr is PER TOKEN (B, T, H) and so is its gradient, stored per step instead of summed.
    pid = tl.program_id(0)""")
# both the replay and the backward walk load the gate this way: load this token's fr right after
rep(P, """            graw = tl.load(gbase + t * H)
""",
"""            graw = tl.load(gbase + t * H)
            if FRT:
                fr = tl.load(fr_ptr + b * T * H + t * H + h)
""", count=2)
rep(P, """            dfr += -tl.sum(occ * dz, axis=0)
""",
"""            if FRT:
                tl.store(dfr_ptr + b * T * H + t * H + h, -tl.sum(occ * dz, axis=0))
            else:
                dfr += -tl.sum(occ * dz, axis=0)
""")
rep(P, """    tl.atomic_add(dfr_ptr, dfr)
    tl.atomic_add(dtemp_ptr, dtemp)""",
"""    if not FRT:
        tl.atomic_add(dfr_ptr, dfr)
    tl.atomic_add(dtemp_ptr, dtemp)""")
rep(P, """        k, q, v, gate = f(k), f(q), f(v), f(gate)
        fr = fr.to(torch.float32).reshape(1)""",
"""        k, q, v, gate = f(k), f(q), f(v), f(gate)
        frt = fr.dim() == 3                        # sh_frtok: one fr per token, shaped like the gate
        if frt:
            assert fr.shape == gate.shape, (fr.shape, gate.shape)
            fr = f(fr)
        else:
            fr = fr.to(torch.float32).reshape(1)""")
rep(P, """                         HARD=bool(hard), HARDG=bool(hard_gate), OCCB=bool(occ_bin), num_warps=nw)
        ctx.save_for_backward(k, q, v, gate, fr, temp, kbuf, vbuf, obuf)
        ctx.meta = (N, int(G), bool(hard), bool(hard_gate), float(eps), bool(detach_addr), int(nw), int(nwb), bool(occ_bin))""",
"""                         HARD=bool(hard), HARDG=bool(hard_gate), OCCB=bool(occ_bin), FRT=frt, num_warps=nw)
        ctx.save_for_backward(k, q, v, gate, fr, temp, kbuf, vbuf, obuf)
        ctx.meta = (N, int(G), bool(hard), bool(hard_gate), float(eps), bool(detach_addr), int(nw), int(nwb), bool(occ_bin), frt)""")
rep(P, """        N, G, hard, hard_gate, eps, detach_addr, nw, nwb, occ_bin = ctx.meta""",
"""        N, G, hard, hard_gate, eps, detach_addr, nw, nwb, occ_bin, frt = ctx.meta""")
rep(P, """        dfr = torch.zeros(1, device=dev, dtype=torch.float32)
        dtemp = torch.zeros(1, device=dev, dtype=torch.float32)
        dkbuf""",
"""        dfr = torch.zeros_like(fr) if frt else torch.zeros(1, device=dev, dtype=torch.float32)
        dtemp = torch.zeros(1, device=dev, dtype=torch.float32)
        dkbuf""")
rep(P, """                           HARD=hard, HARDG=hard_gate, DETACH=detach_addr, OCCB=occ_bin, num_warps=nwb)""",
"""                           HARD=hard, HARDG=hard_gate, DETACH=detach_addr, OCCB=occ_bin, FRT=frt, num_warps=nwb)""")

# ---------------- torch reference (slotkern_kv) ----------------
P = "slotkern_kv.py"
rep(P, """        z = s - fr * ((o >= eps).to(o.dtype) if occ_bin else o)          # free rows preferred""",
"""        _fr = fr[:, t].unsqueeze(-1) if (torch.is_tensor(fr) and fr.dim() == 3) else fr   # sh_frtok: (B,T,H) per token
        z = s - _fr * ((o >= eps).to(o.dtype) if occ_bin else o)         # free rows preferred""")

# ---------------- model (memlm_sh) ----------------
P = "memlm_sh.py"
rep(P, """                    sh_wq_detach=0, sh_occbin=0)
""", """                    sh_wq_detach=0, sh_occbin=0, sh_frtok=0.0, sh_frmax=2.0)
""")
rep(P, """                 sh_wq_detach=0, sh_occbin=0, **kw):""",
"""                 sh_wq_detach=0, sh_occbin=0, sh_frtok=0.0, sh_frmax=2.0, **kw):""")
rep(P, """        self.sh_wq_detach, self.sh_occbin = int(sh_wq_detach), int(sh_occbin)
""", """        self.sh_wq_detach, self.sh_occbin = int(sh_wq_detach), int(sh_occbin)
        # sh_frtok c0 > 0 (sh20, asked 2026-10-20): the FREE-ROW PREFERENCE IS DECIDED PER TOKEN.  Each writer head
        # computes fr_t = temp * frmax * sigmoid(w . x_t.detach() + b), w zero-initialised and b set so the ratio
        # fr/temp starts at c0 for every token.  A taken row then beats the first empty row iff its address cosine
        # exceeds fr_t/temp, so a token can MERGE into a matching row while free rows remain (sh19c: fr 32 > temp,
        # so no token could merge until all rows were full).  c0 must sit inside the range of cosines the address
        # actually produces, or the straight-through gradient on fr is ~exp(-temp*(c0-cos)) and it never moves.
        # 0 = off: the scalar fr path, bit-identical.
        self.sh_frtok, self.sh_frmax = float(sh_frtok), float(sh_frmax)
        assert self.sh_frtok == 0 or 0 < self.sh_frtok < self.sh_frmax, "need 0 < sh_frtok < sh_frmax"
""")
rep(P, """            self.log_free = nn.Parameter(torch.tensor(_lf))                # free-row preference, as memlm_kv
""", """            self.log_free = nn.Parameter(torch.tensor(_lf))                # free-row preference, as memlm_kv
            self.frtok = float(getattr(c, "sh_frtok", 0.0))
            if self.frtok > 0:
                self.frmax = float(getattr(c, "sh_frmax", 2.0))
                self.fr_proj = nn.Linear(D, self.Hs, bias=False)            # ZEROED by MemLM after its global init
                self.fr_b = nn.Parameter(torch.full((self.Hs,), math.log(self.frtok / (self.frmax - self.frtok))))
            self.fr_c = None                                                # mean fr/temp of the last forward (frtok)
""")
rep(P, """            temp = torch.exp(self.log_wsharp.float())
""", """            temp = torch.exp(self.log_wsharp.float())
            if self.frtok > 0:
                _c = self.frmax * torch.sigmoid(self.fr_proj(x.detach()).float() + self.fr_b.float())   # (B,T,Hs)
                fr = _c * temp.detach()                                       # threshold ratio is the only thing learned
                self.fr_c = _c.detach().mean()
""")
rep(P, """            for _n in ("pos_k", "pos_q", "pos_wq"):""",
"""            for _n in ("pos_k", "pos_q", "pos_wq", "fr_proj"):""")

# ---------------- trainer (train_sh) ----------------
P = "train_sh.py"
rep(P, """    ap.add_argument("--sh_occbin", type=int, default=0)""",
"""    ap.add_argument("--sh_frtok", type=float, default=0.0)   # >0: per-token learned free-row preference, fr/temp starts at this (sh20)
    ap.add_argument("--sh_frmax", type=float, default=2.0)   # upper bound of fr/temp under sh_frtok
    ap.add_argument("--sh_occbin", type=int, default=0)""")
rep(P, """              sh_wq_detach=a.sh_wq_detach, sh_occbin=a.sh_occbin)""",
"""              sh_wq_detach=a.sh_wq_detach, sh_occbin=a.sh_occbin, sh_frtok=a.sh_frtok, sh_frmax=a.sh_frmax)""")
print("patched OK")
