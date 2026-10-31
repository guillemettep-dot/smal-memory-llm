"""gqa23: per-token, per-head, per-query learned READ SHARPNESS (--sh_rsharp C > 0; 0 = off, bit-identical).
Every reader's score  rsel * cos(q, k)  becomes  rsel * exp(C tanh(s/C)) * cos(q, k),  s = w . x.detach(),  w zero-init
(so the multiplier is exactly 1 at the switch).  argv: none (patches /home/paul/memprobe/gqa23 in place)."""
import sys
D = "/home/paul/memprobe/gqa23/"
def rep(path, old, new):
    s = open(D + path).read()
    n = s.count(old)
    assert n == 1, (path, n, old[:80])
    open(D + path, "w").write(s.replace(old, new))

M = "memlm_sh.py"
rep(M, "sh_wq_detach=0, sh_occbin=0, sh_frtok=0.0, sh_frmax=2.0, sh_wqk=0)\n",
       "sh_wq_detach=0, sh_occbin=0, sh_frtok=0.0, sh_frmax=2.0, sh_wqk=0, sh_rsharp=0.0)\n")
rep(M, "sh_wq_detach=0, sh_occbin=0, sh_frtok=0.0, sh_frmax=2.0, sh_wqk=0, **kw):\n",
       "sh_wq_detach=0, sh_occbin=0, sh_frtok=0.0, sh_frmax=2.0, sh_wqk=0, sh_rsharp=0.0, **kw):\n")
rep(M, "        self.sh_wqk = int(sh_wqk)\n",
       "        self.sh_wqk = int(sh_wqk)\n"
       "        # sh_rsharp C > 0 (sh23, asked 2026-10-25): each reader's score is multiplied by exp(C tanh(s/C)), s = w . x.detach()\n"
       "        # per (token, head, read query), w zero-init -> x1 exactly at the switch.  One scalar rsel per layer forces every\n"
       "        # query to the same sharpness; a dot product would give each query its own (|q|) but drops the normalisation that\n"
       "        # keeps the scores bounded, so the sharpness is learned per query instead, bounded to exp(+-C).  0 = off.\n"
       "        self.sh_rsharp = float(sh_rsharp)\n"
       "        assert self.sh_rsharp >= 0, 'sh_rsharp must be >= 0'\n")
rep(M, "        self._wqk = []         # sh_wqk: the key-share Linear lives in MemLM.write_qk (registered after write_q)\n",
       "        self._wqk = []         # sh_wqk: the key-share Linear lives in MemLM.write_qk (registered after write_q)\n"
       "        self._rsq = []         # sh_rsharp: the read-sharpness Linear lives in MemLM.read_s (registered LAST)\n"
       "        self.rsharp = float(getattr(c, 'sh_rsharp', 0.0))\n"
       "        self.rs_a = None       # sh_rsharp: detached mean and std of the log-multiplier of the last forward (for the log)\n")
rep(M, "        sc = torch.einsum(\"bjhmd,bjhnd->bjhmn\", q, kbuf).float() * (kinv.unsqueeze(-2) * torch.exp(self.log_rsel.float()))\n",
       "        sc = torch.einsum(\"bjhmd,bjhnd->bjhmn\", q, kbuf).float() * (kinv.unsqueeze(-2) * torch.exp(self.log_rsel.float()))\n"
       "        if self._rsq:\n"
       "            # sh_rsharp: per-token log-multiplier, laid out exactly as q (pad T -> Tp, then (B,NB,Hs,G*NQ)); pad = 0 -> x1\n"
       "            _s = self._rsq[0](x.detach()).float()\n"
       "            _s = self.rsharp * torch.tanh(_s / self.rsharp)                     # (B,T,Hs*NQ), |s| < C\n"
       "            self.rs_a = (_s.detach().mean(), _s.detach().std())\n"
       "            _s = _s.reshape(B, T, Hs, NQ)\n"
       "            if Tp != T:\n"
       "                _s = F.pad(_s, (0, 0, 0, 0, 0, Tp - T))\n"
       "            _s = _s.reshape(B, NB, G, Hs, NQ).permute(0, 1, 3, 2, 4).reshape(B, NB, Hs, G * NQ)\n"
       "            sc = sc * torch.exp(_s).unsqueeze(-1)\n")
rep(M, "                    self.write_qk[str(i)] = lin\n                    mx._wqk = [lin]\n",
       "                    self.write_qk[str(i)] = lin\n                    mx._wqk = [lin]\n"
       "        # sh_rsharp: registered LAST (after write_q / write_qk) so the new weights are the LAST ids of the decay group and a\n"
       "        # resumed optimizer state can be re-indexed exactly (convert_rs.py); weight ZERO -> multiplier exp(0) = 1, bitwise\n"
       "        if float(getattr(c, \"sh_rsharp\", 0.0)) > 0:\n"
       "            self.read_s = nn.ModuleDict()\n"
       "            for i, b in enumerate(self.blocks):\n"
       "                mx = b.mix\n"
       "                if mx.is_reader:\n"
       "                    lin = nn.Linear(c.d_model, mx.Hs * mx.NQ, bias=False)\n"
       "                    nn.init.zeros_(lin.weight)\n"
       "                    self.read_s[str(i)] = lin\n"
       "                    mx._rsq = [lin]\n")
T = "train_sh.py"
rep(T, "    ap.add_argument(\"--sh_wqk\", type=int, default=0)",
       "    ap.add_argument(\"--sh_rsharp\", type=float, default=0.0)   # >0: per-token learned read sharpness, bounded exp(+-C) (sh23)\n"
       "    ap.add_argument(\"--sh_wqk\", type=int, default=0)")
rep(T, "sh_frtok=a.sh_frtok, sh_frmax=a.sh_frmax, sh_wqk=a.sh_wqk)\n",
       "sh_frtok=a.sh_frtok, sh_frmax=a.sh_frmax, sh_wqk=a.sh_wqk, sh_rsharp=a.sh_rsharp)\n")
rep(T, "            lsum = float(_win_l) / _win_n\n",
       "            if getattr(a, \"sh_rsharp\", 0):\n"
       "                # the learned read-sharpness log-multiplier, first and last reader of each memory: mean/std (last micro-batch)\n"
       "                _rl = [(_i, _b.mix) for _i, _b in enumerate(model.blocks) if getattr(_b.mix, \"rs_a\", None) is not None]\n"
       "                _rl = [_rl[0], _rl[3], _rl[4], _rl[-1]] if len(_rl) >= 5 else _rl\n"
       "                ustr += \"\".join(f\" rs{_i} {float(_m.rs_a[0]):+.3f}/{float(_m.rs_a[1]):.3f}\" for _i, _m in _rl)\n"
       "            lsum = float(_win_l) / _win_n\n")
print("patched")
