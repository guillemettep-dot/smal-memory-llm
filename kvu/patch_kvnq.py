"""patch gqa25 (a copy of gqa23) with --kv_nq Q: Q queries per rolling-window K/V head (GQA inside the 16-token window).
Exact-anchor replacements, each must match exactly once.  argv: codedir"""
import sys, os
D = sys.argv[1]
assert D.rstrip("/").endswith("gqa25"), D
def patch(fn, pairs):
    p = os.path.join(D, fn); s = open(p).read()
    for old, new in pairs:
        c = s.count(old); assert c == 1, (fn, c, old[:100])
        s = s.replace(old, new)
    open(p, "w").write(s); print("patched", fn, len(pairs))

M = "memlm_sh.py"
patch(M, [
 # defaults table + Cfg signature
 ("sh_wq_detach=0, sh_occbin=0, sh_frtok=0.0, sh_frmax=2.0, sh_wqk=0, sh_rsharp=0.0)\n",
  "sh_wq_detach=0, sh_occbin=0, sh_frtok=0.0, sh_frmax=2.0, sh_wqk=0, sh_rsharp=0.0, kv_nq=1)\n"),
 ("sh_wq_detach=0, sh_occbin=0, sh_frtok=0.0, sh_frmax=2.0, sh_wqk=0, sh_rsharp=0.0, **kw):",
  "sh_wq_detach=0, sh_occbin=0, sh_frtok=0.0, sh_frmax=2.0, sh_wqk=0, sh_rsharp=0.0, kv_nq=1, **kw):"),
 ("        assert self.sh_rsharp >= 0, 'sh_rsharp must be >= 0'\n",
  "        assert self.sh_rsharp >= 0, 'sh_rsharp must be >= 0'\n"
  "        # kv_nq Q > 1 (sh25, asked 2026-10-28): every rolling-window K/V head gets Q queries instead of 1 (grouped-query\n"
  "        # attention inside the 16-token window), so a layer can pull up to Q x kv_heads distinct recent tokens.  The\n"
  "        # original query of each K/V head is untouched; the Q-1 extra ones live in MemLM.win_q2 / win_o2 (registered\n"
  "        # LAST), and win_o2 starts at ZERO, so the model is the kv_nq=1 function exactly at the switch.  1 = off.\n"
  "        self.kv_nq = int(kv_nq)\n"
  "        assert self.kv_nq >= 1, 'kv_nq must be >= 1'\n"),
 ('f"= {self.shared_state} | window {self.kv_window}x{self.kv_heads or self.heads}x{self.kv_dim}")',
  'f"= {self.shared_state} | window {self.kv_window}x{self.kv_heads or self.heads}x{self.kv_dim}"\n'
  '                f"{(\' x%d queries\' % self.kv_nq) if self.kv_nq > 1 else \'\'}")'),
 # mixer: holder for the extra-query Linears
 ("        self._rsq = []         # sh_rsharp: the read-sharpness Linear lives in MemLM.read_s (registered LAST)\n",
  "        self._rsq = []         # sh_rsharp: the read-sharpness Linear lives in MemLM.read_s (registered LAST)\n"
  "        self._wq2 = []         # kv_nq: [extra-query Linear, its output Linear], in MemLM.win_q2 / win_o2 (after read_s)\n"
  "        self.kvnq = int(getattr(c, 'kv_nq', 1))\n"),
 # window forward: the extra queries share the K/V heads; head j of the extras uses K/V head j // (Q-1)
 ("            kvr = self._window(cq, ck, cv, T).transpose(1, 2)\n"
  "            delta = self.wo_kv(kvr.reshape(B, T, KVH * kvd))\n",
  "            kvr = self._window(cq, ck, cv, T).transpose(1, 2)\n"
  "            delta = self.wo_kv(kvr.reshape(B, T, KVH * kvd))\n"
  "            if self._wq2:\n"
  "                # kv_nq: Q-1 more queries per K/V head, a SEPARATE window call so the original heads' maths is unchanged\n"
  "                _q2l, _o2l = self._wq2\n"
  "                _r = self.kvnq - 1\n"
  "                cq2 = _rp(_q2l(x).reshape(B, T, KVH * _r, kvd)).transpose(1, 2)\n"
  "                kvr2 = self._window(cq2, ck.repeat_interleave(_r, dim=1), cv.repeat_interleave(_r, dim=1), T).transpose(1, 2)\n"
  "                delta = delta + _o2l(kvr2.reshape(B, T, KVH * _r * kvd))\n"),
 # MemLM: register after read_s (LAST), so the new weights are the last ids of the decay group
 ("                    self.read_s[str(i)] = lin\n"
  "                    mx._rsq = [lin]\n",
  "                    self.read_s[str(i)] = lin\n"
  "                    mx._rsq = [lin]\n"
  "        # kv_nq: registered LAST (after read_s): all q2 weights, then all o2 weights -> the last 2-D ids of the decay group,\n"
  "        # so a resumed optimizer state can be re-indexed exactly (convert_kvnq.py).  o2 ZERO -> the kv_nq=1 function.\n"
  "        if int(getattr(c, 'kv_nq', 1)) > 1:\n"
  "            self.win_q2 = nn.ModuleDict(); self.win_o2 = nn.ModuleDict()\n"
  "            _wl = [(i, b.mix) for i, b in enumerate(self.blocks) if b.mix.wo_kv is not None]\n"
  "            for i, mx in _wl:\n"
  "                lin = nn.Linear(c.d_model, mx.KVH * (c.kv_nq - 1) * mx.kvd, bias=False)\n"
  "                nn.init.normal_(lin.weight, std=0.02)\n"
  "                self.win_q2[str(i)] = lin\n"
  "            for i, mx in _wl:\n"
  "                lin = nn.Linear(mx.KVH * (c.kv_nq - 1) * mx.kvd, c.d_model, bias=False)\n"
  "                nn.init.zeros_(lin.weight)\n"
  "                self.win_o2[str(i)] = lin\n"
  "                mx._wq2 = [self.win_q2[str(i)], lin]\n"),
])

T = "train_sh.py"
patch(T, [
 ('    ap.add_argument("--sh_rsharp", type=float, default=0.0)',
  '    ap.add_argument("--kv_nq", type=int, default=1)   # >1: queries per rolling-window K/V head (sh25)\n'
  '    ap.add_argument("--sh_rsharp", type=float, default=0.0)'),
 ("sh_frtok=a.sh_frtok, sh_frmax=a.sh_frmax, sh_wqk=a.sh_wqk, sh_rsharp=a.sh_rsharp)",
  "sh_frtok=a.sh_frtok, sh_frmax=a.sh_frmax, sh_wqk=a.sh_wqk, sh_rsharp=a.sh_rsharp, kv_nq=a.kv_nq)"),
 ("            lsum = float(_win_l) / _win_n\n",
  "            if getattr(a, 'kv_nq', 1) > 1 and hasattr(model, 'win_o2'):\n"
  "                # kv_nq: RMS of the extra queries' output weights (0 at the switch), layers 0 / 6 / 11\n"
  "                _ko = list(model.win_o2.items())\n"
  "                ustr += ''.join(f' o2_{_k} {float(_l.weight.float().pow(2).mean().sqrt()):.2e}' for _k, _l in (_ko[0], _ko[len(_ko) // 2], _ko[-1]))\n"
  "            lsum = float(_win_l) / _win_n\n"),
])
print("ok")
