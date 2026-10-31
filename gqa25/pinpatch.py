import io, ast
p = "/home/paul/memprobe/gqa/slotkern_kv.py"; s = io.open(p, encoding="utf-8").read()
def rep(a, b):
    global s
    assert s.count(a) == 1, a[:60]; s = s.replace(a, b)
rep("def ref_scan_kv(k, v, gate, fr, N, temp, G=1, hard=False, hard_gate=False, eps=1e-4, return_w=False, detach_addr=False, q=None):",
    "def ref_scan_kv(k, v, gate, fr, N, temp, G=1, hard=False, hard_gate=False, eps=1e-4, return_w=False, detach_addr=False, q=None, pin=True):")
rep("""        return _ref_scan_kv_impl(k.float(), v.float(), gate.float(), fr, N, temp, G, hard, hard_gate, eps, return_w, detach_addr,
                                 None if q is None else q.float())""",
    """        return _ref_scan_kv_impl(k.float(), v.float(), gate.float(), fr, N, temp, G, hard, hard_gate, eps, return_w, detach_addr,
                                 None if q is None else q.float(), pin=pin)""")
rep("def _ref_scan_kv_impl(k, v, gate, fr, N, temp, G=1, hard=False, hard_gate=False, eps=1e-4, return_w=False, detach_addr=False, q=None):",
    "def _ref_scan_kv_impl(k, v, gate, fr, N, temp, G=1, hard=False, hard_gate=False, eps=1e-4, return_w=False, detach_addr=False, q=None, pin=True):")
rep("""        s = torch.where(_Ka.norm(dim=-1) < 1e-6, torch.zeros_like(s), s)
""", """        # 2026-10-02: THE TRITON KERNEL (slotkern_kvt._kvfwd) HAS NO PIN.  It clamps the norm and keeps
        # the score, so a row holding only softmax-tail writes (|K| < 1e-6 but nonzero) scores up to
        # temp*|K|/1e-6 there and exactly 0 here.  Measured on a model's real inputs at 12 x 192:
        # pinned vs kernel 2.0e-1, pin=False vs kernel 1.9e-6.  Training runs the kernel, so pin=False
        # is the function a GPU-trained model learned.  The default stays True only so batteries
        # already in a series keep computing what they computed.
        if pin:
            s = torch.where(_Ka.norm(dim=-1) < 1e-6, torch.zeros_like(s), s)
""")
ast.parse(s); io.open(p, "w", encoding="utf-8").write(s); print("patched")
