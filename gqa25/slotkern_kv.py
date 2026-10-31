"""KEY-MATCHED SLOT SCAN (torch reference).

Difference from slotkern_v.ref_scan_v, which this is derived from and must not replace:
  there, the write address is `wa[:, t]` -- a learned projection to N logits, computed OUTSIDE the
  scan and structurally unrelated to what is stored. Here the address is a COSINE MATCH of the new
  key against the keys already in the buffer, so the same key space serves the write and the read,
  and re-writing the same key lands on the row that already holds it (update in place).

Kept verbatim from the original because it already does the right thing:
  o                 per-row occupancy, accumulated as o <- (1-w)o + w
  z = s - fr*o      a LEARNED preference (fr) for rows that are still free

Added:
  K buffer          evolves beside V; the address depends on it, so it must be computed step by step
  EMPTIES-AS-ONE    every empty row scores identically (cos against a zero key is 0 for all of them),
                    so a SOFT write would split one value across the whole free pool and leave every
                    empty row holding the same direction -- the smear, in its worst form. Only the
                    FIRST free row is offered as a candidate; the rest are masked out. Choosing among
                    identical rows is a pure symmetry, so breaking it arbitrarily costs no gradient.
"""
import torch

try:                                   # import only -- a failure here is a MISSING kernel, not a bug
    from slotkern_kvt import kv_scan as _kv_triton
except Exception:                      # CPU-only boxes (the probes) never have it
    _kv_triton = None


def scan_kv(k, v, gate, fr, N, temp, G=1, hard=False, hard_gate=False, eps=1e-4, detach_addr=False, q=None):
    """Triton on CUDA at G=1, the torch reference otherwise.

    The kernel is validated against ref_scan_kv below: forward to within the scan's own chaotic
    round-off amplification (measured -- the reference diverges from itself by the same amount when
    its inputs are perturbed by 1e-7), and backward to 1e-6 on all five gradients in all four
    hard/soft combinations.  Measured 113-197x faster fwd+bwd at the model's shapes.

    NOTE the errors are NOT swallowed: only the IMPORT is guarded.  A raising kernel must raise,
    rather than silently drop back to a path 100x slower."""
    if _kv_triton is not None and k.is_cuda and G == 1:
        return _kv_triton(k, v, gate, fr, N, temp, hard=hard, hard_gate=hard_gate, eps=eps, detach_addr=detach_addr, q=q)
    return ref_scan_kv(k, v, gate, fr, N, temp, G=G, hard=hard, hard_gate=hard_gate, eps=eps, detach_addr=detach_addr, q=q)

NEG = -1e30          # not -inf: an all-masked row set would make softmax produce NaN


def _first_empty_mask(o, eps):
    """True on occupied rows AND on exactly one free row (the lowest-index one)."""
    empty = o < eps                                  # (B,H,N)
    idx = empty.to(torch.int32).cumsum(-1)           # 1 at the first empty, >1 after
    first = empty & (idx == 1)
    return (~empty) | first


def ref_scan_kv(k, v, gate, fr, N, temp, G=1, hard=False, hard_gate=False, eps=1e-4, return_w=False, detach_addr=False, q=None, pin=True, occ_bin=False):
    """AUTOCAST IS DISABLED FOR THE WHOLE SCAN, deliberately.  torch.einsum is autocast-aware even
    when its inputs are already fp32, so under `--amp fp16` the cosine match below would silently
    run in half precision -- the same class of bug that cost this project three separate failures
    (MoE index_add_, _ChunkedCE forward, _ChunkedCE backward).  The match feeds a softmax at a
    sharpness of ~16, where losing mantissa bits changes which row wins."""
    dev = "cuda" if k.is_cuda else "cpu"
    with torch.autocast(device_type=dev, enabled=False):
        return _ref_scan_kv_impl(k.float(), v.float(), gate.float(), fr, N, temp, G, hard, hard_gate, eps, return_w, detach_addr,
                                 None if q is None else q.float(), pin=pin, occ_bin=occ_bin)


def _ref_scan_kv_impl(k, v, gate, fr, N, temp, G=1, hard=False, hard_gate=False, eps=1e-4, return_w=False, detach_addr=False, q=None, pin=True, occ_bin=False):
    """k:(B,T,H,DK) v:(B,T,H,DV) gate:(B,T,H) fr:(1,) temp: scalar tensor (address sharpness).
    Returns (Kbuf, Vbuf), each (B, ceil(T/G), H, N, D*), PRE-write, matching ref_scan_v's convention."""
    B, T, H, DK = k.shape
    DV = v.shape[-1]
    K = k.new_zeros(B, H, N, DK)
    V = v.new_zeros(B, H, N, DV)
    o = k.new_zeros(B, H, N)
    kb, vb, wb = [], [], []
    for t in range(T):
        if t % G == 0:
            kb.append(K); vb.append(V)
        kt = k[:, t]                                                     # (B,H,DK) -- STORED
        qt = kt if q is None else q[:, t]                                # -- used to SEARCH
        # detach_addr: the write address reads the buffer as a CONSTANT. The forward is bit-identical;
        # it only removes the gradient path 'adjust the keys I stored earlier so this write lands
        # elsewhere' -- the self-referential loop (write -> keys -> next address -> next write).
        # Keys stay learnable through the read matching against them, which is the path that matters.
        _Ka = K.detach() if detach_addr else K
        kn = _Ka / _Ka.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        qn = qt / qt.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        s = torch.einsum("bhnd,bhd->bhn", kn, qn) * temp                 # cosine match, sharpened
        # ZERONAN 2026-09-15: a row that was never written is EXACTLY zero, so its norm hits the
        # 1e-6 clamp and d(K/clamp)/dK becomes 1e6. Under hard writes that compounds to 1e22 by
        # T=8 and overflows to NaN by T=16 -- measured, and the torch reference and the Triton
        # kernel do it identically. Its cosine is 0 either way, so pin the score to a constant and
        # cut the gradient path into the zero row: same forward, finite backward.
        # (Same shape as the s4 NANFIX: the forward was fine and the BACKWARD was the problem.)
        # 2026-10-02: THE TRITON KERNEL (slotkern_kvt._kvfwd) HAS NO PIN.  It clamps the norm and keeps
        # the score, so a row holding only softmax-tail writes (|K| < 1e-6 but nonzero) scores up to
        # temp*|K|/1e-6 there and exactly 0 here.  Measured on a model's real inputs at 12 x 192:
        # pinned vs kernel 2.0e-1, pin=False vs kernel 1.9e-6.  Training runs the kernel, so pin=False
        # is the function a GPU-trained model learned.  The default stays True only so batteries
        # already in a series keep computing what they computed.
        if pin:
            s = torch.where(_Ka.norm(dim=-1) < 1e-6, torch.zeros_like(s), s)
        # occ_bin: the penalty counts a row as taken once written ([o >= eps], the same test as the empty mask),
        # so a faint soft-gate write (o ~ g) still costs the full fr.  A step function: no gradient into o.
        _fr = fr[:, t].unsqueeze(-1) if (torch.is_tensor(fr) and fr.dim() == 3) else fr   # sh_frtok: (B,T,H) per token
        z = s - _fr * ((o >= eps).to(o.dtype) if occ_bin else o)         # free rows preferred
        z = z.masked_fill(~_first_empty_mask(o, eps), NEG)               # empties collapse to one
        a = torch.softmax(z, -1)
        if hard:
            ah = torch.nn.functional.one_hot(z.argmax(-1), N).to(v.dtype)
            a = ah + a - a.detach()                                      # straight-through
        g = gate[:, t]
        if hard_gate:
            g = (g > 0.5).to(v.dtype) + g - g.detach()
        w = g.unsqueeze(-1) * a
        if return_w:
            wb.append(a)                                                 # the write DISTRIBUTION, pre-gate
        ws = w.unsqueeze(-1)
        K = (1 - ws) * K + ws * kt.unsqueeze(2)
        V = (1 - ws) * V + ws * v[:, t].unsqueeze(2)
        o = (1 - w) * o + w
    if return_w:
        return torch.stack(kb, 1), torch.stack(vb, 1), torch.stack(wb, 1)
    return torch.stack(kb, 1), torch.stack(vb, 1)
