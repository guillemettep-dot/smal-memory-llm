import sys
sys.path.insert(0, "/home/paul/memprobe/gqa")
import torch
from slotkern_kv import _ref_scan_kv_impl, _first_empty_mask
import inspect
print(inspect.getsource(_first_empty_mask))
d = torch.load("/home/paul/memprobe/gqa/_scan_inputs.pt")
k, v, g, fr, tp = (d[n].double()[:1] if d[n].dim() > 1 else d[n].double() for n in ("k", "v", "g", "fr", "tp"))
N = 192
def rel(a, b): return ((a - b).abs().max() / b.abs().max().clamp_min(1e-30)).item()
with torch.no_grad():
    rk, rv = _ref_scan_kv_impl(k, v, g, fr, N, tp, 1, False, False, 1e-4, False, True)
    torch.manual_seed(0)
    for eps_pert in (1e-9, 1e-7, 1e-6):
        k2 = k * (1 + eps_pert * torch.randn_like(k))
        pk, pv = _ref_scan_kv_impl(k2, v, g, fr, N, tp, 1, False, False, 1e-4, False, True)
        e = (pv - rv).abs().amax(dim=(0, 2, 3, 4)); bad = (e > 1e-3).nonzero()
        print("fp64 reference, keys perturbed by %.0e relative: vbuf rel diff %.1e, first t with err>1e-3: %s" % (eps_pert, rel(pv, rv), bad[0].item() if len(bad) else None))
    # trace occupancy near the eps threshold, per head, through time
    B, T, H, DK = k.shape
    K = k.new_zeros(B, H, N, DK); o = k.new_zeros(B, H, N)
    near = []
    for t in range(T):
        Ka = K
        kn = Ka / Ka.norm(dim=-1, keepdim=True).clamp_min(1e-6); qn = k[:, t] / k[:, t].norm(dim=-1, keepdim=True).clamp_min(1e-6)
        s = torch.einsum("bhnd,bhd->bhn", kn, qn) * tp
        s = torch.where(Ka.norm(dim=-1) < 1e-6, torch.zeros_like(s), s)
        z = (s - fr * o).masked_fill(~_first_empty_mask(o, 1e-4), -1e30)
        a = torch.softmax(z, -1); w = g[:, t].unsqueeze(-1) * a
        K = (1 - w.unsqueeze(-1)) * K + w.unsqueeze(-1) * k[:, t].unsqueeze(2); o = (1 - w) * o + w
        r = (o / 1e-4).log10().abs()                     # 0 == exactly at the threshold
        near.append(r.min().item())
        if t in (40, 84, 85, 86, 150, 299):
            occ = (o >= 1e-4).sum(-1)
            print("t=%3d occupied rows per head %s | rows with o in (1e-7,1e-4): %d | closest log10 distance to eps %.2e"
                  % (t, occ[0].tolist(), ((o > 1e-7) & (o < 1e-4)).sum().item(), r.min().item()))
