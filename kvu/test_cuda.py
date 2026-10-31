"""kv_scan_c (CUDA) vs kv_scan_g (Triton), hard + detached address, and vs the float64 reference.
Not bitwise (different summation order), so: forward snapshots max |diff|, the share of (snapshot, head, row)
entries that differ by > 1e-3 (a different row choice shows up here), and each gradient's max |diff| relative
to its own max.  The float64 reference arbitrates on the smaller scalar-fr cases."""
import sys, os, torch
sys.path.insert(0, "/home/paul/memprobe/gqa21")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import slotkern_kvg as KG, slotkern_kv as KV, slotkern_cuda as KC
assert "gqa21" in KG.__file__
dev = "cuda"
cases = [  # B, T, H, N, D, G, frt, occb, sq, hardg
    (2, 1024, 12, 192, 64, 8, True, True, True, False),
    (2, 1024, 12, 96, 64, 8, True, True, True, False),
    (2, 1024, 6, 192, 128, 8, True, True, True, False),
    (2, 1024, 6, 96, 128, 8, True, True, True, False),
    (2, 1000, 12, 192, 64, 8, True, True, True, False),
    (1, 300, 4, 40, 64, 16, False, False, False, False),
    (1, 300, 4, 40, 64, 7, False, True, True, True),
    (1, 300, 3, 50, 128, 8, False, False, True, False),
]
def inputs(B, T, H, D, frt, sq, seed):
    gen = torch.Generator(device=dev).manual_seed(seed)
    r = lambda *s: torch.randn(*s, device=dev, generator=gen)
    k = r(B, T, H, D); q = r(B, T, H, D) if sq else None; v = r(B, T, H, D)
    g = torch.rand(B, T, H, device=dev, generator=gen); m = torch.rand(B, T, H, device=dev, generator=gen)
    g = torch.where(m < 0.15, torch.zeros_like(g), torch.where(m > 0.92, torch.ones_like(g), g))
    fr = (torch.rand(B, T, H, device=dev, generator=gen) * 2) if frt else torch.tensor([1.5], device=dev)
    return k, q, v, g, fr
def run(fn, k, q, v, g, fr, N, G, occb, hardg, dtype=torch.float32, wk=None, wv=None):
    cv = lambda x: None if x is None else x.to(dtype)
    lv = [cv(t).clone().requires_grad_() for t in (k, v, g, fr)] + ([cv(q).clone().requires_grad_()] if q is not None else [])
    tp = torch.tensor(16.0, device=dev, dtype=dtype, requires_grad=True)
    if fn is None:
        kb, vb = KV._ref_scan_kv_impl(lv[0], lv[1], lv[2], lv[3].reshape(1), N, tp, G=G, hard=True, hard_gate=hardg,
                                      detach_addr=True, q=lv[4] if q is not None else None, pin=False, occ_bin=occb)
    else:
        kb, vb = fn(lv[0], lv[1], lv[2], lv[3], N, tp, G=G, hard=True, hard_gate=hardg, detach_addr=True,
                    q=lv[4] if q is not None else None, occ_bin=occb)
    if wk is None:
        gen = torch.Generator(device=dev).manual_seed(7)
        wk = torch.randn(kb.shape, device=dev, generator=gen); wv = torch.randn(vb.shape, device=dev, generator=gen)
    (kb * wk.to(dtype) + vb * wv.to(dtype)).sum().backward()
    return kb.detach(), vb.detach(), [t.grad for t in lv] + [tp.grad], wk, wv
def cmp(ga, gb):
    names = ["dk", "dv", "dgate", "dfr"] + (["dq"] if len(ga) == 6 else []) + ["dtemp"]
    assert len(names) == len(ga) == len(gb)
    out = []
    for nm, a, b in zip(names, ga, gb):
        if a is None or b is None: continue
        a, b = a.double(), b.double()
        out.append((nm, ((a - b).abs().max() / a.abs().max().clamp_min(1e-30)).item()))
    return out
allok = True
for ci, (B, T, H, N, D, G, frt, occb, sq, hardg) in enumerate(cases):
    k, q, v, g, fr = inputs(B, T, H, D, frt, sq, 300 + ci)
    kb_g, vb_g, gr_g, wk, wv = run(KG.kv_scan_g, k, q, v, g, fr, N, G, occb, hardg)
    kb_c, vb_c, gr_c, _, _ = run(KC.kv_scan_c, k, q, v, g, fr, N, G, occb, hardg, wk=wk, wv=wv)
    fin = all(torch.isfinite(x).all() for x in [kb_c, vb_c] + [x for x in gr_c if x is not None])
    fd = max((kb_g - kb_c).abs().max().item(), (vb_g - vb_c).abs().max().item())
    rowdiff = (((kb_g - kb_c).abs().amax(-1) > 1e-3) | ((vb_g - vb_c).abs().amax(-1) > 1e-3)).double().mean().item()
    c = cmp(gr_g, gr_c); worst = max(x for _, x in c)
    ok = fin and rowdiff == 0 and fd < 1e-4 and worst < 1e-3
    allok &= ok
    print(f"[{ci}] B{B} T{T} H{H} N{N} D{D} G{G} frt={int(frt)} occb={int(occb)} q={int(sq)} hardg={int(hardg)} | "
          f"fwd max|d| {fd:.1e} rows differing {rowdiff:.4f} | " + "  ".join(f"{n} {x:.1e}" for n, x in c)
          + ("   OK" if ok else "   <-- CHECK"), flush=True)
    if T <= 300:
        kb_r, vb_r, gr_r, _, _ = run(None, k, q, v, g, fr, N, G, occb, hardg, dtype=torch.float64, wk=wk, wv=wv)
        rg, rc = dict(cmp(gr_r, gr_g)), dict(cmp(gr_r, gr_c))
        print("      vs fp64 ref:  " + "  ".join(f"{n} triton {rg[n]:.1e} cuda {rc[n]:.1e}" for n in rg), flush=True)
print("\nALL OK" if allok else "\nSOME CASES NEED A LOOK")
