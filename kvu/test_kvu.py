"""kv_scan_u (undo-log backward) vs kv_scan_g (replay backward), hard address, on the GPU.
  1. the forward snapshots must be BITWISE equal (same arithmetic);
  2. the gradients: max |diff| relative to the gradient's own max.  If they are not bitwise equal, a float64
     reference (the same kernel family's reference, hard address) arbitrates which one is closer.
Covers: both memory shapes of sh21 and of 6 x 128, a T that is not a multiple of G, per-token and scalar fr,
occupancy indicator on/off, detached/attached address, separate search query, hard gate, and gates that are
EXACTLY 0 and 1 (a skipped write and a full overwrite)."""
import sys, os, torch
sys.path.insert(0, os.environ.get("CODEDIR", "/home/paul/memprobe/gqa21"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import slotkern_kvg as KG, slotkern_kv as KV, slotkern_kvu as KU
assert "gqa21" in KG.__file__, KG.__file__
dev = "cuda"
cases = [  # B, T, H, N, D, G, frt, occb, detach, sq, hardg
    (2, 1024, 12, 192, 64, 8, True, True, True, True, False),     # sh21 main, as trained
    (2, 1024, 12, 96, 64, 8, True, True, True, True, False),      # sh21 encoder, as trained
    (2, 1024, 6, 192, 128, 8, True, True, True, True, False),     # 6 x 128 main
    (2, 1024, 6, 96, 128, 8, True, True, True, True, False),      # 6 x 128 encoder
    (2, 1000, 12, 192, 64, 8, True, True, False, True, False),    # T % G != 0, address grads through K
    (1, 300, 4, 40, 32, 16, False, False, False, False, False),   # scalar fr, soft occupancy, q = k
    (1, 300, 4, 40, 32, 7, False, True, True, True, True),        # hard gate, odd stride
]
def inputs(B, T, H, N, D, frt, sq, seed):
    gen = torch.Generator(device=dev).manual_seed(seed)
    r = lambda *s: torch.randn(*s, device=dev, generator=gen)
    k = r(B, T, H, D); q = r(B, T, H, D) if sq else None; v = r(B, T, H, D)
    g = torch.rand(B, T, H, device=dev, generator=gen)
    m = torch.rand(B, T, H, device=dev, generator=gen)
    g = torch.where(m < 0.15, torch.zeros_like(g), torch.where(m > 0.92, torch.ones_like(g), g))
    fr = (torch.rand(B, T, H, device=dev, generator=gen) * 2) if frt else torch.tensor([1.5], device=dev)
    return k, q, v, g, fr
def run(fn, k, q, v, g, fr, N, G, occb, det, hardg, wk=None, wv=None):
    lv = [t.clone().requires_grad_() for t in (k, v, g, fr)] + ([q.clone().requires_grad_()] if q is not None else [])
    kb, vb = fn(lv[0], lv[1], lv[2], lv[3], N, torch.tensor(16.0, device=dev), G=G, hard=True, hard_gate=hardg,
                detach_addr=det, q=lv[4] if q is not None else None, occ_bin=occb)
    if wk is None:
        gen = torch.Generator(device=dev).manual_seed(7)
        wk = torch.randn(kb.shape, device=dev, generator=gen); wv = torch.randn(vb.shape, device=dev, generator=gen)
    (kb * wk + vb * wv).sum().backward()
    return kb.detach(), vb.detach(), [t.grad for t in lv], wk, wv
names = ["dk", "dv", "dgate", "dfr", "dq"]
allok = True
for ci, (B, T, H, N, D, G, frt, occb, det, sq, hardg) in enumerate(cases):
    k, q, v, g, fr = inputs(B, T, H, N, D, frt, sq, 100 + ci)
    kb_g, vb_g, gr_g, wk, wv = run(KG.kv_scan_g, k, q, v, g, fr, N, G, occb, det, hardg)
    kb_u, vb_u, gr_u, _, _ = run(KU.kv_scan_u, k, q, v, g, fr, N, G, occb, det, hardg, wk, wv)
    fwd_eq = torch.equal(kb_g, kb_u) and torch.equal(vb_g, vb_u)
    line = []
    worst = 0.0
    for nm, a, b in zip(names, gr_g, gr_u):
        if a is None and b is None: continue
        assert torch.isfinite(a).all() and torch.isfinite(b).all(), nm
        if torch.equal(a, b):
            line.append(f"{nm} =="); continue
        rel = ((a - b).abs().max() / a.abs().max().clamp_min(1e-30)).item()
        worst = max(worst, rel); line.append(f"{nm} {rel:.1e}")
    ok = fwd_eq and worst < 1e-5
    allok &= ok
    print(f"[{ci}] B{B} T{T} H{H} N{N} D{D} G{G} frt={int(frt)} occb={int(occb)} det={int(det)} q={int(sq)} hardg={int(hardg)}"
          f" | fwd {'BITWISE' if fwd_eq else 'DIFFERS'} | " + "  ".join(line) + ("   OK" if ok else "   <-- CHECK"), flush=True)
    if worst > 0 and T <= 300:
        # arbitrate with the float64 reference (scalar fr only: the reference has no per-token fr)
        if not frt:
            d = lambda x: None if x is None else x.double()
            lv = [d(t).requires_grad_() for t in (k, v, g, fr)] + ([d(q).requires_grad_()] if q is not None else [])
            kr, vr = KV._ref_scan_kv_impl(lv[0], lv[1], lv[2], lv[3].reshape(1), N, torch.tensor(16.0, device=dev, dtype=torch.float64),
                                          G=G, hard=True, hard_gate=hardg, detach_addr=det, q=lv[4] if q is not None else None,
                                          pin=False, occ_bin=occb)
            fwd_ref = (kr.float() - kb_g).abs().max().item()
            (kr * wk.double() + vr * wv.double()).sum().backward()
            for nm, a, b, r in zip(names, gr_g, gr_u, [t.grad for t in lv]):
                if r is None: continue
                sc = r.abs().max().clamp_min(1e-30)
                print(f"      vs fp64 ref {nm}: replay {((a.double()-r).abs().max()/sc).item():.1e}  undo {((b.double()-r).abs().max()/sc).item():.1e}"
                      f"   (fwd |kbuf-ref| {fwd_ref:.1e})")
print("\nALL OK" if allok else "\nSOME CASES NEED A LOOK")
