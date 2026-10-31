"""kvnq_cmp.py A.pt B.pt bitwise|close  -- compare two kvnq_run.py outputs.
bitwise: logits, loss and every grad A has must be torch.equal in B.  Params only in B are reported (and for the
         switch: win_q2 grads must be exactly 0 and win_o2 grads finite and non-zero).
close:   max |diff| / max |ref| for logits and every shared grad, must be < TOL (default 1e-4, fp32 regrouping)."""
import sys, torch
A, B, how = torch.load(sys.argv[1]), torch.load(sys.argv[2]), sys.argv[3]
TOL = 1e-4
print(sys.argv[1], A["mode"], A["code"], "vs", sys.argv[2], B["mode"], B["code"], "| loss %.7f / %.7f" % (A["loss"], B["loss"]))
bad = []
def rel(a, b):
    return float((a - b).abs().max() / b.abs().max().clamp_min(1e-30))
if how == "bitwise":
    if not torch.equal(A["logits"], B["logits"]): bad.append(("logits", rel(A["logits"], B["logits"])))
    if not torch.equal(A["loss"], B["loss"]): bad.append(("loss", float(A["loss"] - B["loss"])))
    nn = 0
    for n, g in A["grads"].items():
        h = B["grads"][n]
        if g is None or h is None:
            if not (g is None and h is None): bad.append((n, "grad None on one side"))
            continue
        nn += 1
        if not torch.equal(g, h): bad.append((n, rel(g, h)))
    print("bitwise: logits, loss and %d grads compared" % nn)
    for n in sorted(set(B["grads"]) - set(A["grads"])):
        h = B["grads"][n]
        z = float(h.abs().max()) if h is not None else None
        fin = bool(torch.isfinite(h).all()) if h is not None else None
        print("  only in B: %-26s max|grad| %s finite %s" % (n, z, fin))
        if n.startswith("win_q2.") and z != 0.0: bad.append((n, "q2 grad not exactly 0 at the switch"))
        if n.startswith("win_o2.") and not (fin and z > 0): bad.append((n, "o2 grad not finite non-zero"))
else:
    r = rel(B["logits"], A["logits"]); print("logits rel %.2e" % r)
    if not r < TOL: bad.append(("logits", r))
    worst = (0.0, None)
    for n, g in A["grads"].items():
        h = B["grads"].get(n, "missing")
        if isinstance(h, str): bad.append((n, "missing in B")); continue
        if g is None or h is None:
            if not (g is None and h is None): bad.append((n, "None on one side"))
            continue
        e = rel(h, g) if float(g.abs().max()) > 0 else float(h.abs().max())
        worst = max(worst, (e, n))
        if not e < TOL: bad.append((n, e))
    print("grads: %d compared, worst rel %.2e (%s)" % (len(A["grads"]), worst[0], worst[1]))
    for n in sorted(set(B["grads"]) - set(A["grads"])): bad.append((n, "only in B"))
print("FAIL" if bad else "PASS", bad[:10])
sys.exit(1 if bad else 0)
