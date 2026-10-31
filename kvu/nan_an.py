import sys, os, torch
HERE = "/home/paul/memprobe/gqa24"; sys.path.insert(0, HERE)
import slotkern_cw as KW
d = torch.load("/mnt/ssd/memprobe_ckpt/sh24mem2_dbg/nonfinite_bwd_7539.pt", weights_only=False)
dkb, dvb = d["dkbuf"], d["dvbuf"]
print("N", d["N"], "G", d["G"], "dkbuf", tuple(dkb.shape), "nonfinite", (~torch.isfinite(dkb)).sum().item(), "nan", torch.isnan(dkb).sum().item(), "inf", torch.isinf(dkb).sum().item(),
      "| dvbuf nonfinite", (~torch.isfinite(dvb)).sum().item())
bad = ~torch.isfinite(dkb)                                    # (B, NB, H, N, D)
rows = bad.any(-1)
idx = rows.nonzero()
print("bad (b, snap, h, row) entries:", idx.shape[0], " whole rows bad:", (bad.all(-1) == rows).all().item())
print("first few:", idx[:12].tolist())
print("snapshots with a bad row:", sorted(set(idx[:, 1].tolist()))[:20], " rows:", sorted(set(idx[:, 3].tolist()))[:30])
# recompute the forward snapshots on the saved inputs to see what those rows held
dev = "cuda"
k, q, v, g, fr, temp = (d[x].to(dev) for x in ("k", "q", "v", "g", "fr", "temp"))
kb, vb = KW.kv_scan_cw(k, v, g, fr, d["N"], temp.reshape(()), G=d["G"], hard=True, detach_addr=True, q=q, occ_bin=d["occ_bin"])
kb = kb.cpu(); n = kb.norm(dim=-1)
print("|K| of the bad rows: min %.3e max %.3e ; rows exactly zero among them: %d of %d" % (
      n[rows].min().item(), n[rows].max().item(), (n[rows] == 0).sum().item(), rows.sum().item()))
print("all-zero rows overall in the snapshots: %d of %d" % ((n == 0).sum().item(), n.numel()))
