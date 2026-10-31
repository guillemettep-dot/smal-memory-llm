import sys, torch
sys.path.insert(0, "/home/paul/memprobe/gqa24")
import slotkern_kv as KV
d = torch.load("/mnt/ssd/memprobe_ckpt/sh24mem2_dbg/nonfinite_bwd_7539.pt", weights_only=False)
dkb = d["dkbuf"]; bad = ~torch.isfinite(dkb); rows = bad.any(-1)          # (B, NB, H, N)
B, NB, H, N, D = dkb.shape; G = d["G"]
k, q, v, g, fr, temp = (d[x].double() for x in ("k", "q", "v", "g", "fr", "temp"))
kb, vb, w = KV._ref_scan_kv_impl(k, v, g, fr, N, temp.reshape(()), G=G, hard=True, detach_addr=True, q=q, pin=False, occ_bin=d["occ_bin"], return_w=True)
am = w.argmax(-1)                                   # (B, T, H)
nsq = kb.pow(2).sum(-1)                             # (B, NB, H, N)
for b in range(B):
    for j in sorted(set(rows[b].nonzero()[:, 0].tolist())):
        br = rows[b, j, 0].nonzero().flatten().tolist()
        nvalid = (nsq[b, j, 0] > 1e-12).sum().item()
        recent = am[b, max(0, j * G - 16):j * G, 0].tolist()
        print(f"b{b} snap {j} (t={j*G}): {len(br)} bad rows {br[:16]}...  valid rows {nvalid}  rows written in the 16 tokens before: {recent}")
print("am (row written) for b0, first 64 tokens:", am[0, :64, 0].tolist())
print("dkbuf bad count by snapshot, b0:", rows[0, :, 0].sum(-1).tolist())
