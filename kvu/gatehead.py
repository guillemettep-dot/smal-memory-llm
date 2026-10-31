"""gatehead.py -- per-head write-gate use of the shared memories (does a token write in all heads, or a few?).
Reads the gate exactly as memlm_sh computes it: g = sigmoid(w_in(x)[gate cols] + g_bias), (B,T,Hs), via a
forward hook on each WRITER layer's fused input GEMM.  Self-check: the hook's (g>0.5).mean() must equal the
module's own self.wrate for every forward.  Held-out windows (first 10M tokens of shard 0), doc-aware.
Usage: CODE=gqa23 python gatehead.py OUT.json CKPT [CKPT ...]   (refuses if OUT exists)"""
import os, sys, inspect, json
CODE = os.environ["CODE"]
os.environ.update(MEMPROBE_ROOT="/home/paul/memprobe", MEMLM_MOD="memlm_sh", KV_DETACH="1", KV_WQ="0",
                  SLOT_ROPE="1", SLOT_ROPE_FRAC="0.5")
sys.path.insert(0, "/home/paul/memprobe/" + CODE)
os.chdir("/home/paul/memprobe/" + CODE)
import numpy as np, torch
import memlm_sh as S
import train_sh as TS
OUT = sys.argv[1]
assert not os.path.exists(OUT), "exists " + OUT
NW, T = int(os.environ.get("NW", "16")), 1024
sh = TS.Shards(doc_aware=True)
rng = np.random.default_rng(7)
X = torch.from_numpy(np.stack([sh._window(T, rng, True) for _ in range(NW)])[:, :T].astype(np.int64))
assert X.shape == (NW, T), X.shape
ok = set(inspect.signature(S.Cfg.__init__).parameters) - {"self"}
res = {}
for path in sys.argv[2:]:
    st = torch.load(path, map_location="cpu", weights_only=False)
    cc = st["cfg"]
    m = S.MemLM(S.Cfg(**{k: v for k, v in cc.items() if k in ok}))
    mis, unx = m.load_state_dict(st["model"], strict=False)
    assert not mis and not unx, (mis[:5], unx[:5])
    m = m.cuda().eval()
    writers = [(n, mod) for n, mod in m.named_modules() if getattr(mod, "is_writer", False)]
    assert len(writers) in (1, 2), [n for n, _ in writers]
    got = {n: [] for n, _ in writers}; chk = []
    def mk(n, mod):
        def hook(_m, _i, out):
            pg = out.split(mod.sizes, -1)[5]
            g = torch.sigmoid(pg.float() + mod.g_bias.float())
            got[n].append(g.detach().cpu())
        return hook
    hs = [mod.w_in.register_forward_hook(mk(n, mod)) for n, mod in writers]
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for b in range(0, NW, 2):
            m(X[b:b + 2].cuda())
            for n, mod in writers:
                chk.append((float((got[n][-1] > 0.5).float().mean()), float(mod.wrate)))
    for h in hs: h.remove()
    for a, b in chk:
        assert abs(a - b) < 1e-6, ("hook gate != module wrate", a, b)
    print("\n##### %s step %s  heads %s  gate_init %s floor %s nogate %s hard_gate %s  | wrate self-check ok on %d forwards"
          % (os.path.basename(path), st["step"], cc.get("sh_heads"), cc.get("gate_init"), cc.get("sh_gate_floor", 0.0),
             cc.get("sh_nogate", 0), cc.get("slot_hard_gate"), len(chk)))
    R = {}
    for n, mod in writers:
        g = torch.cat(got[n]).reshape(-1, mod.Hs).numpy().astype(np.float64)    # (tokens, H)
        N, H = g.shape
        name = "MAIN (layer %d)" % cc["sh_write"] if mod.mid == "main" else "ENC (layer %d)" % cc["sh_enc_write"]
        live = g > 0.5
        nh = live.sum(1)
        effh = (g.sum(1) ** 2) / np.maximum((g ** 2).sum(1), 1e-30)
        top = g.max(1) / np.maximum(g.sum(1), 1e-30)
        r = dict(H=H, tokens=N,
                 mean=g.mean(0).tolist(), median=np.median(g, 0).tolist(),
                 p_gt05=live.mean(0).tolist(), p_gt01=(g > 0.1).mean(0).tolist(), p_lt001=(g < 0.01).mean(0).tolist(),
                 nheads_hist=np.bincount(nh, minlength=H + 1).tolist(),
                 eff_heads_mean=float(effh.mean()), top_share_mean=float(top.mean()),
                 bias_sig=torch.sigmoid(mod.g_bias.detach().float().cpu()).tolist())
        print("  %s  H=%d  %d tokens  sigmoid(g_bias) %s" % (name, H, N, " ".join("%.3f" % v for v in r["bias_sig"])))
        print("    per head mean g    : " + " ".join("%6.3f" % v for v in r["mean"]))
        print("    per head median g  : " + " ".join("%6.3f" % v for v in r["median"]))
        print("    per head %% g>0.5   : " + " ".join("%6.1f" % (100 * v) for v in r["p_gt05"]))
        print("    per head %% g>0.1   : " + " ".join("%6.1f" % (100 * v) for v in r["p_gt01"]))
        print("    per head %% g<0.01  : " + " ".join("%6.1f" % (100 * v) for v in r["p_lt001"]))
        if H > 1:
            C = np.corrcoef(g.T)
            r["corr"] = C.tolist()
            print("    heads with g>0.5 per token (0..%d): %s  (%% of tokens)" % (H, " ".join("%.1f" % (100 * c / N) for c in r["nheads_hist"])))
            print("    effective heads per token (sum g)^2/sum g^2: %.2f of %d   top head's share of sum g: %.2f" % (r["eff_heads_mean"], H, r["top_share_mean"]))
            print("    gate correlation between heads:")
            for i in range(H):
                print("      " + " ".join("%5.2f" % C[i, j] for j in range(H)))
        R[name] = r
    res[os.path.basename(path)] = R
    del m, st; torch.cuda.empty_cache()
json.dump(res, open(OUT, "w"))
print("\nwrote", OUT)
