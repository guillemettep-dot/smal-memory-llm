import sys, math, torch
sys.path.insert(0, "/home/paul/memprobe/gqa")
def show(path, names):
    st = torch.load(path, map_location="cpu", weights_only=False); m = st["model"]
    print("==", path.split("/")[-1], "step", st["step"])
    for n in names(m):
        t = m[n].float()
        if t.numel() <= 16:
            print("   %-38s %s" % (n, [round(x, 4) for x in t.flatten().tolist()]))
        else:
            print("   %-38s shape %-18s rms %.5f" % (n, tuple(t.shape), t.pow(2).mean().sqrt().item()))
    return m
print("init values: log_wsharp = log16 = %.4f, log_free = 1.5, log_rsel = log8 = %.4f, g_bias = logit(0.42) = %.4f, weights rms 0.02"
      % (math.log(16), math.log(8), math.log(0.42 / 0.58)))
sh = show("/home/paul/memprobe/bank_sh3/sh3_step3000.pt", lambda m: [n for n in m if any(s in n for s in
     ("blocks.2.mix.log_", "blocks.2.mix.g_bias", "blocks.6.mix.log_", "blocks.6.mix.g_bias", "blocks.6.mix.w_in", "blocks.9.mix.log_rsel", "blocks.9.mix.tk_q", "blocks.9.mix.wo_tk", "blocks.0.mix.w_in"))])
w = sh["blocks.6.mix.w_in.weight"].float()
print("   writer w_in rows: window k/q/v rms %.5f | value %.5f | key %.5f | gate %.5f" % (
    w[:1536].pow(2).mean().sqrt(), w[1536:1536+768].pow(2).mean().sqrt(), w[1536+768:1536+1536].pow(2).mean().sqrt(), w[1536+1536:].pow(2).mean().sqrt()))
show("/mnt/ssd/memprobe_ckpt/gqa2k/gqa2k_step3000.pt", lambda m: [n for n in m if any(s in n for s in
     ("blocks.6.mix.log_", "blocks.6.mix.g_bias", "blocks.6.mix.tk_q", "blocks.6.mix.wo_tk"))])
