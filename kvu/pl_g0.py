import json, collections
D = "/home/paul/memprobe/needle_out/"
try:
    import tiktoken; dec = tiktoken.get_encoding("gpt2").decode
except Exception:
    from transformers import GPT2TokenizerFast; dec = GPT2TokenizerFast.from_pretrained("gpt2").decode
for n in ("sh23_5000", "sh23_6000", "sh20_6000"):
    r = json.load(open(D + f"needle_{n}_pl.json"))["per_trial"]
    for dist in ("64", "1024"):
        v = r[dist]
        for key in ("g1", "g2"):
            c = collections.Counter(repr(dec([g[0]])) for g, a in zip(v[key], v["a"]) if len(a) == 3)
            dig = sum(1 for g, a in zip(v[key], v["a"]) if len(a) == 3 and dec([g[0]]).strip().isdigit())
            print(n, dist, key, "digit-greedy", dig, "/125  top:", c.most_common(5))
v = r["64"]; print("example answers:", [[dec([t]) for t in a] for a in v["a"][:4]])
