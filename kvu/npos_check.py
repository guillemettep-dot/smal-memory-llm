# verify a needle_pos json: ts/to recomputed from saved tokens, and print digits-only summary
import json, sys
d = json.load(open(sys.argv[1]))["per_trial"]
for k, v in d.items():
    for i in range(len(v["ts"])):
        a, g1, g2 = v["a"][i], v["g1"][i], v["g2"][i]
        assert abs(sum(x == y for x, y in zip(g1, a)) / len(a) - v["ts"][i]) < 1e-9
        assert abs(sum(x == y for x, y in zip(g2, a)) / len(a) - v["to"][i]) < 1e-9
        assert int(g1 == a) == v["es"][i]
    print(k, "ok", len(v["ts"]), "trials")
