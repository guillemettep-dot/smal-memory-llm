# verify needle_pl json: mean of per-token CE == stored loss; tokens consistent
import json, sys
d = json.load(open(sys.argv[1]))["per_trial"]
for k, v in d.items():
    for i in range(len(v["ls"])):
        assert len(v["p1"][i]) == len(v["a"][i]) == len(v["p2"][i])
        assert abs(sum(v["p1"][i]) / len(v["p1"][i]) - v["ls"][i]) < 1e-4, (k, i)
        assert abs(sum(v["p2"][i]) / len(v["p2"][i]) - v["lo"][i]) < 1e-4, (k, i)
        assert abs(sum(x == y for x, y in zip(v["g1"][i], v["a"][i])) / len(v["a"][i]) - v["ts"][i]) < 1e-9
    print(k, "ok", len(v["ls"]))
