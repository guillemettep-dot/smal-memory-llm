import sys, numpy as np, collections
sys.path.insert(0, "/home/paul/memprobe/gqa")
import factgen as F, factgen2 as F2
r = np.random.default_rng(77)
N = 6000
ent_bad_contra = ent_bad_det = ent_bad_pool = 0
rel_bad_same = rel_bad_ent = 0
ex_e, ex_r = [], []
for _ in range(N):
    st, cu, v, e, m = F.simple_meta(r)
    # entity-sharing distractor, called exactly as sibling_fact calls it, but keeping meta
    s1, _, _, _, m1 = F2.sample(r, F.SPLIT, F.ANCHOR_P, q_frac=F.Q_FRAC, pool=m["pool"],
                                stmt_entity=m["E"], exclude_key=m["rel"])
    ent_bad_contra += int(m1["rel"] == m["rel"])                       # same entity AND relation
    ent_bad_det += int(m["E"] not in s1 and m["E"][0].upper() + m["E"][1:] not in s1)
    ent_bad_pool += int(m1["pool"] != m["pool"])
    # relation-sharing distractor
    s2, _, _, _, m2 = F2.sample(r, F.SPLIT, F.ANCHOR_P, q_frac=F.Q_FRAC, R_force=m["R"],
                                exclude_entity=m["ent"])
    rel_bad_same += int(m2["rel"] != m["rel"])
    rel_bad_ent += int(m2["ent"] == m["ent"])
    if len(ex_e) < 5: ex_e.append((st, s1))
    if len(ex_r) < 5 and m["novel"]: ex_r.append((st, s2))
print("ENTITY-sharing distractors (%d):" % N)
print("   same relation too (a contradiction) : %d" % ent_bad_contra)
print("   entity not written as the target    : %d" % ent_bad_det)
print("   different entity pool               : %d" % ent_bad_pool)
print("RELATION-sharing distractors (%d):" % N)
print("   relation NOT the target's           : %d   (was ~62%% before: invented relations)" % rel_bad_same)
print("   same entity as the target           : %d" % rel_bad_ent)
print("\nexamples, entity-sharing  (target  ||  distractor):")
for a, b in ex_e: print("   %-58s || %s" % (a[:58], b[:70]))
print("examples, relation-sharing on INVENTED relations:")
for a, b in ex_r: print("   %-58s || %s" % (a[:58], b[:70]))
