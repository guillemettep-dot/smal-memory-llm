cd /home/paul/memprobe/gqa
E="MEMPROBE_ROOT=/home/paul/memprobe MEMLM_MOD=memlm_kv DEV=cpu THREADS=3 KV_DETACH=1 KV_WQ=0 SLOT_ROPE=1 SLOT_ROPE_FRAC=0.5"
CK=/mnt/ssd/memprobe_ckpt/gqa2k/gqa2k_step3000.pt
P=/home/paul/memprobe/.venv/bin/python3
echo "### reference: needle_cpu.py (batch 1, full head)"
S=$(date +%s)
env $E CKPT=$CK TRIALS=16 DISTS=64,256 TRAIN_MAXLEN=1024 BOOT=200 OUTDIR=/tmp/vref $P -u needle_cpu.py > /tmp/vref.log 2>&1
echo "  took $(( $(date +%s) - S ))s"
echo "### fast: needle_fast.py (BATCH=8, gathered head)"
S=$(date +%s)
env $E BATCH=8 CKPT=$CK TRIALS=16 DISTS=64,256 TRAIN_MAXLEN=1024 BOOT=200 OUTDIR=/tmp/vfast $P -u needle_fast.py > /tmp/vfast.log 2>&1
echo "  took $(( $(date +%s) - S ))s"
echo "### do they agree?"
$P - <<'PY'
import json, glob
def load(d):
    f = glob.glob(d + "/needle_hi_*.json")
    return json.load(open(f[0]))["per_trial"] if f else None
a, b = load("/tmp/vref"), load("/tmp/vfast")
if a is None or b is None:
    print("  MISSING OUTPUT: ref=%s fast=%s" % (a is not None, b is not None)); raise SystemExit(1)
bad = 0
for d in a:
    for k in ("es", "eo", "ts", "to"):
        if a[d][k] != b[d][k]:
            print("  MISMATCH dist %s field %s" % (d, k)); bad += 1
    for k in ("ls", "lo"):
        m = max(abs(x - y) for x, y in zip(a[d][k], b[d][k]))
        print("  dist %-5s %s max|delta| = %.2e" % (d, k, m))
        if m > 1e-3: print("    *** EXCEEDS 1e-3 ***"); bad += 1
print("  VERDICT:", "IDENTICAL - safe to use" if bad == 0 else "DIVERGED (%d) - do NOT use" % bad)
PY
