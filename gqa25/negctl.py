# Negative control for battery3's filler self-check: collapse FRAMES to ONE entry -- literally the
# v2 defect -- and confirm the check REFUSES.  A check that only ever passes is not a check.
import io, os, sys, re
os.environ.setdefault("GPT2", "gpt2"); os.environ.setdefault("TASKS", "code")
os.environ.setdefault("DISTS", "1024"); os.environ.setdefault("N", "2")
sys.path.insert(0, "/home/paul/memprobe/gqa"); os.chdir("/home/paul/memprobe/gqa")
src = io.open("battery3.py", encoding="utf-8").read()
start = src.index("FRAMES = [")
end   = src.index("]\n", src.index("needed a new fuse")) + 1
one   = 'FRAMES = [" The {thing} in the {place} had been awkward since the winter."]'
src2  = src[:start] + one + src[end:]
assert src2.count("FRAMES = [") == 1 and "needed a new fuse" not in src2, "surgery failed"
try:
    exec(compile(src2, "battery3-DEGENERATE", "exec"), {"__name__": "__main__"})
except AssertionError as e:
    print("NEGATIVE CONTROL PASSED -- check refused:\n   %s" % str(e)[:200]); raise SystemExit(0)
print("NEGATIVE CONTROL FAILED -- a one-frame filler was accepted"); raise SystemExit(1)
