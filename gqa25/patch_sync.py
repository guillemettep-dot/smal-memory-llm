import io, ast
p = "/home/paul/memprobe/gqa/train_sh.py"; s = io.open(p, encoding="utf-8").read()
def rep(a, b):
    global s
    assert s.count(a) == 1, (s.count(a), a[:60]); s = s.replace(a, b)
rep("            _plain_acc += float(_plain); _hot_acc += float(_hot); _nmb += 1\n",
    "            # GPU-side accumulators: float() here was a host sync between the forward and the backward,\n"
    "            # and with loss.item() below it left the CPU building the next batch while the GPU sat idle\n"
    "            _plain_acc += _plain.detach().float(); _hot_acc += _hot.detach().float(); _nmb += 1\n")
rep("            lsum += loss.item()\n", "            lsum += loss.detach().float()\n")
rep('                ustr += f" ce {_plain_acc/_nmb:.4f} recall {100*_hot_acc/_nmb:.1f}%"',
    '                ustr += f" ce {float(_plain_acc)/_nmb:.4f} recall {100*float(_hot_acc)/_nmb:.1f}%"')
rep('            print(f"step {step:6d}/{total_steps} loss {lsum:.4f} ppl {math.exp(min(lsum,20)):8.2f}{ustr} "',
    '            lsum = float(lsum)\n'
    '            print(f"step {step:6d}/{total_steps} loss {lsum:.4f} ppl {math.exp(min(lsum,20)):8.2f}{ustr} "')
ast.parse(s); io.open(p, "w", encoding="utf-8").write(s); print("sync patch applied")
