# -*- coding: utf-8 -*-
"""Build needle_fast.py from needle_cpu.py. Same protocol, same seeds, same stats, same output.
Two changes only:
  1. GATHER HEAD -- ans_loss only reads the last ~3 positions, but the head was projecting
     2048 -> 50304 at every one of T positions. Now it runs only on the scored positions.
  2. BATCHING -- 3,072 batch-1 forwards become ceil(3072/BATCH). Right-padding is safe because the
     model is causal: nothing after position p can change the output at p.
"""
import io
src = io.open("needle_cpu.py", encoding="utf-8").read()
def one(o, n, w):
    global src
    c = src.count(o); assert c == 1, "%s: found %d" % (w, c)
    src = src.replace(o, n)

one('BOOT = int(os.environ.get("BOOT", "10000"))',
    'BOOT = int(os.environ.get("BOOT", "10000"))\nBATCH = int(os.environ.get("BATCH", "8"))',
    "BATCH knob")

one("model.requires_grad_(False)",
"""model.requires_grad_(False)

class _GatherHead(torch.nn.Module):
    \"\"\"Run the LM head ONLY at the positions that get scored.\"\"\"
    def __init__(self, h):
        super().__init__(); self.h = h; self.idx = None
    def forward(self, x):
        if self.idx is None:
            return self.h(x)
        i = self.idx.unsqueeze(-1).expand(-1, -1, x.size(-1))
        return self.h(torch.gather(x, 1, i))
model.head = _GatherHead(model.head)""", "gather head")

old = src[src.index("@torch.no_grad()\ndef ans_loss(ctx, ans):"):src.index("# ---------------------------------------------------------------- statistics")]
src = src.replace(old, '''@torch.no_grad()
def ans_loss_batch(items):
    """[(ctx, ans), ...] -> [(mean_loss, greedy_tokens), ...]"""
    seqs = [np.concatenate([c, a]) for c, a in items]
    ns = [len(a) for _, a in items]
    Ls = [len(s) for s in seqs]
    B, T, K = len(items), max(Ls), max(ns)
    seq = torch.zeros(B, T, dtype=torch.long, device=DEV)
    for b, s in enumerate(seqs):
        seq[b, :len(s)] = torch.as_tensor(s, dtype=torch.long, device=DEV)
    idx = torch.zeros(B, K, dtype=torch.long, device=DEV)
    tgt = torch.zeros(B, K, dtype=torch.long, device=DEV)
    for b in range(B):
        n, L = ns[b], Ls[b]
        for k in range(n):
            p = L - n + k
            idx[b, k] = p - 1
            tgt[b, k] = seq[b, p]
    model.head.idx = idx
    lg = model(seq[:, :-1]).float()
    model.head.idx = None
    lp = F.cross_entropy(lg.reshape(-1, lg.size(-1)), tgt.reshape(-1),
                         reduction="none").reshape(B, K)
    gr = lg.argmax(-1)
    out = [(lp[b, :ns[b]].mean().item(), gr[b, :ns[b]].tolist()) for b in range(B)]
    del lg, seq
    return out


@torch.no_grad()
def ans_loss(ctx, ans):
    return ans_loss_batch([(ctx, ans)])[0]


''')

one("""    ls, lo, es, eo, ts, to = [], [], [], [], [], []
    for t in range(TRIALS):
        fact, ofact, q, ans, pre, stream = trial_content(t)
        mid = stream[:dist]
        l1, g1 = ans_loss(np.concatenate([pre, fact, mid, q]), ans)
        l2, g2 = ans_loss(np.concatenate([pre, ofact, mid, q]), ans)
        a = ans.tolist()
        ls.append(l1); lo.append(l2)
        es.append(int(g1 == a)); eo.append(int(g2 == a))
        ts.append(float(np.mean([x == y for x, y in zip(g1, a)])))
        to.append(float(np.mean([x == y for x, y in zip(g2, a)])))
        if (t + 1) % 64 == 0:
            print(f"    dist {dist}: {t+1}/{TRIALS}", flush=True)""",
"""    ls, lo, es, eo, ts, to = [], [], [], [], [], []
    _items, _ans = [], []
    for t in range(TRIALS):
        fact, ofact, q, ans, pre, stream = trial_content(t)
        mid = stream[:dist]
        _items.append((np.concatenate([pre, fact, mid, q]), ans))    # saw
        _items.append((np.concatenate([pre, ofact, mid, q]), ans))   # control
        _ans.append(ans.tolist())
    _res = []
    for i in range(0, len(_items), BATCH):
        _res += ans_loss_batch(_items[i:i + BATCH])
        done = len(_res) // 2
        if done % 64 < BATCH // 2 and done > 0:
            print(f"    dist {dist}: {done}/{TRIALS}", flush=True)
    for t in range(TRIALS):
        (l1, g1), (l2, g2) = _res[2 * t], _res[2 * t + 1]
        a = _ans[t]
        ls.append(l1); lo.append(l2)
        es.append(int(g1 == a)); eo.append(int(g2 == a))
        ts.append(float(np.mean([x == y for x, y in zip(g1, a)])))
        to.append(float(np.mean([x == y for x, y in zip(g2, a)])))""", "batched loop")

io.open("needle_fast.py", "w", encoding="utf-8").write(src)
print("  needle_fast.py written")
