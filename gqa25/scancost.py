# -*- coding: utf-8 -*-
"""How much of a step is the memory scan? Measured at the PRODUCTION shape, alongside the live run.

Deliberately small: ~400 MB, a few seconds, so it cannot OOM the training job sharing this card.
Compares the recurrence against a GEMM of the size one layer's MoE actually does, on the same card,
so the answer is "the scan costs X of what the trunk costs", not an absolute nobody can interpret.
"""
import os, time, torch
os.environ.setdefault("KV_DETACH", "1")
from slotkern_kv import scan_kv

dev = "cuda"
B, T, H, N, DK, DV = 1, 1024, 2, 96, 64, 64
D, F, L, TOPK = 2048, 1408, 12, 2

def timed(fn, iters=8, warmup=3):
    for _ in range(warmup): fn()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(iters): fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters

k = torch.randn(B, T, H, DK, device=dev, requires_grad=True)
v = torch.randn(B, T, H, DV, device=dev, requires_grad=True)
g = torch.rand(B, T, H, device=dev, requires_grad=True)
fr = torch.rand(1, device=dev) * 0.1   # scalar free-rate, not per-head
temp = torch.ones(1, device=dev)

def scan_fwd():
    kb, bf = scan_kv(k.float(), v.float(), g, fr, N, temp, detach_addr=True)
    return kb, bf
def scan_fwdbwd():
    kb, bf = scan_fwd(); (kb.sum() + bf.sum()).backward()

t_s_f = timed(scan_fwd); t_s_fb = timed(scan_fwdbwd)

# one layer's MoE work: top_k=2 experts x 3 matrices, as dense GEMMs at the same token count
x = torch.randn(B * T, D, device=dev, dtype=torch.bfloat16, requires_grad=True)
Ws = [torch.randn(D, F, device=dev, dtype=torch.bfloat16, requires_grad=True) for _ in range(3 * TOPK)]
def moe_fwd():
    h = x
    for W in Ws: h = (x @ W) @ W.T
    return h
def moe_fwdbwd():
    moe_fwd().sum().backward()
t_m_f = timed(moe_fwd); t_m_fb = timed(moe_fwdbwd)

print("  PRODUCTION SHAPE  B=%d T=%d H=%d N=%d DK=%d DV=%d" % (B, T, H, N, DK, DV))
print("  %-38s %8.2f ms fwd   %8.2f ms fwd+bwd" % ("memory scan (one layer)", t_s_f*1e3, t_s_fb*1e3))
print("  %-38s %8.2f ms fwd   %8.2f ms fwd+bwd" % ("MoE GEMMs (one layer, top_k=2)", t_m_f*1e3, t_m_fb*1e3))
print()
print("  scan / MoE ratio, fwd+bwd:  %.1fx" % (t_s_fb / t_m_fb))
print("  scan x %d layers, fwd+bwd:  %8.1f ms per micro-step of %d tokens" % (L, t_s_fb*L*1e3, B*T))
obs = 65536 / 1500.0                      # measured steady-state seconds per optimizer step
micro = obs / (65536 / (B*T))             # seconds per micro-step
print("  measured micro-step (65,536 tok/step at 1,500 tok/s): %8.1f ms" % (micro*1e3))
print("  -> scan is %.0f%% of a micro-step; MoE GEMMs are %.0f%%" %
      (100*t_s_fb*L/micro, 100*t_m_fb*L/micro))
print("  -> everything else (proj, attn, router, optimizer, launch overhead): %.0f%%" %
      (100*(1 - (t_s_fb*L + t_m_fb*L)/micro)))
