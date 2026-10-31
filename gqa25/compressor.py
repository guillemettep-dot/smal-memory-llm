"""Per-big-head-per-layer compressor (once per token), grouped so each big head has its own weights:
    [ big-head slot state 3k | KV-cache reads + fill ] -> sparse 512 -> sparse 128 (own-layer tap) -> dense 64 (cross)
    +  SELECTION ATTENTION over the slots: queries from the KV-context reads, keys computed from each slot's VALUE
       on the fly (nothing stored in the memory), n_sel heads -> the selected values (n_sel*dv) enter the tap DENSELY.
Why the selection: "pick the fact that responds best to the immediate context" is a comparison of the context against
every slot; in the fixed random block-sparse stage only ~1% of units see a given slot together with the context blocks,
so it cannot do that comparison for most slots (s1 @4k: needle lift 0 while m12 had +4.2). The sparse funnel stays as
the broad summary; the selection is the filter. LayerNorms keep the tap's scale fill-invariant. groups = n_big_heads."""
import torch, torch.nn as nn, torch.nn.functional as F, math
import bsparse
from bsparse import make_wiring, inverse_map


class GroupBSP(nn.Module):
    """grouped block-sparse linear over [xs | xr]: (N, G, KS) [+ (N, G, KR)] -> (N, G, n_out) fp32."""
    def __init__(self, groups, n_in, n_out, nb, seed):
        super().__init__()
        assert n_in % 32 == 0 and n_out % 32 == 0
        idx = make_wiring(groups, n_in, n_out, nb, seed)
        self.register_buffer("idx", idx)
        self.register_buffer("inv", inverse_map(idx, n_in))
        bd = 1.0 / (idx.shape[2] * 32) ** 0.5
        self.w = nn.Parameter(torch.empty(groups, n_out // 32, idx.shape[2], 32, 32).uniform_(-bd, bd))
        self.n_in, self.n_out = n_in, n_out

    def forward(self, xs, xr=None, gelu=False):
        return bsparse.bs_funnel(xs, xr, None, self.w, self.idx, self.inv, gelu=gelu)


class GroupLinear(nn.Module):
    """grouped dense linear: (N, G, in) -> (N, G, out), weight (G, in, out)."""
    def __init__(self, groups, n_in, n_out, bias=True):
        super().__init__()
        self.w = nn.Parameter(torch.empty(groups, n_in, n_out).normal_(0, n_in ** -0.5))
        self.b = nn.Parameter(torch.zeros(groups, n_out)) if bias else None

    def forward(self, x):
        y = torch.einsum("ngi,gio->ngo", x, self.w)
        return y + self.b if self.b is not None else y


class Compressor(nn.Module):
    def __init__(self, groups, state_dim=3072, kv_dim=160, nb=5, seed=1000, w1=512, w2=128, out=64, dv=16, n_sel=4):
        super().__init__()
        assert state_dim % 32 == 0 and kv_dim % 32 == 0, "segments must be 32-multiples (block-sparse kernel)"
        assert state_dim % dv == 0
        self.G, self.KS, self.KR, self.own_dim, self.out_dim = groups, state_dim, kv_dim, w2, out
        self.dv, self.n_slots, self.n_sel = dv, state_dim // dv, n_sel
        self.s1 = GroupBSP(groups, state_dim + kv_dim, w1, nb, seed + 1)   # block ids: [state | reads+fill]
        self.s2 = GroupBSP(groups, w1, w2, nb, seed + 2)                   # -> own-layer tap (128)
        self.d3 = GroupLinear(groups, w2, out)                             # -> cross-layer broadcast (64)
        self.own_scale = nn.Parameter(torch.full((groups,), 0.15))         # the LayerNormed tap starts SMALL in the residual
        # selection attention over the slots, scored in VALUE space: q (n_sel x dv) from the context, score = q . v_slot.
        # No key tensor: 192 x dv MACs per head per token. n_sel = 0 disables the selection (funnel only).
        if n_sel > 0:
            self.sel_q = GroupLinear(groups, kv_dim, n_sel * dv, bias=False)   # context -> n_sel queries in value space
            self.sel_o = GroupLinear(groups, n_sel * dv, w2, bias=False)       # selected values -> the tap, densely

    def select(self, state, kv, qsel=None):
        # state (N,G,KS) kv (N,G,KR) -> selected values (N,G,n_sel*dv) fp32. Query: qsel (N,G,n_sel*dv) if given
        # (from the token's own hidden state, as attention does), else derived from the KV-context reads.
        N, G, _ = state.shape
        v = state.float().reshape(N, G, self.n_slots, self.dv)                              # (N,G,T,dv)
        q = (qsel.float() if qsel is not None else torch.einsum("ngi,gio->ngo", kv.float(), self.sel_q.w.float())).reshape(N, G, self.n_sel, self.dv)
        sc = torch.einsum("ngsd,ngtd->ngst", q, v) / math.sqrt(self.dv)                    # (N,G,S,T)
        valid = (v.abs().amax(-1) > 0)                                                      # (N,G,T)
        sc = sc.masked_fill(~valid.unsqueeze(2), float("-inf"))
        a = torch.nan_to_num(torch.softmax(sc, -1))                                          # no filled slot -> zeros
        return torch.einsum("ngst,ngtd->ngsd", a, v).reshape(N, G, self.n_sel * self.dv)

    def forward(self, state, kv, qsel=None):
        # state:(B,T,G,KS) kv:(B,T,G,KR) [qsel:(B,T,G,n_sel*dv) query from the token] in the model dtype.
        # Returns own (B,T,G,128), cross (B,T,G,64), sel (B,T,G,n_sel*dv) -- the selected values also go to the
        # residual DIRECTLY in the mixer (a read path), not only through the funnel tap.
        B, T, G, _ = state.shape
        xs = state.reshape(B * T, G, self.KS); xr = kv.reshape(B * T, G, self.KR)
        h = self.s1(xs, xr, gelu=False)                                     # fp32
        h = F.layer_norm(h, (h.shape[-1],))                                 # fixed internal scale whatever the slot fill
        tap = self.s2(h, None, gelu=True)                                   # fp32 (N,G,128)
        sel = None
        if self.n_sel > 0:
            sel = self.select(xs, xr, None if qsel is None else qsel.reshape(B * T, G, -1))
            tap = tap + torch.einsum("ngi,gio->ngo", sel, self.sel_o.w.float())
        own = F.layer_norm(tap, (self.own_dim,))
        own = (own * self.own_scale.float().view(1, G, 1)).to(state.dtype)
        cross = self.d3(F.gelu(own))
        sel = None if sel is None else sel.reshape(B, T, G, self.n_sel * self.dv).to(state.dtype)
        return own.reshape(B, T, G, self.own_dim), cross.reshape(B, T, G, self.out_dim), sel


if __name__ == "__main__":
    torch.manual_seed(0)
    dev = "cuda"
    B, T, BH, SD, KV = 2, 16, 8, 3072, 160
    c = Compressor(BH, SD, KV).to(dev).to(torch.bfloat16)
    n = sum(p.numel() for p in c.parameters())
    print("compressor params: %.3fM per LAYER (groups=%d; selection %.3fM) | x12 = %.1fM" % (
        n / 1e6, BH, (c.sel_q.w.numel() + c.sel_o.w.numel()) / 1e6, n / 1e6 * 12))
    state = torch.randn(B, T, BH, SD, device=dev, dtype=torch.bfloat16, requires_grad=True)
    kv = torch.randn(B, T, BH, KV, device=dev, dtype=torch.bfloat16, requires_grad=True)
    own, cross, _ = c(state, kv)
    (own.float().square().mean() + cross.float().square().mean()).backward()
    gfin = bool(all(torch.isfinite(p.grad).all() for p in c.parameters() if p.grad is not None) and torch.isfinite(state.grad).all() and torch.isfinite(kv.grad).all())
    from bsparse import expand_dense
    def dense(stage):
        return expand_dense(stage.w.detach().float(), stage.idx.cpu(), stage.n_in).to(dev)
    with torch.no_grad():
        X = torch.cat([state.float(), kv.float()], -1).reshape(B * T, BH, -1)
        h = F.layer_norm(torch.einsum("ngk,gko->ngo", X, dense(c.s1)), (512,))
        sel = c.select(state.reshape(B * T, BH, SD), kv.reshape(B * T, BH, KV))
        own_r = F.layer_norm(torch.einsum("ngk,gko->ngo", F.gelu(h), dense(c.s2)) + torch.einsum("ngi,gio->ngo", sel, c.sel_o.w.float()), (128,)) * c.own_scale.float().view(1, BH, 1)
    rel = ((own.reshape(B * T, BH, 128).float() - own_r).abs().max() / (own_r.abs().max() + 1e-9)).item()
    # selection sanity: with one slot planted per group, the selection must return that slot's value
    st = torch.zeros(1, 1, BH, SD, device=dev); planted = torch.randn(BH, 16, device=dev)
    for g in range(BH): st[0, 0, g, 7 * 16:8 * 16] = planted[g]
    s2 = c.select(st.reshape(1, BH, SD).to(torch.bfloat16), torch.randn(1, BH, KV, device=dev, dtype=torch.bfloat16))
    sel_ok = bool(torch.allclose(s2.reshape(BH, c.n_sel, 16)[:, 0], planted, atol=0.05))
    print("shapes own %s cross %s | grads finite %s | vs dense fp32 reference (bf16 kernels): rel %.2e | single-slot selection exact: %s" % (
        tuple(own.shape), tuple(cross.shape), gfin, rel, sel_ok))
    print("COMPRESSOR_OK" if own.shape[-1] == 128 and cross.shape[-1] == 64 and gfin and rel < 5e-2 and sel_ok else "COMPRESSOR_CHECK")
