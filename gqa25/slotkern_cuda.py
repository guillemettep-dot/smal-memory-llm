"""Key-matched slot scan, HARD address + DETACHED address gradient (sh21's training mode), as four CUDA kernels.
Same contract as slotkern_kvg.kv_scan_g(hard=True, detach_addr=True): returns (kbuf, vbuf), the PRE-write state
every G steps, with gradients for k, q, v, gate, fr, temp.  Any other mode falls back to kv_scan_g.

WHY FOUR KERNELS.  With a hard address every step changes ONE row, and with the address detached the key/value
gradient recursion  dK <- (1 - w) dK + dkbuf,  dV <- (1 - w) dV + dvbuf  depends only on WHICH row was written and
the gate -- never on the address gradient.  So the work splits into:
  addr_fwd   (one block per (b, h), sequential in t, 1 barrier/step): keys + occupancy only; picks the row (argmax),
             logs (row, old key row, old occupancy), writes the key snapshots.
  val_fwd    (parallel over rows, no barriers): replays each row's value from the log, writes value snapshots and
             the old value row at every write.
  rows_bwd   (parallel over rows, no barriers, reverse in t): undoes each row's key/value from the log, runs the
             dK/dV recursion, emits dk, dv, and D[t, n] = sum (k_t - K_n) dK_n + sum (v_t - V_n) dV_n.
  addr_bwd   (one block per (b, h), reverse in t, 2 barriers/step): undoes keys + occupancy from the log, rebuilds
             the soft address, and with dw = D + (1 - o) do runs the softmax adjoint: dgate, dfr, dtemp, dq.
The sequential kernels hold N x DK floats instead of the Triton kernels' N x (DK + DV) (+ the gradient in the
backward), use exactly N rows (no power-of-two padding), reduce within a row by warp shuffles and across rows with
one shared-memory exchange.

Arithmetic follows slotkern_kvt/_kvg line by line in ORDER OF OPERATIONS PER ELEMENT, but sums are taken in a
different order, so results agree to rounding, not bitwise; an argmax near-tie can therefore pick a different row
than Triton does.  test_cuda.py checks against the float64 reference and against kv_scan_g."""
import ctypes
import os
import torch
# torch's NVRTC wrapper asks for CUDA_HOME/include even when the source includes nothing; this machine has no
# toolkit, but torch's own wheel ships the headers under nvidia/cu13.
if "CUDA_HOME" not in os.environ:
    import nvidia
    _cu = os.path.join(list(nvidia.__path__)[0], "cu13")
    if os.path.isdir(os.path.join(_cu, "include")):
        os.environ["CUDA_HOME"] = _cu
from slotkern_kvg import kv_scan_g

_SRC = r"""
#define N_   {N}
#define D_   {D}
#define EPT  8
#define TPR  (D_ / EPT)
#define NT   {NT}
#define NG   (NT / TPR)
#define RPT  ((N_ + NG - 1) / NG)
#define NW   (NT / 32)
#define RB   {RB}
#define NTR  (RB * TPR)
#define EPS  {EPS}f
#define OCCB {OCCB}
#define FRT  {FRT}
#define FULL 0xffffffffu
#define BIGI 0x7fffffff
#define INFINITY __int_as_float(0x7f800000)

__device__ __forceinline__ float rsum(float v) {{       // sum over the TPR lanes of one row
#pragma unroll
    for (int off = TPR / 2; off > 0; off >>= 1) v += __shfl_xor_sync(FULL, v, off);
    return v;
}}
__device__ __forceinline__ void ld8(float* d, const float* s) {{
    float4 a = *reinterpret_cast<const float4*>(s), b = *reinterpret_cast<const float4*>(s + 4);
    d[0] = a.x; d[1] = a.y; d[2] = a.z; d[3] = a.w; d[4] = b.x; d[5] = b.y; d[6] = b.z; d[7] = b.w;
}}
__device__ __forceinline__ void st8(float* d, const float* s) {{
    *reinterpret_cast<float4*>(d) = make_float4(s[0], s[1], s[2], s[3]);
    *reinterpret_cast<float4*>(d + 4) = make_float4(s[4], s[5], s[6], s[7]);
}}
// occupied candidate (z, i): larger z wins, equal z -> smaller index (tl.argmax returns the first maximum)
__device__ __forceinline__ void best(float& z, int& i, float z2, int i2) {{
    if (z2 > z || (z2 == z && i2 < i)) {{ z = z2; i = i2; }}
}}

// =====================================================================================================
extern "C" __global__ void __launch_bounds__(NT, 1) addr_fwd(
    const float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ gate,
    const float* __restrict__ fr, const float* __restrict__ temp_p,
    float* __restrict__ kbuf, int* __restrict__ am_out, float* __restrict__ kold, float* __restrict__ oold,
    float* __restrict__ kfin, float* __restrict__ ofin, int T, int H, int G, int NB)
{{
    const int pid = blockIdx.x, b = pid / H, h = pid % H;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int grp = tid / TPR, c = tid % TPR;
    __shared__ float s_z[2][NW], s_ez[2][NW];
    __shared__ int s_i[2][NW], s_e[2][NW];
    float K[RPT][EPT], o[RPT];
#pragma unroll
    for (int i = 0; i < RPT; i++) {{ o[i] = 0.f;
#pragma unroll
        for (int j = 0; j < EPT; j++) K[i][j] = 0.f; }}
    const float temp = temp_p[0];
    float frv = FRT ? 0.f : fr[0];
    float qn_[EPT], kt[EPT], qa[EPT], ka[EPT];
    float ga, fa;
    // prefetch t = 0
    {{ size_t ix = ((size_t)b * T + 0) * H + h;
       ld8(qa, q + ix * D_ + c * EPT); ld8(ka, k + ix * D_ + c * EPT); ga = gate[ix]; fa = FRT ? fr[ix] : 0.f; }}
    int par = 0;
    for (int t = 0; t < T; t++) {{
        const size_t ix = ((size_t)b * T + t) * H + h;
        float qt[EPT];
#pragma unroll
        for (int j = 0; j < EPT; j++) {{ qt[j] = qa[j]; kt[j] = ka[j]; }}
        const float g = ga;
        if (FRT) frv = fa;
        if (t + 1 < T) {{ size_t ix2 = ix + H;
            ld8(qa, q + ix2 * D_ + c * EPT); ld8(ka, k + ix2 * D_ + c * EPT); ga = gate[ix2]; if (FRT) fa = fr[ix2]; }}
        if (t % G == 0) {{
            float* kb = kbuf + ((size_t)(b * NB + t / G) * H + h) * N_ * D_;
#pragma unroll
            for (int i = 0; i < RPT; i++) {{ int n = grp + i * NG; if (n < N_) st8(kb + n * D_ + c * EPT, K[i]); }}
        }}
        float qq = 0.f;
#pragma unroll
        for (int j = 0; j < EPT; j++) qq += qt[j] * qt[j];
        qq = rsum(qq);
        const float qinv = 1.f / fmaxf(sqrtf(qq), 1e-6f);
#pragma unroll
        for (int j = 0; j < EPT; j++) qn_[j] = qt[j] * qinv;
        float bz = -INFINITY; int bi = BIGI; int ei = BIGI; float ez = 0.f;
#pragma unroll
        for (int i = 0; i < RPT; i++) {{
            const int n = grp + i * NG;
            float kk = 0.f;
#pragma unroll
            for (int j = 0; j < EPT; j++) kk += K[i][j] * K[i][j];
            kk = rsum(kk);
            const float kinv = 1.f / fmaxf(sqrtf(kk), 1e-6f);
            float dt = 0.f;
#pragma unroll
            for (int j = 0; j < EPT; j++) dt += (K[i][j] * kinv) * qn_[j];
            dt = rsum(dt);
            const float s = temp * dt;
            const float z = OCCB ? (s - frv * (o[i] >= EPS ? 1.f : 0.f)) : (s - frv * o[i]);
            if (n < N_) {{
                if (o[i] < EPS) {{ if (n < ei) {{ ei = n; ez = z; }} }}
                else best(bz, bi, z, n);
            }}
        }}
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {{
            float z2 = __shfl_xor_sync(FULL, bz, off); int i2 = __shfl_xor_sync(FULL, bi, off);
            int e2 = __shfl_xor_sync(FULL, ei, off); float ez2 = __shfl_xor_sync(FULL, ez, off);
            best(bz, bi, z2, i2);
            if (e2 < ei) {{ ei = e2; ez = ez2; }}
        }}
        if (lane == 0) {{ s_z[par][warp] = bz; s_i[par][warp] = bi; s_e[par][warp] = ei; s_ez[par][warp] = ez; }}
        __syncthreads();
        bz = -INFINITY; bi = BIGI; ei = BIGI; ez = 0.f;
#pragma unroll
        for (int w = 0; w < NW; w++) {{
            best(bz, bi, s_z[par][w], s_i[par][w]);
            if (s_e[par][w] < ei) {{ ei = s_e[par][w]; ez = s_ez[par][w]; }}
        }}
        par ^= 1;
        int am = bi;
        if (ei != BIGI && (bi == BIGI || ez > bz || (ez == bz && ei < bi))) am = ei;
        if (tid == 0) am_out[ix] = am;
        const float omw = 1.f - g;
#pragma unroll
        for (int i = 0; i < RPT; i++) {{
            const int n = grp + i * NG;
            if (n == am) {{
                st8(kold + ix * D_ + c * EPT, K[i]);
                if (c == 0) oold[ix] = o[i];
#pragma unroll
                for (int j = 0; j < EPT; j++) K[i][j] = omw * K[i][j] + g * kt[j];
                o[i] = omw * o[i] + g;
            }}
        }}
    }}
#pragma unroll
    for (int i = 0; i < RPT; i++) {{
        const int n = grp + i * NG;
        if (n < N_) {{ st8(kfin + ((size_t)pid * N_ + n) * D_ + c * EPT, K[i]); if (c == 0) ofin[pid * N_ + n] = o[i]; }}
    }}
}}

// =====================================================================================================
// one thread-group of TPR lanes per row, RB rows per block, grid (B*H, ceil(N/RB)); the log lives in shared memory
extern "C" __global__ void __launch_bounds__(NTR) val_fwd(
    const float* __restrict__ v, const float* __restrict__ gate, const int* __restrict__ am,
    float* __restrict__ vbuf, float* __restrict__ vold, float* __restrict__ vfin, int T, int H, int G, int NB)
{{
    extern __shared__ unsigned char sm_[];
    int* s_am = reinterpret_cast<int*>(sm_);
    float* s_g = reinterpret_cast<float*>(sm_ + sizeof(int) * T);
    const int pid = blockIdx.x, b = pid / H, h = pid % H;
    const int tid = threadIdx.x, r = tid / TPR, c = tid % TPR, n = blockIdx.y * RB + r;
    for (int t = tid; t < T; t += NTR) {{ size_t ix = ((size_t)b * T + t) * H + h; s_am[t] = am[ix]; s_g[t] = gate[ix]; }}
    __syncthreads();
    if (n >= N_) return;
    float V[EPT];
#pragma unroll
    for (int j = 0; j < EPT; j++) V[j] = 0.f;
    for (int t = 0; t < T; t++) {{
        if (t % G == 0) st8(vbuf + ((size_t)(b * NB + t / G) * H + h) * N_ * D_ + n * D_ + c * EPT, V);
        if (s_am[t] == n) {{
            const size_t ix = ((size_t)b * T + t) * H + h;
            const float g = s_g[t], omw = 1.f - g;
            float vv[EPT];
            ld8(vv, v + ix * D_ + c * EPT);
            st8(vold + ix * D_ + c * EPT, V);
#pragma unroll
            for (int j = 0; j < EPT; j++) V[j] = omw * V[j] + g * vv[j];
        }}
    }}
    st8(vfin + ((size_t)pid * N_ + n) * D_ + c * EPT, V);
}}

// =====================================================================================================
extern "C" __global__ void __launch_bounds__(NTR) rows_bwd(
    const float* __restrict__ k, const float* __restrict__ v, const float* __restrict__ gate, const int* __restrict__ am,
    const float* __restrict__ kold, const float* __restrict__ vold, const float* __restrict__ kfin,
    const float* __restrict__ vfin, const float* __restrict__ dkbuf, const float* __restrict__ dvbuf,
    float* __restrict__ dk, float* __restrict__ dv, float* __restrict__ Dout, int T, int H, int G, int NB)
{{
    extern __shared__ unsigned char sm_[];
    int* s_am = reinterpret_cast<int*>(sm_);
    float* s_g = reinterpret_cast<float*>(sm_ + sizeof(int) * T);
    const int pid = blockIdx.x, b = pid / H, h = pid % H;
    const int tid = threadIdx.x, r = tid / TPR, c = tid % TPR, n0 = blockIdx.y * RB + r;
    const bool live = n0 < N_;
    const int n = live ? n0 : 0;           // dead rows still run the shuffles; they read row 0 and write nothing
    for (int t = tid; t < T; t += NTR) {{ size_t ix = ((size_t)b * T + t) * H + h; s_am[t] = am[ix]; s_g[t] = gate[ix]; }}
    __syncthreads();
    float K[EPT], V[EPT], dK[EPT], dV[EPT];
    ld8(K, kfin + ((size_t)pid * N_ + n) * D_ + c * EPT);
    ld8(V, vfin + ((size_t)pid * N_ + n) * D_ + c * EPT);
#pragma unroll
    for (int j = 0; j < EPT; j++) {{ dK[j] = 0.f; dV[j] = 0.f; }}
    float ka[EPT], va[EPT];
    {{ size_t ix = ((size_t)b * T + (T - 1)) * H + h; ld8(ka, k + ix * D_ + c * EPT); ld8(va, v + ix * D_ + c * EPT); }}
    for (int t = T - 1; t >= 0; t--) {{
        const size_t ix = ((size_t)b * T + t) * H + h;
        float kt[EPT], vv[EPT];
#pragma unroll
        for (int j = 0; j < EPT; j++) {{ kt[j] = ka[j]; vv[j] = va[j]; }}
        if (t > 0) {{ size_t ix2 = ix - H; ld8(ka, k + ix2 * D_ + c * EPT); ld8(va, v + ix2 * D_ + c * EPT); }}
        const bool hit = live && (s_am[t] == n);
        if (hit) {{ ld8(K, kold + ix * D_ + c * EPT); ld8(V, vold + ix * D_ + c * EPT); }}
        float p = 0.f, pv = 0.f;
#pragma unroll
        for (int j = 0; j < EPT; j++) p += (kt[j] - K[j]) * dK[j];
#pragma unroll
        for (int j = 0; j < EPT; j++) pv += (vv[j] - V[j]) * dV[j];
        p = rsum(p); pv = rsum(pv);
        if (live && c == 0) Dout[((size_t)pid * T + t) * N_ + n] = p + pv;
        if (hit) {{
            const float g = s_g[t], omw = 1.f - g;
            float a[EPT], bb[EPT];
#pragma unroll
            for (int j = 0; j < EPT; j++) {{ a[j] = g * dK[j]; bb[j] = g * dV[j]; dK[j] = omw * dK[j]; dV[j] = omw * dV[j]; }}
            st8(dk + ix * D_ + c * EPT, a); st8(dv + ix * D_ + c * EPT, bb);
        }}
        if (live && t % G == 0) {{
            const size_t so = ((size_t)(b * NB + t / G) * H + h) * N_ * D_ + n * D_ + c * EPT;
            float a[EPT], bb[EPT];
            ld8(a, dkbuf + so); ld8(bb, dvbuf + so);
#pragma unroll
            for (int j = 0; j < EPT; j++) {{ dK[j] += a[j]; dV[j] += bb[j]; }}
        }}
    }}
}}

// =====================================================================================================
extern "C" __global__ void __launch_bounds__(NT, 1) addr_bwd(
    const float* __restrict__ q, const float* __restrict__ gate, const float* __restrict__ fr,
    const float* __restrict__ temp_p, const int* __restrict__ am, const float* __restrict__ kold,
    const float* __restrict__ oold, const float* __restrict__ kfin, const float* __restrict__ ofin,
    const float* __restrict__ Din, float* __restrict__ dq, float* __restrict__ dg, float* __restrict__ dfr,
    float* __restrict__ dtemp, int T, int H)
{{
    const int pid = blockIdx.x, b = pid / H, h = pid % H;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int grp = tid / TPR, c = tid % TPR;
    __shared__ float r_m[2][NW], r_s[2][NW], r_sd[2][NW], r_ez[2][NW], r_ed[2][NW];
    __shared__ int r_e[2][NW];
    __shared__ float r_q[2][NW][D_ / EPT * EPT];
    __shared__ float r_fr[2][NW];
    __shared__ float r_dt[NW];
    float K[RPT][EPT], o[RPT], dO[RPT];
#pragma unroll
    for (int i = 0; i < RPT; i++) {{
        const int n = grp + i * NG, nn = n < N_ ? n : 0;
        ld8(K[i], kfin + ((size_t)pid * N_ + nn) * D_ + c * EPT);
        o[i] = n < N_ ? ofin[pid * N_ + nn] : 0.f;
        dO[i] = 0.f;
    }}
    const float temp = temp_p[0];
    float frv = FRT ? 0.f : fr[0];
    float dtemp_acc = 0.f, dfr_acc = 0.f;
    int par = 0;
    for (int t = T - 1; t >= 0; t--) {{
        const size_t ix = ((size_t)b * T + t) * H + h;
        const int a = am[ix];
        const float g = gate[ix];
        if (FRT) frv = fr[ix];
        float qt[EPT];
        ld8(qt, q + ix * D_ + c * EPT);
        float Dn[RPT];
#pragma unroll
        for (int i = 0; i < RPT; i++) {{
            const int n = grp + i * NG;
            Dn[i] = n < N_ ? Din[((size_t)pid * T + t) * N_ + n] : 0.f;
            if (n == a) {{ ld8(K[i], kold + ix * D_ + c * EPT); o[i] = oold[ix]; }}
        }}
        float qq = 0.f;
#pragma unroll
        for (int j = 0; j < EPT; j++) qq += qt[j] * qt[j];
        qq = rsum(qq);
        const float qnorm = sqrtf(qq), qinv = 1.f / fmaxf(qnorm, 1e-6f);
        float qn_[EPT];
#pragma unroll
        for (int j = 0; j < EPT; j++) qn_[j] = qt[j] * qinv;
        float kinv[RPT], sv[RPT], zv[RPT], dw[RPT];
        float M = -INFINITY, S = 0.f, SD = 0.f; int ei = BIGI; float ez = 0.f, ed = 0.f;
#pragma unroll
        for (int i = 0; i < RPT; i++) {{
            const int n = grp + i * NG;
            float kk = 0.f;
#pragma unroll
            for (int j = 0; j < EPT; j++) kk += K[i][j] * K[i][j];
            kk = rsum(kk);
            kinv[i] = 1.f / fmaxf(sqrtf(kk), 1e-6f);
            float dt = 0.f;
#pragma unroll
            for (int j = 0; j < EPT; j++) dt += (K[i][j] * kinv[i]) * qn_[j];
            dt = rsum(dt);
            sv[i] = temp * dt;
            const float occ = OCCB ? (o[i] >= EPS ? 1.f : 0.f) : o[i];
            zv[i] = sv[i] - frv * occ;
            dw[i] = Dn[i] + (1.f - o[i]) * dO[i];
            if (n < N_ && c == 0) {{
                if (o[i] < EPS) {{ if (n < ei) {{ ei = n; ez = zv[i]; ed = dw[i]; }} }}
                else {{
                    const float m2 = fmaxf(M, zv[i]);
                    const float sc = (M == -INFINITY) ? 0.f : expf(M - m2);
                    const float e = expf(zv[i] - m2);
                    S = S * sc + e; SD = SD * sc + e * dw[i]; M = m2;
                }}
            }}
        }}
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {{
            const float M2 = __shfl_xor_sync(FULL, M, off), S2 = __shfl_xor_sync(FULL, S, off), SD2 = __shfl_xor_sync(FULL, SD, off);
            const int e2 = __shfl_xor_sync(FULL, ei, off);
            const float ez2 = __shfl_xor_sync(FULL, ez, off), ed2 = __shfl_xor_sync(FULL, ed, off);
            const float m = fmaxf(M, M2);
            if (m != -INFINITY) {{
                const float s1 = (M == -INFINITY) ? 0.f : expf(M - m), s2 = (M2 == -INFINITY) ? 0.f : expf(M2 - m);
                S = S * s1 + S2 * s2; SD = SD * s1 + SD2 * s2; M = m;
            }}
            if (e2 < ei) {{ ei = e2; ez = ez2; ed = ed2; }}
        }}
        if (lane == 0) {{ r_m[par][warp] = M; r_s[par][warp] = S; r_sd[par][warp] = SD;
                          r_e[par][warp] = ei; r_ez[par][warp] = ez; r_ed[par][warp] = ed; }}
        __syncthreads();
        M = -INFINITY; S = 0.f; SD = 0.f; ei = BIGI; ez = 0.f; ed = 0.f;
#pragma unroll
        for (int w = 0; w < NW; w++) {{
            const float M2 = r_m[par][w], S2 = r_s[par][w], SD2 = r_sd[par][w];
            const float m = fmaxf(M, M2);
            if (m != -INFINITY) {{
                const float s1 = (M == -INFINITY) ? 0.f : expf(M - m), s2 = (M2 == -INFINITY) ? 0.f : expf(M2 - m);
                S = S * s1 + S2 * s2; SD = SD * s1 + SD2 * s2; M = m;
            }}
            if (r_e[par][w] < ei) {{ ei = r_e[par][w]; ez = r_ez[par][w]; ed = r_ed[par][w]; }}
        }}
        if (ei != BIGI) {{                                   // the first free row competes too
            const float m = fmaxf(M, ez);
            const float s1 = (M == -INFINITY) ? 0.f : expf(M - m), e = expf(ez - m);
            S = S * s1 + e; SD = SD * s1 + e * ed; M = m;
        }}
        const float sa = g * SD / S;                       // sum_n asoft_n * (g * dw_n)
        float pq[EPT];
#pragma unroll
        for (int j = 0; j < EPT; j++) pq[j] = 0.f;
        float frp = 0.f;
#pragma unroll
        for (int i = 0; i < RPT; i++) {{
            const int n = grp + i * NG;
            const bool ok = n < N_ && (o[i] >= EPS || n == ei);
            const float as = ok ? expf(zv[i] - M) / S : 0.f;
            const float dz = ok ? as * (g * dw[i] - sa) : 0.f;
            if (n == a && c == 0) dg[ix] = dw[i];
            if (c == 0) {{
                dtemp_acc += dz * sv[i] / temp;
                const float occ = OCCB ? (o[i] >= EPS ? 1.f : 0.f) : o[i];
                frp += -occ * dz;
            }}
#pragma unroll
            for (int j = 0; j < EPT; j++) pq[j] += dz * temp * (K[i][j] * kinv[i]);
            if (!OCCB) {{ const float omw = (n == a) ? 1.f - g : 1.f; dO[i] = omw * dO[i] - frv * dz; }}
            else if (n == a) dO[i] = (1.f - g) * dO[i];
        }}
#pragma unroll
        for (int off = TPR; off < 32; off <<= 1) {{
#pragma unroll
            for (int j = 0; j < EPT; j++) pq[j] += __shfl_xor_sync(FULL, pq[j], off);
        }}
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) frp += __shfl_xor_sync(FULL, frp, off);
        if (lane < TPR) {{
#pragma unroll
            for (int j = 0; j < EPT; j++) r_q[par][warp][c * EPT + j] = pq[j];
        }}
        if (lane == 0) r_fr[par][warp] = frp;
        __syncthreads();
        if (warp == 0) {{
            float dqn[D_ / 32], qd[D_ / 32], pr = 0.f;
#pragma unroll
            for (int m = 0; m < D_ / 32; m++) {{
                const int d = lane + 32 * m;
                float acc = 0.f;
#pragma unroll
                for (int w = 0; w < NW; w++) acc += r_q[par][w][d];
                dqn[m] = acc; qd[m] = q[ix * D_ + d] * qinv; pr += acc * qd[m];
            }}
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) pr += __shfl_xor_sync(FULL, pr, off);
#pragma unroll
            for (int m = 0; m < D_ / 32; m++) {{
                const int d = lane + 32 * m;
                dq[ix * D_ + d] = (qnorm < 1e-6f) ? qinv * dqn[m] : qinv * (dqn[m] - qd[m] * pr);
            }}
            if (lane == 0) {{
                float f = 0.f;
#pragma unroll
                for (int w = 0; w < NW; w++) f += r_fr[par][w];
                if (FRT) dfr[ix] = f; else dfr_acc += f;
            }}
        }}
        par ^= 1;
    }}
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) dtemp_acc += __shfl_xor_sync(FULL, dtemp_acc, off);
    if (lane == 0) r_dt[warp] = dtemp_acc;
    __syncthreads();
    if (tid == 0) {{
        float s = 0.f;
        for (int w = 0; w < NW; w++) s += r_dt[w];
        atomicAdd(dtemp, s);
        if (!FRT) atomicAdd(dfr, dfr_acc);
    }}
}}
"""

_cache = {}


def _cfg(N, D):
    # sequential kernels: 384 threads; rows-parallel kernels: 32 rows per block
    return dict(NT=384, RB=32)


def _kernels(N, D, occb, frt, eps):
    key = (N, D, bool(occb), bool(frt), float(eps))
    if key not in _cache:
        assert D % 8 == 0 and 32 % (D // 8) == 0 and D % 32 == 0, D
        c = _cfg(N, D)
        src = _SRC.format(N=N, D=D, NT=c["NT"], RB=c["RB"], EPS=repr(float(eps)), OCCB=int(bool(occb)), FRT=int(bool(frt)))
        ks = {}
        for name in ("addr_fwd", "val_fwd", "rows_bwd", "addr_bwd"):
            ks[name] = torch.cuda._compile_kernel(src, name)
        _cache[key] = (ks, c)
    return _cache[key]


def kernel_attrs(N, D, occb=True, frt=True, eps=1e-4):
    """(registers, local-memory bytes per thread = spills) of each compiled kernel."""
    ks, _ = _kernels(N, D, occb, frt, eps)
    lib = torch.cuda._utils._get_gpu_runtime_library()
    out = {}
    for name, kern in ks.items():
        vals = []
        for attr in (4, 3):     # CU_FUNC_ATTRIBUTE_NUM_REGS, CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES
            v = ctypes.c_int()
            lib.cuFuncGetAttribute(ctypes.byref(v), attr, kern.func)
            vals.append(v.value)
        out[name] = tuple(vals)
    return out


class _KVScanC(torch.autograd.Function):
    @staticmethod
    def forward(ctx, k, q, v, gate, fr, temp, N, G, hard_gate, eps, occ_bin):
        B, T, H, D = k.shape
        assert v.shape[-1] == D, "the CUDA scan needs DK == DV"
        NB = (T + G - 1) // G
        f = lambda x: x.contiguous().to(torch.float32)
        k, q, v = f(k), f(q), f(v)
        g = (gate > 0.5).to(torch.float32).contiguous() if hard_gate else f(gate)
        frt = fr.dim() == 3
        if frt:
            assert fr.shape == gate.shape, (fr.shape, gate.shape)
            fr = f(fr)
        else:
            fr = fr.to(torch.float32).reshape(1).contiguous()
        temp = torch.as_tensor(temp, dtype=torch.float32, device=k.device).reshape(1).contiguous()
        ks, c = _kernels(N, D, occ_bin, frt, eps)
        dev = k.device
        e = lambda *s, dt=torch.float32: torch.empty(*s, device=dev, dtype=dt)
        kbuf, vbuf = e(B, NB, H, N, D), e(B, NB, H, N, D)
        am = e(B, T, H, dt=torch.int32)
        kold, vold, oold = e(B, T, H, D), e(B, T, H, D), e(B, T, H)
        kfin, vfin, ofin = e(B * H, N, D), e(B * H, N, D), e(B * H, N)
        ks["addr_fwd"](grid=(B * H, 1, 1), block=(c["NT"], 1, 1),
                       args=[q, k, g, fr, temp, kbuf, am, kold, oold, kfin, ofin, T, H, G, NB])
        smem = 8 * T
        ks["val_fwd"](grid=(B * H, (N + c["RB"] - 1) // c["RB"], 1), block=(c["RB"] * (D // 8), 1, 1),
                      args=[v, g, am, vbuf, vold, vfin, T, H, G, NB], shared_mem=smem)
        ctx.save_for_backward(k, q, v, g, fr, temp, am, kold, vold, oold, kfin, vfin, ofin)
        ctx.meta = (N, int(G), NB, frt, float(eps), bool(occ_bin))
        return kbuf, vbuf

    @staticmethod
    def backward(ctx, dkbuf, dvbuf):
        k, q, v, g, fr, temp, am, kold, vold, oold, kfin, vfin, ofin = ctx.saved_tensors
        N, G, NB, frt, eps, occ_bin = ctx.meta
        B, T, H, D = k.shape
        ks, c = _kernels(N, D, occ_bin, frt, eps)
        dev = k.device
        dkbuf = torch.zeros(B, NB, H, N, D, device=dev) if dkbuf is None else dkbuf.contiguous().float()
        dvbuf = torch.zeros(B, NB, H, N, D, device=dev) if dvbuf is None else dvbuf.contiguous().float()
        dk = torch.empty_like(k); dv = torch.empty_like(v)
        Dn = torch.empty(B * H, T, N, device=dev, dtype=torch.float32)
        ks["rows_bwd"](grid=(B * H, (N + c["RB"] - 1) // c["RB"], 1), block=(c["RB"] * (D // 8), 1, 1),
                       args=[k, v, g, am, kold, vold, kfin, vfin, dkbuf, dvbuf, dk, dv, Dn, T, H, G, NB],
                       shared_mem=8 * T)
        dq = torch.empty_like(q); dg = torch.empty_like(g)
        dfr = torch.empty_like(fr) if frt else torch.zeros(1, device=dev, dtype=torch.float32)
        dtemp = torch.zeros(1, device=dev, dtype=torch.float32)
        ks["addr_bwd"](grid=(B * H, 1, 1), block=(c["NT"], 1, 1),
                       args=[q, g, fr, temp, am, kold, oold, kfin, ofin, Dn, dq, dg, dfr, dtemp, T, H])
        return dk, dq, dv, dg, dfr, dtemp.reshape(()), None, None, None, None, None


def kv_scan_c(k, v, gate, fr, N, temp, G=16, hard=False, hard_gate=False, eps=1e-4, detach_addr=False, q=None,
              nw=None, nwb=None, occ_bin=False):
    """Drop-in for kv_scan_g.  Only hard + detached address with DK == DV runs the CUDA kernels; anything else
    falls back to the Triton scan unchanged."""
    if not (hard and detach_addr and k.shape[-1] == v.shape[-1] and k.shape[-1] in (64, 128)):
        return kv_scan_g(k, v, gate, fr, N, temp, G=G, hard=hard, hard_gate=hard_gate, eps=eps,
                         detach_addr=detach_addr, q=q, nw=nw, nwb=nwb, occ_bin=occ_bin)
    return _KVScanC.apply(k, k if q is None else q, v, gate, fr, temp, N, G, hard_gate, eps, bool(occ_bin))
