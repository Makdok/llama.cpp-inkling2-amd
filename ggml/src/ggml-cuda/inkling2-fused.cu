#include "inkling2-fused.cuh"

// Design rule for these kernels: a thread reads a few CONTIGUOUS elements. A thread that walks memory in strides pays one
// memory latency per access, in series; with strided loops the first versions of the mix and gate kernels took 20 to 80 us
// per launch and were slower than the generic ops they replaced.

// ---------------------------------------------------------------------------------------------------------------------
// L2 clip of rows: one warp per row, each lane owns a contiguous chunk of the row
// ---------------------------------------------------------------------------------------------------------------------

// scale (optional): per-channel weights {d, n_scale_rows} applied first, row r of x uses scale row r % n_scale_rows
static __global__ void ink2_l2_clip_f32(const float * x, const float * scale, float * dst, const int d, const int n_scale_rows,
        const float threshold, const float eps) {
    ggml_cuda_pdl_lc();
    const int64_t row   = blockIdx.x;
    const int     chunk = (d + WARP_SIZE - 1)/WARP_SIZE;
    const int     e0    = threadIdx.x*chunk;
    const int     e1    = min(d, e0 + chunk);

    const float * xr = x   + row*d;
    float       * dr = dst + row*d;
    const float * sr = scale ? scale + (row % n_scale_rows)*d : nullptr;

    ggml_cuda_pdl_sync();

    float sum = 0.0f;
    for (int e = e0; e < e1; ++e) {
        const float v = sr ? xr[e]*sr[e] : xr[e];
        sum += v*v;
    }
    sum = warp_reduce_sum(sum);

    // every lane has read its elements before the reduction returns, so dst may alias x
    const float factor = fminf(1.0f, fmaxf(0.0f, threshold/(sqrtf(sum) + eps)));
    for (int e = e0; e < e1; ++e) {
        const float v = sr ? xr[e]*sr[e] : xr[e];
        dr[e] = v*factor;
    }
}

void ggml_cuda_op_ink2_l2_clip(ggml_backend_cuda_context & ctx, const ggml_tensor * x, ggml_tensor * dst, const float threshold, const float eps,
        const ggml_tensor * scale) {
    GGML_ASSERT(x->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(x) && ggml_is_contiguous(dst) && ggml_are_same_shape(x, dst));

    const int     d      = x->ne[0];
    const int64_t n_rows = ggml_nelements(x)/d;
    int n_scale_rows = 1;
    if (scale) {
        GGML_ASSERT(scale->type == GGML_TYPE_F32 && ggml_is_contiguous(scale) && scale->ne[0] == d);
        n_scale_rows = (int) (ggml_nelements(scale)/d);
        GGML_ASSERT(n_rows % n_scale_rows == 0);
    }

    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params((int) n_rows, WARP_SIZE, 0, ctx.stream());
    ggml_cuda_kernel_launch(ink2_l2_clip_f32, launch_params, (const float *) x->data, scale ? (const float *) scale->data : nullptr,
        (float *) dst->data, d, n_scale_rows, threshold, eps);
}

// ---------------------------------------------------------------------------------------------------------------------
// depth attention mix: one block of 256 threads per token, each thread owns a contiguous chunk of the embedding
// ---------------------------------------------------------------------------------------------------------------------

// The number of states NS and of queries NQ are template parameters: with run-time sizes the per-thread accumulators are
// dynamically indexed arrays, which the compiler puts in scratch memory (the first version cost as much as the 7 ops it
// replaced). With constant bounds they stay in registers.
struct ink2_mix_args {
    const float * s[GGML_CUDA_INK2_MIX_MAX_STATES];
    int   n_embd;
    float eps;
    float scale;
};

static constexpr int INK2_MIX_BLOCK = 256;

// dst element (e, q, t) at t*dst_st + q*dst_sq + e: [n_embd, n_q, n_tokens] (the MUL_MAT layout) or [n_embd, n_tokens, n_q]
// (the permuted copy of inkling2.cpp, whose rows are contiguous)
template <int NS, int NQ>
static __global__ void ink2_depth_mix_f32(const ink2_mix_args a, const float * query, float * dst, const int64_t dst_st, const int64_t dst_sq) {
    ggml_cuda_pdl_lc();
    constexpr int NRED = 1 + NQ; // per state: sum of squares, then one dot per query

    const int64_t t     = blockIdx.x;
    const int     tid   = threadIdx.x;
    const int     lane  = tid % WARP_SIZE;
    const int     warp  = tid / WARP_SIZE;
    const int     chunk = (a.n_embd + INK2_MIX_BLOCK - 1)/INK2_MIX_BLOCK;
    const int     e0    = tid*chunk;
    const int     e1    = min(a.n_embd, e0 + chunk);

    __shared__ float red[INK2_MIX_BLOCK/WARP_SIZE][NS*NRED];
    __shared__ float coef[NS*NQ];

    ggml_cuda_pdl_sync();

    float acc[NS*NRED];
#pragma unroll
    for (int k = 0; k < NS*NRED; ++k) {
        acc[k] = 0.0f;
    }
    for (int e = e0; e < e1; ++e) {
        float qv[NQ];
#pragma unroll
        for (int q = 0; q < NQ; ++q) {
            qv[q] = query[(int64_t) q*a.n_embd + e];
        }
#pragma unroll
        for (int i = 0; i < NS; ++i) {
            const float v = a.s[i][t*a.n_embd + e];
            acc[i*NRED] += v*v;
#pragma unroll
            for (int q = 0; q < NQ; ++q) {
                acc[i*NRED + 1 + q] += v*qv[q];
            }
        }
    }
#pragma unroll
    for (int k = 0; k < NS*NRED; ++k) {
        const float r = warp_reduce_sum(acc[k]);
        if (lane == 0) {
            red[warp][k] = r;
        }
    }
    __syncthreads();

    // thread i < NS finishes state i: rms and scaled logits, written back to red[0] once every thread has read red
    float fin[NRED];
#pragma unroll
    for (int k = 0; k < NRED; ++k) {
        fin[k] = 0.0f;
    }
    if (tid < NS) {
#pragma unroll
        for (int k = 0; k < NRED; ++k) {
#pragma unroll
            for (int w = 0; w < INK2_MIX_BLOCK/WARP_SIZE; ++w) {
                fin[k] += red[w][tid*NRED + k];
            }
        }
        fin[0] = rsqrtf(fin[0]/a.n_embd + a.eps);
#pragma unroll
        for (int q = 0; q < NQ; ++q) {
            fin[1 + q] *= fin[0]*a.scale;
        }
    }
    __syncthreads();
    if (tid < NS) {
#pragma unroll
        for (int k = 0; k < NRED; ++k) {
            red[0][tid*NRED + k] = fin[k];
        }
    }
    __syncthreads();

    // thread q < NQ: softmax over the states for query q
    if (tid < NQ) {
        float l[NS];
        float mx = -INFINITY;
#pragma unroll
        for (int i = 0; i < NS; ++i) {
            l[i] = red[0][i*NRED + 1 + tid];
            mx   = fmaxf(mx, l[i]);
        }
        float sum = 0.0f;
#pragma unroll
        for (int i = 0; i < NS; ++i) {
            l[i] = expf(l[i] - mx);
            sum += l[i];
        }
#pragma unroll
        for (int i = 0; i < NS; ++i) {
            // softmax weight times the rms normalization of the state
            coef[i*NQ + tid] = l[i]/sum*red[0][i*NRED];
        }
    }
    __syncthreads();

    float cf[NS*NQ];
#pragma unroll
    for (int k = 0; k < NS*NQ; ++k) {
        cf[k] = coef[k];
    }
    for (int e = e0; e < e1; ++e) {
        float o[NQ];
#pragma unroll
        for (int q = 0; q < NQ; ++q) {
            o[q] = 0.0f;
        }
#pragma unroll
        for (int i = 0; i < NS; ++i) {
            const float v = a.s[i][t*a.n_embd + e];
#pragma unroll
            for (int q = 0; q < NQ; ++q) {
                o[q] += cf[i*NQ + q]*v;
            }
        }
#pragma unroll
        for (int q = 0; q < NQ; ++q) {
            dst[t*dst_st + q*dst_sq + e] = o[q];
        }
    }
}

bool ggml_cuda_ink2_depth_mix_supported(const int n_states, const int n_q) {
    return n_states >= 2 && n_states <= 6 && n_q >= 1 && n_q <= 3;
}

void ggml_cuda_op_ink2_depth_mix(ggml_backend_cuda_context & ctx, const ggml_tensor * const * states, const int n_states,
        const ggml_tensor * query, ggml_tensor * dst, const float eps, const float scale, const bool rows_first) {
    GGML_ASSERT(query->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 && ggml_is_contiguous(query) && ggml_is_contiguous(dst));

    ink2_mix_args a = {};
    a.n_embd = (int) query->ne[0];
    a.eps    = eps;
    a.scale  = scale;
    const int n_q = (int) (ggml_nelements(query)/query->ne[0]);
    GGML_ASSERT(ggml_cuda_ink2_depth_mix_supported(n_states, n_q));

    const int64_t n_tokens = ggml_nelements(dst)/((int64_t) a.n_embd*n_q);
    for (int i = 0; i < n_states; ++i) {
        GGML_ASSERT(states[i]->type == GGML_TYPE_F32 && ggml_is_contiguous(states[i]));
        GGML_ASSERT(states[i]->ne[0] == a.n_embd && ggml_nelements(states[i]) == a.n_embd*n_tokens);
        a.s[i] = (const float *) states[i]->data;
    }

    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params((int) n_tokens, INK2_MIX_BLOCK, 0, ctx.stream());
    const float * q_d = (const float *) query->data;
    float       * d_d = (float *) dst->data;
    const int64_t st  = rows_first ? a.n_embd : (int64_t) n_q*a.n_embd;
    const int64_t sq  = rows_first ? n_tokens*a.n_embd : a.n_embd;

#define INK2_MIX_CASE(NS, NQ) case (NS)*8 + (NQ): ggml_cuda_kernel_launch(ink2_depth_mix_f32<NS, NQ>, launch_params, a, q_d, d_d, st, sq); break
    switch (n_states*8 + n_q) {
        INK2_MIX_CASE(2, 1); INK2_MIX_CASE(2, 2); INK2_MIX_CASE(2, 3);
        INK2_MIX_CASE(3, 1); INK2_MIX_CASE(3, 2); INK2_MIX_CASE(3, 3);
        INK2_MIX_CASE(4, 1); INK2_MIX_CASE(4, 2); INK2_MIX_CASE(4, 3);
        INK2_MIX_CASE(5, 1); INK2_MIX_CASE(5, 2); INK2_MIX_CASE(5, 3);
        INK2_MIX_CASE(6, 1); INK2_MIX_CASE(6, 2); INK2_MIX_CASE(6, 3);
        default: GGML_ABORT("fatal error");
    }
#undef INK2_MIX_CASE
}

// ---------------------------------------------------------------------------------------------------------------------
// loop message gate, two kernels:
//   1) h[r] = <D[:n, r], cur> * inv_rms(cur) + <D[n:, r], msg> * inv_rms(msg): one block per (token, r)
//   2) dst[e] = (1 - tanh(<F[:, e], h>)) * cur[e] + tanh(<G[:, e], h>) * msg[e]: one thread per element
// ---------------------------------------------------------------------------------------------------------------------

static constexpr int INK2_GATE_BLOCK = 256;

static __global__ void ink2_loop_gate_h_f32(const float * cur, const float * msg, const float * down, float * h,
        const int n_embd, const int n_rank, const float eps) {
    ggml_cuda_pdl_lc();
    const int64_t t     = blockIdx.x / n_rank;
    const int     r     = blockIdx.x % n_rank;
    const int     tid   = threadIdx.x;
    const int     lane  = tid % WARP_SIZE;
    const int     warp  = tid / WARP_SIZE;
    const int     chunk = (n_embd + INK2_GATE_BLOCK - 1)/INK2_GATE_BLOCK;
    const int     e0    = tid*chunk;
    const int     e1    = min(n_embd, e0 + chunk);

    __shared__ float red[INK2_GATE_BLOCK/WARP_SIZE][4];

    const float * c = cur  + t*n_embd;
    const float * m = msg  + t*n_embd;
    const float * d = down + (int64_t) r*2*n_embd;

    ggml_cuda_pdl_sync();

    float acc[4] = { 0.0f, 0.0f, 0.0f, 0.0f }; // |cur|^2, |msg|^2, <D, cur>, <D, msg>
    for (int e = e0; e < e1; ++e) {
        const float vc = c[e];
        const float vm = m[e];
        acc[0] += vc*vc;
        acc[1] += vm*vm;
        acc[2] += d[e]*vc;
        acc[3] += d[n_embd + e]*vm;
    }
    for (int k = 0; k < 4; ++k) {
        const float s = warp_reduce_sum(acc[k]);
        if (lane == 0) {
            red[warp][k] = s;
        }
    }
    __syncthreads();

    if (tid == 0) {
        float tot[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
        for (int w = 0; w < INK2_GATE_BLOCK/WARP_SIZE; ++w) {
            for (int k = 0; k < 4; ++k) {
                tot[k] += red[w][k];
            }
        }
        h[t*n_rank + r] = tot[2]*rsqrtf(tot[0]/n_embd + eps) + tot[3]*rsqrtf(tot[1]/n_embd + eps);
    }
}

static __global__ void ink2_loop_gate_out_f32(const float * cur, const float * msg, const float * gain, const float * forget,
        const float * h, float * dst, const int n_embd, const int n_rank, const int64_t n_total) {
    ggml_cuda_pdl_lc();
    const int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n_total) {
        return;
    }
    const int64_t t = i / n_embd;
    const int     e = i % n_embd;

    const float * g  = gain   + (int64_t) e*n_rank;
    const float * f  = forget + (int64_t) e*n_rank;
    const float * ht = h + t*n_rank;

    ggml_cuda_pdl_sync();

    float sg = 0.0f;
    float sf = 0.0f;
    for (int r = 0; r < n_rank; ++r) {
        const float hv = ht[r];
        sg += g[r]*hv;
        sf += f[r]*hv;
    }
    // same index read and written: dst may alias cur or msg
    dst[i] = (1.0f - tanhf(sf))*cur[i] + tanhf(sg)*msg[i];
}

// Multi-token version (every weight read once per gate, whatever the number of tokens; also faster for one token):
//   1) h kernel: one block per rank r, float4 loads, all NT tokens at once (the norms of cur and msg come with the dots)
//   2) out kernel: 8 lanes per element e, each lane a contiguous quarter... eighth of the rows G[:, e] and F[:, e] (float4
//      loads), h of all tokens in shared memory, butterfly sums; lane k of the group writes the tokens t = k mod 8
static constexpr int INK2_GATE_LPE = 8; // lanes per output element

// 4 consecutive weights i4*4 .. i4*4 + 3 as floats: F32 (one float4 load) or BF16 stored as uint16_t (one 8-byte load; the
// loop gate weights of Inkling 2 Mini are exact in BF16, so BF16 storage halves their bytes without changing them)
static __device__ __forceinline__ float4 ink2_w4(const float * p, const int64_t i4) {
    return ((const float4 *) p)[i4];
}
static __device__ __forceinline__ float4 ink2_w4(const uint16_t * p, const int64_t i4) {
    const uint2 u = ((const uint2 *) p)[i4];
    return make_float4(__uint_as_float(u.x << 16), __uint_as_float(u.x & 0xffff0000u), __uint_as_float(u.y << 16), __uint_as_float(u.y & 0xffff0000u));
}

template <int NT, typename WT>
static __global__ void ink2_loop_gate_h_mt_f32(const float * cur, const float * msg, const WT * down, float * h,
        const int n_embd, const int n_rank, const float eps) {
    ggml_cuda_pdl_lc();
    constexpr int NACC = 4*NT; // per token: |cur|^2, |msg|^2, <D, cur>, <D, msg>
    const int r    = blockIdx.x;
    const int tid  = threadIdx.x;
    const int lane = tid % WARP_SIZE;
    const int warp = tid / WARP_SIZE;
    const int n4   = n_embd/4;

    const float4 * c4 = (const float4 *) cur;
    const float4 * m4 = (const float4 *) msg;
    const WT     * dr = down + (int64_t) r*2*n_embd;

    __shared__ float red[32][NACC];

    ggml_cuda_pdl_sync();

    float acc[NACC];
#pragma unroll
    for (int k = 0; k < NACC; ++k) {
        acc[k] = 0.0f;
    }
    for (int j = tid; j < n4; j += blockDim.x) {
        const float4 dc = ink2_w4(dr, j);
        const float4 dm = ink2_w4(dr, n4 + j);
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            const float4 c = c4[(int64_t) t*n4 + j];
            const float4 m = m4[(int64_t) t*n4 + j];
            acc[4*t + 0] += c.x*c.x + c.y*c.y + c.z*c.z + c.w*c.w;
            acc[4*t + 1] += m.x*m.x + m.y*m.y + m.z*m.z + m.w*m.w;
            acc[4*t + 2] += dc.x*c.x + dc.y*c.y + dc.z*c.z + dc.w*c.w;
            acc[4*t + 3] += dm.x*m.x + dm.y*m.y + dm.z*m.z + dm.w*m.w;
        }
    }
#pragma unroll
    for (int k = 0; k < NACC; ++k) {
        const float s = warp_reduce_sum(acc[k]);
        if (lane == 0) {
            red[warp][k] = s;
        }
    }
    __syncthreads();
    if (tid < NT) {
        const int nw = blockDim.x / WARP_SIZE;
        float tot[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
        for (int w = 0; w < nw; ++w) {
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                tot[k] += red[w][4*tid + k];
            }
        }
        h[(int64_t) tid*n_rank + r] = tot[2]*rsqrtf(tot[0]/n_embd + eps) + tot[3]*rsqrtf(tot[1]/n_embd + eps);
    }
}

// NV = float4 loads per lane and weight row (n_rank = 32*NV)
template <int NT, int NV, typename WT>
static __global__ void ink2_loop_gate_out_mt_f32(const float * cur, const float * msg, const WT * gain, const WT * forget,
        const float * h, float * dst, const int n_embd) {
    ggml_cuda_pdl_lc();
    constexpr int n_rank = INK2_GATE_LPE*4*NV;
    const int tid = threadIdx.x;
    const int sub = tid % INK2_GATE_LPE;                       // lane in the group of the element
    const int e   = (blockIdx.x*blockDim.x + tid)/INK2_GATE_LPE;

    __shared__ float hs[NT*n_rank];

    ggml_cuda_pdl_sync();

    for (int k = tid; k < NT*n_rank; k += blockDim.x) {
        hs[k] = h[k];
    }
    __syncthreads();
    if (e >= n_embd) {
        return;
    }

    const WT * gr = gain   + (int64_t) e*n_rank;
    const WT * fr = forget + (int64_t) e*n_rank;
    float4 g[NV];
    float4 f[NV];
#pragma unroll
    for (int v = 0; v < NV; ++v) {
        g[v] = ink2_w4(gr, sub*NV + v);
        f[v] = ink2_w4(fr, sub*NV + v);
    }

    float sg[NT];
    float sf[NT];
#pragma unroll
    for (int t = 0; t < NT; ++t) {
        const float * ht = hs + t*n_rank + sub*4*NV;
        float a = 0.0f;
        float b = 0.0f;
#pragma unroll
        for (int v = 0; v < NV; ++v) {
            a += g[v].x*ht[4*v + 0] + g[v].y*ht[4*v + 1] + g[v].z*ht[4*v + 2] + g[v].w*ht[4*v + 3];
            b += f[v].x*ht[4*v + 0] + f[v].y*ht[4*v + 1] + f[v].z*ht[4*v + 2] + f[v].w*ht[4*v + 3];
        }
#pragma unroll
        for (int o = INK2_GATE_LPE/2; o > 0; o >>= 1) {
            a += ggml_cuda_shfl_xor(a, o, WARP_SIZE);
            b += ggml_cuda_shfl_xor(b, o, WARP_SIZE);
        }
        sg[t] = a;
        sf[t] = b;
    }
#pragma unroll
    for (int t = 0; t < NT; ++t) {
        if (t % INK2_GATE_LPE == sub) {
            const int64_t i = (int64_t) t*n_embd + e;
            // same index read and written: dst may alias cur or msg
            dst[i] = (1.0f - tanhf(sf[t]))*cur[i] + tanhf(sg[t])*msg[i];
        }
    }
}

template <int NT, typename WT>
static void ink2_loop_gate_mt_launch(ggml_backend_cuda_context & ctx, const float * cur, const float * msg, const WT * down,
        const WT * gain, const WT * forget, float * h, float * dst, const int n_embd, const int n_rank, const float eps) {
    {
        const int n4    = n_embd/4;
        const int iters = (n4 + 1023)/1024;
        const int block = std::min(1024, ((n4 + iters - 1)/iters + WARP_SIZE - 1)/WARP_SIZE*WARP_SIZE);
        const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(n_rank, block, 0, ctx.stream());
        ggml_cuda_kernel_launch(ink2_loop_gate_h_mt_f32<NT, WT>, launch_params, cur, msg, down, h, n_embd, n_rank, eps);
    }
    {
        constexpr int block = 128;
        const int grid = (n_embd*INK2_GATE_LPE + block - 1)/block;
        const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid, block, 0, ctx.stream());
        switch (n_rank) {
            case  32: ggml_cuda_kernel_launch(ink2_loop_gate_out_mt_f32<NT, 1, WT>, launch_params, cur, msg, gain, forget, (const float *) h, dst, n_embd); break;
            case  64: ggml_cuda_kernel_launch(ink2_loop_gate_out_mt_f32<NT, 2, WT>, launch_params, cur, msg, gain, forget, (const float *) h, dst, n_embd); break;
            case 128: ggml_cuda_kernel_launch(ink2_loop_gate_out_mt_f32<NT, 4, WT>, launch_params, cur, msg, gain, forget, (const float *) h, dst, n_embd); break;
            default: GGML_ABORT("fatal error");
        }
    }
}

template <typename WT>
static void ink2_loop_gate_mt(ggml_backend_cuda_context & ctx, const float * cur, const float * msg, const WT * dw, const WT * gw,
        const WT * fw, float * h, float * dst, const int n_embd, const int n_rank, const int64_t n_tokens, const float eps) {
    for (int64_t t0 = 0; t0 < n_tokens; t0 += 8) {
        const int nt = (int) std::min<int64_t>(8, n_tokens - t0);
        const float * c  = cur + t0*n_embd;
        const float * mg = msg + t0*n_embd;
        float       * d  = dst + t0*n_embd;
        float       * hh = h   + t0*n_rank;
        switch (nt) {
            case 1: ink2_loop_gate_mt_launch<1, WT>(ctx, c, mg, dw, gw, fw, hh, d, n_embd, n_rank, eps); break;
            case 2: ink2_loop_gate_mt_launch<2, WT>(ctx, c, mg, dw, gw, fw, hh, d, n_embd, n_rank, eps); break;
            case 3: ink2_loop_gate_mt_launch<3, WT>(ctx, c, mg, dw, gw, fw, hh, d, n_embd, n_rank, eps); break;
            case 4: ink2_loop_gate_mt_launch<4, WT>(ctx, c, mg, dw, gw, fw, hh, d, n_embd, n_rank, eps); break;
            case 5: ink2_loop_gate_mt_launch<5, WT>(ctx, c, mg, dw, gw, fw, hh, d, n_embd, n_rank, eps); break;
            case 6: ink2_loop_gate_mt_launch<6, WT>(ctx, c, mg, dw, gw, fw, hh, d, n_embd, n_rank, eps); break;
            case 7: ink2_loop_gate_mt_launch<7, WT>(ctx, c, mg, dw, gw, fw, hh, d, n_embd, n_rank, eps); break;
            case 8: ink2_loop_gate_mt_launch<8, WT>(ctx, c, mg, dw, gw, fw, hh, d, n_embd, n_rank, eps); break;
            default: GGML_ABORT("fatal error");
        }
    }
}

bool ggml_cuda_ink2_loop_gate_bf16_supported(const int n_embd, const int n_rank) {
    return n_embd % 4 == 0 && (n_rank == 32 || n_rank == 64 || n_rank == 128);
}

void ggml_cuda_op_ink2_loop_gate(ggml_backend_cuda_context & ctx, const ggml_tensor * cur, const ggml_tensor * msg,
        const ggml_tensor * down, const ggml_tensor * gain, const ggml_tensor * forget, ggml_tensor * dst, const float eps) {
    const int     n_embd   = (int) dst->ne[0];
    const int     n_rank   = (int) down->ne[1];
    const int64_t n_tokens = ggml_nelements(dst)/n_embd;

    for (const ggml_tensor * t : { cur, msg, (const ggml_tensor *) dst }) {
        GGML_ASSERT(t->type == GGML_TYPE_F32 && ggml_is_contiguous(t));
    }
    // the weights: all F32, or all BF16 (multi-token kernels only)
    const bool bf16 = down->type == GGML_TYPE_BF16;
    for (const ggml_tensor * t : { down, gain, forget }) {
        GGML_ASSERT(t->type == (bf16 ? GGML_TYPE_BF16 : GGML_TYPE_F32) && ggml_is_contiguous(t));
    }
    GGML_ASSERT(n_rank >= 1);
    GGML_ASSERT(down->ne[0] == 2*n_embd && gain->ne[0] == n_rank && gain->ne[1] == n_embd && forget->ne[0] == n_rank && forget->ne[1] == n_embd);
    GGML_ASSERT(ggml_nelements(cur) == ggml_nelements(dst) && ggml_nelements(msg) == ggml_nelements(dst));

    ggml_cuda_pool_alloc<float> h(ctx.pool(), n_tokens*n_rank);

    // multi-token kernels: float4 rows (16-byte aligned), n_rank 32/64/128, chunks of at most 8 tokens.
    // GGML_CUDA_INK2_GATE_MT=0 keeps the first kernels (F32 weights only).
    static const bool mt_enabled = []() { const char * e = getenv("GGML_CUDA_INK2_GATE_MT"); return !(e && e[0] == '0'); }();
    const auto al16 = [](const void * p) { return ((uintptr_t) p) % 16 == 0; };
    const bool mt_ok = n_embd % 4 == 0 && (n_rank == 32 || n_rank == 64 || n_rank == 128) && al16(cur->data) && al16(msg->data) &&
        al16(down->data) && al16(gain->data) && al16(forget->data);
    if (bf16) {
        GGML_ASSERT(mt_ok);
        ink2_loop_gate_mt<uint16_t>(ctx, (const float *) cur->data, (const float *) msg->data, (const uint16_t *) down->data,
            (const uint16_t *) gain->data, (const uint16_t *) forget->data, h.get(), (float *) dst->data, n_embd, n_rank, n_tokens, eps);
        return;
    }
    if (mt_enabled && mt_ok) {
        ink2_loop_gate_mt<float>(ctx, (const float *) cur->data, (const float *) msg->data, (const float *) down->data,
            (const float *) gain->data, (const float *) forget->data, h.get(), (float *) dst->data, n_embd, n_rank, n_tokens, eps);
        return;
    }

    {
        const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params((int) (n_tokens*n_rank), INK2_GATE_BLOCK, 0, ctx.stream());
        ggml_cuda_kernel_launch(ink2_loop_gate_h_f32, launch_params, (const float *) cur->data, (const float *) msg->data,
            (const float *) down->data, h.get(), n_embd, n_rank, eps);
    }
    {
        const int64_t n_total = n_tokens*n_embd;
        const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params((int) ((n_total + INK2_GATE_BLOCK - 1)/INK2_GATE_BLOCK), INK2_GATE_BLOCK, 0, ctx.stream());
        ggml_cuda_kernel_launch(ink2_loop_gate_out_f32, launch_params, (const float *) cur->data, (const float *) msg->data,
            (const float *) gain->data, (const float *) forget->data, (const float *) h.get(), (float *) dst->data, n_embd, n_rank, n_total);
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// short convolution with its residual: dst[c, t] = (cur[c, t] + sum_j x[t + j, c] * w[j, c]) * scale (+ res[c, t])
// x is {n_conv - 1 + n_tokens, n_channels}, one thread per output element
// ---------------------------------------------------------------------------------------------------------------------

static __global__ void ink2_sconv_res_f32(const float * x, const float * w, const float * cur, const float * res, float * dst,
        const int n_conv, const int n_x, const int n_channels, const int64_t n_total, const float scale) {
    ggml_cuda_pdl_lc();
    const int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n_total) {
        return;
    }
    const int64_t t = i / n_channels;
    const int     c = i % n_channels;

    const float * xs = x + (int64_t) c*n_x + t;
    const float * ws = w + (int64_t) c*n_conv;

    ggml_cuda_pdl_sync();

    float sum = 0.0f;
    for (int j = 0; j < n_conv; ++j) {
        sum += xs[j]*ws[j];
    }
    // same index read and written: dst may alias cur or res
    const float v = (cur[i] + sum)*scale;
    dst[i] = res ? v + res[i] : v;
}

void ggml_cuda_op_ink2_sconv_res(ggml_backend_cuda_context & ctx, const ggml_tensor * x, const ggml_tensor * w, const ggml_tensor * cur,
        const ggml_tensor * res, ggml_tensor * dst, const float scale) {
    for (const ggml_tensor * t : { x, w, cur, (const ggml_tensor *) dst }) {
        GGML_ASSERT(t->type == GGML_TYPE_F32 && ggml_is_contiguous(t));
    }
    GGML_ASSERT(res == nullptr || (res->type == GGML_TYPE_F32 && ggml_is_contiguous(res) && ggml_nelements(res) == ggml_nelements(dst)));
    const int     n_conv     = (int) w->ne[0];
    const int     n_x        = (int) x->ne[0];
    const int     n_channels = (int) x->ne[1];
    const int64_t n_total    = ggml_nelements(dst);
    GGML_ASSERT(w->ne[1] == n_channels && n_total == (int64_t) n_channels*(n_x - n_conv + 1) && ggml_nelements(cur) == n_total);

    constexpr int block = 256;
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params((int) ((n_total + block - 1)/block), block, 0, ctx.stream());
    ggml_cuda_kernel_launch(ink2_sconv_res_f32, launch_params, (const float *) x->data, (const float *) w->data, (const float *) cur->data,
        res ? (const float *) res->data : nullptr, (float *) dst->data, n_conv, n_x, n_channels, n_total, scale);
}

// ---------------------------------------------------------------------------------------------------------------------
// SITU activation: dst = beta*tanh(gate/beta)*sigmoid(gate) * lb*tanh(up/lb); gate and up are row views (stride in floats)
// ---------------------------------------------------------------------------------------------------------------------

static __global__ void ink2_situ_f32(const float * gate, const float * up, float * dst, const int n_ff, const int64_t row_stride,
        const int64_t n_total, const float beta, const float lb) {
    ggml_cuda_pdl_lc();
    const int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n_total) {
        return;
    }
    const int64_t row = i / n_ff;
    const int     e   = i % n_ff;

    ggml_cuda_pdl_sync();

    const float g = gate[row*row_stride + e];
    const float u = up[row*row_stride + e];
    const float a = beta*tanhf(g/beta)/(1.0f + expf(-g));
    dst[i] = a*(lb*tanhf(u/lb));
}

void ggml_cuda_op_ink2_situ(ggml_backend_cuda_context & ctx, const ggml_tensor * gate, const ggml_tensor * up, ggml_tensor * dst,
        const float beta, const float lb) {
    GGML_ASSERT(gate->type == GGML_TYPE_F32 && up->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 && ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_are_same_shape(gate, dst) && ggml_are_same_shape(up, dst));
    GGML_ASSERT(gate->nb[0] == sizeof(float) && up->nb[0] == sizeof(float) && gate->nb[1] == up->nb[1] && gate->nb[1] % sizeof(float) == 0);
    GGML_ASSERT(gate->nb[2] == gate->nb[1]*gate->ne[1] && up->nb[2] == up->nb[1]*up->ne[1] && gate->ne[3] == 1);

    const int     n_ff    = (int) dst->ne[0];
    const int64_t n_total = ggml_nelements(dst);
    constexpr int block = 256;
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params((int) ((n_total + block - 1)/block), block, 0, ctx.stream());
    ggml_cuda_kernel_launch(ink2_situ_f32, launch_params, (const float *) gate->data, (const float *) up->data, (float *) dst->data,
        n_ff, (int64_t) (gate->nb[1]/sizeof(float)), n_total, beta, lb);
}

// ---------------------------------------------------------------------------------------------------------------------
// MoE weight normalisation: dst[k, t] = w[k, t] / (sum_k w[k, t] + eps) * scale, one warp per token
// ---------------------------------------------------------------------------------------------------------------------

static __global__ void ink2_moe_wnorm_f32(const float * w, float * dst, const int n_used, const float eps, const float scale) {
    ggml_cuda_pdl_lc();
    const int64_t t = blockIdx.x;
    const float * wt = w + t*n_used;

    ggml_cuda_pdl_sync();

    float sum = 0.0f;
    float v[4];
    int   n = 0;
    for (int k = threadIdx.x; k < n_used && n < 4; k += WARP_SIZE) {
        v[n] = wt[k];
        sum += v[n];
        n++;
    }
    sum = warp_reduce_sum(sum);
    // the lanes have read before the reduction returns: dst may alias w
    const float f = scale/(sum + eps);
    n = 0;
    for (int k = threadIdx.x; k < n_used && n < 4; k += WARP_SIZE) {
        dst[t*n_used + k] = v[n]*f;
        n++;
    }
}

void ggml_cuda_op_ink2_moe_wnorm(ggml_backend_cuda_context & ctx, const ggml_tensor * w, ggml_tensor * dst, const float eps, const float scale) {
    GGML_ASSERT(w->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 && ggml_is_contiguous(w) && ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_nelements(w) == ggml_nelements(dst));
    // [1, n_used, n_tokens] or [n_used, n_tokens]
    const int     n_used   = (int) (dst->ne[0] == 1 ? dst->ne[1] : dst->ne[0]);
    const int64_t n_tokens = ggml_nelements(dst)/n_used;
    GGML_ASSERT(n_used <= 4*WARP_SIZE);

    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params((int) n_tokens, WARP_SIZE, 0, ctx.stream());
    ggml_cuda_kernel_launch(ink2_moe_wnorm_f32, launch_params, (const float *) w->data, (float *) dst->data, n_used, eps, scale);
}

// ---------------------------------------------------------------------------------------------------------------------
// short conv input: x = concat(past {n_past, C}, cur^T {T, C}) along dim 0, and the new state = last n_past rows of x
// past, cur^T and the state are strided views (strides in floats); one thread per (row, channel) of x
// ---------------------------------------------------------------------------------------------------------------------

struct ink2_sconv_in_args {
    const float * past;
    const float * cur;
    float       * x;
    float       * state;
    int64_t past_s0, past_s1; // strides of past along (row, channel)
    int64_t cur_s0,  cur_s1;  // strides of cur^T along (token, channel)
    int64_t st_s0,   st_s1;   // strides of the state along (row, channel)
    int n_past;
    int n_tokens;
    int n_channels;
};

static __global__ void ink2_sconv_in_f32(const ink2_sconv_in_args a) {
    ggml_cuda_pdl_lc();
    const int     n_x     = a.n_past + a.n_tokens;
    const int64_t n_total = (int64_t) n_x*a.n_channels;
    const int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n_total) {
        return;
    }
    const int c = i / n_x;
    const int r = i % n_x;

    ggml_cuda_pdl_sync();

    const float v = r < a.n_past ? a.past[r*a.past_s0 + c*a.past_s1] : a.cur[(r - a.n_past)*a.cur_s0 + c*a.cur_s1];
    a.x[i] = v;
    if (r >= a.n_tokens) {
        // row r of x is row r - n_tokens of the new state
        a.state[(r - a.n_tokens)*a.st_s0 + c*a.st_s1] = v;
    }
}

void ggml_cuda_op_ink2_sconv_in(ggml_backend_cuda_context & ctx, const ggml_tensor * past, const ggml_tensor * cur_t, ggml_tensor * x,
        ggml_tensor * state) {
    for (const ggml_tensor * t : { past, cur_t, (const ggml_tensor *) x }) {
        GGML_ASSERT(t->type == GGML_TYPE_F32 && t->nb[0] % sizeof(float) == 0 && t->nb[1] % sizeof(float) == 0);
    }
    // the state cell is the flat copy of the last rows: element (row, channel) at row + n_past*channel
    GGML_ASSERT(state->type == GGML_TYPE_F32 && ggml_is_contiguous(state));
    GGML_ASSERT(ggml_is_contiguous(x) && x->ne[2] == 1 && x->ne[3] == 1);
    ink2_sconv_in_args a = {};
    a.past       = (const float *) past->data;
    a.cur        = (const float *) cur_t->data;
    a.x          = (float *) x->data;
    a.state      = (float *) state->data;
    a.past_s0    = past->nb[0]/sizeof(float);
    a.past_s1    = past->nb[1]/sizeof(float);
    a.cur_s0     = cur_t->nb[0]/sizeof(float);
    a.cur_s1     = cur_t->nb[1]/sizeof(float);
    a.st_s0      = 1;
    a.st_s1      = past->ne[0];
    a.n_past     = (int) past->ne[0];
    a.n_tokens   = (int) cur_t->ne[0];
    a.n_channels = (int) x->ne[1];
    GGML_ASSERT(x->ne[0] == a.n_past + a.n_tokens && past->ne[1] == a.n_channels && cur_t->ne[1] == a.n_channels);
    GGML_ASSERT(ggml_nelements(state) == (int64_t) a.n_past*a.n_channels);

    const int64_t n_total = (int64_t) (a.n_past + a.n_tokens)*a.n_channels;
    constexpr int block = 256;
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params((int) ((n_total + block - 1)/block), block, 0, ctx.stream());
    ggml_cuda_kernel_launch(ink2_sconv_in_f32, launch_params, a);
}

// ---------------------------------------------------------------------------------------------------------------------
// relative-position bias + mask, one thread per (key, token, head): 16 contiguous floats of the bias row of its distance and
// 16 contiguous floats of r; keys past the extent (most of a long context) only read the mask and the sink
// ---------------------------------------------------------------------------------------------------------------------

static __global__ void ink2_rel_bias_f16(const float * bias, const float * R, const float * sinks, const int32_t * rel_pos,
        const half * mask, half * dst, const int n_rel, const int extent, const int n_head, const int n_kv, const int n_tok) {
    ggml_cuda_pdl_lc();
    const int64_t idx = int64_t(blockIdx.x)*blockDim.x + threadIdx.x;
    if (idx >= int64_t(n_kv)*n_tok*n_head) {
        return;
    }
    const int k  = int(idx % n_kv);
    const int th = int(idx / n_kv); // t + n_tok*h
    const int t  = th % n_tok;
    const int h  = th / n_tok;

    ggml_cuda_pdl_sync();

    const int d = rel_pos[int64_t(t)*n_kv + k];
    float v = __half2float(mask[int64_t(t)*n_kv + k]) - sinks[t*n_head + h];
    if (d >= 0 && d < extent) {
        const float * b = bias + int64_t(d)*n_rel;
        const float * r = R + (int64_t(t)*n_head + h)*n_rel;
        float sum = 0.0f;
        for (int i = 0; i < n_rel; ++i) {
            sum += b[i]*r[i];
        }
        v += sum;
    }
    dst[idx] = __float2half(v);
}

void ggml_cuda_op_ink2_rel_bias(ggml_backend_cuda_context & ctx, const ggml_tensor * bias, const ggml_tensor * R, const ggml_tensor * sinks,
        const ggml_tensor * rel_pos, const ggml_tensor * mask, ggml_tensor * dst) {
    GGML_ASSERT(bias->type == GGML_TYPE_F32 && R->type == GGML_TYPE_F32 && sinks->type == GGML_TYPE_F32);
    GGML_ASSERT(rel_pos->type == GGML_TYPE_I32 && mask->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F16);
    GGML_ASSERT(ggml_is_contiguous(bias) && ggml_is_contiguous(R) && ggml_is_contiguous(sinks) && ggml_is_contiguous(rel_pos) &&
                ggml_is_contiguous(mask) && ggml_is_contiguous(dst));

    const int     n_rel  = bias->ne[0];
    const int     extent = bias->ne[1];
    const int     n_head = R->ne[1];
    const int     n_tok  = R->ne[2];
    const int64_t n_kv   = rel_pos->ne[0];
    const int64_t n      = n_kv*n_tok*n_head;
    GGML_ASSERT(R->ne[0] == n_rel && ggml_nelements(sinks) == int64_t(n_head)*n_tok && rel_pos->ne[1] == n_tok);
    GGML_ASSERT(ggml_nelements(mask) == n_kv*n_tok && ggml_nelements(dst) == n && n < (int64_t(1) << 31));

    constexpr int block = 256;
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params((int) ((n + block - 1)/block), block, 0, ctx.stream());
    ggml_cuda_kernel_launch(ink2_rel_bias_f16, launch_params, (const float *) bias->data, (const float *) R->data, (const float *) sinks->data,
        (const int32_t *) rel_pos->data, (const half *) mask->data, (half *) dst->data, n_rel, extent, n_head, (int) n_kv, n_tok);
}

// ---------------------------------------------------------------------------------------------------------------------
// Q8_0 mat-vec on F32 activations: one warp per row, each lane owns INK2_MMVF8_BYTES contiguous weight bytes per step
// ---------------------------------------------------------------------------------------------------------------------

#ifndef INK2_MMVF8_BYTES
#define INK2_MMVF8_BYTES 16
#endif
#ifndef INK2_MMVF8_WARPS
#define INK2_MMVF8_WARPS 1
#endif

template <int NT>
static __global__ void ink2_mmv_q8_0_f32(const block_q8_0 * x, const float * y, float * dst, const int nblocks, const int64_t stride_row,
        const int stride_y, const int stride_dst) {
    constexpr int nbytes  = INK2_MMVF8_BYTES;          // weight bytes per lane and step: 8 or 16
    constexpr int nwarps  = INK2_MMVF8_WARPS;
    constexpr int lanes_b = QK8_0/nbytes;              // lanes per block
    constexpr int nthr    = nwarps*WARP_SIZE;
    constexpr int step    = nthr/lanes_b;              // blocks per step

    ggml_cuda_pdl_lc();
    const int64_t row = blockIdx.x;
    const int     tid = threadIdx.y*WARP_SIZE + threadIdx.x;
    const int     q   = tid % lanes_b;                 // which part of the block
    const block_q8_0 * xr = x + row*stride_row;

    ggml_cuda_pdl_sync();

    float sum[NT];
#pragma unroll
    for (int t = 0; t < NT; ++t) {
        sum[t] = 0.0f;
    }

    for (int b = tid/lanes_b; b < nblocks; b += step) {
        const block_q8_0 * blk = xr + b;
        const float d = blk->d;
        const uint16_t * w16 = (const uint16_t *) (blk->qs + q*nbytes); // the quants start 2 bytes into the block
        const float * yb = y + b*QK8_0 + q*nbytes;
        int w[nbytes/4];
#pragma unroll
        for (int i = 0; i < nbytes/4; ++i) {
            w[i] = int(w16[2*i]) | (int(w16[2*i + 1]) << 16);
        }
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            const float * yt = yb + int64_t(t)*stride_y;
            float s = 0.0f;
#pragma unroll
            for (int i = 0; i < nbytes; ++i) {
                s += float(int8_t(w[i/4] >> (8*(i % 4))))*yt[i];
            }
            sum[t] += d*s;
        }
    }

    if constexpr (nwarps > 1) {
        __shared__ float part[nwarps - 1 > 0 ? nwarps - 1 : 1][NT][WARP_SIZE];
        if (threadIdx.y > 0) {
#pragma unroll
            for (int t = 0; t < NT; ++t) {
                part[threadIdx.y - 1][t][threadIdx.x] = sum[t];
            }
        }
        __syncthreads();
        if (threadIdx.y > 0) {
            return;
        }
#pragma unroll
        for (int t = 0; t < NT; ++t) {
#pragma unroll
            for (int l = 0; l < nwarps - 1; ++l) {
                sum[t] += part[l][t][threadIdx.x];
            }
        }
    }

#pragma unroll
    for (int t = 0; t < NT; ++t) {
        const float r = warp_reduce_sum(sum[t]);
        if (threadIdx.x == 0) {
            dst[int64_t(t)*stride_dst + row] = r;
        }
    }
}

// Several rows per warp (NR): the activations of a step (NBYTES floats per column and lane, float4 loads) are read once for the
// NR rows instead of once per row; with 4 columns the activations are 4 times the bytes of the weights, so the one-row kernel
// above spends most of its time re-reading them (+60 % per product from 1 to 4 columns). WPB warps per block, each its own rows.
template <int NT, int NR, int NBYTES>
static __device__ __forceinline__ void ink2_mmv_q8_0_f32_rows(const block_q8_0 * x, const float * y, float * dst, const int nblocks,
        const int64_t stride_row, const int stride_y, const int stride_dst, const int64_t row0) {
    constexpr int lanes_b = QK8_0/NBYTES; // lanes per block
    constexpr int step    = WARP_SIZE/lanes_b;

    const int lane = threadIdx.x;
    const int q    = lane % lanes_b;

    float sum[NR*NT];
#pragma unroll
    for (int k = 0; k < NR*NT; ++k) {
        sum[k] = 0.0f;
    }

    for (int b = lane/lanes_b; b < nblocks; b += step) {
        float yv[NT*NBYTES];
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            const float4 * y4 = (const float4 *) (y + (int64_t) t*stride_y + b*QK8_0 + q*NBYTES);
#pragma unroll
            for (int i = 0; i < NBYTES/4; ++i) {
                const float4 v = y4[i];
                yv[t*NBYTES + 4*i + 0] = v.x;
                yv[t*NBYTES + 4*i + 1] = v.y;
                yv[t*NBYTES + 4*i + 2] = v.z;
                yv[t*NBYTES + 4*i + 3] = v.w;
            }
        }
#pragma unroll
        for (int r = 0; r < NR; ++r) {
            const block_q8_0 * blk = x + (row0 + r)*stride_row + b;
            const float d = blk->d;
            const uint16_t * w16 = (const uint16_t *) (blk->qs + q*NBYTES); // the quants start 2 bytes into the block
            int w[NBYTES/4];
#pragma unroll
            for (int i = 0; i < NBYTES/4; ++i) {
                w[i] = int(w16[2*i]) | (int(w16[2*i + 1]) << 16);
            }
#pragma unroll
            for (int t = 0; t < NT; ++t) {
                float s = 0.0f;
#pragma unroll
                for (int i = 0; i < NBYTES; ++i) {
                    s += float(int8_t(w[i/4] >> (8*(i % 4))))*yv[t*NBYTES + i];
                }
                sum[r*NT + t] += d*s;
            }
        }
    }

#pragma unroll
    for (int r = 0; r < NR; ++r) {
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            const float v = warp_reduce_sum(sum[r*NT + t]);
            if (lane == 0) {
                dst[int64_t(t)*stride_dst + row0 + r] = v;
            }
        }
    }
}

template <int NT, int NR, int NBYTES>
static __global__ void ink2_mmv_q8_0_f32_mr(const block_q8_0 * x, const float * y, float * dst, const int nblocks, const int64_t stride_row,
        const int stride_y, const int stride_dst) {
    ggml_cuda_pdl_lc();
    const int64_t row0 = ((int64_t) blockIdx.x*blockDim.y + threadIdx.y)*NR;
    ggml_cuda_pdl_sync();
    ink2_mmv_q8_0_f32_rows<NT, NR, NBYTES>(x, y, dst, nblocks, stride_row, stride_y, stride_dst, row0);
}

// several Q8_0 matrices that multiply the same activations (Q, K, V, gate, r projections of a layer) in one launch: the
// blocks of matrix k are [blk_end[k - 1], blk_end[k]); the matrix is picked with constant indices (no dynamic indexing)
struct ink2_q8_group_args {
    const block_q8_0 * x[GGML_CUDA_INK2_Q8_GROUP_MAX];
    float            * dst[GGML_CUDA_INK2_Q8_GROUP_MAX];
    int64_t            stride_row[GGML_CUDA_INK2_Q8_GROUP_MAX];
    int                stride_dst[GGML_CUDA_INK2_Q8_GROUP_MAX];
    int                blk_end[GGML_CUDA_INK2_Q8_GROUP_MAX];
    int                n;
};

template <int NT, int NR, int NBYTES>
static __global__ void ink2_mmv_q8_0_f32_group(const ink2_q8_group_args a, const float * y, const int nblocks, const int stride_y) {
    ggml_cuda_pdl_lc();
    const block_q8_0 * x  = a.x[0];
    float            * d  = a.dst[0];
    int64_t            sr = a.stride_row[0];
    int                sd = a.stride_dst[0];
    int                b0 = 0;
#pragma unroll
    for (int k = 1; k < GGML_CUDA_INK2_Q8_GROUP_MAX; ++k) {
        if (k < a.n && (int) blockIdx.x >= a.blk_end[k - 1]) {
            x  = a.x[k];
            d  = a.dst[k];
            sr = a.stride_row[k];
            sd = a.stride_dst[k];
            b0 = a.blk_end[k - 1];
        }
    }
    const int64_t row0 = ((int64_t) ((int) blockIdx.x - b0)*blockDim.y + threadIdx.y)*NR;
    ggml_cuda_pdl_sync();
    ink2_mmv_q8_0_f32_rows<NT, NR, NBYTES>(x, y, d, nblocks, sr, stride_y, sd, row0);
}

template <int NT>
static void ink2_mmv_q8_0_f32_mr_launch(const int nr, const int wpb, const int64_t nrows, cudaStream_t stream, const block_q8_0 * x,
        const float * y, float * d, const int nblocks, const int64_t stride_row, const int stride_y, const int stride_dst) {
    constexpr int nbytes = NT <= 4 ? 16 : 8;
    const dim3 grid(nrows/(nr*wpb), 1, 1);
    const dim3 block(WARP_SIZE, wpb, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid, block, 0, stream);
    switch (nr) {
        case 1: ggml_cuda_kernel_launch(ink2_mmv_q8_0_f32_mr<NT, 1, nbytes>, launch_params, x, y, d, nblocks, stride_row, stride_y, stride_dst); break;
        case 2: ggml_cuda_kernel_launch(ink2_mmv_q8_0_f32_mr<NT, 2, nbytes>, launch_params, x, y, d, nblocks, stride_row, stride_y, stride_dst); break;
        case 4: ggml_cuda_kernel_launch(ink2_mmv_q8_0_f32_mr<NT, 4, nbytes>, launch_params, x, y, d, nblocks, stride_row, stride_y, stride_dst); break;
        default: GGML_ABORT("fatal error");
    }
}

template <int NT>
static void ink2_mmv_q8_0_f32_group_launch(const int nr, const int wpb, const int nblk, cudaStream_t stream, const ink2_q8_group_args & a,
        const float * y, const int nblocks, const int stride_y) {
    constexpr int nbytes = NT <= 4 ? 16 : 8;
    const dim3 grid(nblk, 1, 1);
    const dim3 block(WARP_SIZE, wpb, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid, block, 0, stream);
    switch (nr) {
        case 1: ggml_cuda_kernel_launch(ink2_mmv_q8_0_f32_group<NT, 1, nbytes>, launch_params, a, y, nblocks, stride_y); break;
        case 2: ggml_cuda_kernel_launch(ink2_mmv_q8_0_f32_group<NT, 2, nbytes>, launch_params, a, y, nblocks, stride_y); break;
        case 4: ggml_cuda_kernel_launch(ink2_mmv_q8_0_f32_group<NT, 4, nbytes>, launch_params, a, y, nblocks, stride_y); break;
        default: GGML_ABORT("fatal error");
    }
}

// rows per warp and warps per block of the multi-row kernel, per number of columns; 0 rows = the one-row kernel above.
// GGML_CUDA_INK2_Q8_NR="n1,n2,...,n8" and GGML_CUDA_INK2_Q8_WPB="w1,...,w8" override them (tuning).
static void ink2_mmv_q8_shape(const int ncols, int & nr, int & wpb) {
    // test-backend-ops perf (us per product, weights in the infinity cache), m 3072 x k 2304, columns 1/2/4/8:
    // one-row kernel 5.42/8.05/13.08/23.17, 2 rows per warp 4.86/6.25/9.76/15.98, 4 rows 5.86/7.50/11.77/19.22
    static int s_nr[9]  = { 0, 2, 2, 2, 2, 2, 2, 2, 2 };
    static int s_wpb[9] = { 1, 4, 4, 1, 1, 1, 1, 1, 1 };
    static bool init = false;
    if (!init) {
        init = true;
        for (int which = 0; which < 2; ++which) {
            const char * e = getenv(which ? "GGML_CUDA_INK2_Q8_WPB" : "GGML_CUDA_INK2_Q8_NR");
            int * arr = which ? s_wpb : s_nr;
            for (int k = 1; e && *e && k <= 8; ++k) {
                arr[k] = atoi(e);
                while (*e && *e != ',') {
                    ++e;
                }
                if (*e == ',') {
                    ++e;
                }
            }
        }
    }
    nr  = s_nr[ncols];
    wpb = std::max(1, s_wpb[ncols]);
}

bool ggml_cuda_ink2_mmv_q8_f32_supported(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst) {
    static const bool enabled = []() { const char * e = getenv("GGML_CUDA_INK2_MMV_F32"); return !(e && e[0] == '0'); }();
    return enabled && src0->type == GGML_TYPE_Q8_0 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
        src0->ne[2] == 1 && src0->ne[3] == 1 && src1->ne[2] == 1 && src1->ne[3] == 1 && src1->ne[1] >= 1 && src1->ne[1] <= 8 &&
        src0->ne[0] % QK8_0 == 0 && src1->ne[0] == src0->ne[0] && dst->ne[0] == src0->ne[1] && dst->ne[1] == src1->ne[1] &&
        ggml_is_contiguous(src0) && src1->nb[0] == sizeof(float) && src1->nb[1] % sizeof(float) == 0 &&
        dst->nb[0] == sizeof(float) && dst->nb[1] % sizeof(float) == 0 && src0->ne[1] < (int64_t(1) << 30);
}

void ggml_cuda_op_ink2_mmv_q8_f32(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(ggml_cuda_ink2_mmv_q8_f32_supported(src0, src1, dst));
    const int     nblocks    = src0->ne[0]/QK8_0;
    const int64_t stride_row = src0->nb[1]/sizeof(block_q8_0);
    const int     stride_y   = src1->nb[1]/sizeof(float);
    const int     stride_dst = dst->nb[1]/sizeof(float);
    const dim3 grid(src0->ne[1], 1, 1);
    const dim3 block(WARP_SIZE, INK2_MMVF8_WARPS, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid, block, 0, ctx.stream());
    const block_q8_0 * x = (const block_q8_0 *) src0->data;
    const float      * y = (const float *) src1->data;
    float            * d = (float *) dst->data;

    int nr  = 0;
    int wpb = 1;
    ink2_mmv_q8_shape((int) src1->ne[1], nr, wpb);
    const int64_t nrows = src0->ne[1];
    if (nr > 0 && nrows % (nr*wpb) == 0 && ((uintptr_t) y) % 16 == 0 && stride_y % 4 == 0) {
        switch (src1->ne[1]) {
            case 1: ink2_mmv_q8_0_f32_mr_launch<1>(nr, wpb, nrows, ctx.stream(), x, y, d, nblocks, stride_row, stride_y, stride_dst); break;
            case 2: ink2_mmv_q8_0_f32_mr_launch<2>(nr, wpb, nrows, ctx.stream(), x, y, d, nblocks, stride_row, stride_y, stride_dst); break;
            case 3: ink2_mmv_q8_0_f32_mr_launch<3>(nr, wpb, nrows, ctx.stream(), x, y, d, nblocks, stride_row, stride_y, stride_dst); break;
            case 4: ink2_mmv_q8_0_f32_mr_launch<4>(nr, wpb, nrows, ctx.stream(), x, y, d, nblocks, stride_row, stride_y, stride_dst); break;
            case 5: ink2_mmv_q8_0_f32_mr_launch<5>(nr, wpb, nrows, ctx.stream(), x, y, d, nblocks, stride_row, stride_y, stride_dst); break;
            case 6: ink2_mmv_q8_0_f32_mr_launch<6>(nr, wpb, nrows, ctx.stream(), x, y, d, nblocks, stride_row, stride_y, stride_dst); break;
            case 7: ink2_mmv_q8_0_f32_mr_launch<7>(nr, wpb, nrows, ctx.stream(), x, y, d, nblocks, stride_row, stride_y, stride_dst); break;
            case 8: ink2_mmv_q8_0_f32_mr_launch<8>(nr, wpb, nrows, ctx.stream(), x, y, d, nblocks, stride_row, stride_y, stride_dst); break;
            default: GGML_ABORT("fatal error");
        }
        return;
    }

    switch (src1->ne[1]) {
        case 1: ggml_cuda_kernel_launch(ink2_mmv_q8_0_f32<1>, launch_params, x, y, d, nblocks, stride_row, stride_y, stride_dst); break;
        case 2: ggml_cuda_kernel_launch(ink2_mmv_q8_0_f32<2>, launch_params, x, y, d, nblocks, stride_row, stride_y, stride_dst); break;
        case 3: ggml_cuda_kernel_launch(ink2_mmv_q8_0_f32<3>, launch_params, x, y, d, nblocks, stride_row, stride_y, stride_dst); break;
        case 4: ggml_cuda_kernel_launch(ink2_mmv_q8_0_f32<4>, launch_params, x, y, d, nblocks, stride_row, stride_y, stride_dst); break;
        case 5: ggml_cuda_kernel_launch(ink2_mmv_q8_0_f32<5>, launch_params, x, y, d, nblocks, stride_row, stride_y, stride_dst); break;
        case 6: ggml_cuda_kernel_launch(ink2_mmv_q8_0_f32<6>, launch_params, x, y, d, nblocks, stride_row, stride_y, stride_dst); break;
        case 7: ggml_cuda_kernel_launch(ink2_mmv_q8_0_f32<7>, launch_params, x, y, d, nblocks, stride_row, stride_y, stride_dst); break;
        case 8: ggml_cuda_kernel_launch(ink2_mmv_q8_0_f32<8>, launch_params, x, y, d, nblocks, stride_row, stride_y, stride_dst); break;
        default: GGML_ABORT("fatal error");
    }
}

bool ggml_cuda_ink2_mmv_q8_f32_group_ok(const ggml_tensor * a, const ggml_tensor * b) {
    return a->op == GGML_OP_MUL_MAT && b->op == GGML_OP_MUL_MAT && a->src[1] == b->src[1] &&
        ggml_cuda_ink2_mmv_q8_f32_supported(a->src[0], a->src[1], a) && ggml_cuda_ink2_mmv_q8_f32_supported(b->src[0], b->src[1], b) &&
        a->src[0]->ne[0] == b->src[0]->ne[0] && a->src[0]->ne[1] % 8 == 0 && b->src[0]->ne[1] % 8 == 0 &&
        ((uintptr_t) a->src[1]->data) % 16 == 0 && (a->src[1]->nb[1]/sizeof(float)) % 4 == 0;
}

void ggml_cuda_op_ink2_mmv_q8_f32_group(ggml_backend_cuda_context & ctx, ggml_tensor * const * dsts, const int n) {
    GGML_ASSERT(n >= 1 && n <= GGML_CUDA_INK2_Q8_GROUP_MAX);
    const ggml_tensor * src1 = dsts[0]->src[1];
    const int ncols = (int) src1->ne[1];
    int nr  = 0;
    int wpb = 1;
    ink2_mmv_q8_shape(ncols, nr, wpb);
    if (nr == 0) {
        nr  = 2;
        wpb = 4;
    }
    ink2_q8_group_args a = {};
    a.n = n;
    int nblk = 0;
    for (int k = 0; k < n; ++k) {
        const ggml_tensor * dst  = dsts[k];
        const ggml_tensor * src0 = dst->src[0];
        GGML_ASSERT(dst->src[1] == src1 && ggml_cuda_ink2_mmv_q8_f32_supported(src0, src1, dst));
        GGML_ASSERT(src0->ne[1] % (nr*wpb) == 0 && src0->ne[0] == dsts[0]->src[0]->ne[0]);
        a.x[k]          = (const block_q8_0 *) src0->data;
        a.dst[k]        = (float *) dst->data;
        a.stride_row[k] = src0->nb[1]/sizeof(block_q8_0);
        a.stride_dst[k] = (int) (dst->nb[1]/sizeof(float));
        nblk           += (int) (src0->ne[1]/(nr*wpb));
        a.blk_end[k]    = nblk;
    }
    const float * y        = (const float *) src1->data;
    const int     nblocks  = (int) (dsts[0]->src[0]->ne[0]/QK8_0);
    const int     stride_y = (int) (src1->nb[1]/sizeof(float));
    GGML_ASSERT(((uintptr_t) y) % 16 == 0 && stride_y % 4 == 0);
    switch (ncols) {
        case 1: ink2_mmv_q8_0_f32_group_launch<1>(nr, wpb, nblk, ctx.stream(), a, y, nblocks, stride_y); break;
        case 2: ink2_mmv_q8_0_f32_group_launch<2>(nr, wpb, nblk, ctx.stream(), a, y, nblocks, stride_y); break;
        case 3: ink2_mmv_q8_0_f32_group_launch<3>(nr, wpb, nblk, ctx.stream(), a, y, nblocks, stride_y); break;
        case 4: ink2_mmv_q8_0_f32_group_launch<4>(nr, wpb, nblk, ctx.stream(), a, y, nblocks, stride_y); break;
        case 5: ink2_mmv_q8_0_f32_group_launch<5>(nr, wpb, nblk, ctx.stream(), a, y, nblocks, stride_y); break;
        case 6: ink2_mmv_q8_0_f32_group_launch<6>(nr, wpb, nblk, ctx.stream(), a, y, nblocks, stride_y); break;
        case 7: ink2_mmv_q8_0_f32_group_launch<7>(nr, wpb, nblk, ctx.stream(), a, y, nblocks, stride_y); break;
        case 8: ink2_mmv_q8_0_f32_group_launch<8>(nr, wpb, nblk, ctx.stream(), a, y, nblocks, stride_y); break;
        default: GGML_ABORT("fatal error");
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// conv input + rollback snapshots: one thread per element of x, which also writes it to the (at most n_past) windows it is in
// ---------------------------------------------------------------------------------------------------------------------

struct ink2_sconv_in_rb_args {
    const float   * past;
    const float   * cur;
    float         * x;
    float         * slots;
    const int32_t * slot_order;
    int64_t past_s0, past_s1; // strides of past along (row, channel)
    int64_t cur_s0,  cur_s1;  // strides of cur^T along (token, channel)
    int64_t sl_s0, sl_s1, sl_s2; // strides of the slots along (row, slot, channel)
    int n_past;
    int n_tokens;
    int n_channels;
    int n_win;
};

static __global__ void ink2_sconv_in_rb_f32(const ink2_sconv_in_rb_args a) {
    ggml_cuda_pdl_lc();
    const int     n_x     = a.n_past + a.n_tokens;
    const int64_t n_total = (int64_t) n_x*a.n_channels;
    const int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n_total) {
        return;
    }
    const int c = i / n_x;
    const int r = i % n_x;

    ggml_cuda_pdl_sync();

    const float v = r < a.n_past ? a.past[r*a.past_s0 + c*a.past_s1] : a.cur[(r - a.n_past)*a.cur_s0 + c*a.cur_s1];
    a.x[i] = v;
    // window w covers rows base + w .. base + w + n_past - 1
    const int base = a.n_tokens - (a.n_win - 1);
    const int w0   = max(0, r - base - a.n_past + 1);
    const int w1   = min(a.n_win - 1, r - base);
    for (int w = w0; w <= w1; ++w) {
        a.slots[(r - base - w)*a.sl_s0 + a.slot_order[w]*a.sl_s1 + c*a.sl_s2] = v;
    }
}

void ggml_cuda_op_ink2_sconv_in_rb(ggml_backend_cuda_context & ctx, const ggml_tensor * past, const ggml_tensor * cur_t, ggml_tensor * x,
        const ggml_tensor * slots, const ggml_tensor * slot_order, const int n_win) {
    for (const ggml_tensor * t : { past, cur_t, (const ggml_tensor *) x, slots }) {
        GGML_ASSERT(t->type == GGML_TYPE_F32 && t->nb[0] % sizeof(float) == 0 && t->nb[1] % sizeof(float) == 0 && t->nb[2] % sizeof(float) == 0);
    }
    GGML_ASSERT(slot_order->type == GGML_TYPE_I32 && slot_order->nb[0] == sizeof(int32_t) && slot_order->ne[0] == n_win);
    GGML_ASSERT(ggml_is_contiguous(x) && x->ne[2] == 1 && x->ne[3] == 1 && slots->ne[3] == 1);
    ink2_sconv_in_rb_args a = {};
    a.past       = (const float *) past->data;
    a.cur        = (const float *) cur_t->data;
    a.x          = (float *) x->data;
    a.slots      = (float *) slots->data;
    a.slot_order = (const int32_t *) slot_order->data;
    a.past_s0    = past->nb[0]/sizeof(float);
    a.past_s1    = past->nb[1]/sizeof(float);
    a.cur_s0     = cur_t->nb[0]/sizeof(float);
    a.cur_s1     = cur_t->nb[1]/sizeof(float);
    a.sl_s0      = slots->nb[0]/sizeof(float);
    a.sl_s1      = slots->nb[1]/sizeof(float);
    a.sl_s2      = slots->nb[2]/sizeof(float);
    a.n_past     = (int) past->ne[0];
    a.n_tokens   = (int) cur_t->ne[0];
    a.n_channels = (int) x->ne[1];
    a.n_win      = n_win;
    GGML_ASSERT(x->ne[0] == a.n_past + a.n_tokens && past->ne[1] == a.n_channels && cur_t->ne[1] == a.n_channels);
    GGML_ASSERT(slots->ne[0] == a.n_past && slots->ne[2] == a.n_channels && n_win >= 1 && n_win <= a.n_tokens && n_win <= slots->ne[1]);

    const int64_t n_total = (int64_t) (a.n_past + a.n_tokens)*a.n_channels;
    constexpr int block = 256;
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params((int) ((n_total + block - 1)/block), block, 0, ctx.stream());
    ggml_cuda_kernel_launch(ink2_sconv_in_rb_f32, launch_params, a);
}

// ---------------------------------------------------------------------------------------------------------------------
// short convolution in one kernel: its input x = [past ; cur^T] with the state store (plain or with rollback snapshots), and
// the convolution with its residual, dst[c, t] = (cur[c, t] + sum_j x[t + j, c] w[j, c]) * scale (+ res[c, t]). x is not
// written (nothing else reads it). One thread per element of x; the threads of rows n_past.. also compute output t = r - n_past.
// dst must not overlap cur or past (other threads read them); it may be res (read and written by the same thread).
// ---------------------------------------------------------------------------------------------------------------------

struct ink2_sconv_full_args {
    const float   * past;
    const float   * cur_t;
    float         * slots;      // rollback form: windows of x into the slots
    const int32_t * slot_order;
    float         * state;      // plain form: the last n_past rows of x
    int64_t past_s0, past_s1;
    int64_t cur_s0,  cur_s1;
    int64_t sl_s0, sl_s1, sl_s2;
    int64_t st_s0, st_s1;
    int n_past;
    int n_tokens;
    int n_channels;
    int n_win;
    const float * w;
    const float * cur;
    const float * res;
    float       * dst;
    float         scale;
};

static __global__ void ink2_sconv_full_f32(const ink2_sconv_full_args a) {
    ggml_cuda_pdl_lc();
    const int     n_x     = a.n_past + a.n_tokens;
    const int64_t n_total = (int64_t) n_x*a.n_channels;
    const int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n_total) {
        return;
    }
    // channel fastest: the reads of cur^T and the writes of dst are coalesced (with the row fastest, a 512-token batch read
    // and wrote one token row per thread and the prompt reading lost 2.5 %)
    const int c = i % a.n_channels;
    const int r = i / a.n_channels;

    ggml_cuda_pdl_sync();

    const auto xv = [&](const int row) {
        return row < a.n_past ? a.past[row*a.past_s0 + c*a.past_s1] : a.cur_t[(row - a.n_past)*a.cur_s0 + c*a.cur_s1];
    };
    const float v = xv(r);
    if (a.slots) {
        const int base = a.n_tokens - (a.n_win - 1);
        const int w0   = max(0, r - base - a.n_past + 1);
        const int w1   = min(a.n_win - 1, r - base);
        for (int w = w0; w <= w1; ++w) {
            a.slots[(r - base - w)*a.sl_s0 + a.slot_order[w]*a.sl_s1 + c*a.sl_s2] = v;
        }
    } else if (r >= a.n_tokens) {
        a.state[(r - a.n_tokens)*a.st_s0 + c*a.st_s1] = v;
    }
    if (r >= a.n_past) {
        const int     t  = r - a.n_past;
        const int     nc = a.n_past + 1;
        const float * ws = a.w + (int64_t) c*nc;
        float sum = 0.0f;
        for (int j = 0; j < nc; ++j) {
            sum += xv(t + j)*ws[j];
        }
        const int64_t o = (int64_t) t*a.n_channels + c;
        const float   y = (a.cur[o] + sum)*a.scale;
        a.dst[o] = a.res ? y + a.res[o] : y;
    }
}

void ggml_cuda_op_ink2_sconv_full(ggml_backend_cuda_context & ctx, const ggml_tensor * past, const ggml_tensor * cur_t,
        const ggml_tensor * slots, const ggml_tensor * slot_order, const int n_win, ggml_tensor * state,
        const ggml_tensor * w, const ggml_tensor * cur, const ggml_tensor * res, ggml_tensor * dst, const float scale) {
    GGML_ASSERT((slots != nullptr) != (state != nullptr));
    ink2_sconv_full_args a = {};
    a.past       = (const float *) past->data;
    a.cur_t      = (const float *) cur_t->data;
    a.past_s0    = past->nb[0]/sizeof(float);
    a.past_s1    = past->nb[1]/sizeof(float);
    a.cur_s0     = cur_t->nb[0]/sizeof(float);
    a.cur_s1     = cur_t->nb[1]/sizeof(float);
    a.n_past     = (int) past->ne[0];
    a.n_tokens   = (int) cur_t->ne[0];
    a.n_channels = (int) cur_t->ne[1];
    if (slots) {
        GGML_ASSERT(slot_order->type == GGML_TYPE_I32 && slot_order->nb[0] == sizeof(int32_t) && slot_order->ne[0] == n_win);
        GGML_ASSERT(slots->ne[0] == a.n_past && slots->ne[2] == a.n_channels && n_win >= 1 && n_win <= a.n_tokens && n_win <= slots->ne[1]);
        a.slots      = (float *) slots->data;
        a.slot_order = (const int32_t *) slot_order->data;
        a.sl_s0      = slots->nb[0]/sizeof(float);
        a.sl_s1      = slots->nb[1]/sizeof(float);
        a.sl_s2      = slots->nb[2]/sizeof(float);
        a.n_win      = n_win;
    } else {
        GGML_ASSERT(state->type == GGML_TYPE_F32 && ggml_is_contiguous(state) && ggml_nelements(state) == (int64_t) a.n_past*a.n_channels);
        a.state = (float *) state->data;
        a.st_s0 = 1;
        a.st_s1 = a.n_past;
    }
    for (const ggml_tensor * t : { w, cur, (const ggml_tensor *) dst }) {
        GGML_ASSERT(t->type == GGML_TYPE_F32 && ggml_is_contiguous(t));
    }
    GGML_ASSERT(w->ne[0] == a.n_past + 1 && w->ne[1] == a.n_channels);
    GGML_ASSERT(ggml_nelements(cur) == (int64_t) a.n_channels*a.n_tokens && ggml_nelements(dst) == ggml_nelements(cur));
    GGML_ASSERT(res == nullptr || (res->type == GGML_TYPE_F32 && ggml_is_contiguous(res) && ggml_nelements(res) == ggml_nelements(dst)));
    a.w     = (const float *) w->data;
    a.cur   = (const float *) cur->data;
    a.res   = res ? (const float *) res->data : nullptr;
    a.dst   = (float *) dst->data;
    a.scale = scale;

    const int64_t n_total = (int64_t) (a.n_past + a.n_tokens)*a.n_channels;
    constexpr int block = 256;
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params((int) ((n_total + block - 1)/block), block, 0, ctx.stream());
    ggml_cuda_kernel_launch(ink2_sconv_full_f32, launch_params, a);
}

// ---------------------------------------------------------------------------------------------------------------------
// dynamic short convolution of the full-attention layers (K and V): static taps, input-dependent taps per head and residual
// dst[c, t] = (cur[c, t] + (conv + sum_lag x[t + n_conv - 1 - lag, c] * D[lag, h(c), t])) * scale, conv = sum_j x[t + j, c] w[j, c]
// x {n_x, n_channels} (n_x = n_conv - 1 + n_tokens), w {n_conv, n_channels}, D {n_conv, n_heads, n_tokens}; one thread per element.
// The sums are added in the order of the generic graph (conv, then lag 0, 1, ...).
// ---------------------------------------------------------------------------------------------------------------------

static __global__ void ink2_dconv_f32(const float * x, const float * w, const float * D, const float * cur, float * dst,
        const int n_conv, const int n_x, const int n_channels, const int ch_per_head, const int n_heads, const int64_t n_total,
        const float scale) {
    ggml_cuda_pdl_lc();
    const int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n_total) {
        return;
    }
    const int64_t t = i / n_channels;
    const int     c = i % n_channels;
    const int     h = c / ch_per_head;

    const float * xs = x + (int64_t) c*n_x + t;
    const float * ws = w + (int64_t) c*n_conv;
    const float * Dt = D + (int64_t) n_conv*(h + (int64_t) n_heads*t);

    ggml_cuda_pdl_sync();

    float conv = 0.0f;
    for (int j = 0; j < n_conv; ++j) {
        conv += xs[j]*ws[j];
    }
    float v = conv;
    for (int lag = 0; lag < n_conv; ++lag) {
        v += xs[n_conv - 1 - lag]*Dt[lag];
    }
    // same index read and written: dst may alias cur
    dst[i] = (cur[i] + v)*scale;
}

void ggml_cuda_op_ink2_dconv(ggml_backend_cuda_context & ctx, const ggml_tensor * x, const ggml_tensor * w, const ggml_tensor * D,
        const ggml_tensor * cur, ggml_tensor * dst, const int n_heads, const float scale) {
    for (const ggml_tensor * t : { x, w, D, cur, (const ggml_tensor *) dst }) {
        GGML_ASSERT(t->type == GGML_TYPE_F32 && ggml_is_contiguous(t));
    }
    const int     n_conv     = (int) w->ne[0];
    const int     n_x        = (int) x->ne[0];
    const int     n_channels = (int) x->ne[1];
    const int64_t n_tokens   = n_x - n_conv + 1;
    const int64_t n_total    = ggml_nelements(dst);
    GGML_ASSERT(w->ne[1] == n_channels && n_total == n_channels*n_tokens && ggml_nelements(cur) == n_total);
    GGML_ASSERT(n_heads >= 1 && n_channels % n_heads == 0 && ggml_nelements(D) == (int64_t) n_conv*n_heads*n_tokens);

    constexpr int block = 256;
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params((int) ((n_total + block - 1)/block), block, 0, ctx.stream());
    ggml_cuda_kernel_launch(ink2_dconv_f32, launch_params, (const float *) x->data, (const float *) w->data, (const float *) D->data,
        (const float *) cur->data, (float *) dst->data, n_conv, n_x, n_channels, n_channels/n_heads, n_heads, n_total, scale);
}

// ---------------------------------------------------------------------------------------------------------------------
// loop gate tail of a prompt batch (the products D, F and G stay generic GEMMs): dst = (1 - tanh(f)) * cur + tanh(g) * msg
// ---------------------------------------------------------------------------------------------------------------------

static __global__ void ink2_gate_tail_f32(const float * f, const float * g, const float * cur, const float * msg, float * dst, const int64_t n) {
    ggml_cuda_pdl_lc();
    const int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    ggml_cuda_pdl_sync();
    // same index read and written: dst may be any of the inputs
    dst[i] = (1.0f - tanhf(f[i]))*cur[i] + tanhf(g[i])*msg[i];
}

void ggml_cuda_op_ink2_gate_tail(ggml_backend_cuda_context & ctx, const ggml_tensor * f, const ggml_tensor * g, const ggml_tensor * cur,
        const ggml_tensor * msg, ggml_tensor * dst) {
    for (const ggml_tensor * t : { f, g, cur, msg, (const ggml_tensor *) dst }) {
        GGML_ASSERT(t->type == GGML_TYPE_F32 && ggml_is_contiguous(t) && ggml_nelements(t) == ggml_nelements(dst));
    }
    const int64_t n = ggml_nelements(dst);
    constexpr int block = 256;
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params((int) ((n + block - 1)/block), block, 0, ctx.stream());
    ggml_cuda_kernel_launch(ink2_gate_tail_f32, launch_params, (const float *) f->data, (const float *) g->data, (const float *) cur->data,
        (const float *) msg->data, (float *) dst->data, n);
}

// ---------------------------------------------------------------------------------------------------------------------
// first index of the maximum of one row (the token selection of the DSpark drafting graph): one block, each thread scans its
// elements in increasing order keeping the first maximum, then the block keeps the largest value and, on a tie, the lowest
// index. Elements equal to the maximum (including infinities) are what the generic chain selects; rows hold no NaN.
// ---------------------------------------------------------------------------------------------------------------------

#define INK2_FIRST_MAX_THREADS 1024

static __device__ __forceinline__ void ink2_first_max_pick(float & best, int & bi, const float v, const int k) {
    if (v > best || (v == best && k < bi)) {
        best = v;
        bi   = k;
    }
}

static __global__ void __launch_bounds__(INK2_FIRST_MAX_THREADS) ink2_first_max_f32(const float * __restrict__ x, int32_t * __restrict__ dst, const int n) {
    float best = -INFINITY;
    int   bi   = INT_MAX;
    const int n4 = (((uintptr_t) x & 15) == 0) ? n/4 : 0;
    for (int k4 = threadIdx.x; k4 < n4; k4 += INK2_FIRST_MAX_THREADS) {
        const float4 v = ((const float4 *) x)[k4];
        ink2_first_max_pick(best, bi, v.x, 4*k4 + 0);
        ink2_first_max_pick(best, bi, v.y, 4*k4 + 1);
        ink2_first_max_pick(best, bi, v.z, 4*k4 + 2);
        ink2_first_max_pick(best, bi, v.w, 4*k4 + 3);
    }
    for (int k = 4*n4 + threadIdx.x; k < n; k += INK2_FIRST_MAX_THREADS) {
        ink2_first_max_pick(best, bi, x[k], k);
    }
#pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
        const float v = ggml_cuda_shfl_xor(best, offset, WARP_SIZE);
        const int   k = ggml_cuda_shfl_xor(bi, offset, WARP_SIZE);
        ink2_first_max_pick(best, bi, v, k);
    }
    constexpr int n_warps = INK2_FIRST_MAX_THREADS/WARP_SIZE;
    __shared__ float s_best[n_warps];
    __shared__ int   s_bi  [n_warps];
    const int lane = threadIdx.x % WARP_SIZE;
    const int warp = threadIdx.x / WARP_SIZE;
    if (lane == 0) {
        s_best[warp] = best;
        s_bi  [warp] = bi;
    }
    __syncthreads();
    if (warp == 0) {
        best = lane < n_warps ? s_best[lane] : -INFINITY;
        bi   = lane < n_warps ? s_bi  [lane] : INT_MAX;
#pragma unroll
        for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
            const float v = ggml_cuda_shfl_xor(best, offset, WARP_SIZE);
            const int   k = ggml_cuda_shfl_xor(bi, offset, WARP_SIZE);
            ink2_first_max_pick(best, bi, v, k);
        }
        if (lane == 0) {
            // a row of n -inf values: every element is "the maximum", the first one is 0
            dst[0] = bi == INT_MAX ? 0 : bi;
        }
    }
}

void ggml_cuda_op_ink2_first_max(ggml_backend_cuda_context & ctx, const ggml_tensor * row, ggml_tensor * dst) {
    GGML_ASSERT(row->type == GGML_TYPE_F32 && row->nb[0] == sizeof(float) && ggml_nrows(row) == 1 && row->ne[0] < INT_MAX);
    GGML_ASSERT(dst->type == GGML_TYPE_I32 && ggml_nelements(dst) == 1);
    static_assert(INK2_FIRST_MAX_THREADS/WARP_SIZE <= WARP_SIZE, "the second reduction is one warp");
    ink2_first_max_f32<<<1, INK2_FIRST_MAX_THREADS, 0, ctx.stream()>>>((const float *) row->data, (int32_t *) dst->data, (int) row->ne[0]);
}

// ---------------------------------------------------------------------------------------------------------------------
// NVFP4 expert products of prompt batches on FP16 WMMA (RDNA3)
// ---------------------------------------------------------------------------------------------------------------------
// MMQ multiplies the int8 E2M1 values with q8_1 activations by 16-deep int8 WMMA and rescales every 16x16 result (one NVFP4
// scale per 16 weights): on RDNA3 the rescale (8 converts, 8 multiplies and 8 FMAs per lane) runs on the same SIMD as the WMMA
// and costs about as much. Here a tile of 128 weight rows x 64 tokens decodes its weights to FP16 in shared memory once, the
// scale folded in (exact: E2M1 x UE4M3 fits FP16), and accumulates 16-deep FP16 WMMA in FP32 without any rescale. The
// activations are rounded to FP16 (more precise than q8_1). Tiles come from the expert sort of mmid (tile_spec, 64 columns).

#include "mmid.cuh"

#define INK2_MOE_TM   128               // weight rows per block
#define INK2_MOE_TN   64                // tokens (expert slots) per block
#define INK2_MOE_TK   64                // k per step = one NVFP4 block of a row
#define INK2_MOE_LD   (INK2_MOE_TK + 8) // padded row of the shared tiles, in halves
#define INK2_MOE_LDC  (INK2_MOE_TM + 4) // padded row of the staged result, in floats
#define INK2_MOE_NT   256               // threads: 8 warps, 4 along the rows x 2 along the tokens, 32 x 32 each

static __global__ void ink2_f32_to_f16(const float * __restrict__ x, half * __restrict__ y, const int64_t n8) {
    const int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n8) {
        return;
    }
    const float4 a = ((const float4 *) x)[2*i + 0];
    const float4 b = ((const float4 *) x)[2*i + 1];
    half2 h[4] = { __floats2half2_rn(a.x, a.y), __floats2half2_rn(a.z, a.w), __floats2half2_rn(b.x, b.y), __floats2half2_rn(b.z, b.w) };
    ((uint4 *) y)[i] = *(const uint4 *) h;
}

#if defined(RDNA3)
typedef _Float16 ink2_h16 __attribute__((ext_vector_type(16)));
typedef float    ink2_f8  __attribute__((ext_vector_type(8)));

// 4 E2M1 codes (one per byte, low nibble; bits 4..7 ignored) -> 2 half2 bit patterns of twice the E2M1 value (the integers of
// kvalues_fp4), by byte permutation from a table of the high bytes of 0, 1, 2, 3, 4, 6, 8, 12 in FP16
static __device__ __forceinline__ void ink2_e2m1x4_to_h2x2(const uint32_t q, uint32_t & h01, uint32_t & h23) {
    const uint32_t L0 = 0x42403C00u, L1 = 0x4A484644u;
    const uint32_t n = q & 0x07070707u, s = q & 0x08080808u;
    const uint32_t sel01 = 0x000C000Cu | ((n & 0xFFu) << 8) | ((n & 0xFF00u) << 16);
    const uint32_t sel23 = 0x000C000Cu | ((n >> 8) & 0xFF00u) | (n & 0xFF000000u);
    h01 = __builtin_amdgcn_perm(L1, L0, sel01) | ((s & 0x08u) << 12) | ((s & 0x0800u) << 20);
    h23 = __builtin_amdgcn_perm(L1, L0, sel23) | ((s & 0x080000u) >> 4) | ((s & 0x08000000u) << 4);
}
#endif // defined(RDNA3)

// SITU activation of one element, the formula of ink2_situ_f32
static __device__ __forceinline__ float ink2_situ1(const float g, const float u, const float beta, const float lb) {
    return beta*tanhf(g/beta)/(1.0f + expf(-g))*(lb*tanhf(u/lb));
}

// wtype: NVFP4, Q8_0 or Q5_K weights (each decoded to FP16 in shared memory).
// situ: the merged gate_up projection [gate rows 0..n_ff-1 ; up rows n_ff..2 n_ff-1] is not written; a block computes the gate
// rows m0..m0+TM/2-1 and the matching up rows, and writes situ(gate, up) for its TM/2 rows (dst rows of n_ff = s1 floats)
// lds1: one shared-memory buffer instead of two (27 KB instead of 54 KB: twice the blocks per WGP), at the cost of a second
// barrier per K step; the result is then staged in two passes of TN/2 tokens
template <ggml_type wtype, bool situ, bool lds1>
static __global__ void __launch_bounds__(INK2_MOE_NT, 2) ink2_moe_f16(
        const char * __restrict__ W, const half * __restrict__ X, const int32_t * __restrict__ ids_src1,
        const int32_t * __restrict__ ids_dst, const ggml_cuda_mm_id_tile_spec * __restrict__ tile_spec, float * __restrict__ dst,
        const float * __restrict__ x_scale, const int64_t nb01, const int64_t nb02, const int64_t s1, const int K,
        const int n_ff, const float beta, const float lb) {
#if defined(RDNA3)
    constexpr int TM = INK2_MOE_TM, TN = INK2_MOE_TN, TK = INK2_MOE_TK, LD = INK2_MOE_LD, LDC = INK2_MOE_LDC;
    constexpr int XCH = TN*TK/8/INK2_MOE_NT; // 16-byte chunks of the token tile per thread
    constexpr int NB  = lds1 ? 1 : 2;        // shared-memory buffers
    constexpr int TS  = lds1 ? TN/2 : TN;    // tokens per staging pass of the result
    __shared__ __align__(16) _Float16 smem[NB*TM*LD + NB*TN*LD];
    static_assert(TS*LDC*sizeof(float) <= sizeof(smem), "staged result does not fit");
    // buffer b: weights at smem + b*TM*LD, tokens at smem + NB*TM*LD + b*TN*LD (no pointer arrays: they become static initializers)

    const ggml_cuda_mm_id_tile_spec spec = tile_spec[blockIdx.y];
    const int n = spec.col_high - spec.col_low;
    if (n <= 0) {
        return;
    }
    const int e    = spec.expert;
    const int m0   = blockIdx.x*(situ ? TM/2 : TM);
    const int tid  = threadIdx.x;
    const int lane = tid % 32;
    const int warp = tid / 32;
    const int nkb  = K / TK;

    // weights: 2 threads per row, 2 sub-blocks (32 values) each; tokens: XCH chunks of 8 halves per thread
    const int wr = tid / 2, wh = tid % 2;
    const int64_t grow = situ ? (wr < TM/2 ? m0 + wr : n_ff + m0 + wr - TM/2) : m0 + wr;
    const char * wrow = W + e*nb02 + grow*nb01;
    const half * xrow[XCH];
    int          xkc[XCH], xt[XCH];
#pragma unroll
    for (int i = 0; i < XCH; ++i) {
        const int c = tid + i*INK2_MOE_NT;
        xt[i]  = c / (TK/8);
        xkc[i] = c % (TK/8);
        xrow[i] = xt[i] < n ? X + (int64_t) ids_src1[spec.col_low + xt[i]]*K : nullptr;
    }

    // raw weight words of one K step for this thread: NVFP4 5 (scales + 16 qs bytes), Q8_0 9 (the 34-byte block of 32 values
    // and its neighbour's bytes, 4-byte aligned), Q5_K 20 (header, qh, the 32 qs bytes of the step)
    constexpr int NW = wtype == GGML_TYPE_NVFP4 ? 5 : (wtype == GGML_TYPE_Q8_0 ? 9 : 20);
    uint32_t wreg[NW];
    int      wstep = 0; // Q5_K: quarter of the super-block of the loaded step
    uint4    xr[XCH];
    const auto gload = [&](const int kb) {
        if constexpr (wtype == GGML_TYPE_NVFP4) {
            const uint32_t * b = (const uint32_t *) (wrow + (int64_t) kb*sizeof(block_nvfp4));
            wreg[4] = b[0];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                wreg[i] = b[1 + 4*wh + i];
            }
        } else if constexpr (wtype == GGML_TYPE_Q8_0) {
            // two blocks per step: thread 0 reads bytes 0..35 (d0, qs0, d1), thread 1 bytes 32..67 (.., d1, qs1)
            const uint32_t * b = (const uint32_t *) (wrow + (int64_t) kb*(2*sizeof(block_q8_0)) + 32*wh);
#pragma unroll
            for (int i = 0; i < 9; ++i) {
                wreg[i] = b[i];
            }
        } else {
            // a step is a quarter of a 256-value super-block: d, dmin, scales (16 bytes), qh (32), qs bytes 32 j .. 32 j + 31
            const char * b = wrow + (int64_t) (kb/4)*sizeof(block_q5_K);
            wstep = kb % 4;
            const uint4 u0 = *(const uint4 *) b;
            const uint4 u1 = *(const uint4 *) (b + 16);
            const uint4 u2 = *(const uint4 *) (b + 32);
            const uint4 u3 = *(const uint4 *) (b + 48 + 32*wstep);
            const uint4 u4 = *(const uint4 *) (b + 64 + 32*wstep);
            wreg[ 0] = u0.x; wreg[ 1] = u0.y; wreg[ 2] = u0.z; wreg[ 3] = u0.w;
            wreg[ 4] = u1.x; wreg[ 5] = u1.y; wreg[ 6] = u1.z; wreg[ 7] = u1.w;
            wreg[ 8] = u2.x; wreg[ 9] = u2.y; wreg[10] = u2.z; wreg[11] = u2.w;
            wreg[12] = u3.x; wreg[13] = u3.y; wreg[14] = u3.z; wreg[15] = u3.w;
            wreg[16] = u4.x; wreg[17] = u4.y; wreg[18] = u4.z; wreg[19] = u4.w;
        }
#pragma unroll
        for (int i = 0; i < XCH; ++i) {
            xr[i] = xrow[i] ? *(const uint4 *) (xrow[i] + kb*TK + xkc[i]*8) : make_uint4(0, 0, 0, 0);
        }
    };
    // 4 bytes -> 2 half2 bit patterns of 1024 + byte (exact), by byte permutation with the high byte 0x64
    const auto b4_to_h2x2 = [](const uint32_t u, half2 & h01, half2 & h23) {
        const uint32_t a = __builtin_amdgcn_perm(0x64646464u, u, 0x04010400u);
        const uint32_t b = __builtin_amdgcn_perm(0x64646464u, u, 0x04030402u);
        memcpy(&h01, &a, sizeof(a));
        memcpy(&h23, &b, sizeof(b));
    };
    const auto sstore = [&](const int buf) {
        if constexpr (wtype == GGML_TYPE_NVFP4) {
            // sub-block s = qs bytes 8 s .. 8 s + 7: low nibbles are values 16 s + 0..7, high nibbles 16 s + 8..15
#pragma unroll
            for (int ss = 0; ss < 2; ++ss) {
                const int s = 2*wh + ss;
                const half2 d2 = __float2half2_rn(ggml_cuda_ue4m3_to_fp32((wreg[4] >> (8*s)) & 0xFF));
                uint32_t o[8];
#pragma unroll
                for (int w = 0; w < 2; ++w) {
                    const uint32_t q = wreg[2*ss + w];
                    uint32_t h[4];
                    ink2_e2m1x4_to_h2x2(q,      h[0], h[1]);
                    ink2_e2m1x4_to_h2x2(q >> 4, h[2], h[3]);
#pragma unroll
                    for (int u = 0; u < 4; ++u) {
                        half2 v;
                        memcpy(&v, &h[u], sizeof(v));
                        v = __hmul2(v, d2);
                        memcpy(&h[u], &v, sizeof(v));
                    }
                    o[2*w + 0]     = h[0];
                    o[2*w + 1]     = h[1];
                    o[4 + 2*w + 0] = h[2];
                    o[4 + 2*w + 1] = h[3];
                }
                _Float16 * p = smem + buf*(TM*LD) + wr*LD + 16*s;
                *(uint4 *) (p + 0) = make_uint4(o[0], o[1], o[2], o[3]);
                *(uint4 *) (p + 8) = make_uint4(o[4], o[5], o[6], o[7]);
            }
        } else {
            // 32 values k = 32 wh .. 32 wh + 31 of the step: Q8_0 x*d (one rounding: x is exact in FP16), Q5_K q*(d sc) - dmin m
            // (one FMA rounding; d sc and dmin m rounded to FP16 once per 32 values)
            half2 mul2, add2 = __float2half2_rn(0.0f);
            uint32_t q[8];
            if constexpr (wtype == GGML_TYPE_Q8_0) {
                const uint16_t dbits = (uint16_t) (wh ? wreg[0] >> 16 : wreg[0] & 0xFFFF);
                half d;
                memcpy(&d, &dbits, sizeof(d));
                mul2 = __half2half2(d);
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    // int8 -> unsigned byte + 128
                    q[i] = (wh ? wreg[i + 1] : __builtin_amdgcn_alignbyte(wreg[i + 1], wreg[i], 2)) ^ 0x80808080u;
                }
            } else {
                const int is = 2*wstep + wh; // sub-block of the super-block
                // 6-bit scale and min of sub-block is (get_scale_min_k4) from the scale bytes s[0..11] = wreg[1..3]: every byte it
                // reads is byte is % 4 of one of the three words (no indexed register array: it would go to scratch memory)
                const int      sh = 8*(is & 3);
                const uint32_t b0 = (wreg[1] >> sh) & 0xFFu, b1 = (wreg[2] >> sh) & 0xFFu, b2 = (wreg[3] >> sh) & 0xFFu;
                const uint32_t sc = is < 4 ? b0 & 63 : (b2 & 0xF) | ((b0 >> 6) << 4);
                const uint32_t mn = is < 4 ? b1 & 63 : (b2 >> 4)  | ((b1 >> 6) << 4);
                const uint16_t dbits = (uint16_t) (wreg[0] & 0xFFFF), mbits = (uint16_t) (wreg[0] >> 16);
                half d, dmin;
                memcpy(&d, &dbits, sizeof(d));
                memcpy(&dmin, &mbits, sizeof(dmin));
                mul2 = __float2half2_rn(__half2float(d)*(float) sc);
                add2 = __float2half2_rn(-__half2float(dmin)*(float) mn);
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    // low (wh 0) or high (wh 1) nibbles of qs, bit is of qh moved to bit 4 of each byte
                    const uint32_t nib = (wreg[12 + i] >> (4*wh)) & 0x0F0F0F0Fu;
                    const uint32_t hb  = __builtin_amdgcn_alignbit(wreg[4 + i], wreg[4 + i], (is - 4) & 31) & 0x10101010u;
                    q[i] = nib | hb;
                }
            }
            const half2 off2 = __float2half2_rn(wtype == GGML_TYPE_Q8_0 ? 1152.0f : 1024.0f);
            uint32_t o[16];
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                half2 h01, h23;
                b4_to_h2x2(q[i], h01, h23);
                h01 = __hsub2(h01, off2);
                h23 = __hsub2(h23, off2);
                if constexpr (wtype == GGML_TYPE_Q8_0) {
                    h01 = __hmul2(h01, mul2);
                    h23 = __hmul2(h23, mul2);
                } else {
                    h01 = __hfma2(h01, mul2, add2);
                    h23 = __hfma2(h23, mul2, add2);
                }
                memcpy(&o[2*i + 0], &h01, sizeof(h01));
                memcpy(&o[2*i + 1], &h23, sizeof(h23));
            }
            _Float16 * p = smem + buf*(TM*LD) + wr*LD + 32*wh;
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                *(uint4 *) (p + 8*c) = make_uint4(o[4*c + 0], o[4*c + 1], o[4*c + 2], o[4*c + 3]);
            }
        }
#pragma unroll
        for (int i = 0; i < XCH; ++i) {
            *(uint4 *) (smem + NB*TM*LD + buf*(TN*LD) + xt[i]*LD + xkc[i]*8) = xr[i];
        }
    };

    const int wm = (warp % 4)*32, wn = (warp / 4)*32;
    ink2_f8 acc[2][2];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            acc[i][j] = ink2_f8{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        }
    }

    gload(0);
    sstore(0);
    __syncthreads();
    for (int kb = 0; kb < nkb; ++kb) {
        const int buf = lds1 ? 0 : kb & 1;
        if (kb + 1 < nkb) {
            gload(kb + 1);
        }
#pragma unroll
        for (int k0 = 0; k0 < TK; k0 += 16) {
            ink2_h16 a[2], b[2];
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                a[i] = *(const ink2_h16 *) (smem + buf*(TM*LD) + (wm + i*16 + lane % 16)*LD + k0);
                b[i] = *(const ink2_h16 *) (smem + NB*TM*LD + buf*(TN*LD) + (wn + i*16 + lane % 16)*LD + k0);
            }
            // the last tile of an expert is often partly empty: skip the 16-token sub-tiles without tokens (uniform per warp)
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                if (wn + j*16 < n) {
#pragma unroll
                    for (int i = 0; i < 2; ++i) {
                        acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a[i], b[j], acc[i][j]);
                    }
                }
            }
        }
        if constexpr (lds1) {
            // the next step overwrites the only buffer: every warp must be done reading it
            __syncthreads();
            if (kb + 1 < nkb) {
                sstore(0);
                __syncthreads();
            }
        } else {
            if (kb + 1 < nkb) {
                sstore(buf ^ 1);
            }
            __syncthreads();
        }
    }

    // stage the result as [token][row] (element v of lane l: row 2 v + l/16, token l % 16 of its 16x16 tile), then write rows;
    // with lds1 in two passes: the warps of tokens p TS .. p TS + TS - 1 stage them in pass p (wn is a multiple of TS)
    float * Cs = (float *) smem;
    const float scale = x_scale ? x_scale[e] : 1.0f;
#pragma unroll
    for (int p = 0; p < TN/TS; ++p) {
        if (p > 0) {
            __syncthreads(); // the rows of the previous pass are written out
        }
        if (wn >= p*TS && wn < (p + 1)*TS) {
#pragma unroll
            for (int i = 0; i < 2; ++i) {
#pragma unroll
                for (int j = 0; j < 2; ++j) {
#pragma unroll
                    for (int v = 0; v < 8; ++v) {
                        Cs[(wn - p*TS + j*16 + lane % 16)*LDC + wm + i*16 + 2*v + lane/16] = acc[i][j][v];
                    }
                }
            }
        }
        __syncthreads();
        if constexpr (situ) {
            // gate row r of the tile is staged at r, its up row at TM/2 + r
            constexpr int TPT = TM/2/4; // threads per token, 4 rows each
#pragma unroll
            for (int t0 = 0; t0 < TS; t0 += INK2_MOE_NT/TPT) {
                const int t = t0 + tid / TPT;
                const int m = 4*(tid % TPT);
                if (p*TS + t < n) {
                    const float4 g = *(const float4 *) (Cs + t*LDC + m);
                    const float4 u = *(const float4 *) (Cs + t*LDC + TM/2 + m);
                    float4 v;
                    v.x = ink2_situ1(g.x*scale, u.x*scale, beta, lb);
                    v.y = ink2_situ1(g.y*scale, u.y*scale, beta, lb);
                    v.z = ink2_situ1(g.z*scale, u.z*scale, beta, lb);
                    v.w = ink2_situ1(g.w*scale, u.w*scale, beta, lb);
                    *(float4 *) (dst + (int64_t) ids_dst[spec.col_low + p*TS + t]*s1 + m0 + m) = v;
                }
            }
        } else {
#pragma unroll
            for (int t0 = 0; t0 < TS; t0 += INK2_MOE_NT/(TM/4)) {
                const int t = t0 + tid / (TM/4);
                const int m = 4*(tid % (TM/4));
                if (p*TS + t < n) {
                    float4 v = *(const float4 *) (Cs + t*LDC + m);
                    v.x *= scale; v.y *= scale; v.z *= scale; v.w *= scale;
                    *(float4 *) (dst + (int64_t) ids_dst[spec.col_low + p*TS + t]*s1 + m0 + m) = v;
                }
            }
        }
    }
#else
    GGML_UNUSED_VARS(W, X, ids_src1, ids_dst, tile_spec, dst, x_scale, nb01, nb02, s1, K, n_ff, beta, lb);
    NO_DEVICE_CODE;
#endif // defined(RDNA3)
}

bool ggml_cuda_ink2_moe_f16_supported(const int device, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids,
        const ggml_tensor * dst, const ggml_tensor * x_scale, const bool situ) {
    static const int min_tokens = []() { const char * e = getenv("GGML_CUDA_INK2_MOE_F16"); return e ? atoi(e) : 32; }();
    // GGML_CUDA_INK2_MOE_F16_Q = minimum number of tokens for the Q8_0 / Q5_K experts (default 256: from there on faster than
    // MMQ, equal for Q5_K at 256), 0 keeps them on MMQ
    static const int min_tokens_q = []() { const char * e = getenv("GGML_CUDA_INK2_MOE_F16_Q"); return e ? atoi(e) : 256; }();
    const ggml_type t = src0->type;
    const bool q_type = t == GGML_TYPE_Q8_0 || t == GGML_TYPE_Q5_K;
    if (min_tokens <= 0 || !ids || (t != GGML_TYPE_NVFP4 && !(q_type && min_tokens_q > 0)) ||
            src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 || !ggml_cuda_ink2_arch_ok(device)) {
        return false;
    }
    const int64_t K = src0->ne[0], M = src0->ne[1], n_experts = src0->ne[2];
    const int64_t n_used = ids->ne[0], n_tokens = src1->ne[2];
    if (q_type && n_tokens < min_tokens_q) {
        return false;
    }
    // Q5_K: whole super-blocks, read with 16-byte loads
    const size_t align = t == GGML_TYPE_Q5_K ? 16 : 4;
    if (t == GGML_TYPE_Q5_K && (K % QK_K != 0 || (uintptr_t) src0->data % 16 != 0)) {
        return false;
    }
    if (n_tokens < min_tokens || K % INK2_MOE_TK != 0 || M % INK2_MOE_TM != 0 || src0->ne[3] != 1 || src1->ne[3] != 1 ||
            src0->nb[1] % align != 0 || src0->nb[2] % align != 0 || !ggml_is_contiguous(src1) || !ggml_is_contiguous(dst) ||
            dst->ne[0] != (situ ? M/2 : M) || dst->ne[1] != n_used || dst->ne[2] != n_tokens || ids->ne[1] != n_tokens ||
            src1->ne[0] != K || (src1->ne[1] != 1 && src1->ne[1] != n_used) || M/INK2_MOE_TM > 65535) {
        return false;
    }
    if (x_scale && (x_scale->type != GGML_TYPE_F32 || !ggml_is_contiguous(x_scale) || ggml_nelements(x_scale) != n_experts)) {
        return false;
    }
    return true;
}

bool ggml_cuda_ink2_moe_f16(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids,
        ggml_tensor * dst, const ggml_tensor * x_scale, const float * situ) {
    if (!ggml_cuda_ink2_moe_f16_supported(ctx.device, src0, src1, ids, dst, x_scale, situ != nullptr)) {
        return false;
    }
    const int64_t K = src0->ne[0], M = src0->ne[1], n_experts = src0->ne[2];
    const int64_t n_used = ids->ne[0], n_tokens = src1->ne[2];
    cudaStream_t stream = ctx.stream();

    // activations in FP16, rows as in src1 (the expert sort below maps each slot to its src1 row)
    const int64_t n_src1 = ggml_nelements(src1);
    ggml_cuda_pool_alloc<half> x_f16(ctx.pool(), n_src1);
    {
        const int64_t n8 = n_src1/8;
        ink2_f32_to_f16<<<(n8 + 255)/256, 256, 0, stream>>>((const float *) src1->data, x_f16.get(), n8);
    }

    const int64_t n_slots = n_tokens*n_used;
    ggml_cuda_pool_alloc<int32_t> ids_src1(ctx.pool(), n_slots);
    ggml_cuda_pool_alloc<int32_t> ids_dst(ctx.pool(), n_slots);
    const int64_t ntj = (n_slots + INK2_MOE_TN - 1)/INK2_MOE_TN + n_experts;
    ggml_cuda_pool_alloc<ggml_cuda_mm_id_tile_spec> tile_spec(ctx.pool(), ntj);
    ggml_cuda_launch_mm_ids_helper(ctx.pool(), (const int32_t *) ids->data, ids_src1.get(), ids_dst.get(), tile_spec.get(),
        n_experts, n_tokens, n_used, src1->ne[1], INK2_MOE_TN, ids->nb[1]/ggml_element_size(ids), src1->nb[2]/src1->nb[1],
        /*write_inverse =*/ false, stream);
    CUDA_CHECK(cudaGetLastError());

    // with situ the grid is the same: M/2 output rows in tiles of TM/2
    const dim3 grid(M/INK2_MOE_TM, ntj);
    const auto launch = [&](auto kernel) {
        kernel<<<grid, INK2_MOE_NT, 0, stream>>>((const char *) src0->data, x_f16.get(), ids_src1.get(), ids_dst.get(),
            tile_spec.get(), (float *) dst->data, x_scale ? (const float *) x_scale->data : nullptr, src0->nb[1], src0->nb[2],
            dst->nb[1]/sizeof(float), (int) K, (int) (M/2), situ ? situ[0] : 0.0f, situ ? situ[1] : 0.0f);
    };
    // NVFP4: one shared-memory buffer (see ink2_moe_f16; ~4 % faster prompt reading on RDNA3), GGML_CUDA_INK2_MOE_LDS1=0 restores
    // two; the Q8_0 / Q5_K instances keep two buffers (not measured with one)
    static const bool lds1 = []() { const char * e = getenv("GGML_CUDA_INK2_MOE_LDS1"); return !e || atoi(e) != 0; }();
    switch (src0->type) {
        case GGML_TYPE_NVFP4:
            if (lds1) { situ ? launch(ink2_moe_f16<GGML_TYPE_NVFP4, true, true>) : launch(ink2_moe_f16<GGML_TYPE_NVFP4, false, true>); }
            else      { situ ? launch(ink2_moe_f16<GGML_TYPE_NVFP4, true, false>) : launch(ink2_moe_f16<GGML_TYPE_NVFP4, false, false>); }
            break;
        case GGML_TYPE_Q8_0:  situ ? launch(ink2_moe_f16<GGML_TYPE_Q8_0, true, false>) : launch(ink2_moe_f16<GGML_TYPE_Q8_0, false, false>); break;
        case GGML_TYPE_Q5_K:  situ ? launch(ink2_moe_f16<GGML_TYPE_Q5_K, true, false>) : launch(ink2_moe_f16<GGML_TYPE_Q5_K, false, false>); break;
        default: GGML_ABORT("unsupported type");
    }
    CUDA_CHECK(cudaGetLastError());
    return true;
}
