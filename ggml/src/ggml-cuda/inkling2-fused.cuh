#include "common.cuh"

// Local additions (tuned on RDNA3, gfx1100): fused kernels for the small-op patterns of the Inkling 2 graph. On HIP every kernel
// launch costs a few microseconds whatever its size, and one decoded token runs about 8100 of them: these replace runs of 6 to
// 12 generic ops by one kernel each. The patterns are matched in ggml-cuda.cu (ggml_cuda_try_fuse / graph_optimize).

// The local paths of the CUDA/HIP backend (these fusions, the Q8_0 x F32 mat-vec, the tuned tiles, the far attention
// extensions) were only run on RDNA3 (gfx1100, wave32): they are taken on RDNA3.0 GPUs (gfx110x) only, every other GPU keeps
// the stock paths (RDNA3.5, RDNA4, RDNA1/2, GCN, CDNA untested; the fused kernels assume 32-lane warps), and CUDA builds never
// take them. GGML_CUDA_INK2_ANY_AMD=1 also allows them on the other AMD architectures (untested).
static inline bool ggml_cuda_ink2_arch_ok(const int device) {
#if defined(GGML_USE_HIP)
    static signed char ok[GGML_CUDA_MAX_DEVICES] = {}; // 0 = unknown, 1 = yes, -1 = no
    if (device < 0 || device >= GGML_CUDA_MAX_DEVICES) {
        return false;
    }
    if (ok[device] == 0) {
        // GGML_CUDA_INK2_FORCE_STOCK=1 takes the stock paths here too (to test the fallback that the other GPUs get)
        static const bool any_amd    = getenv("GGML_CUDA_INK2_ANY_AMD") != nullptr;
        static const bool force_stock = getenv("GGML_CUDA_INK2_FORCE_STOCK") != nullptr;
        const int cc = ggml_cuda_info().devices[device].cc;
        ok[device] = !force_stock && (GGML_CUDA_CC_IS_RDNA3_0(cc) || (any_amd && GGML_CUDA_CC_IS_AMD(cc))) ? 1 : -1;
    }
    return ok[device] > 0;
#else
    GGML_UNUSED(device);
    return false;
#endif
}

// The switches that skip work to time it (wrong output) and the experimental head-mask WMMA path (can hang the GPU) only
// exist in builds with GGML_CUDA_INK2_COST_TOOLS defined (off by default).

#define GGML_CUDA_INK2_MIX_MAX_STATES 8
#define GGML_CUDA_INK2_MIX_MAX_Q      4

// dst = x * clamp(threshold / (sqrt(sum_rows(x^2)) + eps), 0, 1), rows along dim 0; with scale {d, n} (a weight broadcast over
// the rows, row r using scale row r % n), x is first multiplied by it
void ggml_cuda_op_ink2_l2_clip(ggml_backend_cuda_context & ctx, const ggml_tensor * x, ggml_tensor * dst, float threshold, float eps,
        const ggml_tensor * scale = nullptr);

// depth attention: states s_i [n_embd, n_tokens], query [n_embd, n_q] -> dst [n_embd, n_q, n_tokens] ([n_embd, n_tokens, n_q]
// with rows_first: the permuted copy of the result, whose rows are contiguous)
// dst[:, q, t] = sum_i softmax_i(<rms(s_i), query_q> * scale) * rms(s_i), rms(x) = x / sqrt(mean(x^2) + eps)
bool ggml_cuda_ink2_depth_mix_supported(int n_states, int n_q);
void ggml_cuda_op_ink2_depth_mix(ggml_backend_cuda_context & ctx, const ggml_tensor * const * states, int n_states,
        const ggml_tensor * query, ggml_tensor * dst, float eps, float scale, bool rows_first = false);

// loop message gate: h = D^T [rms(cur); rms(msg)], dst = (1 - tanh(F^T h)) * cur + tanh(G^T h) * msg
// cur, msg, dst [n_embd, n_tokens]; D [2*n_embd, n_rank]; G, F [n_rank, n_embd]
void ggml_cuda_op_ink2_loop_gate(ggml_backend_cuda_context & ctx, const ggml_tensor * cur, const ggml_tensor * msg,
        const ggml_tensor * down, const ggml_tensor * gain, const ggml_tensor * forget, ggml_tensor * dst, float eps);
// the weights D, G, F may also all be BF16 (lossless for Inkling 2 Mini) for these sizes
bool ggml_cuda_ink2_loop_gate_bf16_supported(int n_embd, int n_rank);
// up to GGML_CUDA_INK2_GATE_N_MAX independent gates of the same shape (1..8 tokens, the multi-token kernels) in one launch of
// each of the two kernels, the same arithmetic as one by one; _ok: whether one gate qualifies
#define GGML_CUDA_INK2_GATE_N_MAX 3
bool ggml_cuda_ink2_loop_gate_n_ok(const ggml_tensor * cur, const ggml_tensor * msg, const ggml_tensor * down, const ggml_tensor * gain,
        const ggml_tensor * forget, const ggml_tensor * dst);
void ggml_cuda_op_ink2_loop_gate_n(ggml_backend_cuda_context & ctx, const ggml_tensor * const * cur, const ggml_tensor * const * msg,
        const ggml_tensor * const * down, const ggml_tensor * const * gain, const ggml_tensor * const * forget, ggml_tensor * const * dst,
        int n, float eps);

// short convolution tail: dst = (cur + ssm_conv(x, w)) * scale (+ res); x {n_conv - 1 + n_tokens, n_channels}, w {n_conv, n_channels}
void ggml_cuda_op_ink2_sconv_res(ggml_backend_cuda_context & ctx, const ggml_tensor * x, const ggml_tensor * w, const ggml_tensor * cur,
        const ggml_tensor * res, ggml_tensor * dst, float scale);

// SITU activation of the MoE: gate and up are the two halves (row views) of the gate_up projection
void ggml_cuda_op_ink2_situ(ggml_backend_cuda_context & ctx, const ggml_tensor * gate, const ggml_tensor * up, ggml_tensor * dst, float beta, float lb);

// router weights of the selected experts: w / (sum + eps) * scale
void ggml_cuda_op_ink2_moe_wnorm(ggml_backend_cuda_context & ctx, const ggml_tensor * w, ggml_tensor * dst, float eps, float scale);

// input of a short convolution: x = [past ; cur^T] and the state update (the last rows of x)
void ggml_cuda_op_ink2_sconv_in(ggml_backend_cuda_context & ctx, const ggml_tensor * past, const ggml_tensor * cur_t, ggml_tensor * x,
        ggml_tensor * state);

// relative-position bias of the dense attention path, per head, already added to the mask and cast for flash attention:
// dst[k, t, h] = f16(mask[k, t] + (d < E ? sum_r bias[r, d]*R[r, h, t] : 0) - sinks[h, t]) with d = rel_pos[k, t] in [0, E]
// bias {n_rel, E}, R {n_rel, n_head, n_tok}, sinks {n_head, n_tok}, rel_pos i32 {n_kv, n_tok}, mask f16 {n_kv, n_tok}
void ggml_cuda_op_ink2_rel_bias(ggml_backend_cuda_context & ctx, const ggml_tensor * bias, const ggml_tensor * R, const ggml_tensor * sinks,
        const ggml_tensor * rel_pos, const ggml_tensor * mask, ggml_tensor * dst);

// Q8_0 matrix times 1..8 F32 vectors without quantizing the vectors first: dst[r, t] = sum_b d[r, b] * sum_i q[r, b, i]*y[b*32 + i, t]
// (one launch instead of two, and no rounding of the activations). Returns false when the shapes are not handled.
bool ggml_cuda_ink2_mmv_q8_f32_supported(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst);
void ggml_cuda_op_ink2_mmv_q8_f32(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);

// up to GGML_CUDA_INK2_Q8_GROUP_MAX such products (MUL_MAT nodes) that read the same activations, in one launch
#define GGML_CUDA_INK2_Q8_GROUP_MAX 6
// same_src1 = false: b may read other activations than a (same shape); each matrix then multiplies its own
bool ggml_cuda_ink2_mmv_q8_f32_group_ok(const ggml_tensor * a, const ggml_tensor * b, bool same_src1 = true);
void ggml_cuda_op_ink2_mmv_q8_f32_group(ggml_backend_cuda_context & ctx, ggml_tensor * const * dsts, int n);

// a whole short convolution: input [past ; cur^T] and state store (rollback slots, or plain state), then
// dst = (cur + ssm_conv(x, w)) * scale (+ res), without writing x
void ggml_cuda_op_ink2_sconv_full(ggml_backend_cuda_context & ctx, const ggml_tensor * past, const ggml_tensor * cur_t,
        const ggml_tensor * slots, const ggml_tensor * slot_order, int n_win, ggml_tensor * state,
        const ggml_tensor * w, const ggml_tensor * cur, const ggml_tensor * res, ggml_tensor * dst, float scale);

// two such convolutions of the same shape (3 past rows) in one launch, the same arithmetic per element; false (nothing done)
// when the shapes do not allow it
struct ggml_cuda_ink2_sconv_desc {
    const ggml_tensor * past;
    const ggml_tensor * cur_t;
    const ggml_tensor * slots;
    const ggml_tensor * slot_order;
    int                 n_win;
    ggml_tensor       * state;
    const ggml_tensor * w;
    const ggml_tensor * cur;
    const ggml_tensor * res;
    ggml_tensor       * dst;
    float               scale;
};
bool ggml_cuda_op_ink2_sconv_full2(ggml_backend_cuda_context & ctx, const ggml_cuda_ink2_sconv_desc & a, const ggml_cuda_ink2_sconv_desc & b);

// dynamic short convolution with its residual: dst = (cur + (ssm_conv(x, w) + sum_lag window_lag(x) * D[lag, head])) * scale
// x {n_conv - 1 + n_tokens, n_channels}, w {n_conv, n_channels}, D {n_conv, n_heads, n_tokens}
void ggml_cuda_op_ink2_dconv(ggml_backend_cuda_context & ctx, const ggml_tensor * x, const ggml_tensor * w, const ggml_tensor * D,
        const ggml_tensor * cur, ggml_tensor * dst, int n_heads, float scale);

// input of a short convolution with rollback snapshots: x = [past ; cur^T], and window w of x (rows base + w .. base + w + n_past - 1,
// base = n_tokens - (n_win - 1)) is stored in slot slot_order[w] of the state: slots {n_past, n_slots, n_channels} (strided view)
void ggml_cuda_op_ink2_sconv_in_rb(ggml_backend_cuda_context & ctx, const ggml_tensor * past, const ggml_tensor * cur_t, ggml_tensor * x,
        const ggml_tensor * slots, const ggml_tensor * slot_order, int n_win);

// loop gate tail (prompt batches): dst = (1 - tanh(f)) * cur + tanh(g) * msg, all F32 of the same size
void ggml_cuda_op_ink2_gate_tail(ggml_backend_cuda_context & ctx, const ggml_tensor * f, const ggml_tensor * g, const ggml_tensor * cur,
        const ggml_tensor * msg, ggml_tensor * dst);

// whole loop gate of a prompt batch (more than 16 tokens) with BF16 weights, n_rank 64 (RDNA3 WMMA, three kernels): the same
// dst = (1 - tanh(F h))*cur + tanh(G h)*msg, h = D [rms(cur); rms(msg)], without writing the normalized input or F h and G h.
// Rounding level against the generic path (BF16 activations, F32 sums in another order; F h and G h are not rounded to BF16).
// GGML_CUDA_INK2_GATE_PP=0 disables it.
bool ggml_cuda_ink2_gate_pp_supported(int device, const ggml_tensor * cur, const ggml_tensor * msg, const ggml_tensor * down,
        const ggml_tensor * gain, const ggml_tensor * forget, const ggml_tensor * dst);
void ggml_cuda_op_ink2_gate_pp(ggml_backend_cuda_context & ctx, const ggml_tensor * cur, const ggml_tensor * msg, const ggml_tensor * down,
        const ggml_tensor * gain, const ggml_tensor * forget, ggml_tensor * dst, float eps);

// first index of the maximum of one F32 row (dst: one I32), the token selection rule of the DSpark drafting graph
void ggml_cuda_op_ink2_first_max(ggml_backend_cuda_context & ctx, const ggml_tensor * row, ggml_tensor * dst);

// row ops that follow the projections of an attention layer, in one launch: tasks of rows [d, rows_per_tok, n_tok] (contiguous
// F32), each an L2 clip (kind 0, optional per-channel scale, d <= 4*WARP_SIZE) or an RMS norm (kind 1, d == 4*WARP_SIZE), then
// optionally the product with a per-token scalar (tok, written to dst_tok) and the store of the rows into a K/V cache (kv rows
// idx[t] of kv_nb1 bytes, F16 or Q8_0, the rows_per_tok rows of a token side by side). Each row is computed as its own kernel
// computes it (ink2_l2_clip_f32, rms_norm_f32 of 32 threads, MUL, SET_ROWS): the same values bit for bit.
#define GGML_CUDA_INK2_POST_MAX 6
struct ggml_cuda_ink2_post_task {
    const float   * x;
    const float   * scale;
    float         * dst;
    const float   * tok;
    float         * dst_tok;
    char          * kv;
    const int64_t * idx;
    int64_t         kv_nb1;
    int64_t         row0; // first row of the task in the launch (filled by the launcher)
    int             n_rows;
    int             d;
    int             rows_per_tok;
    int             n_scale_rows;
    int             kind;
    int             kv_q8; // 0 = F16 cache, 1 = Q8_0 cache
    float           threshold;
    float           eps;
};
struct ggml_cuda_ink2_post_args {
    ggml_cuda_ink2_post_task t[GGML_CUDA_INK2_POST_MAX];
    int n;
};
void ggml_cuda_op_ink2_post(ggml_backend_cuda_context & ctx, ggml_cuda_ink2_post_args & args);

// NVFP4 expert products of prompt batches on FP16 WMMA (RDNA3): MUL_MAT_ID with NVFP4 weights, the weights decoded to FP16 in
// shared memory with their sub-block scale, the activations rounded to FP16 (instead of the q8_1 of MMQ), FP32 accumulation;
// x_scale (optional, one F32 per expert) multiplies the result (the folded MUL of the per-expert scale). Called by
// ggml_cuda_mul_mat_q for the ids case; returns false (nothing done) when the shape or the device does not qualify.
// GGML_CUDA_INK2_MOE_F16 = minimum number of tokens (default 32: faster than MMQ from 32 tokens up), 0 disables it.
// Q8_0 and Q5_K expert weights take the same kernel (decoded to FP16 per 32 values: x*d, q*(d*sc) - dmin*m) from
// GGML_CUDA_INK2_MOE_F16_Q tokens on (default 256), 0 keeps them on MMQ.
// situ = {beta, lb}: src0 is a merged gate_up projection [gate ; up] and dst [n_ff, n_used, n_tokens] receives
// situ(gate, up) = beta*tanh(gate/beta)*sigmoid(gate) * lb*tanh(up/lb) instead of the projection (fused SITU activation).
bool ggml_cuda_ink2_moe_f16_supported(int device, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids,
        const ggml_tensor * dst, const ggml_tensor * x_scale, bool situ);
bool ggml_cuda_ink2_moe_f16(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids,
        ggml_tensor * dst, const ggml_tensor * x_scale, const float * situ = nullptr);
// gate_up + SITU and the down projection of the same experts in one go (both shapes must pass ggml_cuda_ink2_moe_f16_supported):
// the activation is written once in FP16 (the values the down call would convert it to) instead of F32 into act, and one expert
// sort serves both products. Same results bit for bit (GGML_CUDA_INK2_CMP=ffn checks it). act is not written.
bool ggml_cuda_ink2_moe_ffn_f16_ok(const ggml_tensor * gu_w, const ggml_tensor * dn_w);
bool ggml_cuda_ink2_moe_ffn_f16(ggml_backend_cuda_context & ctx, const ggml_tensor * gu_w, const ggml_tensor * gu_x, const ggml_tensor * ids,
        const ggml_tensor * gu_scale, const float * situ, ggml_tensor * act, const ggml_tensor * dn_w, ggml_tensor * dn_out,
        const ggml_tensor * dn_scale);
