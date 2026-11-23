#include "common.cuh"
#include "fattn-common.cuh"
#include "fattn-mma-f16.cuh"
#include "fattn-tile.cuh"
#include "fattn-vec.cuh"
#include "fattn.cuh"
#include "inkling2-fused.cuh" // ggml_cuda_ink2_arch_ok

#if !defined(GGML_USE_MUSA)

// one list per group of ncols1 queries: a column is selected if any query of the group can see it
template <int ncols1, bool oob>
__launch_bounds__(256, 1)
static __global__ void flash_attn_mask_to_sparse_indices(
        const half * mask_ptr, int32_t * indices_ptr, int32_t * counts_ptr, const int ne30, const int n_queries,
        const int n_kv_max, const int64_t s31, const int64_t s33) {
    ggml_cuda_pdl_sync();

    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int values_per_lane = 8;
    const int tid      = threadIdx.x;
    const int warp     = tid / warp_size;
    const int lane     = tid % warp_size;
    const int sequence = blockIdx.y;
    const int group    = blockIdx.x;

    const int q0 = group*ncols1;
    const int q1 = min(q0 + ncols1, n_queries);

    const half * mask = mask_ptr + sequence*s33 + q0*s31;
    int32_t * indices = indices_ptr + (int64_t(sequence)*gridDim.x + group)*n_kv_max;

    __shared__ int warp_offsets[256/warp_size];
    __shared__ int row_count;
    __shared__ int chunk_count;

    if (tid == 0) {
        row_count = 0;
    }
    __syncthreads();

    for (int i0 = 0; i0 < ne30; i0 += blockDim.x*values_per_lane) {
        ggml_cuda_thread_mask_t selected_warp[values_per_lane];
        int warp_count = 0;
#pragma unroll
        for (int item = 0; item < values_per_lane; ++item) {
            const int i = i0 + (warp*values_per_lane + item)*warp_size + lane;
            bool selected = false;
            if (i < ne30) {
#pragma unroll
                for (int q = 0; q < ncols1; ++q) {
                    selected |= (!oob || q < q1 - q0) && isfinite(__half2float(mask[q*s31 + i]));
                }
            }
            selected_warp[item] = ggml_cuda_ballot((int) selected);
            warp_count += ggml_cuda_popc(selected_warp[item]);
        }

        if (lane == 0) {
            warp_offsets[warp] = warp_count;
        }
        __syncthreads();

        if (tid == 0) {
            int offset = 0;
#pragma unroll
            for (int iw = 0; iw < 256/warp_size; ++iw) {
                const int count = warp_offsets[iw];
                warp_offsets[iw] = offset;
                offset += count;
            }
            chunk_count = offset;
        }
        __syncthreads();

        const ggml_cuda_thread_mask_t lane_mask = lane == 0 ? 0 : ((ggml_cuda_thread_mask_t)1 << lane) - 1;
        int warp_item_offset = 0;
#pragma unroll
        for (int item = 0; item < values_per_lane; ++item) {
            const int i = i0 + (warp*values_per_lane + item)*warp_size + lane;
            const int dst = row_count + warp_offsets[warp] + warp_item_offset + ggml_cuda_popc(selected_warp[item] & lane_mask);
            if ((selected_warp[item] & ((ggml_cuda_thread_mask_t)1 << lane)) && dst < n_kv_max) {
                indices[dst] = i;
            }
            warp_item_offset += ggml_cuda_popc(selected_warp[item]);
        }
        __syncthreads();

        if (tid == 0) {
            row_count += chunk_count;
        }
        __syncthreads();
    }

    const int count = min(row_count, n_kv_max);
    for (int i = count + tid; i < n_kv_max; i += blockDim.x) {
        indices[i] = -1;
    }
    if (tid == 0) {
        counts_ptr[int64_t(sequence)*gridDim.x + group] = count;
    }
    __syncthreads();

    // the dependent grid reads indices, signal once the row is complete
    ggml_cuda_pdl_lc();
}
#endif // !defined(GGML_USE_MUSA)

void ggml_cuda_flash_attn_ext_compact_mask(
        const ggml_tensor * mask, int32_t * indices, int32_t * counts, int32_t n_queries, int32_t ncols1, int32_t n_kv_max, cudaStream_t stream) {
#ifdef GGML_USE_MUSA
    GGML_UNUSED_VARS(mask, indices, counts, n_queries, ncols1, n_kv_max, stream);
    GGML_ABORT("sparse flash attention is not supported on MUSA");
#else
    const int64_t s31 = mask->nb[1] / sizeof(half);
    const int64_t s33 = mask->nb[3] / sizeof(half);
    const dim3 blocks_num((n_queries + ncols1 - 1)/ncols1, mask->ne[3], 1);
    const dim3 block_dim(256, 1, 1);
    const ggml_cuda_kernel_launch_params launch_params(blocks_num, block_dim, 0, stream);
    // the last group of queries is partial only if ncols1 does not divide n_queries
    GGML_ASSERT(ncols1 == 1 || ncols1 == 8);
    const auto kernel = ncols1 == 1       ? flash_attn_mask_to_sparse_indices<1, false> :
                        n_queries % 8 != 0 ? flash_attn_mask_to_sparse_indices<8, true>  :
                                             flash_attn_mask_to_sparse_indices<8, false>;
    ggml_cuda_kernel_launch(kernel, launch_params,
        (const half *) mask->data, indices, counts, int(mask->ne[0]), n_queries, n_kv_max, s31, s33);
    CUDA_CHECK(cudaGetLastError());
#endif // !defined(GGML_USE_MUSA)
}

bool ggml_cuda_flash_attn_ext_mma_f16_shall_use_sparse(const int cc, const ggml_tensor * dst, const int ncols1, const int ncols2) {
#ifdef GGML_USE_MUSA
    GGML_UNUSED_VARS(cc, dst, ncols1, ncols2);
    return false;
#else
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    float max_bias = 0.0f;
    float logit_softcap = 0.0f;
    memcpy(&max_bias,      (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));

    // On CDNA only the DKQ=576 kernel variant with ncols2=16 is compiled with sparse support.
    if (amd_mfma_available(cc) && !ggml_cuda_flash_attn_ext_mma_f16_may_use_sparse(int(Q->ne[0]), int(V->ne[0]), 1, 16)) {
        return false;
    }

    const int32_t n_kv_max = ggml_get_op_params_i32(dst, 4);

    // the dense kernel handles up to 64/ncols2 queries per K/V pass, the single-query gather has to beat that
    const int64_t n_gather = (ncols1 == 1 ? std::min<int64_t>(Q->ne[1], 64/ncols2) : ncols1) * (int64_t) n_kv_max;

    return ((GGML_CUDA_CC_IS_NVIDIA(cc) && turing_mma_available(cc)) || amd_mfma_available(cc)) && dst->src[5] == nullptr &&
        mask != nullptr && n_kv_max > 0 && max_bias == 0.0f && logit_softcap == 0.0f &&
        mask->ne[0] == K->ne[1] && mask->ne[1] >= Q->ne[1] && mask->ne[2] == 1 &&
        K->ne[1] >= std::max<int64_t>(4096, GGML_CUDA_CC_IS_NVIDIA(cc) ? 2*n_gather : 2LL*n_kv_max);
#endif // !defined(GGML_USE_MUSA)
}

template <int DKQ, int DV, int ncols2>
static void ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const ggml_tensor * Q = dst->src[0];

#if !defined(GGML_USE_MUSA)
    if constexpr (ggml_cuda_flash_attn_ext_mma_f16_may_use_sparse(DKQ, DV, 1, ncols2)) {
        // a sparse variant at the full tile width gathers the union of its queries once, prefer it for large batches
        constexpr bool has_wide_sparse = ggml_cuda_flash_attn_ext_mma_f16_may_use_sparse(DKQ, DV, 64/ncols2, ncols2);
        if (!(has_wide_sparse && Q->ne[1] > 32/ncols2) && ggml_cuda_flash_attn_ext_mma_f16_shall_use_sparse(cc, dst, 1, ncols2)) {
            ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 1, ncols2>(ctx, dst);
            return;
        }
    }
#endif // !defined(GGML_USE_MUSA)

    if constexpr (ncols2 <= 8) {
        if (turing_mma_available(cc) && Q->ne[1] <= 8/ncols2) {
            ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 8/ncols2, ncols2>(ctx, dst);
            return;
        }
    }

    if constexpr (ncols2 <= 16) {
        if (Q->ne[1] <= 16/ncols2) {
            ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 16/ncols2, ncols2>(ctx, dst);
            return;
        }
    }

    if (Q->ne[1] <= 32/ncols2 || (GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) == GGML_CUDA_CC_TURING) ||
            (GGML_CUDA_CC_IS_AMD(cc) && DKQ > 256)) {
        ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 32/ncols2, ncols2>(ctx, dst);
        return;
    }

    ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 64/ncols2, ncols2>(ctx, dst);
}

template <int DKQ, int DV>
static void ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const ggml_tensor * KQV  = dst;
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    // with kv_rows the kernel iterates over the mask columns
    const int64_t n_kv = dst->src[5] ? dst->src[5]->ne[0] : K->ne[1];

    float max_bias = 0.0f;
    memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

    // Edge cases like no mask, ALiBi, unpadded K/V, or misaligned addresses for large data transfers
    //     are put into the template specialization without GQA optimizations.
    // a mask with a head dimension is not shared by the Q heads of a K/V head
    // (local: the far mode with an implicit causal mask has no mask tensor, it is shared by construction)
    const bool far_causal = dst->op == GGML_OP_FLASH_ATTN_EXT_BANDED && ((const float *) dst->op_params)[6] != 0.0f && !mask;
    bool use_gqa_opt = (far_causal || (mask && mask->ne[2] == 1)) && max_bias == 0.0f && n_kv % FATTN_KQ_STRIDE == 0;
    for (const ggml_tensor * t : {Q, K, V, mask}) {
        if (t == nullptr || ggml_is_quantized(t->type)) {
            continue;
        }
        for (size_t i = 1; i < GGML_MAX_DIMS; ++i) {
            if (t->nb[i] % 16 != 0) {
                use_gqa_opt = false;
                break;
            }
        }
    }

    GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
    const int gqa_ratio = Q->ne[2] / K->ne[2];

    // On Volta the GQA optimizations aren't as impactful vs. minimizing wasted compute:
    if (cc == GGML_CUDA_CC_VOLTA) {
        if (use_gqa_opt && gqa_ratio % 8 == 0) {
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 8>(ctx, dst);
            return;
        }

        if (use_gqa_opt && gqa_ratio % 4 == 0) {
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 4>(ctx, dst);
            return;
        }

        if constexpr (DKQ <= 256) {
            if (use_gqa_opt && gqa_ratio % 2 == 0) {
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 2>(ctx, dst);
                return;
            }

            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 1>(ctx, dst);
            return;
        } else {
            GGML_ABORT("fatal error");
        }
    }

    // On RDNA it is preferable to minimize wasted compute vs. duplicate I/O for the mask.
    if (amd_wmma_available(cc)) {
#if defined(GGML_USE_HIP)
        // local (RDNA3): GQA ratio 3 as 3 heads per tile instead of 4 with a padding column (a quarter less work).
        // GGML_CUDA_INK2_FA_GQA3 = 0 off, 16 or 32 = queries per tile (wider tiles need batches of more than 32 tokens).
        if constexpr (DKQ == 128 && DV == 128) {
            static const int gqa3 = [] {
                const char * e = getenv("GGML_CUDA_INK2_FA_GQA3");
                return e ? atoi(e) : 16;
            }();
            if (use_gqa_opt && gqa_ratio == 3 && gqa3 > 0 && Q->ne[1] > 16 && ggml_cuda_ink2_arch_ok(ggml_cuda_get_device())) {
                if (gqa3 >= 32 && Q->ne[1] > 32) {
                    ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 32, 3>(ctx, dst);
                } else {
                    ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 16, 3>(ctx, dst);
                }
                return;
            }
        }
#endif // defined(GGML_USE_HIP)
        if (use_gqa_opt && gqa_ratio % 8 == 0) {
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 8>(ctx, dst);
            return;
        }

        if (use_gqa_opt && gqa_ratio % 4 == 0) {
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 4>(ctx, dst);
            return;
        }

        if (use_gqa_opt && gqa_ratio % 2 == 0) {
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 2>(ctx, dst);
            return;
        }
    }

    if (use_gqa_opt && gqa_ratio > 4) {
        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 8>(ctx, dst);
        return;
    }

    if (use_gqa_opt && gqa_ratio > 2) {
        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 4>(ctx, dst);
        return;
    }

    if (use_gqa_opt && gqa_ratio > 1) {
        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 2>(ctx, dst);
        return;
    }

    if constexpr (DKQ <= 256) {
        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 1>(ctx, dst);
    } else {
        GGML_ABORT("fatal error");
    }
}

static void ggml_cuda_flash_attn_ext_mma_f16(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const ggml_tensor * KQV  = dst;
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    switch (Q->ne[0]) {
        case 40:
            GGML_ASSERT(V->ne[0] == 40);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2< 40,  40>(ctx, dst);
            break;
        case 64:
            GGML_ASSERT(V->ne[0] == 64);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2< 64,  64>(ctx, dst);
            break;
        case 72:
            GGML_ASSERT(V->ne[0] == 72);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2< 72,  72>(ctx, dst);
            break;
        case 80:
            GGML_ASSERT(V->ne[0] == 80);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2< 80,  80>(ctx, dst);
            break;
        case 96:
            GGML_ASSERT(V->ne[0] == 96);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2< 96,  96>(ctx, dst);
            break;
        case 112:
            GGML_ASSERT(V->ne[0] == 112);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<112, 112>(ctx, dst);
            break;
        case 128:
            GGML_ASSERT(V->ne[0] == 128);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<128, 128>(ctx, dst);
            break;
        case 192: {
            // MiMo-V2.5 / V2.5-Pro / V2-Flash: gqa_ratio is 8 (SWA) or 16 (full attn)
            GGML_ASSERT(V->ne[0] == 128);
            float max_bias = 0.0f;
            memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));
            const bool use_gqa_opt = mask && max_bias == 0.0f;
            GGML_ASSERT(use_gqa_opt);
            GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
            const int gqa_ratio = Q->ne[2] / K->ne[2];
            if (gqa_ratio % 16 == 0) {
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<192, 128, 16>(ctx, dst);
            } else {
                GGML_ASSERT(gqa_ratio % 8 == 0);
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<192, 128,  8>(ctx, dst);
            }
        } break;
        case 256:
            GGML_ASSERT(V->ne[0] == 256);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<256, 256>(ctx, dst);
            break;
        case 320:
            // For Mistral Small 4, go straight to the ncols1 switch (ncols2=32-only build).
            GGML_ASSERT(V->ne[0] == 256);
            {
                float max_bias = 0.0f;
                memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

                const bool use_gqa_opt = mask && max_bias == 0.0f;
                GGML_ASSERT(use_gqa_opt);
                GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
                const int gqa_ratio = Q->ne[2] / K->ne[2];
                GGML_ASSERT(gqa_ratio % 32 == 0);

                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<320, 256, 32>(ctx, dst);
            }
            break;
        case 512:
            GGML_ASSERT(V->ne[0] == 512);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<512, 512>(ctx, dst);
            break;
        case 576: {
            // For Deepseek, go straight to the ncols1 switch to avoid compiling unnecessary kernels.
            GGML_ASSERT(V->ne[0] == 512);
            float max_bias = 0.0f;
            memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

            const bool use_gqa_opt = mask && max_bias == 0.0f;
            GGML_ASSERT(use_gqa_opt);

            GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
            const int gqa_ratio = Q->ne[2] / K->ne[2];
            if (amd_mfma_available(cc) && ggml_cuda_flash_attn_ext_mma_f16_shall_use_sparse(cc, dst, 1, 16)) {
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
            } else if (gqa_ratio == 20 && GGML_CUDA_CC_IS_NVIDIA(cc)) { // GLM 4.7 Flash
                if (cc >= GGML_CUDA_CC_GB10) {
                    if (Q->ne[1] <= 8) {
                        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
                        break;
                    }
                    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
                    break;
                }
                if (cc >= GGML_CUDA_CC_BLACKWELL) {
                    if (Q->ne[1] <= 4 && K->ne[1] >= 65536) {
                        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
                        break;
                    }
                    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
                    break;
                }
                if (cc >= GGML_CUDA_CC_ADA_LOVELACE) {
                    if (Q->ne[1] <= 4) {
                        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
                        break;
                    }
                    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
                    break;
                }
                if (cc >= GGML_CUDA_CC_TURING) {
                    if (Q->ne[1] <= 4) {
                        if (K->ne[1] <= 16384) {
                            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
                            break;
                        }
                        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 32>(ctx, dst);
                        break;
                    }
                    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
                    break;
                }
                // Volta:
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
            } else if (gqa_ratio % 16 == 0) {
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
            } else {
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512,  4>(ctx, dst);
            }
        } break;
        default:
            GGML_ABORT("fatal error");
    }
}

#define FATTN_VEC_CASE(D, type_K_case, type_V_case)                                                                                \
    if constexpr (GGML_CUDA_FA_##type_K_case##_##type_V_case) {                                                                    \
        if (head_size == (D) && type_K == GGML_TYPE_##type_K_case && type_V == GGML_TYPE_##type_V_case) {                          \
            return ggml_cuda_flash_attn_ext_vec_case<D, GGML_TYPE_##type_K_case, GGML_TYPE_##type_V_case>;                         \
        }                                                                                                                          \
    }                                                                                                                              \

#define FATTN_VEC_CASES_ALL_D(type_K_case, type_V_case) \
    FATTN_VEC_CASE( 64, type_K_case, type_V_case)       \
    FATTN_VEC_CASE(128, type_K_case, type_V_case)       \
    FATTN_VEC_CASE(256, type_K_case, type_V_case)       \

typedef void (* fattn_vec_case_t)(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// local: q8_0 K with q4_0 V at head size 128 (a 4-bit V for very long contexts with the K precision of q8_0) is compiled
// here when the build does not list it in GGML_CUDA_FA_QUANTS: without it a decoded row would copy the whole K/V view to F16
#if !GGML_CUDA_FA_Q8_0_Q4_0
DECL_FATTN_VEC_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0);
#endif // !GGML_CUDA_FA_Q8_0_Q4_0

// Vector kernel for the given head size and K/V types, nullptr if its template instance was not compiled:
static fattn_vec_case_t ggml_cuda_get_fattn_vec_case(const int64_t head_size, const ggml_type type_K_in, const ggml_type type_V_in) {
    const ggml_type type_K = type_K_in == GGML_TYPE_F32 ? GGML_TYPE_F16 : type_K_in;
    const ggml_type type_V = type_V_in == GGML_TYPE_F32 ? GGML_TYPE_F16 : type_V_in;
#if !GGML_CUDA_FA_Q8_0_Q4_0
    if (head_size == 128 && type_K == GGML_TYPE_Q8_0 && type_V == GGML_TYPE_Q4_0) {
        return ggml_cuda_flash_attn_ext_vec_case<128, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0>;
    }
#endif // !GGML_CUDA_FA_Q8_0_Q4_0
    FATTN_VEC_CASES_ALL_D(F16,  F16)
    FATTN_VEC_CASES_ALL_D(Q4_0, F16)
    FATTN_VEC_CASES_ALL_D(Q4_1, F16)
    FATTN_VEC_CASES_ALL_D(Q5_0, F16)
    FATTN_VEC_CASES_ALL_D(Q5_1, F16)
    FATTN_VEC_CASES_ALL_D(Q8_0, F16)
    FATTN_VEC_CASES_ALL_D(BF16, F16)
    FATTN_VEC_CASES_ALL_D(F8_E4M3, F16)

    FATTN_VEC_CASES_ALL_D(F16,  Q4_0)
    FATTN_VEC_CASES_ALL_D(Q4_0, Q4_0)
    FATTN_VEC_CASES_ALL_D(Q4_1, Q4_0)
    FATTN_VEC_CASES_ALL_D(Q5_0, Q4_0)
    FATTN_VEC_CASES_ALL_D(Q5_1, Q4_0)
    FATTN_VEC_CASES_ALL_D(Q8_0, Q4_0)
    FATTN_VEC_CASES_ALL_D(BF16, Q4_0)
    FATTN_VEC_CASES_ALL_D(F8_E4M3, Q4_0)

    FATTN_VEC_CASES_ALL_D(F16,  Q4_1)
    FATTN_VEC_CASES_ALL_D(Q4_0, Q4_1)
    FATTN_VEC_CASES_ALL_D(Q4_1, Q4_1)
    FATTN_VEC_CASES_ALL_D(Q5_0, Q4_1)
    FATTN_VEC_CASES_ALL_D(Q5_1, Q4_1)
    FATTN_VEC_CASES_ALL_D(Q8_0, Q4_1)
    FATTN_VEC_CASES_ALL_D(BF16, Q4_1)
    FATTN_VEC_CASES_ALL_D(F8_E4M3, Q4_1)

    FATTN_VEC_CASES_ALL_D(F16,  Q5_0)
    FATTN_VEC_CASES_ALL_D(Q4_0, Q5_0)
    FATTN_VEC_CASES_ALL_D(Q4_1, Q5_0)
    FATTN_VEC_CASES_ALL_D(Q5_0, Q5_0)
    FATTN_VEC_CASES_ALL_D(Q5_1, Q5_0)
    FATTN_VEC_CASES_ALL_D(Q8_0, Q5_0)
    FATTN_VEC_CASES_ALL_D(BF16, Q5_0)
    FATTN_VEC_CASES_ALL_D(F8_E4M3, Q5_0)

    FATTN_VEC_CASES_ALL_D(F16,  Q5_1)
    FATTN_VEC_CASES_ALL_D(Q4_0, Q5_1)
    FATTN_VEC_CASES_ALL_D(Q4_1, Q5_1)
    FATTN_VEC_CASES_ALL_D(Q5_0, Q5_1)
    FATTN_VEC_CASES_ALL_D(Q5_1, Q5_1)
    FATTN_VEC_CASES_ALL_D(Q8_0, Q5_1)
    FATTN_VEC_CASES_ALL_D(BF16, Q5_1)
    FATTN_VEC_CASES_ALL_D(F8_E4M3, Q5_1)

    FATTN_VEC_CASES_ALL_D(F16,  Q8_0)
    FATTN_VEC_CASES_ALL_D(Q4_0, Q8_0)
    FATTN_VEC_CASES_ALL_D(Q4_1, Q8_0)
    FATTN_VEC_CASES_ALL_D(Q5_0, Q8_0)
    FATTN_VEC_CASES_ALL_D(Q5_1, Q8_0)
    FATTN_VEC_CASES_ALL_D(Q8_0, Q8_0)
    FATTN_VEC_CASES_ALL_D(BF16, Q8_0)
    FATTN_VEC_CASES_ALL_D(F8_E4M3, Q8_0)

    FATTN_VEC_CASES_ALL_D(F16,  BF16)
    FATTN_VEC_CASES_ALL_D(Q4_0, BF16)
    FATTN_VEC_CASES_ALL_D(Q4_1, BF16)
    FATTN_VEC_CASES_ALL_D(Q5_0, BF16)
    FATTN_VEC_CASES_ALL_D(Q5_1, BF16)
    FATTN_VEC_CASES_ALL_D(Q8_0, BF16)
    FATTN_VEC_CASES_ALL_D(BF16, BF16)
    FATTN_VEC_CASES_ALL_D(F8_E4M3, BF16)

    FATTN_VEC_CASES_ALL_D(F16, F8_E4M3)
    FATTN_VEC_CASES_ALL_D(Q4_0, F8_E4M3)
    FATTN_VEC_CASES_ALL_D(Q4_1, F8_E4M3)
    FATTN_VEC_CASES_ALL_D(Q5_0, F8_E4M3)
    FATTN_VEC_CASES_ALL_D(Q5_1, F8_E4M3)
    FATTN_VEC_CASES_ALL_D(Q8_0, F8_E4M3)
    FATTN_VEC_CASES_ALL_D(BF16, F8_E4M3)
    FATTN_VEC_CASES_ALL_D(F8_E4M3, F8_E4M3)

    return nullptr;
}

static void ggml_cuda_flash_attn_ext_vec(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    fattn_vec_case_t vec_case = ggml_cuda_get_fattn_vec_case(Q->ne[0], K->type, V->type);
    if (vec_case == nullptr) {
        static bool warned = false;
        if (!warned) {
            GGML_LOG_WARN("%s: no FlashAttention vector kernel compiled for K/V types %s-%s, converting K and V to f16 instead (slow). "
                "Add \"%s-%s\" to GGML_CUDA_FA_QUANTS to compile it.\n",
                __func__, ggml_type_name(K->type), ggml_type_name(V->type), ggml_type_name(K->type), ggml_type_name(V->type));
            warned = true;
        }
        vec_case = ggml_cuda_get_fattn_vec_case(Q->ne[0], GGML_TYPE_F16, GGML_TYPE_F16);
    }
    GGML_ASSERT(vec_case != nullptr);
    vec_case(ctx, dst);
}

// Best FlashAttention kernel for a specific GPU:
enum best_fattn_kernel {
    BEST_FATTN_KERNEL_NONE    =   0,
    BEST_FATTN_KERNEL_TILE    = 200,
    BEST_FATTN_KERNEL_VEC     = 100,
    BEST_FATTN_KERNEL_MMA_F16 = 400,
};

// K/V types for which there is a vector kernel template instance, other kernels convert these to f16:
static bool ggml_cuda_fattn_kv_type_supported(const ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:
        case GGML_TYPE_BF16:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_F8_E4M3:
            return true;
        default:
            return false;
    }
}

bool ggml_cuda_flash_attn_ext_banded_use_mma(const int device, const ggml_tensor * dst) {
    if (dst->op != GGML_OP_FLASH_ATTN_EXT_BANDED) {
        return false;
    }

    const ggml_tensor * Q   = dst->src[0];
    const ggml_tensor * K   = dst->src[1];
    const ggml_tensor * V   = dst->src[2];
    const ggml_tensor * rel = dst->src[6];

    if (!Q || !K || !V || !rel) {
        return false;
    }

    // the MMA kernel passes rel in the sinks slot
    if (dst->src[4]) {
        return false;
    }

    // the MMA ABI indexes rel linearly and by Q's batch
    if (rel->type != GGML_TYPE_F32 || !ggml_is_contiguous(rel) ||
        rel->ne[3] != Q->ne[3] || rel->ne[0] > (1 << 20)) {
        return false;
    }

    if (K->type == GGML_TYPE_F32 || V->type == GGML_TYPE_F32) {
        return false;
    }

    if ((Q->ne[0] != 64 && Q->ne[0] != 128) || V->ne[0] != Q->ne[0]) {
        return false;
    }

    // local (RDNA3): the far mode (rel carries the far bias and the folded sink) runs on the WMMA kernel; the mask must be
    // shared by the heads and K/V padded, as for the GQA optimization, and the batch large enough for the wide tiles. Only
    // run on RDNA3 (ggml_cuda_ink2_arch_ok): other GPUs report the far mode unsupported and the model takes its stock path.
    if (((const float *) dst->op_params)[6] != 0.0f) {
        const ggml_tensor * mask = dst->src[3];
        // local: no mask = implicit causal mask (position-contiguous cache), not for a rotated ring
        const bool mask_ok = mask ? mask->ne[2] == 1 : ggml_flash_attn_ext_banded_get_far_rot(dst) == 0;
        if (!(amd_wmma_available(ggml_cuda_info().devices[device].cc) && ggml_cuda_ink2_arch_ok(device) && mask_ok &&
                K->ne[1] % FATTN_KQ_STRIDE == 0 && Q->ne[1] >= 1 && Q->ne[2] / K->ne[2] > 1)) {
            return false;
        }
        // same alignment rule as the GQA optimization: without it the dispatch would pick ncols2 = 1,
        // a variant that is not compiled for WMMA (local: a q8_0 K/V is read through a contiguous F16 copy made at launch,
        // see launch_fattn, so its own strides do not matter)
        for (const ggml_tensor * t : {Q, K, V, mask}) {
            if (!t || ((t->type == GGML_TYPE_Q8_0 || t->type == GGML_TYPE_Q4_0) && (t == K || t == V))) {
                continue;
            }
            for (size_t i = 1; i < GGML_MAX_DIMS; ++i) {
                if (t->nb[i] % 16 != 0) {
                    return false;
                }
            }
        }
        return true;
    }

    return turing_mma_available(ggml_cuda_info().devices[device].cc);
}

static best_fattn_kernel ggml_cuda_get_best_fattn_kernel(const int device, const ggml_tensor * dst) {
#ifndef FLASH_ATTN_AVAILABLE
    GGML_UNUSED(device); GGML_UNUSED(dst);
    return BEST_FATTN_KERNEL_NONE;
#endif// FLASH_ATTN_AVAILABLE

    const ggml_tensor * KQV   = dst;
    const ggml_tensor * Q     = dst->src[0];
    const ggml_tensor * K     = dst->src[1];
    const ggml_tensor * V     = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];
    const ggml_tensor * kv_rows = dst->src[5];

    // with kv_rows the kernel iterates over the mask columns
    const int64_t n_kv = kv_rows ? kv_rows->ne[0] : K->ne[1];

    const int gqa_ratio = Q->ne[2] / K->ne[2];
    GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);

    float max_bias = 0.0f;
    memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

    // The effective batch size for the kernel can be increased by gqa_ratio.
    // The kernel versions without this optimization are also used for ALiBi, if there is no mask, if the mask has a head dimension,
    //     or if the KV cache is not padded,
    bool gqa_opt_applies = gqa_ratio >= 2 && mask && mask->ne[2] == 1 && max_bias == 0.0f && n_kv % FATTN_KQ_STRIDE == 0;
    for (const ggml_tensor * t : {Q, K, V, mask}) {
        if (t == nullptr || ggml_is_quantized(t->type)) {
            continue;
        }
        for (size_t i = 1; i < GGML_MAX_DIMS; ++i) {
            if (t->nb[i] % 16 != 0) {
                gqa_opt_applies = false;
                break;
            }
        }
    }

    const int cc = ggml_cuda_info().devices[device].cc;

    if (ggml_cuda_flash_attn_ext_banded_use_mma(device, dst)) {
        return BEST_FATTN_KERNEL_MMA_F16;
    }

    switch (K->ne[0]) {
        case  40:
        case  64:
        case  72:
        case  80:
        case  96:
        case 128:
        case 112:
        case 256:
            if (V->ne[0] != K->ne[0]) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        case 192:
            if (V->ne[0] != 128 || !gqa_opt_applies) {
                return BEST_FATTN_KERNEL_NONE;
            }
            if (gqa_ratio % 8 != 0) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        case 320:
            if (V->ne[0] != 256 || !gqa_opt_applies) {
                return BEST_FATTN_KERNEL_NONE;
            }
            if (gqa_ratio % 32 != 0) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        case 512:
            if (V->ne[0] != K->ne[0]) {
                return BEST_FATTN_KERNEL_NONE;
            }
            if (!gqa_opt_applies) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        case 576:
            if (V->ne[0] != 512) {
                return BEST_FATTN_KERNEL_NONE;
            }
            if (!gqa_opt_applies) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        default:
            return BEST_FATTN_KERNEL_NONE;
    }

    if (!ggml_cuda_fattn_kv_type_supported(K->type) || !ggml_cuda_fattn_kv_type_supported(V->type)) {
        return BEST_FATTN_KERNEL_NONE;
    }

    if (mask && mask->ne[2] != 1) {
        // the kv_rows mask is compacted per sequence, it has no head dimension
        if (kv_rows) {
            return BEST_FATTN_KERNEL_NONE;
        }
        // the logit softcap kernels are compiled for these head sizes only
        float logit_softcap;
        memcpy(&logit_softcap, (const float *) KQV->op_params + 2, sizeof(float));
        if (logit_softcap != 0.0f && Q->ne[0] != 128 && Q->ne[0] != 256) {
            return BEST_FATTN_KERNEL_NONE;
        }
    }

    // only the MMA kernel reads K/V through kv_rows
    if (kv_rows) {
        return GGML_CUDA_CC_IS_NVIDIA(cc) && turing_mma_available(cc) && mask && Q->ne[0] != 40 && Q->ne[0] != 72 ?
            BEST_FATTN_KERNEL_MMA_F16 : BEST_FATTN_KERNEL_NONE;
    }

    // For small batch sizes the vector kernel may be preferable over the kernels optimized for large batch sizes:
    // 192 satisfies % 64 == 0 but has no vec instance (DKQ != DV); force it onto the MMA path.
    const bool can_use_vector_kernel = Q->ne[0] <= 256 && Q->ne[0] % 64 == 0 && Q->ne[0] != 192 && K->ne[1] % FATTN_KQ_STRIDE == 0;

    // If Turing tensor cores are available, use them:
    if (turing_mma_available(cc)) {
        if (can_use_vector_kernel) {
            if (!ggml_is_quantized(K->type) && !ggml_is_quantized(V->type)) {
                // the sparse gather exists only in the MMA kernel: (DKQ, DV, 1, 8) with GQA > 4
                const bool sparse_decode = gqa_opt_applies && gqa_ratio > 4 &&
                    ggml_cuda_flash_attn_ext_mma_f16_may_use_sparse(K->ne[0], V->ne[0], 1, 8) &&
                    ggml_cuda_flash_attn_ext_mma_f16_shall_use_sparse(cc, dst, 1, 8);
                if (!sparse_decode && cc >= GGML_CUDA_CC_ADA_LOVELACE && Q->ne[1] == 1 && Q->ne[3] == 1 &&
                        !(gqa_ratio > 4 && (Q->ne[0] >= 256 || K->ne[1] >= 8192))) {
                    return BEST_FATTN_KERNEL_VEC;
                }
            } else {
                if (cc >= GGML_CUDA_CC_ADA_LOVELACE) {
                    if (Q->ne[1] <= 2) {
                        return BEST_FATTN_KERNEL_VEC;
                    }
                } else {
                    if (Q->ne[1] == 1) {
                        return BEST_FATTN_KERNEL_VEC;
                    }
                }
            }
            if (!gqa_opt_applies && Q->ne[1] == 1) {
                return BEST_FATTN_KERNEL_VEC;
            }
        }
        return BEST_FATTN_KERNEL_MMA_F16;
    }

    const int ncols2_max = Q->ne[0] == 320 ? 32 : ((Q->ne[0] == 576 || Q->ne[0] == 192) ? 16 : 8);
    int gqa_ratio_eff = 1;
    while (max_bias == 0.0f && (!mask || mask->ne[2] == 1) && gqa_ratio % (2*gqa_ratio_eff) == 0 && gqa_ratio_eff < ncols2_max) {
        gqa_ratio_eff *= 2;
    }

    if (volta_mma_available(cc) && Q->ne[0] != 40 && Q->ne[0] != 72) {
        if (can_use_vector_kernel && Q->ne[1] * gqa_ratio_eff <= 2) {
            return BEST_FATTN_KERNEL_VEC;
        }
        if (Q->ne[1] * gqa_ratio_eff <= 16) {
            return BEST_FATTN_KERNEL_TILE; // On Volta tensor cores are only faster for sufficiently large matrices.
        }
        return BEST_FATTN_KERNEL_MMA_F16;
    }

    // On CDNA, the sparse MMA kernel is used for DKQ=576 regardless of the head size heuristics.
    if (amd_mfma_available(cc) && ggml_cuda_flash_attn_ext_mma_f16_shall_use_sparse(cc, dst, 1, 16)) {
        return BEST_FATTN_KERNEL_MMA_F16;
    }

    // AMD MFMA needs a certain minimum batch size to outscale the tile kernel for large head sizes.
    if (amd_mfma_available(cc) && Q->ne[0] != 40 && Q->ne[0] != 72) {
        if ((Q->ne[0] <= 64 && Q->ne[1] * gqa_ratio_eff > 8)) {
            return BEST_FATTN_KERNEL_MMA_F16;
        }
        if ((Q->ne[0] <= 128 && Q->ne[1] * gqa_ratio_eff > 16)) {
            return BEST_FATTN_KERNEL_MMA_F16;
        }
        if ((Q->ne[0] <= 256 && Q->ne[1] * gqa_ratio_eff > 64)) {
            return BEST_FATTN_KERNEL_MMA_F16;
        }
        if (Q->ne[0] > 256 && gqa_opt_applies && Q->ne[1] * gqa_ratio_eff > 128) {
            return BEST_FATTN_KERNEL_MMA_F16;
        }
    }

    // AMD WMMA is faster than the tile kernel if the wide tiles with high arithmetic intensity can be utilized.
    // local (RDNA3): a mask with a head dimension (Inkling 2 relative bias) disables the GQA optimization, which sent prefill
    // to the tile kernel. GGML_HIP_FATTN_WMMA_HEADMASK=N lets WMMA take it from N query rows up (0/unset = off); experimental
    // (can hang the GPU), only in builds with GGML_CUDA_INK2_COST_TOOLS.
#if defined(GGML_CUDA_INK2_COST_TOOLS)
    static const int wmma_headmask_min_q = []() {
        const char * e = getenv("GGML_HIP_FATTN_WMMA_HEADMASK");
        return e ? atoi(e) : 0;
    }();
#else
    static const int wmma_headmask_min_q = 0;
#endif
    const bool wmma_headmask = wmma_headmask_min_q > 0 && mask && mask->ne[2] != 1 && max_bias == 0.0f &&
        n_kv % FATTN_KQ_STRIDE == 0 && Q->ne[1] >= wmma_headmask_min_q;
    if ((amd_wmma_available(cc) && (gqa_opt_applies || wmma_headmask) && Q->ne[0] <= 256) && Q->ne[0] != 40 && Q->ne[0] != 72 &&
            Q->ne[1] * gqa_ratio_eff > (Q->ne[0] <= 128 ? 8 : 16)) {
        return BEST_FATTN_KERNEL_MMA_F16;
    }

    // local (RDNA3 only, ggml_cuda_ink2_arch_ok): a single query row with a per-head mask (Inkling 2 decode) goes to the tile
    // kernel, faster than the vector kernel at long context (test-backend-ops perf, 24/8 heads, 51200 keys: 245 us instead of
    // 312; server decode after 51k: 15.4 -> 14.9 ms per token). GGML_CUDA_INK2_FA_TILE1=0 restores the vector kernel.
    // local (RDNA3): a quantized cache (q8_0 for long contexts) is read in place by the vector kernel up to 8 query rows (one
    // token and the verify batches of speculative decoding); the tile kernel would first copy the whole K/V view to F16 at
    // every call. GGML_CUDA_INK2_FA_QVEC=0 restores the stock choice.
    static const bool ink2_fa_qvec = []() { const char * e = getenv("GGML_CUDA_INK2_FA_QVEC"); return !(e && e[0] == '0'); }();
    const bool kv_quantized = ggml_is_quantized(K->type) || ggml_is_quantized(V->type);
    if (ink2_fa_qvec && kv_quantized && ggml_cuda_ink2_arch_ok(device) && can_use_vector_kernel && Q->ne[1] <= 8 &&
            ggml_cuda_get_fattn_vec_case(Q->ne[0], K->type, V->type) != nullptr) {
        return BEST_FATTN_KERNEL_VEC;
    }
    static const bool ink2_fa_tile1 = []() { const char * e = getenv("GGML_CUDA_INK2_FA_TILE1"); return !(e && e[0] == '0'); }();
    if (ink2_fa_tile1 && ggml_cuda_ink2_arch_ok(device) && mask && mask->ne[2] != 1 && Q->ne[1] == 1 && !kv_quantized) {
        return BEST_FATTN_KERNEL_TILE;
    }

    // If there are no tensor cores available, use the generic tile kernel:
    if (can_use_vector_kernel) {
        if (!ggml_is_quantized(K->type) && !ggml_is_quantized(V->type)) {
            if (Q->ne[1] == 1) {
                if (!gqa_opt_applies) {
                    return BEST_FATTN_KERNEL_VEC;
                }
            }
        } else {
            if (Q->ne[1] <= 2) {
                return BEST_FATTN_KERNEL_VEC;
            }
        }
    }
    return BEST_FATTN_KERNEL_TILE;
}

size_t ggml_cuda_flash_attn_ext_get_alloc_size(int device, const ggml_tensor * dst) {
    GGML_ASSERT(dst->op == GGML_OP_FLASH_ATTN_EXT || dst->op == GGML_OP_FLASH_ATTN_EXT_BANDED);

    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    GGML_ASSERT(K != nullptr);
    GGML_ASSERT(V != nullptr);

    const best_fattn_kernel kernel = ggml_cuda_get_best_fattn_kernel(device, dst);

    bool need_f16_K = false;
    bool need_f16_V = false;

    switch (kernel) {
        case BEST_FATTN_KERNEL_TILE:
        case BEST_FATTN_KERNEL_MMA_F16:
            need_f16_K = true;
            need_f16_V = true;
            break;
        case BEST_FATTN_KERNEL_VEC: {
            const bool f16_fallback = ggml_cuda_get_fattn_vec_case(Q->ne[0], K->type, V->type) == nullptr;
            need_f16_K = K->type == GGML_TYPE_F32 || f16_fallback;
            need_f16_V = V->type == GGML_TYPE_F32 || f16_fallback;
        } break;
        case BEST_FATTN_KERNEL_NONE:
            break;
    }

    const ggml_cuda_flash_attn_ext_f16_extra_data f16_extra =
        ggml_cuda_flash_attn_ext_get_f16_extra_data(dst, need_f16_K, need_f16_V);

    return f16_extra.end - (uintptr_t) dst->data;
}

// local (RDNA3): two to eight query rows (the verify batch of speculative decoding) over a q8_0 cache, D 128, G query heads per
// K/V head. The vector kernel runs one block per query head and pair of rows, so every K/V row is loaded and dequantized 2 G
// times and the kernel ends up far from the memory bound (126k keys, 4 rows, 24/8 heads: ~1 ms per call for 274 MB; this
// kernel ~0.5 ms). One decoded row stays on the vector kernel (this kernel is faster per call, 438 vs 519 us at 126k keys,
// but no gain was measurable per token at 51k).
// Here one block takes the T rows of the G heads of one K/V head (T G columns) over a slice of the keys (split-K, combined
// by flash_attn_combine_results like the vector kernel): each lane scores one key for all columns (int8 dot products with
// Q quantized to q8_1 once per block, as in the vector kernel), then the warp accumulates P V with a lane per 4 dimensions
// (FP16 products of two keys at a time, F32 sums). Each warp keeps its own softmax state; the warps of a block are merged at
// the end. Per-head or shared masks. GGML_CUDA_INK2_FA_Q8GQA=0 keeps the vector kernel.
#define INK2_FQ_D   128
#define INK2_FQ_NW  4   // warps per block
#define INK2_FQ_C   32  // keys per warp and step (one per lane)

#if defined(RDNA3)
typedef _Float16 ink2_fq_h2 __attribute__((ext_vector_type(2)));

static __device__ __forceinline__ uint32_t ink2_fq_u32_at(const uint32_t * w, const int off) {
    // off is a compile-time constant after unrolling: aligned word or two words joined at a 2-byte offset
    return (off % 4 == 0) ? w[off/4] : __builtin_amdgcn_alignbyte(w[off/4 + 1], w[off/4], off % 4);
}
#endif // defined(RDNA3)

// V4: q4_0 V instead of q8_0 (K stays q8_0): the lane's 4 values of a key are the low (values 0-15 of the block) or high
// (16-31) nibbles of 4 bytes, turned into FP16 the same way (1024 + n, exact) before the scale
template <int T, int G, bool MS, bool V4>
static __global__ void __launch_bounds__(INK2_FQ_NW*WARP_SIZE, 2) ink2_fattn_q8_gqa(
        const char * __restrict__ Q_ptr, const char * __restrict__ K_ptr, const char * __restrict__ V_ptr,
        const char * __restrict__ mask_ptr, const char * __restrict__ sinks_ptr, const int * __restrict__ KV_max_ptr,
        const int * __restrict__ KV_rows_ptr, float * __restrict__ dst, float2 * __restrict__ dst_meta,
        const float scale, const float max_bias, const float m0, const float m1, const uint32_t n_head_log2, const float logit_softcap,
        const int32_t ne00, const uint3 ne01, const int32_t ne02, const int32_t ne03,
                            const int32_t nb01, const int32_t nb02, const int32_t nb03,
        const int32_t ne10, const int32_t ne11, const int32_t ne12, const int32_t ne13,
                            const int32_t nb11, const int32_t nb12, const int64_t nb13,
                            const int32_t nb21, const int32_t nb22, const int64_t nb23,
        const int32_t ne31, const int32_t ne32, const int32_t ne33,
        const int32_t nb31, const int64_t nb32, const int64_t nb33) {
#if defined(RDNA3)
    constexpr int D = INK2_FQ_D, NW = INK2_FQ_NW, C = INK2_FQ_C, NC = T*G;
    GGML_UNUSED_VARS(KV_max_ptr, KV_rows_ptr, max_bias, m0, m1, n_head_log2, logit_softcap, ne00, ne03, ne10, ne13, ne31);

    __shared__ __align__(16) int   q_qs[NC][D/4];       // Q*scale as int8, 4 per int
    __shared__ __align__(16) float q_d[NC][D/32];       // q8_1 block scales
    __shared__ __align__(16) half  p_h[NW][C/2][NC][2]; // probabilities of a step, per warp: [key pair][column][key of the pair]
    __shared__ __align__(16) float o_s[NW*D];           // merge of the warps, one column at a time
    __shared__ float m_s[NW][NC], l_s[NW][NC];

    const int lane = threadIdx.x, warp = threadIdx.y;
    const int n_q  = int(ne01.z);
    const int ic0  = blockIdx.x*T;
    const int seq  = blockIdx.z / ne12;
    const int kvh  = blockIdx.z - seq*ne12;

    // column c = (row c % T, head kvh G + c / T)
    for (int c = warp; c < NC; c += NW) {
        const int t = c % T, h = kvh*G + c / T;
        float4 x = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (ic0 + t < n_q) {
            x = *(const float4 *) (Q_ptr + seq*nb03 + h*nb02 + (ic0 + t)*nb01 + 16*lane);
        }
        x.x *= scale; x.y *= scale; x.z *= scale; x.w *= scale;
        float amax = fmaxf(fmaxf(fabsf(x.x), fabsf(x.y)), fmaxf(fabsf(x.z), fabsf(x.w)));
#pragma unroll
        for (int o = 1; o < 8; o <<= 1) {
            amax = fmaxf(amax, __shfl_xor(amax, o, WARP_SIZE));
        }
        const float d  = amax/127.0f;
        const float id = d > 0.0f ? 1.0f/d : 0.0f;
        const int q0 = __float2int_rn(x.x*id), q1 = __float2int_rn(x.y*id), q2 = __float2int_rn(x.z*id), q3 = __float2int_rn(x.w*id);
        q_qs[c][lane] = (q0 & 0xFF) | ((q1 & 0xFF) << 8) | ((q2 & 0xFF) << 16) | (int(uint32_t(q3) << 24));
        if (lane % 8 == 0) {
            q_d[c][lane/8] = d;
        }
    }
    __syncthreads();

    float m[NC], l[NC], o[NC][4];
#pragma unroll
    for (int c = 0; c < NC; ++c) {
        m[c] = -FLT_MAX/2.0f;
        l[c] = 0.0f;
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            o[c][e] = 0.0f;
        }
    }

    const char * Kh = K_ptr + seq*nb13 + kvh*nb12;
    const char * Vh = V_ptr + seq*nb23 + kvh*nb22;
    // MS: several streams, one mask per stream (or one for all)
    const char * Mh = mask_ptr ? mask_ptr + (MS && ne33 > 1 ? seq*nb33 : 0) : nullptr;
    const int vb = lane/8, vj = lane%8; // V: the lane's 4 dimensions are values 4 vj .. 4 vj + 3 of block vb

    for (int k0 = (blockIdx.y*NW + warp)*C; k0 < ne11; k0 += gridDim.y*NW*C) {
        const int key = k0 + lane;
        float s[NC];
        if (key < ne11) {
            // the 136-byte row of 4 q8_0 blocks (8-byte aligned)
            uint32_t w[34];
            const uint2 * kr = (const uint2 *) (Kh + (int64_t) key*nb11);
#pragma unroll
            for (int i = 0; i < 17; ++i) {
                const uint2 u = kr[i];
                w[2*i + 0] = u.x;
                w[2*i + 1] = u.y;
            }
            float dk[4];
#pragma unroll
            for (int b = 0; b < 4; ++b) {
                const uint32_t bits = ((34*b) % 4 == 0) ? (w[(34*b)/4] & 0xFFFF) : (w[(34*b)/4] >> 16);
                half hd;
                const uint16_t b16 = (uint16_t) bits;
                memcpy(&hd, &b16, sizeof(hd));
                dk[b] = __half2float(hd);
            }
#pragma unroll
            for (int c = 0; c < NC; ++c) {
                float acc = 0.0f;
#pragma unroll
                for (int b = 0; b < 4; ++b) {
                    const int4 qa = *(const int4 *) &q_qs[c][8*b + 0];
                    const int4 qb = *(const int4 *) &q_qs[c][8*b + 4];
                    int isum = 0;
                    isum = ggml_cuda_dp4a((int) ink2_fq_u32_at(w, 34*b + 2 +  0), qa.x, isum);
                    isum = ggml_cuda_dp4a((int) ink2_fq_u32_at(w, 34*b + 2 +  4), qa.y, isum);
                    isum = ggml_cuda_dp4a((int) ink2_fq_u32_at(w, 34*b + 2 +  8), qa.z, isum);
                    isum = ggml_cuda_dp4a((int) ink2_fq_u32_at(w, 34*b + 2 + 12), qa.w, isum);
                    isum = ggml_cuda_dp4a((int) ink2_fq_u32_at(w, 34*b + 2 + 16), qb.x, isum);
                    isum = ggml_cuda_dp4a((int) ink2_fq_u32_at(w, 34*b + 2 + 20), qb.y, isum);
                    isum = ggml_cuda_dp4a((int) ink2_fq_u32_at(w, 34*b + 2 + 24), qb.z, isum);
                    isum = ggml_cuda_dp4a((int) ink2_fq_u32_at(w, 34*b + 2 + 28), qb.w, isum);
                    acc += dk[b]*q_d[c][b]*(float) isum;
                }
                const int t = c % T, h = kvh*G + c / T;
                if (ic0 + t >= n_q) {
                    acc = -INFINITY;
                } else if (Mh) {
                    acc += __half2float(*(const half *) (Mh + (h % ne32)*nb32 + (int64_t) (ic0 + t)*nb31 + (int64_t) key*sizeof(half)));
                }
                s[c] = acc;
            }
        } else {
#pragma unroll
            for (int c = 0; c < NC; ++c) {
                s[c] = -INFINITY;
            }
        }

        // online softmax per column over the warp's keys; the lane keeps a partial row sum (reduced at the end)
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            const float m_new = fmaxf(m[c], warp_reduce_max(s[c]));
            const float f     = expf(m[c] - m_new);
            const float p     = expf(s[c] - m_new);
            m[c] = m_new;
            l[c] = l[c]*f + p;
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                o[c][e] *= f;
            }
            p_h[warp][lane/2][c][lane%2] = __float2half(p);
        }
        ggml_cuda_syncwarp();

        // P V: the lane adds its 4 dimensions of the warp's keys, two keys per dot product (FP16 products as in the other
        // attention kernels, F32 sums): V of a key pair as half2 (v of key 2 kp, v of key 2 kp + 1) per dimension
#pragma unroll 4
        for (int kp = 0; kp < C/2; ++kp) {
            const int ka = k0 + 2*kp;
            if (ka >= ne11) {
                break;
            }
            ink2_fq_h2 v2[4];
            if constexpr (!V4) {
            const char * va = Vh + (int64_t) ka*nb21 + 34*vb;
            const char * vb2 = ka + 1 < ne11 ? va + nb21 : va; // past the end: any row, its probability is 0
            const uint32_t dd = uint32_t(*(const uint16_t *) va) | (uint32_t(*(const uint16_t *) vb2) << 16);
            const uint32_t qa = (uint32_t(*(const uint16_t *) (va  + 2 + 4*vj)) | (uint32_t(*(const uint16_t *) (va  + 4 + 4*vj)) << 16)) ^ 0x80808080u;
            const uint32_t qb = (uint32_t(*(const uint16_t *) (vb2 + 2 + 4*vj)) | (uint32_t(*(const uint16_t *) (vb2 + 4 + 4*vj)) << 16)) ^ 0x80808080u;
            ink2_fq_h2 d2;
            memcpy(&d2, &dd, sizeof(d2));
            const ink2_fq_h2 off2 = { (_Float16) 1152.0f, (_Float16) 1152.0f };
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                // bytes e of the two keys as 0x64xx: 1024 + (q + 128) in FP16, exact
                const uint32_t ab = __builtin_amdgcn_perm(qb, qa, uint32_t(e) | (uint32_t(4 + e) << 8));
                const uint32_t h  = __builtin_amdgcn_perm(0x64646464u, ab, 0x04010400u);
                ink2_fq_h2 x;
                memcpy(&x, &h, sizeof(x));
                v2[e] = (x - off2)*d2;
            }
            } else {
            // q4_0 block of 18 bytes: scale, then 16 bytes holding values j (low nibble) and j + 16 (high nibble)
            const char * va = Vh + (int64_t) ka*nb21 + 18*vb;
            const char * vb2 = ka + 1 < ne11 ? va + nb21 : va; // past the end: any row, its probability is 0
            const uint32_t dd = uint32_t(*(const uint16_t *) va) | (uint32_t(*(const uint16_t *) vb2) << 16);
            const int qo = 2 + 4*(vj & 3);
            const int sh = vj >= 4 ? 4 : 0;
            const uint32_t qa = ((uint32_t(*(const uint16_t *) (va  + qo)) | (uint32_t(*(const uint16_t *) (va  + qo + 2)) << 16)) >> sh) & 0x0F0F0F0Fu;
            const uint32_t qb = ((uint32_t(*(const uint16_t *) (vb2 + qo)) | (uint32_t(*(const uint16_t *) (vb2 + qo + 2)) << 16)) >> sh) & 0x0F0F0F0Fu;
            ink2_fq_h2 d2;
            memcpy(&d2, &dd, sizeof(d2));
            const ink2_fq_h2 off2 = { (_Float16) 1032.0f, (_Float16) 1032.0f };
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                // nibbles e of the two keys as 0x64xx: 1024 + n in FP16, exact; n - 8 is the q4_0 value
                const uint32_t ab = __builtin_amdgcn_perm(qb, qa, uint32_t(e) | (uint32_t(4 + e) << 8));
                const uint32_t h  = __builtin_amdgcn_perm(0x64646464u, ab, 0x04010400u);
                ink2_fq_h2 x;
                memcpy(&x, &h, sizeof(x));
                v2[e] = (x - off2)*d2;
            }
            }
            ink2_fq_h2 p2[NC];
#pragma unroll
            for (int c4 = 0; c4 < NC; c4 += 4) {
                if (c4 + 4 <= NC) {
                    const uint4 u = *(const uint4 *) &p_h[warp][kp][c4][0];
                    memcpy(&p2[c4 + 0], &u.x, 4);
                    memcpy(&p2[c4 + 1], &u.y, 4);
                    memcpy(&p2[c4 + 2], &u.z, 4);
                    memcpy(&p2[c4 + 3], &u.w, 4);
                } else {
#pragma unroll
                    for (int cc = 0; c4 + cc < NC; ++cc) {
                        memcpy(&p2[c4 + cc], &p_h[warp][kp][c4 + cc][0], 4);
                    }
                }
            }
#pragma unroll
            for (int c = 0; c < NC; ++c) {
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    o[c][e] = __builtin_amdgcn_fdot2(p2[c], v2[e], o[c][e], false);
                }
            }
        }
        ggml_cuda_syncwarp();
    }

    // row sums of the warp; the sink (one warp of the first key slice)
#pragma unroll
    for (int c = 0; c < NC; ++c) {
        l[c] = warp_reduce_sum(l[c]);
    }
    if (sinks_ptr && blockIdx.y == 0 && warp == 0) {
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            const float sink  = ((const float *) sinks_ptr)[kvh*G + c / T];
            const float m_new = fmaxf(m[c], sink);
            const float f     = expf(m[c] - m_new);
            l[c] = l[c]*f + expf(sink - m_new);
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                o[c][e] *= f;
            }
            m[c] = m_new;
        }
    }
    if (lane == 0) {
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            m_s[warp][c] = m[c];
            l_s[warp][c] = l[c];
        }
    }
    __syncthreads();

    // merge the warps: column by column, every warp scales its part, the block sums them
#pragma unroll
    for (int c = 0; c < NC; ++c) {
        float mb = m_s[0][c];
#pragma unroll
        for (int w2 = 1; w2 < NW; ++w2) {
            mb = fmaxf(mb, m_s[w2][c]);
        }
        float lb = 0.0f;
#pragma unroll
        for (int w2 = 0; w2 < NW; ++w2) {
            lb += l_s[w2][c]*expf(m_s[w2][c] - mb);
        }
        const float f = expf(m[c] - mb);
        *(float4 *) &o_s[warp*D + 4*lane] = make_float4(o[c][0]*f, o[c][1]*f, o[c][2]*f, o[c][3]*f);
        __syncthreads();
        const int t = c % T, h = kvh*G + c / T;
        if (ic0 + t < n_q) {
            const int i = warp*WARP_SIZE + lane; // dimension
            float r = 0.0f;
#pragma unroll
            for (int w2 = 0; w2 < NW; ++w2) {
                r += o_s[w2*D + i];
            }
            const int64_t j = ((int64_t) (seq*n_q + ic0 + t)*ne02 + h)*gridDim.y + blockIdx.y;
            dst[j*D + i] = gridDim.y == 1 ? r/lb : r;
            if (gridDim.y > 1 && i == 0) {
                dst_meta[j] = make_float2(mb, lb);
            }
        }
        __syncthreads();
    }
#else
    GGML_UNUSED_VARS(Q_ptr, K_ptr, V_ptr, mask_ptr, sinks_ptr, KV_max_ptr, KV_rows_ptr, dst, dst_meta, scale, max_bias, m0, m1,
        n_head_log2, logit_softcap, ne00, ne01, ne02, ne03, nb01, nb02, nb03, ne10, ne11, ne12, ne13, nb11, nb12, nb13,
        nb21, nb22, nb23, ne31, ne32, ne33, nb31, nb32, nb33);
    NO_DEVICE_CODE;
#endif // defined(RDNA3)
}

// whether the kernel above takes this op; GGML_CUDA_INK2_FA_Q8GQA=0 leaves it to the vector kernel
static bool ggml_cuda_ink2_fattn_q8_gqa_ok(const int device, const ggml_tensor * dst) {
    static const bool enabled = []() { const char * e = getenv("GGML_CUDA_INK2_FA_Q8GQA"); return !(e && e[0] == '0'); }();
    const ggml_tensor * Q = dst->src[0], * K = dst->src[1], * V = dst->src[2], * mask = dst->src[3], * sinks = dst->src[4];
    if (!enabled || dst->op != GGML_OP_FLASH_ATTN_EXT || dst->src[5] || !ggml_cuda_ink2_arch_ok(device)) {
        return false;
    }
    float max_bias, logit_softcap;
    memcpy(&max_bias,      (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));
    const auto al8 = [](size_t x) { return x % 8 == 0; };
    // several streams (slots of a server decoding at once): one or more rows per stream, each stream with its own K/V and mask
    // (one decoded row per stream included: the vector kernel there runs one block per query head and stream).
    // GGML_CUDA_INK2_FA_Q8GQA_MS=0 keeps them on the vector kernel.
    static const bool ms_enabled = []() { const char * e = getenv("GGML_CUDA_INK2_FA_Q8GQA_MS"); return !(e && e[0] == '0'); }();
    const int64_t n_stream = Q->ne[3];
    const bool    rows_ok  = n_stream == 1 ? Q->ne[1] >= 2 && Q->ne[1] <= 8 : ms_enabled && Q->ne[1] >= 1 && Q->ne[1] <= 8;
    // a q4_0 V (K q8_0) is read by the V4 variant; GGML_CUDA_INK2_FA_Q8GQA_V4=0 leaves it to the vector kernel
    static const bool v4_enabled = []() { const char * e = getenv("GGML_CUDA_INK2_FA_Q8GQA_V4"); return !(e && e[0] == '0'); }();
    const bool v_ok = V->type == GGML_TYPE_Q8_0 || (v4_enabled && V->type == GGML_TYPE_Q4_0);
    return K->type == GGML_TYPE_Q8_0 && v_ok && Q->type == GGML_TYPE_F32 &&
        Q->ne[0] == INK2_FQ_D && K->ne[0] == INK2_FQ_D && V->ne[0] == INK2_FQ_D && K->ne[2] == V->ne[2] &&
        K->ne[2] > 0 && Q->ne[2] == 3*K->ne[2] && rows_ok && n_stream >= 1 && K->ne[3] == n_stream && V->ne[3] == n_stream &&
        max_bias == 0.0f && logit_softcap == 0.0f && Q->nb[0] == sizeof(float) && Q->nb[1] % 16 == 0 && Q->nb[2] % 16 == 0 &&
        Q->nb[3] % 16 == 0 && (uintptr_t) Q->data % 16 == 0 && al8((uintptr_t) K->data) && al8(K->nb[1]) && al8(K->nb[2]) && al8(K->nb[3]) &&
        (uintptr_t) V->data % 2 == 0 && V->nb[1] % 2 == 0 && V->nb[2] % 2 == 0 && V->nb[3] % 2 == 0 && K->ne[1] < INT_MAX &&
        (!mask || (mask->type == GGML_TYPE_F16 && (mask->ne[3] == 1 || mask->ne[3] == n_stream) && (mask->ne[2] == 1 || mask->ne[2] == Q->ne[2]) &&
                   mask->ne[0] >= K->ne[1] && mask->ne[1] >= Q->ne[1])) &&
        (!sinks || (sinks->type == GGML_TYPE_F32 && ggml_is_contiguous(sinks)));
}

template <bool V4>
static void ggml_cuda_ink2_fattn_q8_gqa_v(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];
    // one row is taken only with several streams (see ggml_cuda_ink2_fattn_q8_gqa_ok)
    if (Q->ne[3] > 1) {
        if (Q->ne[1] == 1) {
            launch_fattn<INK2_FQ_D, 1, 3>(ctx, dst, ink2_fattn_q8_gqa<1, 3, true, V4>, INK2_FQ_NW, 0, INK2_FQ_NW*INK2_FQ_C, false, false, false, false);
        } else if (Q->ne[1] <= 4) {
            launch_fattn<INK2_FQ_D, 4, 3>(ctx, dst, ink2_fattn_q8_gqa<4, 3, true, V4>, INK2_FQ_NW, 0, INK2_FQ_NW*INK2_FQ_C, false, false, false, false);
        } else {
            launch_fattn<INK2_FQ_D, 8, 3>(ctx, dst, ink2_fattn_q8_gqa<8, 3, true, V4>, INK2_FQ_NW, 0, INK2_FQ_NW*INK2_FQ_C, false, false, false, false);
        }
    } else if (Q->ne[1] <= 4) {
        launch_fattn<INK2_FQ_D, 4, 3>(ctx, dst, ink2_fattn_q8_gqa<4, 3, false, V4>, INK2_FQ_NW, 0, INK2_FQ_NW*INK2_FQ_C, false, false, false, false);
    } else {
        launch_fattn<INK2_FQ_D, 8, 3>(ctx, dst, ink2_fattn_q8_gqa<8, 3, false, V4>, INK2_FQ_NW, 0, INK2_FQ_NW*INK2_FQ_C, false, false, false, false);
    }
}

static void ggml_cuda_ink2_fattn_q8_gqa(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    if (dst->src[2]->type == GGML_TYPE_Q4_0) {
        ggml_cuda_ink2_fattn_q8_gqa_v<true>(ctx, dst);
    } else {
        ggml_cuda_ink2_fattn_q8_gqa_v<false>(ctx, dst);
    }
}

void ggml_cuda_flash_attn_ext(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_set_device(ctx.device);
    switch (ggml_cuda_get_best_fattn_kernel(ggml_cuda_get_device(), dst)) {
        case BEST_FATTN_KERNEL_NONE:
            GGML_ABORT("fatal error");
        case BEST_FATTN_KERNEL_TILE:
            ggml_cuda_flash_attn_ext_tile(ctx, dst);
            break;
        case BEST_FATTN_KERNEL_VEC:
            if (ggml_cuda_ink2_fattn_q8_gqa_ok(ggml_cuda_get_device(), dst)) {
                ggml_cuda_ink2_fattn_q8_gqa(ctx, dst);
                break;
            }
            ggml_cuda_flash_attn_ext_vec(ctx, dst);
            break;
        case BEST_FATTN_KERNEL_MMA_F16:
            ggml_cuda_flash_attn_ext_mma_f16(ctx, dst);
            break;
    }
}

bool ggml_cuda_flash_attn_ext_supported(int device, const ggml_tensor * dst) {
    return ggml_cuda_get_best_fattn_kernel(device, dst) != BEST_FATTN_KERNEL_NONE;
}
