#include "models.h"
#include "../llama-memory-hybrid-iswa.h"

#include <algorithm>
#include <cstdlib>
#include <vector>

// max elements of the dense relative bias per attention call, and max chunks per layer
// local: on a device that runs the far op, the dense path is only the fallback of large ubatches when a cache is not
// position-contiguous: there the budget is 2^25 (with 2^29 a 1024-token ubatch made [2048, 1024, 24] F32 bias tensors,
// 192 MiB each, that grew the compute buffer by 117 MiB after load, and a fallback on the global cache would make 1.5 GiB
// ones; with 2^25 the bias is chunked, 96 / 384 MiB). Elsewhere the dense path is the main one and keeps the stock 2^29.
// INKLING2_KQ_BIAS_LOG2 sets both.
static const int64_t n_kq_bias_max_far = []() {
    const char * e = std::getenv("INKLING2_KQ_BIAS_LOG2");
    const int    n = e ? atoi(e) : 25;
    return int64_t(1) << (n >= 16 && n <= 32 ? n : 25);
}();
static const int64_t n_kq_bias_max = []() {
    const char * e = std::getenv("INKLING2_KQ_BIAS_LOG2");
    const int    n = e ? atoi(e) : 29;
    return int64_t(1) << (n >= 16 && n <= 32 ? n : 29);
}();
static constexpr int64_t n_kq_chunk_max = 64;
// ubatches up to this many tokens keep the dense path
static constexpr int64_t n_tokens_banded_min = 32;
// local (RDNA3): the far banded op also serves small ubatches (the verify batches of speculative decoding), from
// INKLING2_FAR_MIN tokens (default 2; a single token keeps the dense path and the vector kernel). INKLING2_FAR=0 disables far.
static const bool inkling2_far_enabled = []() {
    const char * e = std::getenv("INKLING2_FAR");
    return !(e && e[0] == '0');
}();
static const int64_t n_tokens_far_min = []() {
    const char * e = std::getenv("INKLING2_FAR_MIN");
    const int    n = e ? atoi(e) : 64; // off by default: it prevents the graph reuse of the verify batches
    return int64_t(n >= 1 ? n : 64);
}();
// local: INKLING2_BANDED=0 keeps the (chunked) dense path even where the banded op is supported
static const bool inkling2_banded_enabled = []() {
    const char * e = std::getenv("INKLING2_BANDED");
    return !(e && e[0] == '0');
}();

void llama_model_inkling2::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_ATTENTION_QK_CLIP,           hparams.f_qk_clip);
    ml.get_key(LLM_KV_ATTN_RES_BLOCK_SIZE,         hparams.attn_res_block_size);
    ml.get_key(LLM_KV_SHORTCONV_L_CACHE,           hparams.n_shortconv_l_cache);

    // relative-position bias, and the log-scaled positions of the full-attention layers
    ml.get_key(LLM_KV_ATTENTION_RELATIVE_LENGTH, hparams.n_relative);
    ml.get_key(LLM_KV_ATTENTION_RELATIVE_EXTENT, hparams.n_relative_extent);
    ml.get_key(LLM_KV_ATTENTION_RELATIVE_CLIP,   hparams.f_relative_clip);
    ml.get_key(LLM_KV_ATTENTION_LOG_SCALE_FLOOR, hparams.n_log_scale_floor);
    ml.get_key(LLM_KV_ATTENTION_LOG_SCALE_ALPHA, hparams.f_log_scale_alpha);

    // the embedding and output tensors are padded past the vocabulary
    ml.get_key(LLM_KV_VOCAB_SIZE,          hparams.n_vocab_padded);
    ml.get_key(LLM_KV_VOCAB_SIZE_UNPADDED, hparams.n_vocab_unpadded);

    // MoE
    ml.get_key_or_arr(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp_arr, hparams.n_layer_all);

    ml.get_key(LLM_KV_ACTIVATION_SITU_BETA,        hparams.situ_beta);
    ml.get_key(LLM_KV_ACTIVATION_SITU_LINEAR_BETA, hparams.situ_linear_beta);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE,        hparams.expert_weights_scale, false);

    hparams.f_final_logit_softcapping = 0.0f;
    ml.get_key(LLM_KV_FINAL_LOGIT_SOFTCAPPING, hparams.f_final_logit_softcapping, false);

    // sliding window
    ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW,         hparams.n_swa);
    ml.get_arr(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, hparams.is_swa_impl);

    uint32_t n_pattern = 0;
    ml.get_arr_n(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, n_pattern);

    if (n_pattern != hparams.n_layer() || !hparams.n_relative || !hparams.n_relative_extent || !hparams.n_swa ||
        !hparams.attn_res_block_size || !hparams.n_log_scale_floor || hparams.n_shortconv_l_cache < 2 ||
        hparams.n_shortconv_l_cache > 64 || hparams.n_relative_extent > INT32_MAX || hparams.n_swa > INT32_MAX ||
        !hparams.n_vocab_unpadded || hparams.n_vocab_unpadded > hparams.n_vocab_padded) {
        throw std::runtime_error("inkling2: invalid dimensions, vocabulary, or layer pattern");
    }

    for (float value : {hparams.f_norm_rms_eps, hparams.f_qk_clip, hparams.f_relative_clip, hparams.situ_beta, hparams.situ_linear_beta}) {
        if (!std::isfinite(value) || value <= 0.0f) {
            throw std::runtime_error("inkling2: normalization, clipping and scaling parameters must be finite and positive");
        }
    }

    if (!std::isfinite(hparams.f_log_scale_alpha) || hparams.f_log_scale_alpha < 0.0f) {
        throw std::runtime_error("inkling2: invalid logarithmic scale");
    }

    if (hparams.loop_n_pass > 1) {
        ml.get_key(LLM_KV_LOOP_GATE_RANK, hparams.loop_gate_rank);
        if (!hparams.loop_core_start || !hparams.loop_gate_rank || hparams.loop_core_start % hparams.attn_res_block_size ||
            (hparams.loop_core_end - hparams.loop_core_start) % hparams.attn_res_block_size) {
            throw std::runtime_error("inkling2: the looped core must start after the first block and span whole blocks");
        }
    }

    hparams.swa_type                = LLAMA_SWA_TYPE_STANDARD;
    hparams.expert_router_f32       = true;
    hparams.expert_weights_norm_eps = 1e-20f;

    // no RoPE, the relative-position bias carries the positions
    hparams.rope_pattern.fill(false);

    // the convolution state keeps K, V, the attention output and the FFN output of the widest layer
    uint64_t n_channels_max = 0;

    for (uint32_t il = 0; il < hparams.n_layer(); ++il) {
        if (!hparams.n_head_kv(il) || hparams.n_head(il) % hparams.n_head_kv(il) ||
            hparams.n_embd_head_k(il) != hparams.n_embd_head_v(il) || !hparams.n_expert_used(il) ||
            hparams.n_expert_used(il) > hparams.n_expert || !hparams.n_ff_exp(il)) {
            throw std::runtime_error("inkling2: invalid attention or expert dimensions");
        }

        n_channels_max = std::max<uint64_t>(n_channels_max, 2ull * hparams.n_embd_k_gqa(il) + 2ull * hparams.n_embd);
    }

    if (n_channels_max * (hparams.n_shortconv_l_cache - 1) > UINT32_MAX) {
        throw std::runtime_error("inkling2: convolution state is too large");
    }

    hparams.n_sconv_channels = n_channels_max;

    type = LLM_TYPE_UNKNOWN;
}

void llama_model_inkling2::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    if (hparams.n_vocab_unpadded != vocab.n_tokens()) {
        throw std::runtime_error("inkling2: tokenizer size must equal the unpadded vocabulary size");
    }

    const int64_t n_vocab_padded = hparams.n_vocab_padded;
    const int64_t n_rel          = hparams.n_relative;
    const int64_t n_conv         = hparams.n_shortconv_l_cache;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD,      "weight"),    {n_embd, n_vocab_padded}, 0);
    tok_norm = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD_NORM, "weight", 0), {n_embd}, 0);

    // output
    output           = create_tensor(tn(LLM_TENSOR_OUTPUT,           "weight"), {n_embd, n_vocab_padded}, 0);
    output_norm      = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM,      "weight"), {n_embd}, 0);
    output_res_query = create_tensor(tn(LLM_TENSOR_OUTPUT_RES_QUERY, "weight"), {n_embd}, 0);

    const bool loop = hparams.loop_n_pass > 1;

    for (int i = 0; i < n_layer - (int) hparams.loop_n_extra(); ++i) {
        auto & layer = layers[i];

        const int iv = hparams.loop_visit(i);

        const int64_t n_head      = hparams.n_head(iv);
        const int64_t n_head_kv   = hparams.n_head_kv(iv);
        const int64_t n_embd_head = hparams.n_embd_head_k(iv);
        const int64_t n_ff_exp    = hparams.n_ff_exp(iv);
        const int64_t n_extent    = hparams.is_swa(iv) ? hparams.n_swa : hparams.n_relative_extent;

        // attention
        layer.wq        = create_tensor(tn(LLM_TENSOR_ATTN_Q,    "weight", i), {n_embd, n_embd_head*n_head}, 0);
        layer.wk        = create_tensor(tn(LLM_TENSOR_ATTN_K,    "weight", i), {n_embd, n_embd_head*n_head_kv}, 0);
        layer.wv        = create_tensor(tn(LLM_TENSOR_ATTN_V,    "weight", i), {n_embd, n_embd_head*n_head_kv}, 0);
        layer.wo        = create_tensor(tn(LLM_TENSOR_ATTN_OUT,  "weight", i), {n_embd_head*n_head, n_embd}, 0);
        layer.wqkv_gate = create_tensor(tn(LLM_TENSOR_ATTN_GATE, "weight", i), {n_embd, n_embd_head*n_head}, 0);

        layer.attn_q_clip    = create_tensor(tn(LLM_TENSOR_ATTN_Q_CLIP,    "weight", i), {n_embd_head, n_head}, 0);
        layer.attn_sink_proj = create_tensor(tn(LLM_TENSOR_ATTN_SINK_PROJ, "weight", i), {n_embd, n_head}, 0);

        // relative-position bias
        layer.attn_rel      = create_tensor(tn(LLM_TENSOR_ATTN_REL,      "weight", i), {n_embd, n_rel*n_head}, 0);
        layer.attn_rel_bias = create_tensor(tn(LLM_TENSOR_ATTN_REL_BIAS, "weight", i), {n_rel, n_extent}, 0);
        layer.attn_rel_clip = create_tensor(tn(LLM_TENSOR_ATTN_REL_CLIP, "weight", i), {n_rel, n_head}, 0);

        // short convolutions
        layer.attn_k_conv    = create_tensor(tn(LLM_TENSOR_ATTN_K_CONV,    "weight", i), {n_conv, n_embd_head*n_head_kv}, 0);
        layer.attn_v_conv    = create_tensor(tn(LLM_TENSOR_ATTN_V_CONV,    "weight", i), {n_conv, n_embd_head*n_head_kv}, 0);
        layer.attn_post_conv = create_tensor(tn(LLM_TENSOR_ATTN_POST_CONV, "weight", i), {n_conv, n_embd}, 0);
        layer.ffn_post_conv  = create_tensor(tn(LLM_TENSOR_FFN_POST_CONV,  "weight", i), {n_conv, n_embd}, 0);

        if (hparams.is_swa(iv)) {
            layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);
        } else {
            // full-attention layers: separate Q, K and V input norms, and input-dependent taps for the K and V convolutions
            layer.attn_q_inp_norm  = create_tensor(tn(LLM_TENSOR_ATTN_Q_INP_NORM,  "weight", i), {n_embd}, 0);
            layer.attn_k_inp_norm  = create_tensor(tn(LLM_TENSOR_ATTN_K_INP_NORM,  "weight", i), {n_embd}, 0);
            layer.attn_v_inp_norm  = create_tensor(tn(LLM_TENSOR_ATTN_V_INP_NORM,  "weight", i), {n_embd}, 0);
            layer.attn_k_conv_proj = create_tensor(tn(LLM_TENSOR_ATTN_K_CONV_PROJ, "weight", i), {n_embd, n_conv*n_head_kv}, 0);
            layer.attn_v_conv_proj = create_tensor(tn(LLM_TENSOR_ATTN_V_CONV_PROJ, "weight", i), {n_embd, n_conv*n_head_kv}, 0);
        }

        // block attention residuals
        if (i > 0) {
            layer.attn_res_query = create_tensor(tn(LLM_TENSOR_ATTN_RES_QUERY, "weight", i), {n_embd, hparams.is_swa(iv) ? 1 : 3}, 0);
        }
        layer.ffn_res_query = create_tensor(tn(LLM_TENSOR_FFN_RES_QUERY, "weight", i), {n_embd}, 0);

        // MoE
        layer.ffn_norm         = create_tensor(tn(LLM_TENSOR_FFN_NORM,         "weight", i), {n_embd}, 0);
        layer.ffn_gate_inp     = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,     "weight", i), {n_embd, n_expert}, 0);
        layer.ffn_exp_probs_b  = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B,  "bias",   i), {n_expert}, TENSOR_NOT_REQUIRED);
        layer.ffn_gate_up_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_UP_EXPS, "weight", i), {n_embd, 2*n_ff_exp, n_expert}, 0);
        layer.ffn_down_exps    = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS,    "weight", i), {n_ff_exp, n_embd, n_expert}, 0);

        // message controllers of the looped core: (attn, k, v, mlp) x (down, gain, forget)
        if (loop && i >= (int) hparams.loop_core_start && i < (int) hparams.loop_core_end) {
            static const llm_tensor down[4]   = {LLM_TENSOR_LOOP_ATTN_DOWN,   LLM_TENSOR_LOOP_K_DOWN,   LLM_TENSOR_LOOP_V_DOWN,   LLM_TENSOR_LOOP_MLP_DOWN};
            static const llm_tensor gain[4]   = {LLM_TENSOR_LOOP_ATTN_GAIN,   LLM_TENSOR_LOOP_K_GAIN,   LLM_TENSOR_LOOP_V_GAIN,   LLM_TENSOR_LOOP_MLP_GAIN};
            static const llm_tensor forget[4] = {LLM_TENSOR_LOOP_ATTN_FORGET, LLM_TENSOR_LOOP_K_FORGET, LLM_TENSOR_LOOP_V_FORGET, LLM_TENSOR_LOOP_MLP_FORGET};

            const int64_t n_rank = hparams.loop_gate_rank;

            for (int m = 0; m < 4; ++m) {
                layer.loop_down[m]   = create_tensor(tn(down[m],   "weight", i), {2*n_embd, n_rank}, 0);
                layer.loop_gain[m]   = create_tensor(tn(gain[m],   "weight", i), {n_rank, n_embd}, 0);
                layer.loop_forget[m] = create_tensor(tn(forget[m], "weight", i), {n_rank, n_embd}, 0);
            }
        }
    }

    if (loop) {
        const int64_t n_query = 2*(hparams.loop_core_end - hparams.loop_core_start);

        loop_res_query    = create_tensor(tn(LLM_TENSOR_LOOP_RES_QUERY,    "weight"), {n_embd, n_query}, 0);
        loop_kv_res_query = create_tensor(tn(LLM_TENSOR_LOOP_KV_RES_QUERY, "weight"), {n_embd, n_query}, 0);
    }
}

std::unique_ptr<llm_graph_context> llama_model_inkling2::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(params);
}

// scale each row down to an L2 norm of at most threshold, after the optional per-channel scale
ggml_tensor * llama_model_inkling2::graph::clip(ggml_tensor * cur, ggml_tensor * scale, float threshold) {
    if (scale) {
        cur = ggml_mul(ctx0, cur, scale);
    }

    ggml_tensor * norm = ggml_sqrt(ctx0, ggml_sum_rows(ctx0, ggml_sqr(ctx0, cur)));
    norm = ggml_scale_bias(ctx0, norm, 1.0f, 1e-6f);

    ggml_tensor * factor = ggml_div(ctx0, ggml_fill(ctx0, norm, threshold), norm);

    return ggml_mul(ctx0, cur, ggml_clamp(ctx0, factor, 0.0f, 1.0f));
}

// attention over the depth: mix the earlier block outputs and cur with learned queries
ggml_tensor * llama_model_inkling2::graph::mix(const std::vector<ggml_tensor *> & blocks, ggml_tensor * cur, ggml_tensor * query) {
    ggml_tensor * states = ggml_reshape_3d(ctx0, cur, n_embd, 1, n_tokens);
    for (ggml_tensor * block : blocks) {
        states = ggml_concat(ctx0, ggml_reshape_3d(ctx0, block, n_embd, 1, n_tokens), states, 1);
    }
    states = ggml_rms_norm(ctx0, states, 1e-6f);

    ggml_tensor * queries = ggml_repeat(ctx0, query, ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, n_embd, query->ne[1], n_tokens));

    ggml_tensor * scores = ggml_mul_mat(ctx0, states, queries);
    scores = ggml_soft_max_ext(ctx0, scores, nullptr, 1.0f/sqrtf(n_embd), 0.0f);

    return ggml_mul_mat(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, states)), scores);
}

// loop message controller: (1 - tanh(F h))*cur + tanh(G h)*message, h = D [rms(cur); rms(message)]; module: q, k, v, mlp
ggml_tensor * llama_model_inkling2::graph::control(ggml_tensor * cur, ggml_tensor * message, const llama_layer & layer, int module) {
    ggml_tensor * h = ggml_concat(ctx0, ggml_reshape_3d(ctx0, cur, n_embd, 1, n_tokens), ggml_reshape_3d(ctx0, message, n_embd, 1, n_tokens), 1);
    h = ggml_reshape_2d(ctx0, ggml_rms_norm(ctx0, h, 1e-6f), 2*n_embd, n_tokens);
    h = build_lora_mm(layer.loop_down[module], h);

    ggml_tensor * gain   = ggml_tanh(ctx0, build_lora_mm(layer.loop_gain[module],   h));
    ggml_tensor * forget = ggml_tanh(ctx0, build_lora_mm(layer.loop_forget[module], h));

    return ggml_add(ctx0, ggml_mul(ctx0, ggml_scale_bias(ctx0, forget, -1.0f, 1.0f), cur), ggml_mul(ctx0, gain, message));
}

// causal short convolution with a residual; dynamic adds input-dependent taps per head
// the last n_conv - 1 inputs of each sequence live in the recurrent state at channel offset `offset`
ggml_tensor * llama_model_inkling2::graph::convolve(
        ggml_tensor * cur,
        ggml_tensor * weight,
        ggml_tensor * dynamic,
        ggml_tensor * state,
        ggml_tensor * history,
        llm_graph_input_rs * inp,
        int64_t offset) {
    const int64_t n_conv       = hparams.n_shortconv_l_cache;
    const int64_t n_channels   = cur->ne[0];
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;
    const int64_t n_seqs       = ubatch.n_seqs;

    ggml_tensor * past = ggml_view_3d(ctx0, history, n_conv - 1, n_channels, n_seqs, history->nb[1], history->nb[2], offset*history->nb[1]);

    ggml_tensor * x = ggml_transpose(ctx0, ggml_reshape_3d(ctx0, cur, n_channels, n_seq_tokens, n_seqs));
    x = ggml_concat(ctx0, past, x, 0);

    build_conv_state_store(inp, state, x, n_conv - 1, offset*(n_conv - 1)*ggml_element_size(state));

    ggml_tensor * out = ggml_ssm_conv(ctx0, x, weight);
    out = ggml_reshape_2d(ctx0, out, n_channels, n_tokens);

    if (dynamic) {
        const int64_t n_heads = dynamic->ne[0]/n_conv;

        // local: the tap projection goes into the graph first, so that the convolution below is a contiguous run of nodes
        // that the backend fuses into one kernel (ggml-cuda.cu, dconv)
        ggml_build_forward_expand(gf, dynamic);
        dynamic = ggml_reshape_3d(ctx0, dynamic, n_conv, n_heads, n_tokens);

        for (int64_t lag = 0; lag < n_conv; ++lag) {
            ggml_tensor * tap    = ggml_view_3d(ctx0, dynamic, 1, n_heads, n_tokens, dynamic->nb[1], dynamic->nb[2], lag*dynamic->nb[0]);
            ggml_tensor * window = ggml_view_3d(ctx0, x, n_seq_tokens, n_channels, n_seqs, x->nb[1], x->nb[2], (n_conv - 1 - lag)*x->nb[0]);

            window = ggml_cont(ctx0, ggml_transpose(ctx0, window));
            window = ggml_reshape_3d(ctx0, window, n_channels/n_heads, n_heads, n_tokens);

            ggml_tensor * term = ggml_reshape_2d(ctx0, ggml_mul(ctx0, window, tap), n_channels, n_tokens);

            out = ggml_add(ctx0, out, term);
        }
    }

    ggml_tensor * result = ggml_scale(ctx0, ggml_add(ctx0, cur, out), 1.0f/sqrtf((dynamic ? 2 : 1)*n_conv + 1));
    // local: the convolution goes into the graph right after its state store, so that the backend runs the input, the store
    // and the convolution as one kernel (ggml-cuda.cu, sconv_full); the callers expand their residual add right after it
    static const bool batch_opt_off = std::getenv("INKLING2_BATCH_OPT_OFF") != nullptr;
    if (!batch_opt_off) {
        ggml_build_forward_expand(gf, result);
    }
    return result;
}

llama_model_inkling2::graph::graph(const llm_graph_params & params) : llm_graph_context(params) {
    auto * inp  = build_inp_mem_hybrid_iswa_rel(hparams.n_relative_extent, hparams.n_swa);
    auto * attn = inp->get_attn();
    // banded FA needs a position-contiguous cache; small ubatches stay dense so the graph can be reused while decoding
    inp->banded = inkling2_banded_enabled && cparams.flash_attn &&
        (n_tokens > n_tokens_banded_min || (inkling2_far_enabled && n_tokens >= n_tokens_far_min));
    // local: the SWA cache is a ring of n_swa + n_ubatch cells that wraps after the first ~1.5k tokens of a prompt; the far
    // kernel reads it as a rotated ring (ggml_flash_attn_ext_banded_set_far_rot) instead of falling back to the dense bias
    // (+45-60 ms per 512-token ubatch). INKLING2_SWA_ROT=0 restores the fallback.
    static const bool swa_rot = []() { const char * e = std::getenv("INKLING2_SWA_ROT"); return !(e && e[0] == '0'); }();
    inp->allow_rot = swa_rot && inkling2_far_enabled;
    for (int i = 0; i < 2 && inp->banded; ++i) {
        inp->n_kv_banded[i] = (i ? attn->mctx->get_swa() : attn->mctx->get_base())->get_n_kv_banded(inp->kv_rot[i], i == 1 && inp->allow_rot);
    }
    std::vector<std::pair<std::pair<bool, ggml_backend_dev_t>, bool>> banded_dev_ok; // per (swa, device)
    const auto use_banded = [&](int il, bool swa) {
        const auto * cache = swa ? attn->mctx->get_swa() : attn->mctx->get_base();
        const int64_t hd = hparams.n_embd_head_k(il);
        const auto type_ok = [](ggml_type t) {
            return t == GGML_TYPE_F32 || t == GGML_TYPE_F16 || t == GGML_TYPE_BF16;
        };
        // the generic banded op needs contiguous cells (no rotation)
        if (!(inp->n_kv_banded[swa] > 0 && inp->kv_rot[swa] == 0 && (hd == 64 || hd == 128) && hparams.n_embd_head_v(il) == hd &&
                type_ok(cache->type_k()) && type_ok(cache->type_v()) && !cache->get_k_scale(il) && !cache->get_v_scale(il) &&
                (swa ? attn->self_attn_mask_swa.raw : attn->self_attn_mask.raw)->ne[3] == 1)) {
            return false;
        }
        ggml_backend_dev_t dev = model.dev_layer(il);
        if (!dev) {
            return false;
        }
        for (const auto & [key, ok] : banded_dev_ok) {
            if (key.first == swa && key.second == dev) {
                return ok;
            }
        }
        const int64_t nh  = hparams.n_head(il);
        const int64_t nkv = hparams.n_head_kv(il);
        ggml_tensor * pq = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, hd, 1, nh, 1);
        ggml_tensor * pk = ggml_new_tensor_4d(ctx0, cache->type_k(), hd, 1, nkv, 1);
        ggml_tensor * pv = ggml_new_tensor_4d(ctx0, cache->type_v(), hd, 1, nkv, 1);
        ggml_tensor * pm = ggml_new_tensor_4d(ctx0, GGML_TYPE_F16, 1, 1, 1, 1);
        ggml_tensor * pr = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, 1, nh, 1, 1);
        ggml_tensor * op = ggml_flash_attn_ext_banded(ctx0, pq, pk, pv, pm, pr, 1.0f, 1);
        ggml_flash_attn_ext_add_sinks(op, ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, nh, 1, 1));
        ggml_prec_set_acc(op, GGML_PREC_F32);
        const bool ok = ggml_backend_dev_supports_op(dev, op);
        banded_dev_ok.push_back({ { swa, dev }, ok });
        return ok;
    };
    // local (RDNA3): far banded op, see ggml_flash_attn_ext_banded_set_far. The probe uses realistic shapes because the
    // WMMA kernel needs a padded K/V and a minimum number of query rows. INKLING2_FAR=0 disables it.
    static const bool far_enabled = []() {
        const char * e = std::getenv("INKLING2_FAR");
        return !(e && e[0] == '0');
    }();
    std::vector<std::pair<std::pair<bool, ggml_backend_dev_t>, bool>> far_dev_ok; // per (swa, device)
    // whether the device of layer il runs the far op at all (probe with realistic shapes, cached per cache kind and device)
    const auto far_probe = [&](int il, bool swa) {
        const auto * cache = swa ? attn->mctx->get_swa() : attn->mctx->get_base();
        const int64_t hd = hparams.n_embd_head_k(il);
        ggml_backend_dev_t dev = model.dev_layer(il);
        if (!far_enabled || !dev || (hd != 64 && hd != 128)) {
            return false;
        }
        for (const auto & [key, ok] : far_dev_ok) {
            if (key.first == swa && key.second == dev) {
                return ok;
            }
        }
        const int64_t nh  = hparams.n_head(il);
        const int64_t nkv = hparams.n_head_kv(il);
        const int64_t ne  = (swa ? hparams.n_swa : hparams.n_relative_extent) + 1;
        ggml_tensor * pq = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, hd, n_tokens, nh, 1);
        ggml_tensor * pk = ggml_new_tensor_4d(ctx0, cache->type_k(), hd, 256, nkv, 1);
        ggml_tensor * pv = ggml_new_tensor_4d(ctx0, cache->type_v(), hd, 256, nkv, 1);
        ggml_tensor * pm = ggml_new_tensor_4d(ctx0, GGML_TYPE_F16, 256, n_tokens, 1, 1);
        ggml_tensor * pr = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, ne, nh, n_tokens, 1);
        ggml_tensor * op = ggml_flash_attn_ext_banded(ctx0, pq, pk, pv, pm, pr, 1.0f, ne);
        ggml_flash_attn_ext_banded_set_far(op, 0);
        const bool ok = ggml_backend_dev_supports_op(dev, op);
        far_dev_ok.push_back({ { swa, dev }, ok });
        return ok;
    };
    const auto use_far = [&](int il, bool swa) {
        if (!far_enabled || !inp->banded) {
            return false;
        }
        const auto * cache = swa ? attn->mctx->get_swa() : attn->mctx->get_base();
        const int64_t hd = hparams.n_embd_head_k(il);
        const auto type_ok = [](ggml_type t) {
            return t == GGML_TYPE_F16 || t == GGML_TYPE_BF16;
        };
        const ggml_tensor * mask = swa ? attn->self_attn_mask_swa.raw : attn->self_attn_mask.raw;
        if (!(inp->n_kv_banded[swa] > 0 && (hd == 64 || hd == 128) && hparams.n_embd_head_v(il) == hd &&
                type_ok(cache->type_k()) && type_ok(cache->type_v()) && !cache->get_k_scale(il) && !cache->get_v_scale(il) &&
                mask->ne[3] == 1 && mask->type == GGML_TYPE_F16 && GGML_PAD(inp->n_kv_banded[swa], 256) <= mask->ne[0])) {
            return false;
        }
        // a rotated ring is viewed whole: its used cells padded to 256 make the ring (local: the empty cells of a ring that is
        // not full, right after a checkpoint restore, are padding cells at the logical end); the kernel takes rotations below 2048
        if (inp->kv_rot[swa] != 0 && (GGML_PAD(inp->n_kv_banded[swa], 256) != mask->ne[0] || inp->kv_rot[swa] >= 2048)) {
            return false;
        }
        return far_probe(il, swa);
    };
    // local: on an AMD GPU (ROCm device) the generic banded kernel is slower than the dense path (measured on gfx1100); there
    // the banded op is only taken where the far mode exists (RDNA3), other AMD GPUs keep the dense path
    const auto banded_generic_ok = [&](int il, bool swa) {
        ggml_backend_dev_t dev = model.dev_layer(il);
        const bool rocm = dev && strncmp(ggml_backend_dev_name(dev), "ROCm", 4) == 0;
        return !rocm || far_probe(il, swa);
    };
    auto * recr = inp->get_recr();

    ggml_tensor * inp_pos     = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    // log-scaled positions: 1 + alpha*log(max(1, (pos + 1)/floor)), multiplies q and r of the full-attention layers
    ggml_tensor * tau = ggml_scale_bias(ctx0, ggml_cast(ctx0, inp_pos, GGML_TYPE_F32), 1.0f, 1.0f);
    tau = ggml_log(ctx0, ggml_clamp(ctx0, ggml_scale(ctx0, tau, 1.0f/hparams.n_log_scale_floor), 1.0f, INFINITY));
    tau = ggml_reshape_3d(ctx0, ggml_scale_bias(ctx0, tau, hparams.f_log_scale_alpha, 1.0f), 1, 1, n_tokens);

    ggml_tensor * cur = build_inp_embd(model.tok_embd);

    // important: do not normalize raw embeddings input (i.e. encoded image embeddings)
    if (ubatch.token) {
        cur = build_norm(cur, model.tok_norm, nullptr, LLM_NORM_RMS, -1);
    }

    std::vector<ggml_tensor *> blocks;

    // local: a mix result [n_embd, n_q, n_tokens] laid out as [n_embd, n_tokens, n_q], so that each row j is a contiguous
    // view. A single token or a single query is already in that layout (a reshape); otherwise one permuted copy, which the
    // backend writes directly from the fused mix kernel, replaces one copy kernel per row (~200 per verify batch).
    // INKLING2_BATCH_OPT_OFF=1 restores the copy per row (and one FILL of the zero sinks per attention call, see below).
    static const bool mix_row_cont = std::getenv("INKLING2_BATCH_OPT_OFF") != nullptr;
    const auto mix_rows = [&](ggml_tensor * mixed) {
        if (mix_row_cont) {
            return mixed;
        }
        if (n_tokens == 1 || mixed->ne[1] == 1) {
            return ggml_reshape_3d(ctx0, mixed, n_embd, n_tokens, mixed->ne[1]);
        }
        return ggml_cont(ctx0, ggml_permute(ctx0, mixed, 0, 2, 1, 3));
    };
    const auto mix_row = [&](ggml_tensor * rows, int64_t j) {
        if (mix_row_cont) {
            ggml_tensor * v = ggml_view_2d(ctx0, rows, n_embd, n_tokens, rows->nb[2], j*rows->nb[1]);
            return n_tokens == 1 ? v : ggml_cont(ctx0, v);
        }
        return ggml_view_2d(ctx0, rows, n_embd, n_tokens, rows->nb[1], j*rows->nb[2]);
    };
    // local: one zero sink tensor for every dense attention call (it was one FILL kernel per layer application)
    ggml_tensor * zero_sinks = nullptr;
    // local: [base, swa] the mask of the far attention calls, shared by the calls of a graph
    ggml_tensor * far_mask[2] = { nullptr, nullptr };
    // local: the row of -1 that folds the sinks into the far relative-bias product (see rel_far)
    ggml_tensor * far_minus_one = nullptr;

    // each pass of the looped core restarts from the prelude state and reads the previous pass's blocks through messages
    const int64_t n_loop_core = hparams.loop_n_pass > 1 ? hparams.loop_core_end - hparams.loop_core_start : 0;

    std::vector<ggml_tensor *> prelude_blocks;
    ggml_tensor *              prelude_cur = nullptr;
    std::vector<ggml_tensor *> bank;
    ggml_tensor *              bank_last = nullptr;

    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model.layers[il];
        const int    ip    = hparams.loop_layer(il);

        ggml_tensor * messages[4] = {nullptr, nullptr, nullptr, nullptr};

        if (n_loop_core > 0 && ip >= (int) hparams.loop_core_start && ip < (int) hparams.loop_core_end) {
            const int pass = (il - (int) hparams.loop_core_start)/n_loop_core;

            if (ip == (int) hparams.loop_core_start) {
                if (pass == 0) {
                    prelude_blocks = blocks;
                    prelude_cur    = cur;
                } else {
                    bank.assign(blocks.begin() + prelude_blocks.size() + 1, blocks.end());
                    bank_last = cur;
                    blocks    = prelude_blocks;
                    cur       = prelude_cur;
                }
            }

            if (pass > 0) {
                const int64_t row = 2*(ip - (int) hparams.loop_core_start);

                ggml_tensor * q_attn = ggml_view_2d(ctx0, model.loop_res_query,    n_embd, 2, model.loop_res_query->nb[1],    row*model.loop_res_query->nb[1]);
                ggml_tensor * q_kv   = ggml_view_2d(ctx0, model.loop_kv_res_query, n_embd, 2, model.loop_kv_res_query->nb[1], row*model.loop_kv_res_query->nb[1]);

                // rows: (attn, mlp) and (k, v)
                ggml_tensor * m_attn = mix_rows(mix(bank, bank_last, q_attn));
                ggml_tensor * m_kv   = mix_rows(mix(bank, bank_last, q_kv));

                for (int j = 0; j < 2; ++j) {
                    messages[j == 0 ? 0 : 3] = mix_row(m_attn, j);
                    messages[j == 0 ? 1 : 2] = mix_row(m_kv, j);
                }
            }
        }

        const bool    is_swa       = hparams.is_swa(il);
        const int64_t n_head       = hparams.n_head(il);
        const int64_t n_head_kv    = hparams.n_head_kv(il);
        const int64_t n_embd_head  = hparams.n_embd_head_k(il);
        const int64_t n_embd_k_gqa = n_embd_head*n_head_kv;

        ggml_tensor * state   = recr->mctx->get_r_l(il);
        ggml_tensor * history = build_rs(recr, state, hparams.n_embd_r(), ubatch.n_seqs);
        history = ggml_reshape_3d(ctx0, history, hparams.n_shortconv_l_cache - 1, hparams.n_sconv_channels, ubatch.n_seqs);

        res->t_layer_inp[il] = cur;

        ggml_tensor * inpSA = cur;

        // attention inputs: SWA layers mix one, full-attention layers one each for Q, K and V
        ggml_tensor * inputs[3] = {cur, cur, cur};
        if (layer.attn_res_query) {
            ggml_tensor * mixed = mix_rows(mix(blocks, cur, layer.attn_res_query));
            for (int j = 0; j < 3; ++j) {
                // SWA layers have one query: the three inputs are the same row (one view, one copy at most)
                inputs[j] = is_swa && j > 0 ? inputs[0] : mix_row(mixed, is_swa ? 0 : j);
            }
        }

        if (messages[0]) {
            // the messages enter before the input norms, so SWA layers normalize three inputs
            for (int j = 0; j < 3; ++j) {
                inputs[j] = control(inputs[j], messages[j], layer, j);
            }
            cb(inputs[0], "loop_q_inp", il);
        }

        if (is_swa && messages[0]) {
            for (int j = 0; j < 3; ++j) {
                inputs[j] = build_norm(inputs[j], layer.attn_norm, nullptr, LLM_NORM_RMS, il);
            }
        } else if (is_swa) {
            inputs[0] = build_norm(inputs[0], layer.attn_norm, nullptr, LLM_NORM_RMS, il);
            inputs[1] = inputs[2] = inputs[0];
        } else {
            inputs[0] = build_norm(inputs[0], layer.attn_q_inp_norm, nullptr, LLM_NORM_RMS, il);
            inputs[1] = build_norm(inputs[1], layer.attn_k_inp_norm, nullptr, LLM_NORM_RMS, il);
            inputs[2] = build_norm(inputs[2], layer.attn_v_inp_norm, nullptr, LLM_NORM_RMS, il);
        }

        // self-attention
        {
            ggml_tensor * Qcur = build_lora_mm(layer.wq, inputs[0]);
            ggml_tensor * Kcur = convolve(build_lora_mm(layer.wk, inputs[1]), layer.attn_k_conv,
                    is_swa ? nullptr : build_lora_mm(layer.attn_k_conv_proj, inputs[1]), state, history, recr, 0);
            ggml_tensor * Vcur = convolve(build_lora_mm(layer.wv, inputs[2]), layer.attn_v_conv,
                    is_swa ? nullptr : build_lora_mm(layer.attn_v_conv_proj, inputs[2]), state, history, recr, n_embd_k_gqa);
            ggml_tensor * Rcur = build_lora_mm(layer.attn_rel, inputs[0]);

            Qcur = clip(ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens), layer.attn_q_clip, hparams.f_qk_clip);
            Kcur = clip(ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens), nullptr,           hparams.f_qk_clip);
            Vcur = ggml_rms_norm(ctx0, ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens), hparams.f_norm_rms_eps);
            Rcur = clip(ggml_reshape_3d(ctx0, Rcur, hparams.n_relative, n_head, n_tokens), layer.attn_rel_clip, hparams.f_relative_clip);

            if (!is_swa) {
                Qcur = ggml_mul(ctx0, Qcur, tau);
                Rcur = ggml_mul(ctx0, Rcur, tau);
            }

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);
            cb(Rcur, "Rcur", il);

            ggml_tensor * rel_pos = is_swa ? inp->self_rel_pos_swa : inp->self_rel_pos;

            const int64_t n_stream = (is_swa ? attn->self_attn_mask_swa.raw : attn->self_attn_mask.raw)->ne[3];
            const int64_t n_kv     = rel_pos->ne[0];

            ggml_tensor * sinks = build_lora_mm(layer.attn_sink_proj, inputs[0]);

            if (use_far(il, is_swa)) {
                // local (RDNA3): relative bias inside the WMMA kernel. rel = [bias - sink ; -sink] by distance, the last row
                // for every distance past the extent, plus a sink of logit 0: the dense formulation below without the
                // n_kv x n_tokens x n_head mask. K/V and the mask are viewed up to the next multiple of 256 cells.
                const auto * cache = is_swa ? attn->mctx->get_swa() : attn->mctx->get_base();
                auto * k_rot = is_swa ? attn->self_k_rot_swa[il] : attn->self_k_rot[il];
                auto * v_rot = is_swa ? attn->self_v_rot_swa[il] : attn->self_v_rot[il];
                if (k_rot) {
                    Qcur = llama_mul_mat_hadamard(ctx0, Qcur, k_rot);
                    Kcur = llama_mul_mat_hadamard(ctx0, Kcur, k_rot);
                }
                if (v_rot) {
                    Vcur = llama_mul_mat_hadamard(ctx0, Vcur, v_rot);
                }
                ggml_build_forward_expand(gf, Qcur);
                ggml_build_forward_expand(gf, Kcur);
                ggml_build_forward_expand(gf, Vcur);
                ggml_build_forward_expand(gf, cache->cpy_k(ctx0, Kcur, is_swa ? attn->get_k_idxs_swa() : attn->get_k_idxs(), il));
                ggml_build_forward_expand(gf, cache->cpy_v(ctx0, Vcur, is_swa ? attn->get_v_idxs_swa() : attn->get_v_idxs(), il));

                const int64_t n_kv_b = inp->n_kv_banded[is_swa];
                const int64_t n_kv_p = GGML_PAD(n_kv_b, 256);
                ggml_tensor * k_all = cache->get_k(ctx0, il);
                ggml_tensor * v_all = cache->get_v(ctx0, il);
                GGML_ASSERT(n_kv_p <= k_all->ne[2]);
                ggml_tensor * q_fa = ggml_permute(ctx0, ggml_reshape_4d(ctx0, Qcur, n_embd_head, n_head, n_tokens, 1), 0, 2, 1, 3);
                ggml_tensor * k_fa = ggml_permute(ctx0, ggml_view_4d(ctx0, k_all, k_all->ne[0], k_all->ne[1], n_kv_p, k_all->ne[3],
                        k_all->nb[1], k_all->nb[2], k_all->nb[3], 0), 0, 2, 1, 3);
                ggml_tensor * v_fa;
                if (v_all->nb[1] > v_all->nb[2]) { // transposed V
                    v_fa = ggml_view_4d(ctx0, v_all, n_kv_p, v_all->ne[1], v_all->ne[2], v_all->ne[3],
                            v_all->nb[1], v_all->nb[2], v_all->nb[3], 0);
                    v_fa = ggml_transpose(ctx0, ggml_permute(ctx0, v_fa, 0, 2, 1, 3));
                } else {
                    v_fa = ggml_permute(ctx0, ggml_view_4d(ctx0, v_all, v_all->ne[0], v_all->ne[1], n_kv_p, v_all->ne[3],
                            v_all->nb[1], v_all->nb[2], v_all->nb[3], 0), 0, 2, 1, 3);
                }
                ggml_tensor * mask = is_swa ? attn->self_attn_mask_swa.raw : attn->self_attn_mask.raw;
                // local: the mask built on the device from relative positions (see kq_dev), n_kv_p columns, once per cache
                if (inp->kq_dev[is_swa]) {
                    ggml_tensor *& g = inp->kq_mask_gen[is_swa];
                    if (!g) {
                        const int64_t nq = mask->ne[1];
                        // clamp(k - q, 0, 1): 1 where the cell is kept, 0 where it is masked
                        const auto keep = [&](ggml_tensor * k, ggml_tensor * q) {
                            ggml_tensor * t = ggml_repeat_4d(ctx0, ggml_view_1d(ctx0, k, n_kv_p, 0), n_kv_p, nq, 1, 1);
                            return ggml_clamp(ctx0, ggml_sub_inplace(ctx0, t, q), 0.0f, 1.0f);
                        };
                        ggml_tensor * t = keep(inp->kq_k_neg[is_swa], inp->kq_q_neg1);
                        if (is_swa) {
                            t = ggml_mul_inplace(ctx0, t, keep(inp->kq_k_pos, inp->kq_q_swa));
                        }
                        g = ggml_log_inplace(ctx0, t);
                    }
                    mask = g;
                }
                // local: the mask is the same for every call of a cache: one view (the whole mask when no padding cell is
                // cut, the usual case) or one copy per graph instead of a copy per attention call (8 copies of up to
                // 128 MiB at a full context for the global layers). INKLING2_FAR_MASK_COPY=1 restores the copy per call.
                static const bool far_mask_copy = std::getenv("INKLING2_FAR_MASK_COPY") != nullptr;
                ggml_tensor *& m = far_mask[is_swa];
                if (!m || far_mask_copy) {
                    m = n_kv_p == mask->ne[0] && !far_mask_copy ? mask :
                        ggml_cont(ctx0, ggml_view_4d(ctx0, mask, n_kv_p, mask->ne[1], mask->ne[2], mask->ne[3],
                            mask->nb[1], mask->nb[2], mask->nb[3], 0));
                }
                ggml_tensor * rel_far = nullptr;
                // local: the sink is folded into the table product as a 17th term (a row of -1 under the bias table, the
                // sinks under r) instead of a SUB that read and wrote the whole [E + 1, n_head, n_tokens] F32 table again
                // (100 MB per call for a 1024-token batch). INKLING2_REL_SUB=1 restores the SUB.
                static const bool rel_sub = std::getenv("INKLING2_REL_SUB") != nullptr;
                if (!rel_sub) {
                    if (!far_minus_one) {
                        far_minus_one = ggml_fill(ctx0, ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 1, layer.attn_rel_bias->ne[1] + 1), -1.0f);
                    }
                    ggml_tensor * table = ggml_concat(ctx0, ggml_pad(ctx0, layer.attn_rel_bias, 0, 1, 0, 0), far_minus_one, 0);
                    ggml_tensor * r_s   = ggml_concat(ctx0, Rcur, ggml_reshape_3d(ctx0, sinks, 1, n_head, n_tokens), 0);
                    // local: one flat product over the n_head x n_tokens columns; with the table broadcast over the tokens the
                    // backend ran one small GEMM per token (RDNA3, 1024-token batch: 252 us instead of 80 us per call).
                    // INKLING2_REL_FLAT=0 restores the broadcast form.
                    static const bool rel_flat = []() { const char * e = std::getenv("INKLING2_REL_FLAT"); return !e || atoi(e) != 0; }();
                    if (rel_flat) {
                        r_s = ggml_reshape_2d(ctx0, r_s, r_s->ne[0], r_s->ne[1]*r_s->ne[2]);
                    }
                    rel_far = ggml_mul_mat(ctx0, table, r_s);
                } else {
                    rel_far = ggml_mul_mat(ctx0, ggml_pad(ctx0, layer.attn_rel_bias, 0, 1, 0, 0), Rcur);
                    rel_far = ggml_sub(ctx0, rel_far, ggml_reshape_3d(ctx0, sinks, 1, n_head, n_tokens));
                }
                rel_far = ggml_reshape_4d(ctx0, rel_far, rel_far->ne[0], n_head, n_tokens, 1);
                cur = ggml_flash_attn_ext_banded(ctx0, q_fa, k_fa, v_fa, m, rel_far, 1.0f/sqrtf(n_embd_head), rel_far->ne[0]);
                ggml_flash_attn_ext_banded_set_far_rot(cur, n_kv_p - n_kv_b, inp->kv_rot[is_swa]);
                res->add_fused_node({LLM_FUSED_OP_FLASH_ATTN, cur, il});
                cur = ggml_reshape_2d(ctx0, cur, cur->ne[0]*cur->ne[1], cur->ne[2]*cur->ne[3]);
                cb(cur, "kqv_out", il);
                if (v_rot) {
                    cur = llama_mul_mat_hadamard(ctx0, cur, v_rot);
                }
            } else if (n_tokens > n_tokens_banded_min && use_banded(il, is_swa) && banded_generic_ok(il, is_swa)) {
                const auto * cache = is_swa ? attn->mctx->get_swa() : attn->mctx->get_base();
                auto * k_rot = is_swa ? attn->self_k_rot_swa[il] : attn->self_k_rot[il];
                auto * v_rot = is_swa ? attn->self_v_rot_swa[il] : attn->self_v_rot[il];
                if (k_rot) {
                    Qcur = llama_mul_mat_hadamard(ctx0, Qcur, k_rot);
                    Kcur = llama_mul_mat_hadamard(ctx0, Kcur, k_rot);
                }
                if (v_rot) {
                    Vcur = llama_mul_mat_hadamard(ctx0, Vcur, v_rot);
                }
                ggml_build_forward_expand(gf, Qcur);
                ggml_build_forward_expand(gf, Kcur);
                ggml_build_forward_expand(gf, Vcur);
                ggml_build_forward_expand(gf, cache->cpy_k(ctx0, Kcur, is_swa ? attn->get_k_idxs_swa() : attn->get_k_idxs(), il));
                ggml_build_forward_expand(gf, cache->cpy_v(ctx0, Vcur, is_swa ? attn->get_v_idxs_swa() : attn->get_v_idxs(), il));

                const int64_t n_kv_b = inp->n_kv_banded[is_swa];
                ggml_tensor * k_all = cache->get_k(ctx0, il);
                ggml_tensor * v_all = cache->get_v(ctx0, il);
                GGML_ASSERT(n_kv_b <= k_all->ne[2]);
                ggml_tensor * q_fa = ggml_permute(ctx0, ggml_reshape_4d(ctx0, Qcur, n_embd_head, n_head, n_tokens, 1), 0, 2, 1, 3);
                ggml_tensor * k_fa = ggml_permute(ctx0, ggml_view_4d(ctx0, k_all, k_all->ne[0], k_all->ne[1], n_kv_b, k_all->ne[3],
                        k_all->nb[1], k_all->nb[2], k_all->nb[3], 0), 0, 2, 1, 3);
                ggml_tensor * v_fa;
                if (v_all->nb[1] > v_all->nb[2]) { // transposed V
                    v_fa = ggml_view_4d(ctx0, v_all, n_kv_b, v_all->ne[1], v_all->ne[2], v_all->ne[3],
                            v_all->nb[1], v_all->nb[2], v_all->nb[3], 0);
                    v_fa = ggml_transpose(ctx0, ggml_permute(ctx0, v_fa, 0, 2, 1, 3));
                } else {
                    v_fa = ggml_permute(ctx0, ggml_view_4d(ctx0, v_all, v_all->ne[0], v_all->ne[1], n_kv_b, v_all->ne[3],
                            v_all->nb[1], v_all->nb[2], v_all->nb[3], 0), 0, 2, 1, 3);
                }
                ggml_tensor * mask = is_swa ? attn->self_attn_mask_swa.raw : attn->self_attn_mask.raw;
                ggml_tensor * m = ggml_cont(ctx0, ggml_view_4d(ctx0, mask, n_kv_b, mask->ne[1], mask->ne[2], mask->ne[3],
                        mask->nb[1], mask->nb[2], mask->nb[3], 0));
                // exp(s + b)/(exp(t) + sum(exp(s + b))): the dense mask b - t with a zero sink
                ggml_tensor * rel_b = ggml_mul_mat(ctx0, layer.attn_rel_bias, Rcur);
                rel_b = ggml_reshape_4d(ctx0, rel_b, rel_b->ne[0], n_head, n_tokens, 1);
                cur = ggml_flash_attn_ext_banded(ctx0, q_fa, k_fa, v_fa, m, rel_b, 1.0f/sqrtf(n_embd_head), rel_b->ne[0]);
                ggml_flash_attn_ext_add_sinks(cur, ggml_reshape_3d(ctx0, sinks, n_head, n_tokens, 1));
                ggml_prec_set_acc(cur, GGML_PREC_F32);
                res->add_fused_node({LLM_FUSED_OP_FLASH_ATTN, cur, il});
                cur = ggml_reshape_2d(ctx0, cur, cur->ne[0]*cur->ne[1], cur->ne[2]*cur->ne[3]);
                cb(cur, "kqv_out", il);
                if (v_rot) {
                    cur = llama_mul_mat_hadamard(ctx0, cur, v_rot);
                }
            } else {
                // local: r and the sinks go into the graph first, so that the bias chain below (PAD ... cast of the mask) is
                // contiguous and the backend can fuse it into one kernel
                ggml_build_forward_expand(gf, Rcur);
                ggml_build_forward_expand(gf, sinks);
                // project r on the bias table first, then gather by distance: only the gathered bias is n_kv x n_tokens x n_head
                ggml_tensor * table = ggml_mul_mat(ctx0, ggml_pad(ctx0, layer.attn_rel_bias, 0, 1, 0, 0), Rcur);

                // the sink logit t is per query: exp(s - t)/(1 + sum(exp(s - t))) = exp(s)/(exp(t) + sum(exp(s))), with a zero sink
                table = ggml_sub(ctx0, table, ggml_reshape_3d(ctx0, sinks, 1, n_head, n_tokens));
                table = ggml_cont(ctx0, ggml_permute(ctx0, table, 1, 0, 2, 3));

                if (!zero_sinks || zero_sinks->ne[0] != n_head || mix_row_cont) {
                    zero_sinks = ggml_fill(ctx0, ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, n_head), 0.0f);
                }
                sinks = zero_sinks;

                // the dense bias is n_kv x n_tokens x n_head: past the budget, attend in chunks of query rows
                const int64_t n_seq_tokens = n_tokens/n_stream;
                // sized for a full cache so the chunking does not change while it fills
                const int64_t n_kv_max = std::max<int64_t>(n_kv, (is_swa ? attn->mctx->get_swa() : attn->mctx->get_base())->get_size());
                const int64_t kq_bias_max = far_probe(il, is_swa) ? n_kq_bias_max_far : n_kq_bias_max;
                int64_t n_rows_max = std::max({ int64_t(1), kq_bias_max/(n_kv_max*n_head*n_stream),
                        (n_seq_tokens + n_kq_chunk_max - 1)/n_kq_chunk_max });
                if (n_rows_max < n_seq_tokens) {
                    // equal chunks of whole 64-row tiles
                    const int64_t n_chunks = (n_seq_tokens + n_rows_max - 1)/n_rows_max;
                    n_rows_max = std::min(n_seq_tokens, GGML_PAD((n_seq_tokens + n_chunks - 1)/n_chunks, 64));
                }
                // rows [r0, r0 + nr) of every stream along dim d
                auto rows = [&](ggml_tensor * t, int d, int64_t r0, int64_t nr) {
                    int64_t ne[4] = { t->ne[0], t->ne[1], t->ne[2], t->ne[3] };
                    ne[d] = nr;
                    return ggml_cont(ctx0, ggml_view_4d(ctx0, t, ne[0], ne[1], ne[2], ne[3], t->nb[1], t->nb[2], t->nb[3], r0*t->nb[d]));
                };
                std::vector<ggml_tensor *> outs;
                for (int64_t r0 = 0; r0 < n_seq_tokens; r0 += n_rows_max) {
                    const int64_t nr = std::min(n_rows_max, n_seq_tokens - r0);
                    const bool all = nr == n_seq_tokens;
                    ggml_tensor * q_c = Qcur;
                    ggml_tensor * kq_b = nullptr;
                    if (all) {
                        kq_b = ggml_get_rows(ctx0, table, rel_pos);
                        kq_b = ggml_reshape_4d(ctx0, kq_b, n_head, n_kv, nr, n_stream);
                        kq_b = ggml_cont(ctx0, ggml_permute(ctx0, kq_b, 2, 0, 1, 3));
                    } else {
                        q_c = ggml_reshape_3d(ctx0, rows(ggml_reshape_4d(ctx0, Qcur, n_embd_head, n_head, n_seq_tokens, n_stream), 2, r0, nr),
                                n_embd_head, n_head, nr*n_stream);
                        ggml_tensor * table_c = ggml_reshape_4d(ctx0, table, table->ne[0], table->ne[1], n_seq_tokens, n_stream);
                        table_c = ggml_view_4d(ctx0, table_c, table_c->ne[0], table_c->ne[1], nr, n_stream,
                                table_c->nb[1], table_c->nb[2], table_c->nb[3], r0*table_c->nb[2]);
                        ggml_tensor *& rel_pos_dev = inp->self_rel_pos_dev[is_swa];
                        if (!rel_pos_dev) {
                            rel_pos_dev = ggml_cont(ctx0, rel_pos);
                        }
                        ggml_tensor * rel_pos_c = ggml_view_3d(ctx0, rel_pos_dev, n_kv, nr, n_stream,
                                rel_pos_dev->nb[1], n_seq_tokens*rel_pos_dev->nb[1], r0*rel_pos_dev->nb[1]);
                        kq_b = ggml_cont(ctx0, ggml_permute(ctx0, ggml_get_rows(ctx0, table_c, rel_pos_c), 2, 0, 1, 3));
                    }
                    // local TIMING EXPERIMENT ONLY (wrong output): INKLING2_NOBIAS=1 drops the relative bias so that the mask keeps
                    // no head dimension; it measures what attention costs without building the dense per-head bias. Only in
                    // builds with GGML_CUDA_INK2_COST_TOOLS defined.
#if defined(GGML_CUDA_INK2_COST_TOOLS)
                    static const bool no_bias = std::getenv("INKLING2_NOBIAS") != nullptr;
#else
                    static const bool no_bias = false;
#endif
                    if (no_bias) {
                        kq_b = ggml_fill(ctx0, ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, n_kv, nr, 1, n_stream), 0.0f);
                    }
                    cb(kq_b, "kq_b", il);

                    // only the first chunk stores K and V
                    ggml_tensor * out = build_attn(inp,
                            nullptr, nullptr, nullptr,
                            q_c, r0 == 0 ? Kcur : nullptr, r0 == 0 ? Vcur : nullptr, kq_b, sinks, 1.0f/sqrtf(n_embd_head), il, r0);
                    outs.push_back(all ? out : ggml_reshape_3d(ctx0, out, out->ne[0], nr, n_stream));
                }
                while (outs.size() > 1) {
                    std::vector<ggml_tensor *> next;
                    for (size_t c = 0; c + 1 < outs.size(); c += 2) {
                        next.push_back(ggml_concat(ctx0, outs[c], outs[c + 1], 1));
                    }
                    if (outs.size() % 2) {
                        next.push_back(outs.back());
                    }
                    outs = std::move(next);
                }
                cur = n_rows_max >= n_seq_tokens ? outs[0] : ggml_reshape_2d(ctx0, outs[0], outs[0]->ne[0], n_tokens);
            }

            cur = ggml_mul(ctx0, cur, ggml_sigmoid(ctx0, build_lora_mm(layer.wqkv_gate, inputs[0])));
            cur = convolve(build_lora_mm(layer.wo, cur), layer.attn_post_conv, nullptr, state, history, recr, 2*n_embd_k_gqa);
            cb(cur, "attn_out", il);
        }

        // the first layer of each block starts a new residual stream, the earlier one joins the mixed blocks
        if (ip % hparams.attn_res_block_size == 0) {
            blocks.push_back(inpSA);
        } else {
            cur = ggml_add(ctx0, inpSA, cur);
            // local: right after the convolution (fused with it, see convolve)
            ggml_build_forward_expand(gf, cur);
        }

        ggml_tensor * ffn_inp = cur;
        cb(ffn_inp, "ffn_inp", il);

        // MoE
        cur = ggml_reshape_2d(ctx0, mix(blocks, cur, layer.ffn_res_query), n_embd, n_tokens);
        if (messages[3]) {
            cur = control(cur, messages[3], layer, 3);
            cb(cur, "loop_ffn_inp", il);
        }
        cur = build_norm(cur, layer.ffn_norm, nullptr, LLM_NORM_RMS, il);

        cur = build_moe_ffn(cur,
                layer.ffn_gate_inp,
                nullptr,
                nullptr,
                layer.ffn_down_exps,
                layer.ffn_exp_probs_b,
                hparams.n_expert,
                hparams.n_expert_used(il),
                LLM_FFN_SITU, true,
                hparams.expert_weights_scale,
                LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID,
                il,
                nullptr,
                layer.ffn_gate_up_exps,
                layer.ffn_up_exps_s,
                layer.ffn_gate_exps_s,
                layer.ffn_down_exps_s);

        cur = convolve(cur, layer.ffn_post_conv, nullptr, state, history, recr, 2*n_embd_k_gqa + n_embd);
        cb(cur, "ffn_out", il);

        cur = ggml_add(ctx0, ffn_inp, cur);
        // local: right after the convolution (fused with it, see convolve)
        ggml_build_forward_expand(gf, cur);
        cb(cur, "l_out", il);
    }

    cur = ggml_reshape_2d(ctx0, mix(blocks, cur, model.output_res_query), n_embd, n_tokens);
    cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);

    res->t_layer_inp[n_layer] = cur;

    if (inp_out_ids) {
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    }

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // lm_head
    cur = build_lora_mm(model.output, cur);
    ggml_prec_set_acc(cur, GGML_PREC_F32);

    if (hparams.f_final_logit_softcapping != 0.0f) {
        const float cap = hparams.f_final_logit_softcapping;

        cur = ggml_scale(ctx0, ggml_tanh(ctx0, ggml_scale(ctx0, cur, 1.0f/cap)), cap);
    }

    // drop the padded vocabulary
    cur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, hparams.n_vocab_unpadded, cur->ne[1], cur->nb[1], 0));

    cb(cur, "result_output", -1);
    res->t_logits = cur;
}
