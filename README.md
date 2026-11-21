# Inkling 2 Mini on AMD GPUs (ROCm/HIP): a fast path (branch `inkling2-amd`)

> **This is not upstream llama.cpp.** This branch is the Unsloth llama.cpp release
> [b12568-mix-8eebad4](https://github.com/unslothai/llama.cpp/releases/tag/b12568-mix-8eebad4) (upstream b12568 plus the
> pull requests Unsloth merged for Inkling 2), with changes that make
> [Inkling 2 Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) fast on AMD GPUs under ROCm/HIP. It is
> experimental: it was tuned and measured on one GPU model, the Radeon RX 7900 XTX (RDNA3, gfx1100, 24 GB), and its GPU
> code paths only switch on for RDNA3.0 GPUs (gfx1100-gfx1103); every other GPU runs the stock kernels of the release
> (see "Which AMD GPUs"). It is not affiliated with ggml-org, Unsloth or Thinking Machines. A GGUF file made for this
> branch is described in its model card: `https://huggingface.co/Makdok/Inkling-2-Mini-AMD-GGUF`. Measured on RX 7900 XTX with the
> Unsloth file where the release fits in VRAM: prompt reading 829 -> 2103 tok/s and writing 42 -> 78 tok/s at the
> guide's context of 32768 without a drafter; 984 -> 1884 tok/s and 60 -> 94 tok/s with the DSpark drafter at 8192, the
> largest context where the release with the drafter stays in VRAM (see "Measurements").
> The upstream README follows after the horizontal line.

## What it is

Inkling 2 Mini (Thinking Machines, 26.4B parameters, mixture of experts with 128 experts and 8 per token, 24 stored layers
of which layers 6-17 run three times with shared weights: 48 layer applications per token) ran in llama.cpp only with
Unsloth's merged release when this work started (2026-11-20). Built for HIP, that release is slow on RDNA3 for two
reasons:

1. Inkling 2 adds a relative-position bias per head to the attention logits. The release builds it as a dense per-head
   mask, which also takes the WMMA flash-attention kernel out of play: reading 2048 tokens at a depth of 32k runs at
   500 tok/s (llama-bench, `-ub 512`, measured on RX 7900 XTX).
2. Decoding is bound by kernel launches: about 8100 small kernels per generated token, 46.8 tok/s at depth 0.

The branch, on top of the import of the release archive:

| commit | content |
|---|---|
| 2 ports (author Daniel Han) | unslothai/llama.cpp#385 (dense bias chunked over query rows) and #386 (banded flash attention for the bias), open pull requests that are not in the release |
| ggml far mode | the relative bias evaluated inside the regular MMA/WMMA flash-attention kernel ("far" mode of `FLASH_ATTN_EXT_BANDED`), wrapped sliding-window caches read as rotated rings, fully masked K/V blocks skipped, tiles of 3 query heads for the 24 query heads over 8 K/V heads (instead of 4 with a padding column), branchless NVFP4 scale decode |
| llama reading path | the Inkling 2 graph on the far op (its relative-bias table as one flat product), KQ masks built on the device, follow-up prompts kept on the far path, a graph order that lets the backend fuse |
| ggml-cuda fusions | fused kernels for 13 small-op patterns of the Inkling 2 graph, Q8_0 mat-vec on F32 activations, grouped Q8_0 products, NVFP4 MoE mat-vec tile: about 1660 launches per generated token instead of 8100; expert products of prompt batches on FP16 WMMA instead of int8 MMQ (NVFP4, and Q8_0 / Q5_K from 256 tokens), with the SITU activation of the gate_up projection written by that kernel; one kernel that uploads the staged inputs of a split |
| drafter | DSpark drafter: head computed only for the drafted positions and partly precomputed at start-up, drafted tokens picked on the device (no logits read back), target features injected from the device, compute buffer sized to its real outputs |
| scheduler input uploads | the small inputs of each split staged in pinned memory and uploaded on the GPU stream (one kernel per split) instead of one synchronous copy each; no GPU wait before a CPU split that follows a GPU split (the drafter's) |
| tooling | test-backend-ops cases, batched generation and the rollback depth in llama-bench, allocator diagnostics |
| docs | this section, `scripts/inkling2-amd`, `.gitignore` |

## Which AMD GPUs

The GPU code paths of this branch are gated by one test, `ggml_cuda_ink2_arch_ok` (in
`ggml/src/ggml-cuda/inkling2-fused.cuh`): true on RDNA3.0 GPUs (gfx1100, gfx1101, gfx1102, gfx1103), false on every
other GPU and in CUDA builds, which then run the stock code of the release. Only gfx1100 was built and measured. The
staged input uploads of the scheduler are host code and are used on every ROCm device; only their batched upload kernel
is gated to RDNA3.0 (table).

| change | where it runs | elsewhere |
|---|---|---|
| far mode (bias inside the flash-attention kernel, rotated rings, skip of masked K/V blocks, the bias table as one flat product), 3-head flash-attention tiles for a GQA ratio of 3 (any model with that ratio and head size 128), single-row tile kernel for decode, fused small-op kernels, Q8_0 mat-vec on F32 activations and grouped Q8_0 products (any model with Q8_0 weights), NVFP4 MoE mat-vec with 4 rows per block, expert products of prompt batches on FP16 WMMA (any model with NVFP4 experts, or Q8_0 / Q5_K experts in batches of 256 tokens or more) | RDNA3.0 (gfx110x) | stock kernels. `GGML_CUDA_INK2_ANY_AMD=1` also enables them on the other AMD GPUs (RDNA3.5, RDNA4, RDNA1/2, GCN, CDNA): untested, the fused kernels assume 32-lane warps |
| shape of the stock Q8_0 mat-vec for one column | RDNA3.0 (mat-vec table `RDNA3_0`) | stock |
| branchless decode of the NVFP4 scales (same values) | RDNA3 and RDNA3.5 device code (gfx11) | stock |
| Inkling 2 attention without the far mode | ROCm devices: the dense path (the generic banded kernel of #386 was slower there); NVIDIA: the tensor-core banded kernel of #386; other backends: the path of the ports | |
| Inkling 2 graph changes (device-built masks, graph order), follow-up prompts on the far path, restore order of checkpoints | every backend, Inkling 2 only | |
| DSpark drafter changes | every backend; target features from the device only when the first GPU is a CUDA or ROCm device; the device-side wait for the target batch only on ROCm | host copies and the host wait of the release |
| staged input uploads of the scheduler, no GPU wait before a CPU split after a GPU split | ROCm devices: one upload kernel per split on RDNA3.0, one asynchronous copy per input on the other AMD GPUs (untested there) | the synchronous copies and the wait of the release |

A build only contains kernels for the targets given at build time (`GPU_TARGETS`): build for your GPU. On an RDNA3.0
GPU other than gfx1100 run the verification below before anything else.

## Requirements

- An AMD GPU supported by ROCm/HIP with enough memory: the recommended setting below (16.4 GB file, drafter, 65536
  context) peaked at about 20.2 GB, so a 24 GB GPU is the practical minimum to keep everything on the GPU (see
  "Context, ubatch and VRAM"). The speed-ups need an RDNA3.0 GPU (RX 7900 XTX / XT / GRE, RX 7800 XT, RX 7700 XT,
  RX 7600, Radeon PRO W7000, the Radeon 780M/760M integrated GPUs); others run at the speed of the release.
- The HIP SDK for Windows or ROCm on Linux. Measured with HIP SDK 7.2 on Windows; Linux was not tested with this branch.
- CMake and Ninja; on Windows the Visual Studio C++ build tools.
- Flash attention (`-fa 1`) and an F16 or BF16 KV cache (the default) for the fast path: a quantized KV cache falls back
  to the dense bias.

## Build

Windows (HIP SDK installed, `HIP_PATH` set by its installer), from the root of this repository:

```bat
set GPU_TARGETS=gfx1100
scripts\inkling2-amd\build-hip.cmd [build-dir]
```

Variables of the script: `GPU_TARGETS` (default gfx1100; a list such as `gfx1100;gfx1101` works), `VCVARS` (path of
`vcvars64.bat`; found with `vswhere` when unset), `INK_CMAKE` (extra configure options, e.g. `-DLLAMA_BUILD_TESTS=ON`),
`INK_TARGETS` (default `llama-server llama-bench llama-perplexity llama-cli llama-quantize`; add `test-backend-ops`
with the tests option). The build directory defaults to `build-hip` and the binaries land in `<build-dir>\bin`. The
configure step runs once per build directory. Options: `-DGGML_HIP=ON -DGGML_HIP_GRAPHS=ON -DGGML_HIP_NO_VMM=ON
-DGGML_NATIVE=ON`, the HIP SDK's clang as C and C++ compiler. `GGML_NATIVE=ON` tunes the CPU code for the build
machine: turn it off for binaries meant for other machines. With the default targets and one GPU target a full build
took about 8 minutes.

Linux (ROCm; not tested with this branch, same options except the Windows-only ones):

```sh
GPU_TARGETS=gfx1100     # your GPU: rocminfo | grep -m1 -o 'gfx[0-9a-f]*'
cmake -S . -B build-hip -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_HIP_COMPILER="$(hipconfig -l)/clang" -DGGML_HIP=ON -DGPU_TARGETS="$GPU_TARGETS" \
  -DGGML_HIP_GRAPHS=ON -DLLAMA_CURL=OFF
cmake --build build-hip --target llama-server llama-bench llama-perplexity llama-cli llama-quantize -j "$(nproc)"
```

## Run

```sh
MODEL=<path to Inkling-2-Mini-h6-lbf16-UD-Q4_K_M.gguf or the Unsloth Inkling-2-Mini-UD-Q4_K_M.gguf>
DRAFT=<path to dspark-Inkling-2-Mini-Q8_0.gguf from unsloth/Inkling-2-Mini-GGUF>
llama-server -m "$MODEL" -ngl 99 -fa 1 -c 65536 -np 1 -b 2048 -ub 1024 --jinja \
  -md "$DRAFT" -ngld 99 --spec-type draft-dspark --spec-draft-n-max 3 --spec-draft-n-ubatch 512 \
  --backend-sampling --temp 1.0 --top-p 1.0 --top-k 0 --min-p 0.0 \
  --chat-template-kwargs '{"reasoning_effort":"high"}' --host 127.0.0.1 --port 8080
```

Stop it with Ctrl+C. In the Windows command prompt, end the lines with `^` instead of `\`, use `%MODEL%` and write the
JSON as `"{\"reasoning_effort\":\"high\"}"`.

- `-ub 1024` reads prompts about 20 % faster than 512 and needs about 0.9 GB more; `-ub 2048` needed about 2 GB more
  and was not faster. The drafter reads the prompt by 512 (`--spec-draft-n-ubatch 512`); a drafter ubatch of 2048 was
  slower and used 1.5 GB more.
- `--backend-sampling` samples on the GPU: about 4 % faster writing with the drafter (four alternating A/B pairs on RX
  7900 XTX). The server turns it off by itself for requests with a grammar (tool calls), which then sample on the CPU.
- `--spec-draft-n-max 3` was the best draft length at short context and about as fast as 2 after a long prompt.
  Without a drafter, drop the `-md ... --spec-draft-n-ubatch 512` options.
- Sampling as recommended for the model (temperature 1.0, top-p 1.0, top-k 0, min-p 0). `reasoning_effort` takes none,
  minimal, low, medium, high or xhigh, or a number from 0 to 0.99.

## Context, ubatch and VRAM

Measured on RX 7900 XTX (24 GB) with the h6-lbf16 file and the command line above:

- weights: 16.44 GB file; the DSpark drafter 0.40 GB, plus about 55 MB for its precomputed head.
- KV cache: the 8 global layer applications keep a full F16 cache of 32.8 KB per token, 2 GiB per 65536 tokens; the
  sliding-window cache stays small (1024-token window plus one ubatch).
- the whole server process (drafter on, context 65536, `-ub 1024`): a peak of 19.90 GB dedicated + 0.25 GB shared over
  a long run (19.14 + 0.18 GB at load). Memory becomes resident when it is first used: judge by the peak over a real
  workload, not by the figure at load.

Estimates, not measured: each further 65536 tokens of context adds about 2.1 GB of F16 KV cache, so roughly 21 GB at
98304 and 22 GB at 131072 with the drafter, before any growth of the compute buffers with the context; without the
drafter about 1.5 GB less; `-ub 512` about 0.9 GB less. Leave room for the display and other programs. On Windows an
overflow is not an error: buffers move to shared system memory and the speed collapses, so watch the shared GPU memory
of the process when the speed drops. GPUs with 16 or 20 GB cannot keep these settings on the GPU; partial offload
(`-ngl` below 99) was not measured.

## Environment switches

Everything below is off or at its fast default when unset. The switches of the first table restore the previous
behaviour of one change, for A/B tests and bisection.

| switch | default | when set |
|---|---|---|
| `GGML_CUDA_INK2_ANY_AMD=1` | off | the GPU paths of this branch on every AMD GPU, not only RDNA3.0 (untested) |
| `GGML_CUDA_INK2_FORCE_STOCK=1` | off | the stock paths on RDNA3.0 too (what other GPUs get) |
| `INKLING2_FAR=0` | on | no far mode: back to the banded or dense path of the ports |
| `INKLING2_SWA_ROT=0` | on | a wrapped sliding-window cache falls back to the dense bias |
| `INKLING2_BANDED=0` | on | the chunked dense path even where the banded op is supported |
| `INKLING2_KQ_BIAS_LOG2=n` | 25 / 29 | budget of the dense bias per attention call, 2^n elements (16..32); 25 where the far op runs, 29 elsewhere |
| `INKLING2_FAR_MIN=n` | 64 | far mode for ubatches from n tokens; lower values break the graph reuse of the verify batches and are slower |
| `INKLING2_MASK_DEV=0` | on | KQ masks filled on the host |
| `INKLING2_FAR_MASK_COPY=1` | off | one copy of the mask per far call |
| `INKLING2_REL_SUB=1` | off | the sinks subtracted from the bias table by a separate op |
| `INKLING2_REL_FLAT=0` | on | the bias table of the far op multiplied in the form broadcast over the tokens (on ROCm one small GEMM per token, about 3x slower) |
| `INKLING2_BATCH_OPT_OFF=1` | off | previous graph layout (mix rows copied one by one, one zero-sink tensor per call, convolutions not reordered) |
| `INKLING2_RESTORE_ORDER=0` | on | the cells of a sliding-window sequence restored from a checkpoint keep their saved order (the ring then no longer reads as one rotation) |
| `GGML_CUDA_INK2_NO_FUSE=1` | off | no fused kernel of this branch |
| `GGML_CUDA_INK2_FUSE_MASK=bits` | all | fused kinds: 1 L2 clip, 2 depth mix, 4 loop gate, 8 short-conv tail, 16 SITU activation, 32 MoE weight normalisation, 64 short-conv input and state store, 128 relative-bias chain, 256 short-conv rollback store, 512 dynamic conv, 1024 full short conv, 2048 gate tail, 4096 first maximum of a row (the drafter's token pick), 8192 SITU activation written by the FP16 expert kernel of prompt batches |
| `GGML_CUDA_INK2_CLIP_SCALED=0` | on | the multiplication by the clip weight stays a separate kernel |
| `GGML_CUDA_INK2_GATE_MT=0` | on | the first version of the loop-gate kernels |
| `GGML_CUDA_INK2_Q8_GROUP=0` | on | no grouping of the Q8_0 products that share their activations |
| `GGML_CUDA_INK2_MMV_F32=0` | on | the stock Q8_0 mat-vec (activations quantized to q8_1 first) |
| `GGML_CUDA_INK2_Q8_NR`, `GGML_CUDA_INK2_Q8_WPB` | tuned | rows per warp and warps per block of that kernel, comma lists per column count (tuning) |
| `GGML_CUDA_INK2_MOE_RPB=n` | 4 | rows per block of the NVFP4 MoE mat-vec: 1, 2, 4 or 8 (upstream: 2) |
| `GGML_CUDA_INK2_MOE_F16=n` | 32 | NVFP4 expert products (`MUL_MAT_ID`) of n tokens or more on FP16 WMMA, fewer tokens on MMQ; `=0`: always MMQ (int8 products on q8_1 activations) |
| `GGML_CUDA_INK2_MOE_F16_Q=n` | 256 | Q8_0 and Q5_K expert products (the last 4 layers of the UD-Q4_K_M files) of n tokens or more on the same FP16 kernel; `=0`: always MMQ |
| `GGML_CUDA_INK2_MOE_LDS1=0` | on | two shared-memory buffers in the NVFP4 FP16 expert kernel instead of one (one buffer: twice the blocks per compute unit) |
| `GGML_CUDA_INK2_FA_TILE1=0` | on | a single query row with a per-head mask goes back to the vector kernel |
| `GGML_CUDA_INK2_FA_LIVE=0` | on | no skip of the fully masked K/V blocks (`=2`: host check of the block bits, debug, needs `GGML_CUDA_DISABLE_GRAPHS=1`) |
| `GGML_CUDA_INK2_FA_GQA3=n` | 16 | queries per tile of the 3-head flash-attention tiles (batches of more than 16 tokens): `=0` back to 4 heads per tile with a padding column, `=32` tiles of 32 queries for batches of more than 32 tokens |
| `GGML_CUDA_DISABLE_BANDED_MMA=1` | off | (from #386, NVIDIA) no tensor-core banded kernel |
| `INKLING2_DSPARK_HEADC=0` | on | no precomputed drafter head: one product with the output matrix per drafted position |
| `INKLING2_DSPARK_DEV=0` | on | drafter features through host copies |
| `INKLING2_DSPARK_DEVSEL=0` | on | the drafting graph builds the logits of the drafted positions again and the host samples them (same drafts, slower) |
| `GGML_SCHED_ASYNC_INPUTS=n` | 1 | `=0`: one synchronous copy per input, as in the release; `=3`: one asynchronous copy per staged input instead of the batched upload kernel; `=2`: the staged uploads on any GPU backend with asynchronous copies, not only ROCm (untested) |
| `GGML_SCHED_CPU_SPLIT_WAIT=1` | off | the scheduler waits for the GPU again before a CPU split that follows a GPU split |
| `INKLING2_INJECT_DEVWAIT=0` | on | the drafter waits for the target batch on the host |
| `INKLING2_DSPARK_ALL=1` | off | every position of a draft block is an output |
| `INKLING2_DSPARK_ALL_ROWS=1` | off | logits for every row of large drafter ubatches |
| `INKLING2_DRAFT_RESERVE_ALL=1` | off | drafter compute buffer reserved for n_batch outputs |

Build-time flags (`-DCMAKE_CXX_FLAGS=-D...`, not needed normally): `GGML_CUDA_INK2_UE4M3_LDEXP` (previous ldexpf
decode of the NVFP4 scales), `INK2_Q8_NWARPS` and `INK2_Q8_RPB` (shape of the stock Q8_0 mat-vec for one column, default
2 warps and 1 row), `INK2_MMVF8_BYTES` and `INK2_MMVF8_WARPS` (shape of the F32-activation Q8_0 kernel, default 16 bytes
per lane and 1 warp), `GGML_CUDA_INK2_FA_NO_SCHED` (no scheduling barriers in the 3-head flash-attention tiles; the
barriers keep the fragment loads next to their WMMA and reduce register spills), `GGML_CUDA_INK2_FA_RDNA_48` and
`GGML_CUDA_INK2_FA_RDNA_96` (configs of the 3-head tiles of 16 and 32 queries, default `96,2,64,64,64,64,1,true` and
`192,1,64,64,64,64,1,true`: threads, occupancy, nbatch_fa, nbatch_K2, nbatch_V2, nbatch_combine, nstages_target,
Q_in_reg), `GGML_CUDA_INK2_FA_RDNA_32` and `GGML_CUDA_INK2_FA_RDNA_64` (same format, override the RDNA configs of the
32- and 64-column tiles of head size 128; unset by default: tuning only), `GGML_CUDA_INK2_COST_TOOLS` (compiles in the
timing-only switches of the last table, which give wrong results, the timing-only build flag `GGML_CUDA_INK2_FA_NO_NEAR`
(no near-band relative bias in the far tiles, wrong results), and the head-mask WMMA experiment
`GGML_HIP_FATTN_WMMA_HEADMASK=n`, which can hang the GPU).

Benchmark aids: `INKLING2_N_RS_SEQ=n` (read by llama-bench only) sets the recurrent rollback depth; with `=3`, llama-bench
builds the graph that llama-server uses for the target model with a drafter of n-max 3. `LLAMA_BENCH_GEN_BATCH=B` makes
llama-bench generate in steps of B tokens, all with logits, like verify batches.

Measurement switches, off (correct results, kept for A/B tests; neither paid off on RX 7900 XTX): `GGML_CUDA_INK2_Q8_BLAS=n`
sends the dense Q8_0 products of n columns or more to dequantization + hipBLAS instead of MMQ (faster for some shapes,
slower for others, FP16 accumulation: perplexity +0.1 %); `GGML_CUDA_INK2_BLAS_F32OUT=1` asks F32 output and accumulation
for the F16 / BF16 hipBLAS products (5-6x slower).

<details>
<summary>Diagnostics only (print to stderr; some synchronize the GPU after every node)</summary>

| switch | what it does |
|---|---|
| `GGML_CUDA_INK2_VERIFY=1` | runs every fused pattern with the generic ops too and prints the largest normalized error per kind at exit (with `GGML_CUDA_DISABLE_GRAPHS=1`; used by check.ps1); a comparison with a NaN shows as `max nmse nan, NaN in n comparisons` |
| `GGML_CUDA_INK2_VERIFY_POISON=1` | with the self-check: test of the check itself, the host copy of the first fused output of each kind gets a NaN and every kind must then report `max nmse nan` |
| `GGML_CUDA_INK2_TRACE=1` or `=2` | histogram of the launches of the last large graph, `=2` with a time per launch (synchronizes every node; used by check.ps1) |
| `GGML_CUDA_INK2_SEQ=start,count` | with the trace: the launches of a node range of the second large graph |
| `GGML_CUDA_INK2_TRACE_NODES=min,max` | with the trace: the graphs it looks at, by node count (default 5000 and up, the target model's graphs; a window of a few hundred nodes selects the drafter's drafting graph) |
| `GGML_CUDA_INK2_TRACE_AT=n` | with the trace: also print the histogram of the n-th traced graph while running (a server that is killed runs no exit handler) |
| `GGML_CUDA_INK2_TIMELINE=1` | one line per device interaction (graph launch and end, stream and event waits, copies, batched uploads) and per host stage of a ubatch, microsecond clock; no added synchronization, but the buffered writes add about 0.3-2 ms per decoding step, and lines still in the buffer are lost when the process is killed |
| `GGML_CUDA_INK2_DEBUG=1` or `=2` | matcher logs; `=2` also counts fused launches |
| `GGML_CUDA_INK2_RAW=start,count`, `GGML_CUDA_INK2_DUMP=start,count` | with the debug logs: the nodes of a range, all of them or the real ops with the fused patterns |
| `GGML_CUDA_INK2_TIME=1` or `=2`, `GGML_CUDA_INK2_TIME_MIN=n` | host enqueue time and GPU tail of large graphs (needs `GGML_CUDA_DISABLE_GRAPHS=1`) |
| `GGML_CUDA_INK2_RELBIAS_LOG=1` | shapes of the fused relative-bias calls |
| `GGML_CUDA_POOL_LOG=1` | every new buffer of the CUDA memory pool |
| `GGML_CUDA_MEM_TRACE=1` | nodes after which the free device memory drops (synchronizes every node) |
| `GGML_SCHED_TIME=1` | split, wait and allocation time of the scheduler |
| `GGML_ALLOC_TOP=MiB`, `GGML_ALLOC_GROWTH=1` | large tensors of a reserved graph; growth of a compute buffer after its first allocation |
| `INKLING2_MASK_TIME=1` | host time of the KQ mask fills |
| `INKLING2_RING_LOG=1` | why a ubatch cannot read the sliding-window ring as a rotated ring (cell runs) |
| `INKLING2_HOST_TIME=1` | host stages of each ubatch and of the drafter injections |
| `INKLING2_STEP_LOG=1` | one line per decoding step (costs about 15 % by itself) |
| `INKLING2_INJECT_TIME=1` | time of the drafter injection (serializes the two contexts) |
| `INKLING2_DRAFT_LOG=1` | the drafted tokens of every step |
| `GGML_CUDA_INK2_SKIP_KIND=bits` | builds with `GGML_CUDA_INK2_COST_TOOLS` only; wrong output, timing only: the fused kernels of these kinds are not run |
| `GGML_CUDA_INK2_SKIP_OP="KEY;KEY"` | same build only; wrong output: single nodes with these trace keys are not run |
| `GGML_CUDA_INK2_SKIP_MMVQ=type` | same build only; wrong output: mat-vec products of that weight type are not run |
| `INKLING2_NOBIAS=1` | same build only; wrong output: the relative bias is dropped |
| `INKLING2_SKIP_INJECT=1` | same build only; worse drafts: the drafter does not read the prompt |

</details>

## Verify

Run this first on any GPU, and always under a time limit: a kernel variant that is not compiled for the device prints
"no device code" and then hangs at 0 % GPU (rebuild with the right `GPU_TARGETS`).

Windows: `scripts/inkling2-amd/check.ps1` does all of it with a time guard (paths are parameters; `-Bin` defaults to
`build-hip\bin` of this repository):

```powershell
.\scripts\inkling2-amd\check.ps1 -Model $env:INKLING_MODEL
.\scripts\inkling2-amd\check.ps1 -Model $env:INKLING_MODEL -Env @{INKLING2_N_RS_SEQ='3'}
```

Linux or by hand, the same steps (`MODEL` and `BIN` set by you):

```sh
export GGML_CUDA_DISABLE_GRAPHS=1
GGML_CUDA_INK2_VERIFY=1 timeout 300 "$BIN/llama-bench" -m "$MODEL" -ngl 99 -fa 1 -p 256 -n 24 -r 1
GGML_CUDA_INK2_VERIFY=1 LLAMA_BENCH_GEN_BATCH=4 timeout 300 "$BIN/llama-bench" -m "$MODEL" -ngl 99 -fa 1 -p 0 -n 16 -d 64 -r 1
GGML_CUDA_INK2_TRACE=1 timeout 300 "$BIN/llama-bench" -m "$MODEL" -ngl 99 -fa 1 -p 0 -n 8 -r 1
unset GGML_CUDA_DISABLE_GRAPHS
for f in 1 ""; do GGML_CUDA_INK2_NO_FUSE=$f timeout 600 "$BIN/llama-bench" -m "$MODEL" -ngl 99 -fa 1 -p 2048 -n 128 -b 2048 -ub 512 -r 2; done
```

1. Self-check: every fused pattern is computed with the generic ops and with the fused kernel on the real model data.
   Expected (measured on RX 7900 XTX): one `ink2 verify <kind>` line per kind at exit with a max nmse of 0 or below
   1e-11. `sconv_full` and `sconv_in` report their patterns as not comparable: their output overwrites one of their inputs
   by design. With
   the h6-lbf16 file the loop gate of the 4-token batches shows about 5e-5, because the generic reference rounds its
   activations to BF16. A larger error means a fused kernel is wrong on that GPU: run with `GGML_CUDA_INK2_NO_FUSE=1`
   and report it. No `ink2 verify` line at all means the GPU paths are off (not an RDNA3.0 GPU).
2. Trace: about 1660 launches per generated token.
3. Speed with and without the fusions of this branch (llama-bench, pp2048 and tg128, `-ub 512`).

test-backend-ops (build with `-DLLAMA_BUILD_TESTS=ON`, run it under a time limit too): `test-backend-ops -b ROCm0 -o
FLASH_ATTN_EXT_BANDED` checks the far mode against the CPU (padding, rotations, sliding windows; on gfx1100 two random
cases with 33 792 keys sit at the 1e-3 NMSE threshold), `-o MUL_MAT` and `-o MUL_MAT_ID` cover the Q8_0 and NVFP4
mat-vec kernels and the FP16 expert kernel for NVFP4, Q8_0 and Q5_K experts (`GGML_CUDA_INK2_MOE_F16=1` and
`GGML_CUDA_INK2_MOE_F16_Q=1` take it from one token on).

Perplexity A/B on the same text (`llama-perplexity -f <text> -c 2048 --chunks 4 -ngl 99 -fa 1 -b 2048 -ub 512`) against
the stock paths (`GGML_CUDA_INK2_FORCE_STOCK=1`): expect differences of 0.1 to 0.3 % without a direction (the kernels
round differently), not more.

## Measurements

Measured on Radeon RX 7900 XTX (24 GB), Windows, HIP SDK 7.2, 2026-11-21; this branch built by
`scripts/inkling2-amd/build-hip.cmd` from the published tree; both engines with the Unsloth UD-Q4_K_M file unless said
otherwise. `llama-server`, one slot. A prompt of source code is read once, then 6 answers of up to 300 tokens are
generated after it (temperature 1.0, top-p 0.95, top-k 20, a fixed seed per answer). Reading = the server's prompt
tokens per second; writing = generated tokens per second after the prompt; VRAM = peak of the server process over the
run, dedicated + shared system memory (every run holds about 0.1-0.2 GB of shared memory from load on: pinned host
buffers, not a spill). The release runs with the `llama-server` line of Unsloth's "How to Run" guide
(`--jinja --n-gpu-layers 99 --parallel 1`, `--temp 1.0 --top-p 1.0 --top-k 0 --min-p 0.0`, and `--model-draft ...
--spec-type draft-dspark --spec-draft-n-max 3 --n-gpu-layers-draft 99` for the drafter; default flash-attention, batch
and ubatch settings; without its vision projector), this branch with the command line of "Run" (without its drafter
options in the rows without drafter), each at the context given.

### Where the release fits in VRAM

| engine, context, prompt | reading | writing | tokens per drafter step | VRAM peak |
|---|---|---|---|---|
| release, no drafter, context 32768 (Unsloth's guide), 24 829 tokens | 829 tok/s | 42.2 tok/s | | 19.95 + 0.22 GB |
| this branch, no drafter, context 32768, 24 829 tokens | 2103 tok/s | 77.6 tok/s | | 19.09 + 0.14 GB |
| release, DSpark drafter, context 8192, 6 494 tokens | 984 tok/s | 60.2 tok/s | 2.65 (44.0 ms per step) | 19.07 + 0.37 GB |
| this branch, DSpark drafter, context 8192, 6 494 tokens | 1884 tok/s | 93.5 tok/s | 2.53 (27.1 ms per step) | 17.78 + 0.19 GB |

The release with the drafter does not stay in dedicated memory above 8192 on this card: at 32768 it keeps 1.20 GB in
shared memory from load on (reading 766 tok/s, writing 44.1 tok/s), at 24576 2.72 GB, and at 16384 it fits at load but
grows to 2.12 GB of shared memory while it reads the prompt (reading 54 tok/s on 14 356 tokens). This branch with the
drafter: 2050 tok/s and 87.7 tok/s at 32768 (18.77 + 0.22 GB), 2053 tok/s and 90.0 tok/s at 16384 (18.13 + 0.20 GB).

### At context 65536 the release does not fit on a 24 GB card

Same text, 52 804 tokens; a second identical client run reads only the follow-up after the cached prefix. At this
context the release warns `failed to fit params to free device memory`, keeps every layer on the GPU as asked, and 4 to
6 GB of its buffers sit in shared system memory from the start: its rows measure that spill as much as the engine.
| engine, file, server options | reading | writing | tokens per drafter step | follow-up | VRAM peak |
|---|---|---|---|---|---|
| release b12568-mix-8eebad4 (HIP build), Unsloth UD-Q4_K_M, the options of Unsloth's guide, no drafter | 265 tok/s | 35.3 tok/s | | 514 tokens at 112 tok/s | 20.31 + 4.05 GB |
| same, with the DSpark drafter (n-max 3) | 261 tok/s | 25.3 / 24.9 tok/s | 2.46 / 2.43 | 514 tokens at 110 tok/s | 20.41 + 5.77 GB |
| this branch, Unsloth UD-Q4_K_M, the command line of "Run" | 1898 / 1901 tok/s | 90.3 / 90.2 tok/s | 2.66 (29.5 ms per step) | | 20.06 + 0.25 GB |
| this branch, UD-Q4_K_M-h6-lbf16, the command line of "Run" | 1943 / 1943 tok/s | 87.6 / 87.4 tok/s | 2.55 (29.1 ms per step) | 1026 tokens at 1223 tok/s | 19.90 + 0.25 GB |

- Later engine revision (2026-11-21, after the rows above: 3-head attention tiles for GQA ratio 3, NVFP4 expert products of prompt batches on FP16 WMMA), h6-lbf16 file, the command line of "Run", same 52 804-token prompt: reading 1952 -> 2381 tok/s (2442 in a second session), writing unchanged within noise (83-87 tok/s), VRAM peak 19.92 GB dedicated + 0.25 GB shared; perplexity 6.2610 against 6.2584 with the new paths off (`GGML_CUDA_INK2_FA_GQA3=0 GGML_CUDA_INK2_MOE_F16=0`; same text and settings for both, not those of the other perplexity figures here).
- Next engine revision (2026-11-21, after the line above: drafted tokens picked on the device, input uploads staged and batched on the GPU stream, no GPU wait before the drafter's CPU split), h6-lbf16 file, the command line of "Run", same 52 804-token prompt: reading 2439-2467 tok/s (was 2381-2442), writing 100-103 tok/s (was 83-87), VRAM peak unchanged at 19.92 GB dedicated + 0.25 GB shared.
- Engine revision of 2026-11-22 (after the line above: SITU activation written by the FP16 expert kernel, Q8_0 / Q5_K experts of the last 4 layers on that kernel, the relative-bias table as one flat product, one shared-memory buffer in the NVFP4 expert kernel), h6-lbf16 file, the command line of "Run", same 52 804-token prompt: reading 2593-2596 tok/s (was 2439-2467), writing 98-103 tok/s after the prompt (unchanged within noise), VRAM peak 19.62 GB dedicated + 0.25 GB shared (was 19.92 + 0.25); perplexity 6.2608 (6.2610 before, same text and settings as in the first line).
- Rows of this branch: two alternated runs per file (Unsloth, h6-lbf16, Unsloth, h6-lbf16). With the fixed seeds each
  file gave the same answers in both of its runs; the Unsloth file's answers happened to accept more drafts (2.66 tokens
  per step against 2.55), so its tokens per second are higher while the h6-lbf16 file is 1.2 % faster per drafter step.
  The follow-up of the h6-lbf16 row comes from a separate run of the same setting (reading 1959 tok/s, writing 88.2 /
  90.3 tok/s in that run).
- Without the drafter (same code before the drafter-head change, the command line of "Run" without the drafter options):
  writing 71.1 tok/s with `--backend-sampling`, 70.8 without; prompt reading 2022 tok/s in an earlier revision. At short
  context (8192, no drafter, three short prompts): 80 tok/s with the h6-lbf16 file, against 44 tok/s for the release with the
  Unsloth file.
- llama-bench (check.ps1: depth 0, `-ub 512`, h6-lbf16 file): pp2048 2424 tok/s and tg128 89.5 tok/s, 1840 and 53.0 with
  `GGML_CUDA_INK2_NO_FUSE=1` (the far mode stays on); the release with the Unsloth file: pp512 1606 tok/s, tg 46.8 tok/s.
- Perplexity (`-c 2048 --chunks 4 -ub 512`, start of the WikiText-2 raw test set), Unsloth UD-Q4_K_M file: 6.2387 with
  the release, 6.2364 with this branch (statistical error of the estimate: 0.23). Details: the model card.

## Limitations

- Built and measured on one GPU model (RX 7900 XTX, gfx1100) under Windows with HIP SDK 7.2. Every other GPU runs the
  stock kernels of the release (the staged input uploads of the scheduler run on every ROCm device and were measured on
  gfx1100 only); the other RDNA3.0 chips (gfx1101-gfx1103) take the same paths as gfx1100 but were not
  measured; `GGML_CUDA_INK2_ANY_AMD=1` on other AMD GPUs, Linux, and CUDA, MUSA, Vulkan, Metal or SYCL builds of this
  branch were not tested.
- Quality: checked by perplexity, the self-check and three smoke questions only (see the model card); paths that round
  differently move the perplexity by 0.1 to 0.3 %. No task benchmark compares this branch with the release.
- Contexts from 8192 to 65536 tokens were measured; longer ones were not.
- The diagnostics listed above are still in the code (off by default, read once at start-up); the switches that give
  wrong results are compiled out unless `GGML_CUDA_INK2_COST_TOOLS` is defined.

## Attribution and license

llama.cpp: MIT, the ggml-org/llama.cpp authors (see LICENSE and AUTHORS). The release tree: Unsloth, with the pull
requests listed in the import commit. #385 and #386: Daniel Han (Unsloth). Inkling 2 Mini: Thinking Machines, Apache-2.0.
GGUF files and the DSpark drafter: Unsloth. The changes of this branch are under the MIT license of the project.

The changes of this branch were written with AI assistance (Anthropic's Claude models, in Claude Code) and measured on
one Radeon RX 7900 XTX.

---

# llama.cpp

![llama.cpp](https://raw.githubusercontent.com/ggml-org/llama-brand/refs/heads/master/logo/svg/cover/llama-cpp/cover-llama-cpp-dark.svg)

<div align="center">

<b>LLM inference in C/C++</b>

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Release](https://img.shields.io/github/v/release/ggml-org/llama.cpp?color=brightgreen)](https://github.com/ggml-org/llama.cpp/releases?q=tag:v1)
[![Nightly](https://img.shields.io/github/v/release/ggml-org/llama.cpp?label=nightly&filter=b*&color=orange)](https://github.com/ggml-org/llama.cpp/releases?q=b)
[![Server](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/server.yml?label=Server)](https://github.com/ggml-org/llama.cpp/actions/workflows/server.yml)
[![Docker](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/docker.yml?label=Docker)](https://github.com/ggml-org/llama.cpp/actions/workflows/docker.yml)
[![Winget](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/winget.yml?label=Winget)](https://github.com/ggml-org/llama.cpp/actions/workflows/winget.yml)

[v1.0.0](https://huggingface.co/blog/ggml-org/llama-cpp-v1) / [ggml](https://github.com/ggml-org/ggml) / [ops](https://github.com/ggml-org/llama.cpp/blob/master/docs/ops.md) / [maintainer PRs](https://github.com/ggml-org/llama.cpp/pulls?q=is%3Apr%20is%3Aopen%20draft%3AFalse%20(author%3Asanmai%20OR%20author%3Argerganov%20OR%20author%3AKitaitiMakoto%20OR%20author%3Adanbev%20OR%20author%3Aaldehir%20OR%20author%3Amax-krasnyansky%20OR%20author%3ACISC%20OR%20author%3Aggerganov%20OR%20author%3Aam17an%20OR%20author%3Ajhen0409%20OR%20author%3Abartowski1182%20OR%20author%3Anikwen%20OR%20author%3Ahipudding%20OR%20author%3Aravi9%20OR%20author%3AServeurpersoCom%20OR%20author%3Athelittlefireman%20OR%20author%3Apwilkin%20OR%20author%3Areeselevine%20OR%20author%3Angxson%20OR%20author%3Ajeffbolznv%20OR%20author%3Amarty1885%20OR%20author%3A0cc4m%20OR%20author%3ATitaniumtown%20OR%20author%3Aangt%20OR%20author%3AIMbackK%20OR%20author%3Aarthw%20OR%20author%3AJohannesGaessler%20OR%20author%3AORippler%20OR%20author%3Aruixiang63%20OR%20author%3Axctan%20OR%20author%3AmasterFoad%20OR%20author%3Aallozaur%20OR%20author%3AAndreas-Krebbel%20OR%20author%3Ayomaytk%20OR%20author%3Aaendk%20OR%20author%3Awine99%20OR%20author%3Agaugarg-nv%20OR%20author%3Ataronaeo%20OR%20author%3Aforforever73%20OR%20author%3Achaxu01%20OR%20author%3Alhez%20OR%20author%3AAlekseiNikiforovIBM%20OR%20author%3Anetrunnereve%20OR%20author%3Afairydreaming%20OR%20author%3Aynankani)%20sort%3Aupdated-desc) / [dev stats](https://github.com/ggml-org/llama.cpp-dev) / [lib llama API](https://github.com/ggml-org/llama.cpp/issues/9289) / [llama-server REST API](https://github.com/ggml-org/llama.cpp/issues/9291)

</div>

## Description

The main goal of `llama.cpp` is to enable LLM (and MLLM) inference with minimal setup and state-of-the-art performance on a wide range of hardware - locally and in the cloud.

The `llama.cpp` project is built on top of the [ggml](https://github.com/ggml-org/ggml) library.

## Quick start

A few options to get `llama.cpp` installed on your machine:

```bash
# curl
curl -LsSf https://llama.app/install.sh | sh

# powershell
irm https://llama.app/install.ps1 | iex
```

- Visit https://llama.app and follow the instructions
- Run with Docker - see our [Docker documentation](docs/docker.md)
- Download pre-built binaries from the [releases page](https://github.com/ggml-org/llama.cpp/releases)
- Build from source by cloning this repository - check out [our build guide](docs/build.md)

Once installed:

```sh
# Download and run a model directly from Hugging Face
llama cli -hf ggml-org/Qwen3.5-0.8B-GGUF

# Launch OpenAI-compatible API server
llama serve -hf ggml-org/Qwen3.5-0.8B-GGUF
```

<table align="center">
    <tr>
        <td align="center" width=50%>
            <img width="1310" height="888" alt="VLM session with `llama cli`" src="https://github.com/user-attachments/assets/88726b48-1713-48aa-a525-95a02e78afc4" />
            <i>VLM session with <b>llama cli</b></i>
        </td>
        <td align="center">
            <img width="1392" height="958" alt="Llama Chat, the built-in chat app, against `llama serve` running Qwen 3.6" src="https://github.com/user-attachments/assets/b402f972-2e32-4def-8771-8d849f08cf2e" />
            <i>Llama Chat, the built-in chat app, against <b>llama serve</b></i>
        </td>
    </tr>
<table>

## Documentation

#### Tools

- [cli](tools/cli/README.md)
- [completion](tools/completion/README.md)
- [server](tools/server/README.md)
- [Text-to-speech (TTS) models](docs/multimodal/tts.md)
- [GBNF grammars](grammars/README.md)

#### Development

- [How to build](docs/build.md)
- [Running on Docker](docs/docker.md)
- [Build on Android](docs/android.md)
- [Multi-GPU usage](docs/multi-gpu.md)
- [Performance troubleshooting](docs/development/token_generation_performance_tips.md)
- [GGML tips & tricks](https://github.com/ggml-org/llama.cpp/wiki/GGML-Tips-&-Tricks)
- [XCFramework](docs/xcframework.md)
- [Completions](docs/completions.md)
- [Models](docs/models.md)
- [Release process](docs/release.md)

## Contributing

- Contributors can open PRs
- Collaborators will be invited based on contributions
- Maintainers can push to branches in the `llama.cpp` repo and merge PRs into the `master` branch
- Any help with managing issues, PRs and projects is very appreciated!
- Read the [CONTRIBUTING.md](CONTRIBUTING.md) for more information

## Acknowledgements

- [yhirose/cpp-httplib](https://github.com/yhirose/cpp-httplib) - Single-header HTTP server, used by `llama-server` - MIT license
- [nothings/stb](https://github.com/nothings/stb) - Single-header image format decoder, used by multimodal subsystem - Public domain
- [nlohmann/json](https://github.com/nlohmann/json) - Single-header JSON library, used by various tools/examples - MIT License
- [mackron/miniaudio](https://github.com/mackron/miniaudio) - Single-header audio format decoder, used by multimodal subsystem - Public domain
- [sheredom/subprocess.h](https://github.com/sheredom/subprocess.h) - Single-header process launching solution for C and C++ - Public domain
