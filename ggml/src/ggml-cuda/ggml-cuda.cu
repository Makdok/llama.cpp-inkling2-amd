#include "ggml-cuda.h"
#include "ggml-impl.h"
#include "ggml-backend-impl.h"

#include "ggml-cuda/allreduce.cuh"
#include "ggml-cuda/common.cuh"
#include "ggml-cuda/graph.cuh"
#include "ggml-cuda/acc.cuh"
#include "ggml-cuda/add-id.cuh"
#include "ggml-cuda/arange.cuh"
#include "ggml-cuda/argmax.cuh"
#include "ggml-cuda/argsort.cuh"
#include "ggml-cuda/batch-copy.cuh"
#include "ggml-cuda/binbcast.cuh"
#include "ggml-cuda/clamp.cuh"
#include "ggml-cuda/col2im-1d.cuh"
#include "ggml-cuda/concat.cuh"
#include "ggml-cuda/conv-transpose-1d.cuh"
#include "ggml-cuda/conv2d.cuh"
#include "ggml-cuda/conv2d-dw.cuh"
#include "ggml-cuda/conv2d-transpose.cuh"
#include "ggml-cuda/conv3d.cuh"
#include "ggml-cuda/convert.cuh"
#include "ggml-cuda/count-equal.cuh"
#include "ggml-cuda/cpy.cuh"
#include "ggml-cuda/cross-entropy-loss.cuh"
#include "ggml-cuda/cumsum.cuh"
#include "ggml-cuda/diag.cuh"
#include "ggml-cuda/diffusion-sampling.cuh"
#include "ggml-cuda/fattn.cuh"
#include "ggml-cuda/fattn-banded.cuh"
#include "ggml-cuda/fp8.cuh"
#include "ggml-cuda/fwht.cuh"
#include "ggml-cuda/getrows.cuh"
#include "ggml-cuda/im2col.cuh"
#include "ggml-cuda/mmf.cuh"
#include "ggml-cuda/mmid.cuh"
#include "ggml-cuda/mmq.cuh"
#include "ggml-cuda/mmq-few.cuh"
#include "ggml-cuda/mmvf.cuh"
#include "ggml-cuda/mmvq.cuh"
#include "ggml-cuda/moe-weighted-reduction.cuh"
#include "ggml-cuda/norm.cuh"
#include "ggml-cuda/opt-step-adamw.cuh"
#include "ggml-cuda/opt-step-sgd.cuh"
#include "ggml-cuda/out-prod.cuh"
#include "ggml-cuda/pad.cuh"
#include "ggml-cuda/pool2d.cuh"
#include "ggml-cuda/pool1d.cuh"
#include "ggml-cuda/quantize.cuh"
#include "ggml-cuda/rope.cuh"
#include "ggml-cuda/roll.cuh"
#include "ggml-cuda/scale.cuh"
#include "ggml-cuda/sleep.cuh"
#include "ggml-cuda/snake.cuh"
#include "ggml-cuda/softcap.cuh"
#include "ggml-cuda/inkling2-fused.cuh"
#include "ggml-cuda/softmax.cuh"
#include "ggml-cuda/ssm-conv.cuh"
#include "ggml-cuda/ssm-scan.cuh"
#include "ggml-cuda/sum.cuh"
#include "ggml-cuda/sumrows.cuh"
#include "ggml-cuda/top-k.cuh"
#include "ggml-cuda/mean.cuh"
#include "ggml-cuda/tsembd.cuh"
#include "ggml-cuda/topk-moe.cuh"
#include "ggml-cuda/unary.cuh"
#include "ggml-cuda/upscale.cuh"
#include "ggml-cuda/wkv.cuh"
#include "ggml-cuda/gla.cuh"
#include "ggml-cuda/gated_delta_net.cuh"
#include "ggml-cuda/dsv4-hc.cuh"
#include "ggml-cuda/set.cuh"
#include "ggml-cuda/set-rows.cuh"
#include "ggml-cuda/pad_reflect_1d.cuh"
#include "ggml-cuda/solve_tri.cuh"
#include "ggml-cuda/tri.cuh"
#include "ggml-cuda/cumsum.cuh"
#include "ggml-cuda/fill.cuh"
#include "ggml-cuda/lightning-indexer.cuh"
#include "ggml-cuda/top-k-mask.cuh"
#include "ggml.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <charconv>
#include <cinttypes>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cfloat>
#include <initializer_list>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
#include "ggml-backend-dl.h"
#include <nvml.h>
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

static_assert(sizeof(half) == sizeof(ggml_fp16_t), "wrong fp16 size");

#define GGML_LOG_WARN_ONCE(str) \
    { static std::once_flag warn_flag; std::call_once(warn_flag, []() { GGML_LOG_WARN(str); }); }

[[noreturn]]
void ggml_cuda_error(const char * stmt, const char * func, const char * file, int line, const char * msg) {
    int id = -1; // in case cudaGetDevice fails
    (void)cudaGetDevice(&id);

    GGML_LOG_ERROR(GGML_CUDA_NAME " error: %s\n", msg);
    GGML_LOG_ERROR("  current device: %d, in function %s at %s:%d\n", id, func, file, line);
    GGML_LOG_ERROR("  %s\n", stmt);
    // abort with GGML_ABORT to get a stack trace
    GGML_ABORT(GGML_CUDA_NAME " error");
}

// map a (possibly virtual) device id to the physical CUDA device that backs it
static int ggml_cuda_get_physical_device(int device) {
    const ggml_cuda_device_info & info = ggml_cuda_info();
    GGML_ASSERT(device >= 0 && device < info.device_count);
    return info.devices[device].physical_device;
}

// this is faster on Windows
// probably because the Windows CUDA libraries forget to make this check before invoking the drivers
void ggml_cuda_set_device(int device) {
    // translate the (possibly virtual) device id to the physical CUDA device that backs it
    const int physical_device = ggml_cuda_get_physical_device(device);

    int current_device;
    CUDA_CHECK(cudaGetDevice(&current_device));

    if (physical_device == current_device) {
        return;
    }

    CUDA_CHECK(cudaSetDevice(physical_device));
}

int ggml_cuda_get_device() {
    int id;
    CUDA_CHECK(cudaGetDevice(&id));
    return id;
}

static bool ggml_cuda_env_flag(const char * name) {
    const char * env = getenv(name);
    return env != nullptr && std::atoi(env) != 0;
}

static int ggml_cuda_env_int(const char * name, int default_value) {
    const char * env = getenv(name);
    return env != nullptr ? std::atoi(env) : default_value;
}

static bool ggml_cuda_env_set(const char * name) {
    return getenv(name) != nullptr;
}

// local diagnostic: GGML_CUDA_INK2_TIMELINE=1 writes a host timeline of the device interactions (graph enqueues, waits,
// synchronous and asynchronous copies) to stderr, one line per event: "ink2 tl <us> <who> <event> <a> <b>", buffered and
// written in chunks of 8192 lines (each write costs about a millisecond here; the last partial chunk is lost when the process
// is killed). It adds no synchronization; the gaps between the events of a speculative decoding step show where its host
// time goes.
static void ggml_cuda_ink2_tl(const void * who, const char * ev, long long a = 0, long long b = 0) {
    static const bool on = ggml_cuda_env_flag("GGML_CUDA_INK2_TIMELINE");
    if (!on) {
        return;
    }
    static std::mutex  mtx;
    static std::string buf;
    static int         n = 0;
    char line[128];
    snprintf(line, sizeof(line), "ink2 tl %lld %p %s %lld %lld\n", (long long) ggml_time_us(), who, ev, a, b);
    std::lock_guard<std::mutex> lock(mtx);
    buf += line;
    if (++n % 8192 == 0) {
        fwrite(buf.data(), 1, buf.size(), stderr);
        fflush(stderr);
        buf.clear();
    }
}

static bool ggml_cuda_unified_memory_enabled() {
    static const bool enabled = ggml_cuda_env_flag("GGML_CUDA_ENABLE_UNIFIED_MEMORY");
    return enabled;
}

static bool ggml_cuda_pinned_memory_disabled() {
    static const bool disabled = ggml_cuda_env_flag("GGML_CUDA_NO_PINNED");
    return disabled;
}

static bool ggml_cuda_register_host_enabled() {
    static const bool enabled = ggml_cuda_env_flag("GGML_CUDA_REGISTER_HOST");
    return enabled;
}

static bool ggml_cuda_fusion_disabled() {
    static const bool disabled = ggml_cuda_env_flag("GGML_CUDA_DISABLE_FUSION");
    return disabled;
}

static bool ggml_cuda_p2p_enabled() {
    static const bool enabled = ggml_cuda_env_set("GGML_CUDA_P2P");
    return enabled;
}

bool ggml_cuda_pool_debug_enabled() {
    static const bool enabled = ggml_cuda_env_flag("GGML_CUDA_POOL_DEBUG");
    return enabled;
}

static int ggml_cuda_op_offload_min_batch() {
    static const int min_batch = ggml_cuda_env_int("GGML_OP_OFFLOAD_MIN_BATCH", 32);
    return min_batch;
}

static cudaError_t ggml_cuda_device_malloc(void ** ptr, size_t size, int device) {
    ggml_cuda_set_device(device);
    cudaError_t err;
    if (ggml_cuda_unified_memory_enabled()) {
        err = cudaMallocManaged(ptr, size);
#if defined(GGML_USE_HIP)
        if (err == hipSuccess) {
            // said once: reports of this path going wrong come from users who did not know a front-end set the variable
            GGML_LOG_WARN_ONCE("GGML_CUDA_ENABLE_UNIFIED_MEMORY is set, allocating device memory as managed. "
                               "On integrated GPUs this draws host RAM instead of the device carve-out, which is "
                               "usually slower and can reduce the largest model that will load. Unset the variable "
                               "or set it to 0 to disable it.\n");

            // hipMemAdviseSetCoarseGrain is an optional performance hint;
            // ignore errors (e.g. hipErrorInvalidValue on some APU/iGPU configs).
            (void)cudaMemAdvise(*ptr, size, hipMemAdviseSetCoarseGrain, device);
            (void)hipGetLastError(); // clear any error
        }

        // fall back to cudaMalloc if not supported (e.g. on Windows)
        if (err == hipErrorNotSupported) {
            static bool warned_unsupported = false;
            if (!warned_unsupported) {
                GGML_LOG_WARN("hipMallocManaged unsupported, falling back to hipMalloc.\n");
                warned_unsupported = true;
            }

            err = cudaMalloc(ptr, size);
        }
#endif // defined(GGML_USE_HIP)
    } else {
        err = cudaMalloc(ptr, size);
    }
    return err;
}

#if defined(GGML_USE_HIP)
static int ggml_cuda_parse_id(const char * devName) {
    // A list of possible Target IDs can be found under the rocclr/clr repo in device.cpp
    // these values are not stable so this is susceptible to breakage
    // https://github.com/ROCm/clr/blob/amd-staging/rocclr/device/device.cpp
    // Extract the base architecture model ("gfx908:sramecc+:xnack-" -> "908")
    std::string archName = devName;
    archName.erase(0, 3);
    archName = archName.substr(0, archName.find(':'));

    const auto hex = [](const std::string & s) { return (int) strtoul(s.c_str(), nullptr, 16); };

    int cc = 0;

    // tease out the version information
    if (archName.size() > 8) {
        // versions labeled generic use '-' as delimiter
        if (archName.compare(archName.size() - 8, 8, "-generic") == 0) {
            archName.resize(archName.size() - 8);
            const size_t sep = archName.find('-');
            cc = 0x100 * hex(archName.substr(0, sep));
            if (sep != std::string::npos) {
                cc += 0x10 * hex(archName.substr(sep + 1));
            }
        }
    } else if (archName.size() >= 3) {
        // ordinary IDs are the version in hex
        cc = hex(archName);
    }

    return GGML_CUDA_CC_OFFSET_AMD + cc;
}
#endif // defined(GGML_USE_HIP)

static ggml_cuda_device_info ggml_cuda_init() {
    ggml_cuda_device_info info = {};

    cudaError_t err = cudaGetDeviceCount(&info.physical_device_count);
    if (err != cudaSuccess) {
        GGML_LOG_ERROR("%s: failed to initialize " GGML_CUDA_NAME ": %s\n", __func__, cudaGetErrorString(err));
        return info;
    }

    GGML_ASSERT(info.physical_device_count <= GGML_CUDA_MAX_DEVICES);

    // by default expose exactly the physical devices; GGML_CUDA_DEVICES can request a different
    // number of (virtual) devices to emulate multi-GPU systems on a machine with fewer GPUs
    info.device_count = info.physical_device_count;

    const char * devices_env = getenv("GGML_CUDA_DEVICES");
    if (devices_env != nullptr && info.physical_device_count > 0) {
        const int requested = atoi(devices_env);
        if (requested > 0) {
            info.device_count = requested;
        } else {
            GGML_LOG_WARN("%s: ignoring invalid GGML_CUDA_DEVICES=\"%s\"\n", __func__, devices_env);
        }
    }

    if (info.device_count > GGML_CUDA_MAX_DEVICES) {
        GGML_LOG_WARN("%s: requested %d devices, clamping to GGML_CUDA_MAX_DEVICES=%d\n",
                      __func__, info.device_count, GGML_CUDA_MAX_DEVICES);
        info.device_count = GGML_CUDA_MAX_DEVICES;
    }

    // map each (virtual) device to a backing physical device (round-robin), assign each its index
    // among the (virtual) devices sharing that physical GPU, and store the per-physical share count
    int physical_share_count[GGML_CUDA_MAX_DEVICES] = {};
    GGML_ASSERT(info.device_count == 0 || info.physical_device_count > 0);
    for (int id = 0; id < info.device_count; ++id) {
        info.devices[id].physical_device = id % info.physical_device_count;
        info.devices[id].virtual_index  = physical_share_count[info.devices[id].physical_device]++;
    }

    int64_t total_vram = 0;
    for (int id = 0; id < info.physical_device_count; ++id) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, id));
        total_vram += prop.totalGlobalMem;
    }
    GGML_LOG_INFO("%s: found %d " GGML_CUDA_NAME " devices (Total VRAM: %zu MiB):\n",
                  __func__, info.physical_device_count, (size_t)(total_vram / (1024 * 1024)));
    if (info.device_count != info.physical_device_count) {
        GGML_LOG_INFO("%s: emulating %d virtual device(s) on %d physical device(s) (GGML_CUDA_DEVICES)\n",
                      __func__, info.device_count, info.physical_device_count);
    }
    total_vram = 0;

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
#if defined(_WIN32)
    // NVML is installed by the driver; its NVSMI directory is not added to PATH on Windows.
    dl_handle_ptr library(LoadLibraryExW(L"nvml.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32));
    if (!library) {
        const DWORD size = GetEnvironmentVariableW(L"ProgramW6432", nullptr, 0);
        if (size > 0) {
            std::wstring path(size, L'\0');
            const DWORD length = GetEnvironmentVariableW(L"ProgramW6432", &path[0], size);
            if (length > 0 && length < size) {
                path.resize(length);
                path += L"\\NVIDIA Corporation\\NVSMI\\nvml.dll";
                if (fs::path(path).is_absolute()) {
                    library.reset(LoadLibraryExW(path.c_str(), nullptr, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS));
                }
            }
        }
    }
#else
    dl_handle_ptr library(dlopen("libnvidia-ml.so.1", RTLD_NOW | RTLD_LOCAL));
#endif // defined(_WIN32)
    const auto get_symbol = [&](const char * name) {
#if defined(_WIN32)
        return library ? GetProcAddress(library.get(), name) : nullptr;
#else
        return library ? dlsym(library.get(), name) : nullptr;
#endif // defined(_WIN32)
    };
    const auto init = reinterpret_cast<decltype(&nvmlInit_v2)>(get_symbol("nvmlInit_v2"));
    const auto shutdown = reinterpret_cast<decltype(&nvmlShutdown)>(get_symbol("nvmlShutdown"));
    const auto get_device = reinterpret_cast<decltype(&nvmlDeviceGetHandleByPciBusId_v2)>(get_symbol("nvmlDeviceGetHandleByPciBusId_v2"));
    const auto get_brand = reinterpret_cast<decltype(&nvmlDeviceGetBrand)>(get_symbol("nvmlDeviceGetBrand"));
    const bool nvml_initialized = init && shutdown && get_device && get_brand && init() == NVML_SUCCESS;

    const auto should_use_fp32_acc = [&](const cudaDeviceProp & prop, const int cc) {
        if (cc == GGML_CUDA_CC_GB10) {
            return true;
        }
        if (!fast_bf16_hardware_available(cc) || !nvml_initialized) {
            return false;
        }

        // CUDA and NVML device indices can differ, e.g. with CUDA_VISIBLE_DEVICES.
        char pci_bus_id[32];
        snprintf(pci_bus_id, sizeof(pci_bus_id), "%04x:%02x:%02x.0",
                 (unsigned int) prop.pciDomainID, (unsigned int) prop.pciBusID, (unsigned int) prop.pciDeviceID);
        nvmlDevice_t device = nullptr;
        nvmlBrandType_t brand = NVML_BRAND_UNKNOWN;
        bool prefer_fp32_acc = false;
        if (get_device(pci_bus_id, &device) == NVML_SUCCESS && get_brand(device, &brand) == NVML_SUCCESS) {
            prefer_fp32_acc = brand == NVML_BRAND_QUADRO ||
                              brand == NVML_BRAND_QUADRO_RTX ||
                              brand == NVML_BRAND_NVIDIA_RTX;
        }
        return prefer_fp32_acc;
    };
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

    std::vector<std::pair<int, std::string>> turing_devices_without_mma;
    for (int id = 0; id < info.device_count; ++id) {
        const int physical_id = info.devices[id].physical_device;

        int device_vmm = 0;

#if defined(GGML_USE_VMM)
        CUdevice device;
        CU_CHECK(cuDeviceGet(&device, physical_id));
        CU_CHECK(cuDeviceGetAttribute(&device_vmm, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, device));

        if (device_vmm) {
            CUmemAllocationProp alloc_prop = {};
            alloc_prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
            alloc_prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            alloc_prop.location.id = physical_id;
            CU_CHECK(cuMemGetAllocationGranularity(&info.devices[id].vmm_granularity, &alloc_prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
        }
#endif // defined(GGML_USE_VMM)
        info.devices[id].vmm = !!device_vmm;

        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, physical_id));

        // a virtual device owns only a share of its physical GPU's memory; report that share so the
        // logged per-device VRAM sums to the physical total above.
        GGML_ASSERT(physical_share_count[physical_id] > 0);
        info.devices[id].physical_share_count = physical_share_count[physical_id];
        const size_t device_vram = prop.totalGlobalMem / info.devices[id].physical_share_count;
        const size_t device_vram_mib = device_vram / (1024 * 1024);

        info.default_tensor_split[id] = total_vram;
        total_vram += device_vram;
        info.devices[id].integrated = false; // Temporarily disabled due to issues with corrupted output (e.g. #15034)
        info.devices[id].nsm        = prop.multiProcessorCount;
        info.devices[id].smpb       = prop.sharedMemPerBlock;
        info.devices[id].warp_size  = prop.warpSize;

        int supports_coop_launch = 0;
        CUDA_CHECK(cudaDeviceGetAttribute(&supports_coop_launch, cudaDevAttrCooperativeLaunch, physical_id));
        info.devices[id].supports_cooperative_launch = !!supports_coop_launch;

#if defined(GGML_USE_HIP)
        info.devices[id].smpbo = prop.sharedMemPerBlock;

        info.devices[id].cc = ggml_cuda_parse_id(prop.gcnArchName);
        if ((info.devices[id].cc & 0xff00) == 0x0) {
            GGML_LOG_WARN("invalid architecture ID received for device %d %s: %s  cc %d.%d\n",
                            id, prop.name, prop.gcnArchName, prop.major, prop.minor);

            // Fallback to prop.major and prop.minor
            if (prop.major > 0) {
                info.devices[id].cc = GGML_CUDA_CC_OFFSET_AMD + prop.major * 0x100;
                info.devices[id].cc += prop.minor * 0x10;
            }
        }
        GGML_LOG_INFO("  Device %d: %s, %s (0x%x), VMM: %s, Wave Size: %d, VRAM: %zu MiB\n",
                      id, prop.name, prop.gcnArchName, info.devices[id].cc & 0xffff,
                      device_vmm ? "yes" : "no", prop.warpSize,
                      device_vram_mib);
#elif defined(GGML_USE_MUSA)
        info.devices[id].smpbo = prop.sharedMemPerBlockOptin;
        info.devices[id].cc = GGML_CUDA_CC_OFFSET_MTHREADS + prop.major * 0x100;
        info.devices[id].cc += prop.minor * 0x10;
        GGML_LOG_INFO("  Device %d: %s, compute capability %d.%d, VMM: %s, VRAM: %zu MiB\n",
                      id, prop.name, prop.major, prop.minor, device_vmm ? "yes" : "no",
                      device_vram_mib);
#else
        info.devices[id].smpbo = prop.sharedMemPerBlockOptin;
        info.devices[id].cc = 100*prop.major + 10*prop.minor;
        if (info.devices[id].virtual_index == 0) {
            info.devices[id].prefer_fp32_acc = should_use_fp32_acc(prop, info.devices[id].cc);
        } else {
            info.devices[id].prefer_fp32_acc = info.devices[physical_id].prefer_fp32_acc;
        }
        GGML_LOG_INFO("  Device %d: %s, compute capability %d.%d, VMM: %s, VRAM: %zu MiB\n",
                      id, prop.name, prop.major, prop.minor, device_vmm ? "yes" : "no",
                      device_vram_mib);
        std::string device_name(prop.name);
        if (device_name == "NVIDIA GeForce MX450") {
            turing_devices_without_mma.push_back({ id, device_name });
        } else if (device_name == "NVIDIA GeForce MX550") {
            turing_devices_without_mma.push_back({ id, device_name });
        } else if (device_name.substr(0, 21) == "NVIDIA GeForce GTX 16") {
            turing_devices_without_mma.push_back({ id, device_name });
        }

        // Temporary performance fix:
        // Setting device scheduling strategy for iGPUs with cc121 to "spinning" to avoid delays in cuda synchronize calls.
        // TODO: Check for future drivers the default scheduling strategy and
        // remove this call again when cudaDeviceScheduleSpin is default.
        if (prop.major == 12 && prop.minor == 1) {
            CUDA_CHECK(cudaSetDevice(physical_id));
            CUDA_CHECK(cudaSetDeviceFlags(cudaDeviceScheduleSpin));
        }

#endif  // defined(GGML_USE_HIP)
    }

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    if (nvml_initialized) {
        (void) shutdown();
    }
    library.reset();
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

    if (ggml_cuda_highest_compiled_arch(GGML_CUDA_CC_TURING) >= GGML_CUDA_CC_TURING && !turing_devices_without_mma.empty()) {
        GGML_LOG_INFO("The following devices will have suboptimal performance due to a lack of tensor cores:\n");
        for (size_t device_pos = 0; device_pos < turing_devices_without_mma.size(); device_pos++) {
            GGML_LOG_INFO(
                "  Device %d: %s\n", turing_devices_without_mma[device_pos].first, turing_devices_without_mma[device_pos].second.c_str());
        }
        GGML_LOG_INFO(
            "Consider compiling with CMAKE_CUDA_ARCHITECTURES=61-virtual;80-virtual and DGGML_CUDA_FORCE_MMQ to force the use of the Pascal code for Turing.\n");
    }

    for (int id = 0; id < info.device_count; ++id) {
        info.default_tensor_split[id] /= total_vram;
    }

    // configure logging to stdout
    // CUBLAS_CHECK(cublasLoggerConfigure(1, 1, 0, nullptr));

    if (ggml_cuda_p2p_enabled()) {
        for (int id = 0; id < info.physical_device_count; ++id) {
            CUDA_CHECK(cudaSetDevice(id));
            for (int id_other = 0; id_other < info.physical_device_count; ++id_other) {
                if (id == id_other) {
                    continue;
                }
                int can_access_peer;
                CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access_peer, id, id_other));
                if (can_access_peer) {
                    CUDA_CHECK(cudaDeviceEnablePeerAccess(id_other, 0));
                }
            }
        }
    }

    return info;
}

const ggml_cuda_device_info & ggml_cuda_info() {
    static ggml_cuda_device_info info = ggml_cuda_init();
    return info;
}

// #define DEBUG_CUDA_MALLOC

// buffer pool for cuda (legacy)
struct ggml_cuda_pool_leg : public ggml_cuda_pool {
    static const int MAX_BUFFERS = 256;

    ggml_backend_cuda_context & ctx;
    int device;
    struct ggml_cuda_buffer {
        void * ptr = nullptr;
        size_t size = 0;
    };

    ggml_cuda_buffer buffer_pool[MAX_BUFFERS] = {};
    size_t pool_size = 0;

    ggml_cuda_pool_leg(ggml_backend_cuda_context & ctx, int device) :
        ctx(ctx),
        device(device) {
    }

    ~ggml_cuda_pool_leg() {
        clear_pool();
        GGML_ASSERT(pool_size == 0);
    }

    void clear_pool() {
        ggml_cuda_set_device(device);
        for (int i = 0; i < MAX_BUFFERS; ++i) {
            ggml_cuda_buffer & b = buffer_pool[i];
            if (b.ptr != nullptr) {
                CUDA_CHECK(cudaFree(b.ptr));
                pool_size -= b.size;
                b.ptr  = nullptr;
                b.size = 0;
            }
        }
    }

    // captured CUDA graphs can still use freed buffers, so they must be captured again
    void invalidate_cuda_graphs() {
#ifdef USE_CUDA_GRAPH
        int n_captured = 0;
        for (const auto & [key, graph] : ctx.cuda_graphs) {
            n_captured += graph->instances.size();
        }
        if (n_captured > 0) {
            GGML_LOG_DEBUG(GGML_CUDA_NAME " pool[%d]: memory freed, %d captured CUDA graphs must be captured again\n", device, n_captured);
        }
        ctx.cuda_graphs.clear();
#endif // USE_CUDA_GRAPH
    }

    void * alloc(size_t size, size_t * actual_size) override {
#ifdef DEBUG_CUDA_MALLOC
        int nnz = 0;
        size_t max_size = 0;
#endif
        size_t best_diff = 1ull << 36;
        int ibest = -1;
        for (int i = 0; i < MAX_BUFFERS; ++i) {
            ggml_cuda_buffer& b = buffer_pool[i];
            if (b.ptr != nullptr) {
#ifdef DEBUG_CUDA_MALLOC
                ++nnz;
                if (b.size > max_size) max_size = b.size;
#endif
                if (b.size >= size) {
                    size_t diff = b.size - size;
                    if (diff < best_diff) {
                        best_diff = diff;
                        ibest = i;
                        if (!best_diff) {
                            void * ptr = b.ptr;
                            *actual_size = b.size;
                            b.ptr = nullptr;
                            b.size = 0;
                            return ptr;
                        }
                    }
                }
            }
        }
        if (ibest >= 0) {
            ggml_cuda_buffer& b = buffer_pool[ibest];
            void * ptr = b.ptr;
            *actual_size = b.size;
            b.ptr = nullptr;
            b.size = 0;
            return ptr;
        }
        void * ptr;
        size_t look_ahead_size = (size_t) (1.05 * size);
        look_ahead_size = 256 * ((look_ahead_size + 255)/256);
        ggml_cuda_set_device(device);
        cudaError_t err = ggml_cuda_device_malloc(&ptr, look_ahead_size, device);
        if (err == cudaErrorMemoryAllocation) {
            (void)cudaGetLastError();
            const size_t cached_bytes = pool_size;
            GGML_LOG_DEBUG(GGML_CUDA_NAME " pool[%d]: alloc of %.2f MiB failed, flushing %.2f MiB of cached buffers and retrying\n",
                           device, look_ahead_size/1024.0/1024.0, cached_bytes/1024.0/1024.0);
            CUDA_CHECK(cudaDeviceSynchronize());
            clear_pool();
            invalidate_cuda_graphs();
            err = ggml_cuda_device_malloc(&ptr, look_ahead_size, device);
            if (err == cudaSuccess) {
                GGML_LOG_DEBUG(GGML_CUDA_NAME " pool[%d]: retry succeeded\n", device);
            }
        }
        CUDA_CHECK(err);
        *actual_size = look_ahead_size;
        pool_size += look_ahead_size;
        ggml_backend_ink2_vidmem_log("pool alloc", look_ahead_size);
        // local diagnostic: GGML_CUDA_POOL_LOG=1 prints every new device allocation of the pool
        {
            static const bool pool_log = getenv("GGML_CUDA_POOL_LOG") != nullptr;
            if (pool_log) {
                int nnz_log = 0;
                for (int i = 0; i < MAX_BUFFERS; ++i) {
                    nnz_log += buffer_pool[i].ptr != nullptr;
                }
                fprintf(stderr, "cuda pool[%d] %p: new buffer %.2f MiB, pool %.2f MiB, %d cached buffers\n", device, (void *) this,
                    look_ahead_size/1024.0/1024.0, pool_size/1024.0/1024.0, nnz_log);
            }
        }
#ifdef DEBUG_CUDA_MALLOC
        GGML_LOG_INFO("%s[%d]: %d buffers, max_size = %u MB, pool_size = %u MB, requested %u MB\n", __func__, device, nnz,
                           (uint32_t)(max_size / 1024 / 1024), (uint32_t)(pool_size / 1024 / 1024), (uint32_t)(size / 1024 / 1024));
#endif
        return ptr;
    }

    void free(void * ptr, size_t size) override {
        for (int i = 0; i < MAX_BUFFERS; ++i) {
            ggml_cuda_buffer& b = buffer_pool[i];
            if (b.ptr == nullptr) {
                b.ptr = ptr;
                b.size = size;
                return;
            }
        }
        GGML_LOG_DEBUG(GGML_CUDA_NAME " buffer pool full, increase MAX_CUDA_BUFFERS\n");
        ggml_cuda_set_device(device);
        CUDA_CHECK(cudaFree(ptr));
        pool_size -= size;
        invalidate_cuda_graphs();
    }
};

// pool with virtual memory
#if defined(GGML_USE_VMM)
struct ggml_cuda_pool_vmm : public ggml_cuda_pool {
    static const size_t CUDA_POOL_VMM_MAX_SIZE = 1ull << 35; // 32 GB

    int device;
    int physical_device;
    CUdeviceptr pool_addr = 0;
    size_t pool_used = 0;
    size_t pool_size = 0;
    size_t max_size = CUDA_POOL_VMM_MAX_SIZE;
    size_t granularity;
#if defined(GGML_USE_HIP)
    std::vector<std::pair<CUdeviceptr, size_t>> mappings;
#endif

    explicit ggml_cuda_pool_vmm(int device) :
        device(device),
        physical_device(ggml_cuda_get_physical_device(device)),
        granularity(ggml_cuda_info().devices[device].vmm_granularity) {
    }

    ~ggml_cuda_pool_vmm() {
        if (pool_addr != 0) {
#if defined(GGML_USE_HIP)
            // Workaround for https://github.com/ROCm/ROCR-Runtime/issues/285
            for (std::pair<CUdeviceptr, size_t> & mapping : mappings) {
                CU_CHECK(cuMemUnmap(mapping.first, mapping.second));
            }
#else
            CU_CHECK(cuMemUnmap(pool_addr, pool_size));
#endif
            CU_CHECK(cuMemAddressFree(pool_addr, max_size));
        }
    }

    void * alloc(size_t size, size_t * actual_size) override {
        // round up the allocation size to the alignment to ensure that all allocations are aligned for all data types
        const size_t alignment = 128;
        size = alignment * ((size + alignment - 1) / alignment);

        size_t avail = pool_size - pool_used;

        if (size > avail) {
            // round up to the next multiple of the granularity
            size_t reserve_size = size - avail;
            reserve_size = granularity * ((reserve_size + granularity - 1) / granularity);

            // reserve virtual address space (if not already reserved)
            if (pool_addr == 0) {
                CUresult err = cuMemAddressReserve(&pool_addr, max_size, 0, 0, 0);
                while (err == CUDA_ERROR_OUT_OF_MEMORY && (max_size >> 1) >= reserve_size && (max_size >> 1) >= granularity) {
                    max_size >>= 1;
                    err = cuMemAddressReserve(&pool_addr, max_size, 0, 0, 0);
                }
                CU_CHECK(err);
                if (max_size < CUDA_POOL_VMM_MAX_SIZE) {
                    GGML_LOG_WARN(GGML_CUDA_NAME " pool[%d]: reserved %.2f GiB of virtual memory instead of %.2f GiB\n",
                                  device, max_size / (1024.0 * 1024.0 * 1024.0), CUDA_POOL_VMM_MAX_SIZE / (1024.0 * 1024.0 * 1024.0));
                }
            }

            GGML_ASSERT(pool_size + reserve_size <= max_size);

            // allocate more physical memory
            CUmemAllocationProp prop = {};
            prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
            prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            prop.location.id = physical_device;
            CUmemGenericAllocationHandle handle;
            CU_CHECK(cuMemCreate(&handle, reserve_size, &prop, 0));

            // map at the end of the pool
            CUdeviceptr start_ptr = (CUdeviceptr)((char *)(pool_addr) + pool_size);
            CU_CHECK(cuMemMap(start_ptr, reserve_size, 0, handle, 0));
#if defined(GGML_USE_HIP)
            mappings.push_back({start_ptr, reserve_size});
#endif

            // the memory allocation handle is no longer needed after mapping
            CU_CHECK(cuMemRelease(handle));

            // VMM Bug fix for P2P access if GGML_CUDA_P2P is set, or if NCCL build
            bool use_peer_access = ggml_cuda_p2p_enabled();
#if defined(GGML_USE_NCCL)
            use_peer_access = true;
#endif // defined(GGML_USE_NCCL)

            if (use_peer_access) {
                // NCCL implicitly enables peer access (cudaDeviceEnablePeerAccess), and
                // GGML_CUDA_P2P enables it explicitly. Unlike cudaMalloc buffers, VMM
                // allocations do not become peer-accessible from that alone, so access
                // must be granted explicitly here. With virtual devices, grant access
                // on the backing *physical* devices (deduplicated, since several
                // virtual devices can map to the same physical GPU).
                std::vector<CUmemAccessDesc> access_descs;
                bool physical_seen[GGML_CUDA_MAX_DEVICES] = {};
                const int device_count = ggml_cuda_info().device_count;
                for (int id = 0; id < device_count; ++id) {
                    const int id_physical = ggml_cuda_get_physical_device(id);
                    if (id_physical != physical_device) {
                        int can_access_peer = 0;
                        CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access_peer, id_physical, physical_device));
                        if (!can_access_peer) {
                            continue;
                        }
                    }
                    if (physical_seen[id_physical]) {
                        continue;
                    }
                    physical_seen[id_physical] = true;
                    CUmemAccessDesc access = {};
                    access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
                    access.location.id = id_physical;
                    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
                    access_descs.push_back(access);
                }
                CU_CHECK(cuMemSetAccess(start_ptr, reserve_size, access_descs.data(), access_descs.size()));
            } else {
                // set access for non P2P
                CUmemAccessDesc access = {};
                access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
                access.location.id = physical_device;
                access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
                CU_CHECK(cuMemSetAccess(start_ptr, reserve_size, &access, 1));
            }

            // add to the pool
            pool_size += reserve_size;

            //printf("cuda pool[%d]: size increased to %llu MB (reserved %llu MB)\n",
            //       device, (unsigned long long) (pool_size/1024/1024),
            //       (unsigned long long) (reserve_size/1024/1024));
        }

        GGML_ASSERT(pool_addr != 0);

        void * ptr = (void *) ((CUdeviceptr)((char *)(pool_addr) + pool_used));
        *actual_size = size;
        pool_used += size;

#ifdef DEBUG_CUDA_MALLOC
        printf("cuda pool[%d]: allocated %llu bytes at %llx\n", device, (unsigned long long) size, ptr);
#endif

        return ptr;
    }

    void free(void * ptr, size_t size) override {
#ifdef DEBUG_CUDA_MALLOC
        printf("cuda pool[%d]: freed %llu bytes at %llx\n", device, (unsigned long long) size, ptr);
#endif

        pool_used -= size;

        // all deallocations must be in reverse order of the allocations
        GGML_ASSERT(ptr == (void *) ((char *)(pool_addr) + pool_used));
    }
};
#endif // defined(GGML_USE_VMM)

// Debug pool for NVIDIA compute-sanitizer, e.g.
// `GGML_CUDA_POOL_DEBUG=1 compute-sanitizer --tool memcheck ./build/bin/test-backend-ops -b CUDA0`
// Allocations are exact-size and freed on release,
// allowing memcheck to catch out-of-bounds and use-after-free.
struct ggml_cuda_pool_debug : public ggml_cuda_pool {
    int device;
    size_t pool_size = 0;       // leak check (error if not zero at destruction)

    explicit ggml_cuda_pool_debug(int device) :
        device(device) {
    }

    ~ggml_cuda_pool_debug() {
        GGML_ASSERT(pool_size == 0);
    }

    void * alloc(size_t size, size_t * actual_size) override {
        void * ptr;
        CUDA_CHECK(ggml_cuda_device_malloc(&ptr, size, device));
        *actual_size = size;
        pool_size += size;
        return ptr;
    }

    void free(void * ptr, size_t size) override {
        ggml_cuda_set_device(device);
        CUDA_CHECK(cudaFree(ptr));
        pool_size -= size;
    }
};

std::unique_ptr<ggml_cuda_pool> ggml_backend_cuda_context::new_pool_for_device(ggml_backend_cuda_context & ctx,
                                                                               int                         device,
                                                                               [[maybe_unused]] int        stream_no) {
    if (ggml_cuda_pool_debug_enabled()) {
        return std::unique_ptr<ggml_cuda_pool>(new ggml_cuda_pool_debug(device));
    }
#if defined(GGML_USE_VMM)
    if (ggml_cuda_info().devices[device].vmm) {
        return std::unique_ptr<ggml_cuda_pool>(new ggml_cuda_pool_vmm(device));
    }
#endif // defined(GGML_USE_VMM)
    return std::unique_ptr<ggml_cuda_pool>(new ggml_cuda_pool_leg(ctx, device));
}

ggml_backend_cuda_context::ggml_backend_cuda_context(int device) :
    device(device),
    name(GGML_CUDA_NAME + std::to_string(device)),
    concurrent_stream_context(std::make_unique<ggml_cuda_stream_context>()) {
}

#ifdef USE_CUDA_GRAPH
ggml_cuda_graph * ggml_backend_cuda_context::cuda_graph(uint64_t graph_key, int64_t time_now) {
    // Give all cached graphs 10s to be reused after an idle period, without updating each graph.
    if (!cuda_graphs.empty() && time_now - last_graph_use_time >= 10'000'000) {
        next_graph_eviction_time = time_now + 10'000'000;
    }
    last_graph_use_time = time_now;

    // Sweep every 5s during active use, evicting graphs unused for at least 10s.
    if (time_now >= next_graph_eviction_time) {
        next_graph_eviction_time = time_now + 5'000'000;
        for (auto it = cuda_graphs.begin(); it != cuda_graphs.end(); ) {
            if (it->first != graph_key && time_now - it->second->last_used_time >= 10'000'000) {
                it = cuda_graphs.erase(it);
            } else {
                ++it;
            }
        }
    }

    auto it = cuda_graphs.find(graph_key);
    if (it == cuda_graphs.end()) {
        it = cuda_graphs.emplace(graph_key, std::make_unique<ggml_cuda_graph>()).first;
    }
    it->second->last_used_time = time_now;
    return it->second.get();
}

bool ggml_backend_cuda_context::any_cuda_graph_enabled() const {
    for (const auto & [key, graph] : cuda_graphs) {
        if (graph && graph->is_enabled()) {
            return true;
        }
    }
    return false;
}

bool ggml_backend_cuda_context::any_cuda_graph_has_instance() const {
    for (const auto & [key, graph] : cuda_graphs) {
        if (graph && !graph->instances.empty()) {
            return true;
        }
    }
    return false;
}
#endif // USE_CUDA_GRAPH

#if defined(GGML_CUDA_USE_PDL)
bool ggml_cuda_kernel_can_use_pdl(const void * kernel) {
    static const bool env_pdl_enabled = []() {
        const char * env = getenv("GGML_CUDA_PDL");
        return env == nullptr || std::atoi(env) != 0;
    }();
    if (!env_pdl_enabled) {
        return false;
    }

    const int device = ggml_cuda_get_device();

    struct cache_key {
        int          device;
        const void * kernel;

        bool operator==(const cache_key & other) const { return device == other.device && kernel == other.kernel; }
    };

    struct cache_key_hash {
        // MurmurHash3 mixing function for better hash distribution (vs. just std::hash which in some implementations simply returns the identity)
        static size_t hash_mix(size_t x) {
            std::uint64_t       y = x;
            const std::uint64_t m = 0xe9846af9b1a615d;

            y ^= y >> 32;
            y *= m;
            y ^= y >> 32;
            y *= m;
            y ^= y >> 28;

            return static_cast<size_t>(y);
        }

        size_t operator()(const cache_key & key) const {
            // Use a nonzero seed to avoid mapping all-zero keys to zero
            size_t h = 42;
            h        = hash_mix(h + key.device);
            h        = hash_mix(h + reinterpret_cast<size_t>(key.kernel));
            return h;
        }
    };

    static std::mutex                                          cache_mutex;
    static std::unordered_map<cache_key, bool, cache_key_hash> cache;

    const cache_key             key = { device, kernel };
    std::lock_guard<std::mutex> lock(cache_mutex);
    const auto                  it = cache.find(key);
    if (it != cache.end()) {
        return it->second;
    }

    cudaFuncAttributes attr = {};
    CUDA_CHECK(cudaFuncGetAttributes(&attr, kernel));

    // PDL device-side primitives are emitted only for PTX versions >= 90.
    // We have to guard on a loaded kernel's PTX version so a kernel forward-JIT'ed
    // from pre-Hopper PTX to a Hopper-or-newer GPU does not opt into PDL.
    const bool can_use_pdl = attr.ptxVersion >= 90;
    cache.emplace(key, can_use_pdl);
    return can_use_pdl;
}
#endif // defined(GGML_CUDA_USE_PDL)

// destroying a cuBLAS handle while a graph is being captured in a different thread can result in a CUDA error
// this lock is used to ensure that no cuBLAS handle is destroyed while a graph is being captured

static std::mutex ggml_cuda_lock;
static std::condition_variable ggml_cuda_lock_cv;
static std::atomic<int> ggml_cuda_lock_counter;

ggml_backend_cuda_context::~ggml_backend_cuda_context() {
    std::unique_lock<std::mutex> lock(ggml_cuda_lock);
    ggml_cuda_lock_cv.wait(lock, []{ return ggml_cuda_lock_counter.load(std::memory_order_relaxed) == 0; });

    if (copy_event != nullptr) {
        CUDA_CHECK(cudaEventDestroy(copy_event));
    }
    for (int i = 0; i < GGML_CUDA_MAX_DEVICES; ++i) {
        for (int j = 0; j < GGML_CUDA_MAX_STREAMS; ++j) {
            if (streams[i][j] != nullptr) {
                CUDA_CHECK(cudaStreamDestroy(streams[i][j]));
            }
            if (cublas_handles[i][j] != nullptr) {
                CUBLAS_CHECK(cublasDestroy(cublas_handles[i][j]));
            }
            if (cublas_workspaces[i][j] != nullptr) {
                CUDA_CHECK(cudaFree(cublas_workspaces[i][j]));
            }
        }
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) && CUDART_VERSION >= 11080
        if (cublaslt_handles[i] != nullptr) {
            CUBLAS_CHECK(cublasLtDestroy(cublaslt_handles[i]));
        }
#endif
    }
}


// cuda buffer

struct ggml_backend_cuda_buffer_context {
    int device;
    void * dev_ptr = nullptr;
    std::string name;

    ggml_backend_cuda_buffer_context(int device, void * dev_ptr) :
        device(device), dev_ptr(dev_ptr),
        name(GGML_CUDA_NAME + std::to_string(device)) {
    }

    ~ggml_backend_cuda_buffer_context() {
        CUDA_CHECK(cudaFree(dev_ptr));
        ggml_backend_ink2_vidmem_log("device buffer free", 0);
    }
};

static void ggml_backend_cuda_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;
    delete ctx;
}

static bool ggml_backend_buffer_is_cuda(ggml_backend_buffer_t buffer) {
    return buffer->iface.free_buffer == ggml_backend_cuda_buffer_free_buffer;
}

static void * ggml_backend_cuda_buffer_get_base(ggml_backend_buffer_t buffer) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;
    return ctx->dev_ptr;
}

static enum ggml_status ggml_backend_cuda_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;

    if (tensor->view_src != NULL) {
        assert(tensor->view_src->buffer->buft == buffer->buft);
        return GGML_STATUS_SUCCESS;
    }

    if (ggml_is_quantized(tensor->type) && tensor->view_src == nullptr && ggml_backend_buffer_get_usage(buffer) != GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
        // initialize padding to 0 to avoid possible NaN values
        const size_t original_size = ggml_nbytes(tensor);
        const size_t padded_size = ggml_backend_buft_get_alloc_size(buffer->buft, tensor);

        if (padded_size > original_size) {
            ggml_cuda_set_device(ctx->device);
            CUDA_CHECK(cudaMemsetAsync((char *)tensor->data + original_size, 0, padded_size - original_size, cudaStreamPerThread));
            CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
        }
    }
    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_cuda_buffer_memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *) buffer->context;

    ggml_cuda_set_device(ctx->device);
    CUDA_CHECK(cudaMemsetAsync((char *) tensor->data + offset, value, size, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

static void ggml_backend_cuda_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *) buffer->context;

    ggml_cuda_set_device(ctx->device);
    ggml_cuda_ink2_tl(nullptr, "bset", (long long) size);
    CUDA_CHECK(cudaMemcpyAsync((char *) tensor->data + offset, data, size, cudaMemcpyHostToDevice, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
    ggml_cuda_ink2_tl(nullptr, "bset_end", (long long) size);
    if (size >= (16u << 20)) {
        ggml_backend_ink2_vidmem_log("after set_tensor (no alloc)", 0);
    }
}

static void ggml_backend_cuda_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *) buffer->context;

    ggml_cuda_set_device(ctx->device);
    ggml_cuda_ink2_tl(nullptr, "bget", (long long) size);
    CUDA_CHECK(cudaMemcpyAsync(data, (const char *) tensor->data + offset, size, cudaMemcpyDeviceToHost, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
    ggml_cuda_ink2_tl(nullptr, "bget_end", (long long) size);
    if (size >= (16u << 20)) {
        ggml_backend_ink2_vidmem_log("after get_tensor (no alloc)", 0);
    }
}

static void ggml_backend_cuda_buffer_set_tensor_2d(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, const void * data,
        size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *) buffer->context;

    ggml_cuda_set_device(ctx->device);
    CUDA_CHECK(cudaMemcpy2DAsync(
        (char *) tensor->data + offset, stride_tensor, data, stride_data, size, n_copies, cudaMemcpyHostToDevice, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

static void ggml_backend_cuda_buffer_get_tensor_2d(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor, void * data,
        size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;

    ggml_cuda_set_device(ctx->device);
    CUDA_CHECK(cudaMemcpy2DAsync(
        data, stride_data, (const char *) tensor->data + offset, stride_tensor, size, n_copies, cudaMemcpyDeviceToHost, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

static void ggml_backend_cuda_buffer_set_tensor_n(ggml_backend_buffer_t buffer, struct ggml_tensor * const * tensors, const void * const * data, const size_t * offsets, const size_t * sizes, size_t n_tensors) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;

    ggml_cuda_set_device(ctx->device);

    if (!ggml_cuda_pinned_memory_disabled() &&
            ggml_cuda_set_tensor_n(ctx->device, tensors, data, offsets, sizes, n_tensors)) {
        return;
    }

    for (size_t i = 0; i < n_tensors; i++) {
        if (sizes[i] == 0) {
            continue;
        }
        CUDA_CHECK(cudaMemcpyAsync(
            (char *) tensors[i]->data + offsets[i],
            data[i],
            sizes[i],
            cudaMemcpyHostToDevice,
            cudaStreamPerThread));
    }

    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

static void ggml_backend_cuda_buffer_get_tensor_n(ggml_backend_buffer_t buffer, const struct ggml_tensor * const * tensors, void * const * data, const size_t * offsets, const size_t * sizes, size_t n_tensors) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;

    ggml_cuda_set_device(ctx->device);

    if (!ggml_cuda_pinned_memory_disabled() &&
            ggml_cuda_get_tensor_n(ctx->device, tensors, data, offsets, sizes, n_tensors)) {
        return;
    }

    for (size_t i = 0; i < n_tensors; i++) {
        if (sizes[i] == 0) {
            continue;
        }
        CUDA_CHECK(cudaMemcpyAsync(
            data[i],
            (const char *) tensors[i]->data + offsets[i],
            sizes[i],
            cudaMemcpyDeviceToHost,
            cudaStreamPerThread));
    }

    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

static bool ggml_backend_cuda_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * src, ggml_tensor * dst) {
    if (ggml_backend_buffer_is_cuda(src->buffer)) {
        ggml_backend_cuda_buffer_context * src_ctx = (ggml_backend_cuda_buffer_context *)src->buffer->context;
        ggml_backend_cuda_buffer_context * dst_ctx = (ggml_backend_cuda_buffer_context *)dst->buffer->context;
        // compare the backing physical devices: distinct virtual devices may share one physical GPU,
        // in which case a same-device copy (not a peer copy) is required
        const int src_physical = ggml_cuda_get_physical_device(src_ctx->device);
        const int dst_physical = ggml_cuda_get_physical_device(dst_ctx->device);
        if (src_physical == dst_physical) {
            CUDA_CHECK(cudaMemcpyAsync(dst->data, src->data, ggml_nbytes(src), cudaMemcpyDeviceToDevice, cudaStreamPerThread));
        } else {
#ifdef GGML_CUDA_NO_PEER_COPY
            return false;
#else
            CUDA_CHECK(cudaMemcpyPeerAsync(dst->data, dst_physical, src->data, src_physical, ggml_nbytes(src), cudaStreamPerThread));
#endif
        }
        CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
        return true;
    }
    return false;

    GGML_UNUSED(buffer);
}

static bool ggml_backend_cuda_buffer_cpy_tensor_n(ggml_backend_buffer_t buffer, const struct ggml_tensor * const * srcs, struct ggml_tensor * const * dsts, size_t n_tensors) {
    ggml_backend_cuda_buffer_context * dst_ctx = (ggml_backend_cuda_buffer_context *)buffer->context;

    for (size_t i = 0; i < n_tensors; i++) {
        ggml_backend_buffer_t src_buf = srcs[i]->view_src ? srcs[i]->view_src->buffer : srcs[i]->buffer;
        if (!ggml_backend_buffer_is_cuda(src_buf)) {
            return false;
        }
    }

#ifdef GGML_CUDA_NO_PEER_COPY
    for (size_t i = 0; i < n_tensors; i++) {
        ggml_backend_buffer_t src_buf = srcs[i]->view_src ? srcs[i]->view_src->buffer : srcs[i]->buffer;
        ggml_backend_buffer_t dst_buf = dsts[i]->view_src ? dsts[i]->view_src->buffer : dsts[i]->buffer;
        ggml_backend_cuda_buffer_context * src_ctx = (ggml_backend_cuda_buffer_context *)src_buf->context;
        ggml_backend_cuda_buffer_context * dst_ctx_i = (ggml_backend_cuda_buffer_context *)dst_buf->context;
        const int src_physical = ggml_cuda_get_physical_device(src_ctx->device);
        const int dst_physical = ggml_cuda_get_physical_device(dst_ctx_i->device);
        if (src_physical != dst_physical) {
            return false;
        }
    }
#endif

    ggml_cuda_set_device(dst_ctx->device);

    for (size_t i = 0; i < n_tensors; i++) {
        ggml_backend_buffer_t src_buf = srcs[i]->view_src ? srcs[i]->view_src->buffer : srcs[i]->buffer;
        ggml_backend_buffer_t dst_buf = dsts[i]->view_src ? dsts[i]->view_src->buffer : dsts[i]->buffer;
        ggml_backend_cuda_buffer_context * src_ctx = (ggml_backend_cuda_buffer_context *)src_buf->context;
        ggml_backend_cuda_buffer_context * dst_ctx_i = (ggml_backend_cuda_buffer_context *)dst_buf->context;
        const int src_physical = ggml_cuda_get_physical_device(src_ctx->device);
        const int dst_physical = ggml_cuda_get_physical_device(dst_ctx_i->device);
        if (src_physical == dst_physical) {
            CUDA_CHECK(cudaMemcpyAsync(dsts[i]->data, srcs[i]->data, ggml_nbytes(srcs[i]), cudaMemcpyDeviceToDevice, cudaStreamPerThread));
        } else {
#ifdef GGML_CUDA_NO_PEER_COPY
            GGML_ABORT("peer copy not supported");
#else
            CUDA_CHECK(cudaMemcpyPeerAsync(dsts[i]->data, dst_physical, srcs[i]->data, src_physical, ggml_nbytes(srcs[i]), cudaStreamPerThread));
#endif
        }
    }

    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
    return true;
}

static void ggml_backend_cuda_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;

    ggml_cuda_set_device(ctx->device);
    CUDA_CHECK(cudaMemsetAsync(ctx->dev_ptr, value, buffer->size, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

static const ggml_backend_buffer_i ggml_backend_cuda_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_cuda_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_cuda_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_cuda_buffer_init_tensor,
    /* .memset_tensor   = */ ggml_backend_cuda_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_cuda_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_cuda_buffer_get_tensor,
    /* .set_tensor_2d   = */ ggml_backend_cuda_buffer_set_tensor_2d,
    /* .get_tensor_2d   = */ ggml_backend_cuda_buffer_get_tensor_2d,
    /* .cpy_tensor      = */ ggml_backend_cuda_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_cuda_buffer_clear,
    /* .reset           = */ NULL,
    /* .set_tensor_n    = */ ggml_backend_cuda_buffer_set_tensor_n,
    /* .get_tensor_n    = */ ggml_backend_cuda_buffer_get_tensor_n,
    /* .cpy_tensor_n    = */ ggml_backend_cuda_buffer_cpy_tensor_n,
};

// cuda buffer type
struct ggml_backend_cuda_buffer_type_context {
    int device;
    std::string name;
};

static const char * ggml_backend_cuda_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_cuda_buffer_type_context * ctx = (ggml_backend_cuda_buffer_type_context *)buft->context;

    return ctx->name.c_str();
}

static bool ggml_backend_buft_is_cuda(ggml_backend_buffer_type_t buft) {
    return buft->iface.get_name == ggml_backend_cuda_buffer_type_get_name;
}

static ggml_backend_buffer_t ggml_backend_cuda_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    ggml_backend_cuda_buffer_type_context * buft_ctx = (ggml_backend_cuda_buffer_type_context *)buft->context;

    ggml_cuda_set_device(buft_ctx->device);

    void * dev_ptr;
    cudaError_t err = ggml_cuda_device_malloc(&dev_ptr, size, buft_ctx->device);
    if (err != cudaSuccess) {
        // clear the error
        (void)cudaGetLastError();
        GGML_LOG_ERROR("%s: allocating %.2f MiB on device %d: cudaMalloc failed: %s\n", __func__, size / 1024.0 / 1024.0, buft_ctx->device, cudaGetErrorString(err));
        return nullptr;
    }

    ggml_backend_cuda_buffer_context * ctx = new ggml_backend_cuda_buffer_context(buft_ctx->device, dev_ptr);
    ggml_backend_ink2_vidmem_log("device buffer alloc", size);

    return ggml_backend_buffer_init(buft, ggml_backend_cuda_buffer_interface, ctx, size);
}

static size_t ggml_backend_cuda_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return 128;

    GGML_UNUSED(buft);
}

static size_t ggml_backend_cuda_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    ggml_backend_cuda_buffer_type_context * buft_ctx = (ggml_backend_cuda_buffer_type_context *) buft->context;

    size_t size = (tensor->op == GGML_OP_FLASH_ATTN_EXT || tensor->op == GGML_OP_FLASH_ATTN_EXT_BANDED)
        ? ggml_cuda_flash_attn_ext_get_alloc_size(buft_ctx->device, tensor)
        : ggml_nbytes(tensor);
    int64_t ne0 = tensor->ne[0];

    // [TAG_ALLOC_SIZE_EXPAND]
    if (ggml_is_quantized(tensor->type)) {
        if (ne0 % MATRIX_ROW_PADDING != 0) {
            GGML_ASSERT(tensor->nb[0] == ggml_element_size(tensor));
            size += ggml_row_size(tensor->type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING);
        }
    }

    return size;
}

static const ggml_backend_buffer_type_i ggml_backend_cuda_buffer_type_interface = {
    /* .get_name            = */ ggml_backend_cuda_buffer_type_get_name,
    /* .alloc_buffer        = */ ggml_backend_cuda_buffer_type_alloc_buffer,
    /* .alloc_buffer_n      = */ NULL,
    /* .get_alignment       = */ ggml_backend_cuda_buffer_type_get_alignment,
    /* .get_max_size        = */ NULL, // defaults to SIZE_MAX
    /* .get_alloc_size      = */ ggml_backend_cuda_buffer_type_get_alloc_size,
    /* .get_alloc_size_n    = */ NULL,
    /* .is_host             = */ NULL,
};

ggml_backend_buffer_type_t ggml_backend_cuda_buffer_type(int device) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    if (device >= ggml_backend_cuda_get_device_count()) {
        return nullptr;
    }

    static ggml_backend_buffer_type ggml_backend_cuda_buffer_types[GGML_CUDA_MAX_DEVICES];

    static bool ggml_backend_cuda_buffer_type_initialized = false;

    if (!ggml_backend_cuda_buffer_type_initialized) {
        for (int i = 0; i < ggml_backend_cuda_get_device_count(); i++) {
            ggml_backend_cuda_buffer_types[i] = {
                /* .iface    = */ ggml_backend_cuda_buffer_type_interface,
                /* .device   = */ ggml_backend_reg_dev_get(ggml_backend_cuda_reg(), i),
                /* .context  = */ new ggml_backend_cuda_buffer_type_context{i, GGML_CUDA_NAME + std::to_string(i)},
            };
        }
        ggml_backend_cuda_buffer_type_initialized = true;
    }

    return &ggml_backend_cuda_buffer_types[device];
}

// Communication context for multi-GPU AllReduce during tensor parallelism.
//
// Created once per meta backend instance.  Resources for the selected mode
// (NCCL communicators or the internal AllReduce pipeline) are initialised
// eagerly during comm_init so any init failure surfaces at startup rather
// than mid-run.
struct ggml_backend_cuda_comm_context {
    using try_allreduce_fn = bool(*)(ggml_backend_cuda_comm_context *, struct ggml_tensor **);

    std::vector<ggml_backend_t> backends;
    std::vector<int>            dev_ids;

    // Set by the init chain (comm_init_{nccl, internal, none}) to one of
    // try_allreduce_{nccl, internal, butterfly}.  nccl needs `comms`,
    // internal needs `ar_pipeline`, butterfly needs nothing.  Per-call
    // failures return false; the meta backend's generic implementation then
    // handles that call.
    try_allreduce_fn            try_allreduce = nullptr;

    ggml_cuda_ar_pipeline *     ar_pipeline = nullptr;

    // Number of ranks in the communicator when each rank is a separate process. Zero unless
    // comm_init_rank created this context.
    int                         world_size = 0;

    // Number of ranks taking part in the reduction. For a communicator from comm_init_rank this is
    // world_size. Otherwise it is the number of backends this process holds. The NCCL reduction
    // uses it to choose the tensor size below which it reduces in F32 rather than converting to
    // BF16 first.
    size_t n_ranks() const {
        return world_size > 0 ? (size_t) world_size : backends.size();
    }

#ifdef GGML_USE_NCCL
    std::vector<ncclComm_t>     comms;
#endif // GGML_USE_NCCL

    ~ggml_backend_cuda_comm_context() {
#ifdef GGML_USE_NCCL
        for (ncclComm_t comm : comms) {
            NCCL_CHECK(ncclCommDestroy(comm));
        }
#endif // GGML_USE_NCCL
        ggml_cuda_ar_pipeline_free(ar_pipeline);
    }
};

#ifdef GGML_USE_NCCL
// AllReduce via NCCL. Reduces as FP32 for small tensors and BF16 for large
// tensors (bandwidth-bound), then converts back to FP32.
static bool ggml_backend_cuda_comm_allreduce_nccl(
        ggml_backend_cuda_comm_context * comm_ctx, struct ggml_tensor ** tensors) {
    const int64_t ne = ggml_nelements(tensors[0]);
    // FIXME the input of llm_graph_context::build_in_out_ids can produce a tensor with 0 elements if n_outputs == 0
    // This then causes a crash in this function
    if (ne == 0) {
        return true;
    }

    const size_t n_backends = comm_ctx->backends.size();
    const size_t n_ranks    = comm_ctx->n_ranks();

    for (size_t i = 0; i < n_backends; ++i) {
        GGML_ASSERT(tensors[i] != nullptr);
        GGML_ASSERT(ggml_nelements(tensors[i]) == ne);
        GGML_ASSERT(ggml_is_contiguously_allocated(tensors[i]));
    }

    // For small tensors, simply reduce them as FP32.
    // The following heuristic for how "small" a tensor should be is based on RTX 4090s connected via 16x PCIe 4.0.
    if ((n_ranks <= 2 && ne < 32768) || (n_ranks == 3 && ne < 131072) || (n_ranks >= 4 && ne < 262144)) {
        for (size_t i = 0; i < n_backends; ++i) {
            if ((tensors[i]->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
                ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) comm_ctx->backends[i]->context;
                ggml_cuda_set_device(cuda_ctx->device);
                CUDA_CHECK(cudaMemsetAsync(tensors[i]->data, 0, ggml_nbytes(tensors[i]), cuda_ctx->stream()));
            }
        }
        NCCL_CHECK(ncclGroupStart());
        for (size_t i = 0; i < n_backends; ++i) {
            ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) comm_ctx->backends[i]->context;
            NCCL_CHECK(ncclAllReduce(tensors[i]->data, tensors[i]->data, ne, ncclFloat, ncclSum, comm_ctx->comms[i], cuda_ctx->stream()));
        }
        NCCL_CHECK(ncclGroupEnd());
        return true;
    }

    // For large tensors it's faster to compress them to BF16 for the reduction:
    to_bf16_cuda_t to_bf16 = ggml_get_to_bf16_cuda(GGML_TYPE_F32);
    to_fp32_cuda_t to_fp32 = ggml_get_to_fp32_cuda(GGML_TYPE_BF16);

    ggml_cuda_pool_alloc<nv_bfloat16> tmp[GGML_CUDA_MAX_DEVICES];
    for (size_t i = 0; i < n_backends; ++i) {
        ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) comm_ctx->backends[i]->context;
        tmp[i].pool = &cuda_ctx->pool();
        tmp[i].alloc(ne);

        ggml_cuda_set_device(cuda_ctx->device);
        if (tensors[i]->flags & GGML_TENSOR_FLAG_COMPUTE) {
            to_bf16(tensors[i]->data, tmp[i].get(), ne, cuda_ctx->stream());
        } else {
            CUDA_CHECK(cudaMemsetAsync(tmp[i].get(), 0, ne * sizeof(nv_bfloat16), cuda_ctx->stream()));
        }
        CUDA_CHECK(cudaGetLastError());
    }

    NCCL_CHECK(ncclGroupStart());
    for (size_t i = 0; i < n_backends; ++i) {
        ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) comm_ctx->backends[i]->context;
        NCCL_CHECK(ncclAllReduce(tmp[i].get(), tmp[i].get(), ne, ncclBfloat16, ncclSum, comm_ctx->comms[i], cuda_ctx->stream()));
    }
    NCCL_CHECK(ncclGroupEnd());

    for (size_t i = 0; i < n_backends; ++i) {
        ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) comm_ctx->backends[i]->context;

        ggml_cuda_set_device(cuda_ctx->device);
        to_fp32(tmp[i].get(), (float *) tensors[i]->data, ne, cuda_ctx->stream());
        CUDA_CHECK(cudaGetLastError());
    }

    return true;
}
#endif // GGML_USE_NCCL

// Run the internal AR pipeline.  Returns false on unsupported / failed input
// -- the caller decides whether to abort (env-forced) or fall back silently.
static bool ggml_backend_cuda_comm_allreduce_internal(
        ggml_backend_cuda_comm_context * comm_ctx, struct ggml_tensor ** tensors) {
    GGML_ASSERT(comm_ctx->ar_pipeline != nullptr);

    const size_t n_backends = comm_ctx->backends.size();
    GGML_ASSERT(n_backends == 2);
    GGML_ASSERT(tensors[0] != nullptr);

    const int64_t   ne   = ggml_nelements(tensors[0]);
    const ggml_type type = tensors[0]->type;

    if (type != GGML_TYPE_F32 && type != GGML_TYPE_F16 && type != GGML_TYPE_BF16) {
        GGML_LOG_DEBUG("%s: internal unsupported: type=%d\n", __func__, (int) type);
        return false;
    }

    if (ne == 0) {
        return true;
    }

    for (size_t i = 0; i < n_backends; ++i) {
        if (tensors[i] == nullptr) {
            GGML_LOG_ERROR("%s: internal failed: tensor[%zu] is null\n", __func__, i);
            return false;
        }
        if (ggml_nelements(tensors[i]) != ne || tensors[i]->type != type) {
            GGML_LOG_ERROR("%s: internal failed: tensor[%zu] ne=%" PRId64 " type=%d expected ne=%" PRId64 " type=%d\n",
                           __func__, i, ggml_nelements(tensors[i]), (int) tensors[i]->type, ne, (int) type);
            return false;
        }
        if (!ggml_is_contiguously_allocated(tensors[i])) {
            GGML_LOG_DEBUG("%s: internal unsupported: tensor[%zu] is not contiguously allocated: ne=%" PRId64 " nbytes=%zu packed=%zu type=%d\n",
                           __func__, i, ne, ggml_nbytes(tensors[i]),
                           (size_t) ne * ggml_type_size(type) / ggml_blck_size(type), (int) type);
            return false;
        }
        if (((uintptr_t) tensors[i]->data & 0xF) != 0) {
            GGML_LOG_DEBUG("%s: internal unsupported: tensor[%zu] data pointer is not 16-byte aligned: %p type=%d ne=%" PRId64 "\n",
                           __func__, i, tensors[i]->data, (int) type, ne);
            return false;
        }
        GGML_ASSERT((ggml_nbytes(tensors[i]) & 0xF) == 0);
    }

    return ggml_cuda_ar_allreduce(comm_ctx->ar_pipeline, comm_ctx->backends.data(), tensors);
}

// ---------------------------------------------------------------------------
// Per-call dispatch -- three variants, one per backend.  Each is set as
// comm_ctx->try_allreduce by the matching init step.  Per-call failure
// returns false; the meta backend's generic implementation handles that call.
// ---------------------------------------------------------------------------

#ifdef GGML_USE_NCCL
static bool ggml_backend_cuda_comm_try_allreduce_nccl(
        ggml_backend_cuda_comm_context * comm_ctx, struct ggml_tensor ** tensors) {
    return ggml_backend_cuda_comm_allreduce_nccl(comm_ctx, tensors);
}
#endif // GGML_USE_NCCL

static bool ggml_backend_cuda_comm_try_allreduce_internal(
        ggml_backend_cuda_comm_context * comm_ctx, struct ggml_tensor ** tensors) {
    return ggml_backend_cuda_comm_allreduce_internal(comm_ctx, tensors);
}

static bool ggml_backend_cuda_comm_try_allreduce_butterfly(
        ggml_backend_cuda_comm_context *, struct ggml_tensor **) {
    return false;
}

static void ggml_backend_cuda_comm_free(void * comm_ctx_v) {
    if (comm_ctx_v == nullptr) {
        return;
    }
    delete static_cast<ggml_backend_cuda_comm_context *>(comm_ctx_v);
}

// ---------------------------------------------------------------------------
// Init -- chained nccl -> internal -> none.  Each step tries to bring up its
// resource; on failure it warns and recurses into the next step.
// ---------------------------------------------------------------------------
static void ggml_backend_cuda_comm_init_none(ggml_backend_cuda_comm_context * ret) {
    ret->try_allreduce = ggml_backend_cuda_comm_try_allreduce_butterfly;
}

static void ggml_backend_cuda_comm_init_internal(ggml_backend_cuda_comm_context * ret) {
    ret->ar_pipeline = ggml_cuda_ar_pipeline_init(ret->dev_ids.data(), ret->dev_ids.size());
    if (ret->ar_pipeline) {
        ret->try_allreduce = ggml_backend_cuda_comm_try_allreduce_internal;
        return;
    }

    // Clear sticky CUDA error from the failed init.
    (void) cudaGetLastError();
    GGML_LOG_WARN("internal AllReduce init failed (n_devices != 2?); "
                  "falling back to meta-backend butterfly\n");
    ggml_backend_cuda_comm_init_none(ret);
}

static void ggml_backend_cuda_comm_init_nccl(ggml_backend_cuda_comm_context * ret) {
#ifdef GGML_USE_NCCL
    // Disabling NCCL path when CUDA virtual devices are in use since NCCL requires one distinct physical GPU per rank.
    const ggml_cuda_device_info & info = ggml_cuda_info();
    if (info.device_count > info.physical_device_count) {
        GGML_LOG_WARN("NCCL disabled: virtual devices in use; "
                      "falling back to internal AllReduce\n");
        ggml_backend_cuda_comm_init_internal(ret);
        return;
    }

    const size_t n = ret->dev_ids.size();
    ret->comms.resize(n);
    ncclResult_t rc = ncclCommInitAll(ret->comms.data(), (int) n, ret->dev_ids.data());
    if (rc == ncclSuccess) {
        ret->try_allreduce = ggml_backend_cuda_comm_try_allreduce_nccl;
        return;
    }

    ret->comms.clear();
    GGML_LOG_WARN("NCCL init failed (%s); falling back to internal AllReduce\n",
                  ncclGetErrorString(rc));
#else // GGML_USE_NCCL
#ifndef GGML_USE_HIP
    GGML_LOG_WARN("NCCL not compiled in; falling back to internal AllReduce.  "
                  "Recompile with -DGGML_CUDA_NCCL=ON for best multi-GPU performance.\n");
#endif // !GGML_USE_HIP
#endif // GGML_USE_NCCL

    ggml_backend_cuda_comm_init_internal(ret);
}

// Top-level init.  Picks one of the three init paths based on
// GGML_CUDA_ALLREDUCE (or the platform default) and lets the chain handle
// any fallback.  Unrecognised env values warn and fall through to the
// platform default.
static void * ggml_backend_cuda_comm_init(ggml_backend_t * backends, size_t n_backends) {
    for (size_t i = 0; i < n_backends; i++) {
        if (!ggml_backend_is_cuda(backends[i])) {
            return nullptr;
        }
    }

    auto * ret = new ggml_backend_cuda_comm_context;
    ret->backends.assign(backends, backends + n_backends);
    ret->dev_ids.reserve(n_backends);
    for (size_t i = 0; i < n_backends; i++) {
        ret->dev_ids.push_back(static_cast<ggml_backend_cuda_context *>(backends[i]->context)->device);
    }

    const char * env = getenv("GGML_CUDA_ALLREDUCE");
    if (!env) {
        // Platform default: Linux uses NCCL, otherwise (generally Windows) internal
#if defined(__linux__)
        ggml_backend_cuda_comm_init_nccl(ret);
#else
        ggml_backend_cuda_comm_init_internal(ret);
#endif // defined(__linux__)
    } else {
        std::string env_str(env);
        if (env_str == "nccl") {
            ggml_backend_cuda_comm_init_nccl(ret);
        } else if (env_str == "internal") {
            ggml_backend_cuda_comm_init_internal(ret);
        } else if (env_str == "none") {
            ggml_backend_cuda_comm_init_none(ret);
        } else {
            GGML_LOG_WARN("unknown GGML_CUDA_ALLREDUCE value: %s\n", env);
            ggml_backend_cuda_comm_init_none(ret);
        }
    }

    return ret;
}

#ifdef GGML_USE_NCCL
// Builds the communicator with ncclCommInitRank, for the case where each rank is its own
// process. Blocks until all world_size ranks have joined.
static void * ggml_backend_cuda_comm_init_rank(ggml_backend_t backend, int rank, int world_size,
                                               ggml_backend_comm_bootstrap_callback bootstrap_callback, void * bootstrap_callback_data) {
    if (world_size < 1 || rank < 0 || rank >= world_size) {
        GGML_LOG_ERROR("%s: invalid rank %d for world size %d\n", __func__, rank, world_size);
        return nullptr;
    }

    // Every rank sends whether its local checks passed, and rank 0 also sends the NCCL unique id.
    // So all ranks either join together or fail together, and no rank waits in ncclCommInitRank alone.
    struct bootstrap_msg {
        uint8_t      ok;
        ncclUniqueId id;
    };
    bootstrap_msg local;
    memset(&local, 0, sizeof(local));
    local.ok = 1;

    if (!ggml_backend_is_cuda(backend)) {
        GGML_LOG_ERROR("%s: backend is not a CUDA backend\n", __func__);
        local.ok = 0;
    } else {
        // NCCL requires one distinct physical GPU per rank, see ggml_backend_cuda_comm_init_nccl.
        const ggml_cuda_device_info & info = ggml_cuda_info();
        if (info.device_count > info.physical_device_count) {
            GGML_LOG_ERROR("%s: CUDA virtual devices are in use, which NCCL does not support\n", __func__);
            local.ok = 0;
        }
    }
    if (local.ok && rank == 0) {
        const ncclResult_t rc = ncclGetUniqueId(&local.id);
        if (rc != ncclSuccess) {
            GGML_LOG_ERROR("%s: ncclGetUniqueId failed (%s)\n", __func__, ncclGetErrorString(rc));
            local.ok = 0;
        }
    }

    std::vector<bootstrap_msg> all(world_size);
    if (!bootstrap_callback(bootstrap_callback_data, &local, all.data(), sizeof(local))) {
        GGML_LOG_ERROR("%s: bootstrap exchange failed on rank %d\n", __func__, rank);
        return nullptr;
    }
    for (int i = 0; i < world_size; i++) {
        if (!all[i].ok) {
            GGML_LOG_ERROR("%s: rank %d failed its setup checks\n", __func__, i);
            return nullptr;
        }
    }

    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;

    auto * ret = new ggml_backend_cuda_comm_context;
    ret->backends.push_back(backend);
    ret->dev_ids.push_back(cuda_ctx->device);
    ret->world_size = world_size;
    ret->comms.resize(1);
    ret->try_allreduce = ggml_backend_cuda_comm_try_allreduce_nccl;

    // On DGX Spark, NCCL picks 64 channels by default, which is slower and uses more memory than 8.
    // NCCL_MAX_CTAS still overrides this.
    ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
    const int cc = ggml_cuda_info().devices[cuda_ctx->device].cc;
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && cc == GGML_CUDA_CC_GB10) {
        config.maxCTAs = 8;
    }

    ggml_cuda_set_device(cuda_ctx->device);
    const ncclResult_t rc = ncclCommInitRankConfig(&ret->comms[0], world_size, all[0].id, rank, &config);
    if (rc != ncclSuccess) {
        GGML_LOG_ERROR("%s: ncclCommInitRankConfig failed for rank %d/%d (%s)\n",
                       __func__, rank, world_size, ncclGetErrorString(rc));
        ret->comms.clear();
        delete ret;
        return nullptr;
    }

    GGML_LOG_INFO("%s: joined NCCL communicator as rank %d/%d on device %d\n",
                  __func__, rank, world_size, cuda_ctx->device);
    return ret;
}
#endif // GGML_USE_NCCL

// Top-level dispatch -- calls the function pointer chosen by comm_init.
// Returns false to let the meta-backend's butterfly run.
static bool ggml_backend_cuda_comm_allreduce_tensor(void * comm_ctx_v, struct ggml_tensor ** tensors) {
    if (comm_ctx_v == nullptr) {
        return false;
    }
    auto * comm_ctx = static_cast<ggml_backend_cuda_comm_context *>(comm_ctx_v);
    return comm_ctx->try_allreduce(comm_ctx, tensors);
}

// host buffer type

static const char * ggml_backend_cuda_host_buffer_type_name(ggml_backend_buffer_type_t buft) {
    return GGML_CUDA_NAME "_Host";

    GGML_UNUSED(buft);
}

static bool ggml_backend_buft_is_cuda_host(ggml_backend_buffer_type_t buft) {
    return buft->iface.get_name == ggml_backend_cuda_host_buffer_type_name;
}

static void ggml_backend_cuda_host_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    CUDA_CHECK(cudaFreeHost(buffer->context));
}

static void * ggml_cuda_host_malloc(size_t size) {
    if (ggml_cuda_pinned_memory_disabled()) {
        return nullptr;
    }

    void * ptr = nullptr;
    cudaError_t err = cudaMallocHost((void **) &ptr, size);
    if (err != cudaSuccess) {
        // clear the error
        (void)cudaGetLastError();
        GGML_LOG_DEBUG("%s: failed to allocate %.2f MiB of pinned memory: %s\n", __func__,
                           size / 1024.0 / 1024.0, cudaGetErrorString(err));
        return nullptr;
    }
    ggml_backend_ink2_vidmem_log("pinned host alloc", size);

    return ptr;
}

static ggml_backend_buffer_t ggml_backend_cuda_host_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    void * ptr = ggml_cuda_host_malloc(size);

    if (ptr == nullptr) {
        // fallback to cpu buffer
        return ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), size);
    }

    ggml_backend_buffer_t buffer = ggml_backend_cpu_buffer_from_ptr(ptr, size);
    buffer->buft = buft;
    buffer->iface.free_buffer = ggml_backend_cuda_host_buffer_free_buffer;

    return buffer;
}

static size_t ggml_backend_cuda_host_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    // very large pinned allocations can fail (e.g. on Windows) and fall back to pageable memory, smaller ones succeed
    return 4ull*1024*1024*1024;
}

ggml_backend_buffer_type_t ggml_backend_cuda_host_buffer_type() {
    static struct ggml_backend_buffer_type ggml_backend_cuda_buffer_type_host = {
        /* .iface    = */ {
            /* .get_name            = */ ggml_backend_cuda_host_buffer_type_name,
            /* .alloc_buffer        = */ ggml_backend_cuda_host_buffer_type_alloc_buffer,
            /* .alloc_buffer_n      = */ NULL,
            /* .get_alignment       = */ ggml_backend_cpu_buffer_type()->iface.get_alignment,
            /* .get_max_size        = */ ggml_backend_cuda_host_buffer_type_get_max_size,
            /* .get_alloc_size      = */ ggml_backend_cpu_buffer_type()->iface.get_alloc_size,
            /* .get_alloc_size_n    = */ NULL,
            /* .is_host             = */ ggml_backend_cpu_buffer_type()->iface.is_host,
        },
        /* .device   = */ ggml_backend_reg_dev_get(ggml_backend_cuda_reg(), 0),
        /* .context  = */ nullptr,
    };

    return &ggml_backend_cuda_buffer_type_host;
}

//static bool ggml_backend_buffer_is_cuda_host(ggml_backend_buffer_t buffer) {
//    return buffer->buft->iface.get_name == ggml_backend_cuda_host_buffer_type_name;
//}

/// kernels

typedef void (*ggml_cuda_op_mul_mat_t)(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream);

static __global__ void k_compute_batched_ptrs(
        const void * src0_as_f16, const void * src1_as_f16, char * dst,
        const void ** ptrs_src, void ** ptrs_dst,
        int64_t ne12, int64_t ne13,
        int64_t ne23,
        size_t  nb02, size_t  nb03,
        size_t  nb12, size_t  nb13,
        size_t  nbd2, size_t  nbd3,
        int64_t r2,   int64_t r3) {
    const int64_t i13 = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t i12 = blockIdx.y * blockDim.y + threadIdx.y;

    if (i13 >= ne13 || i12 >= ne12) {
        return;
    }

    const int64_t i03 = i13 / r3;
    const int64_t i02 = i12 / r2;

    ptrs_src[0*ne23 + i12 + i13*ne12] = (const char *) src0_as_f16 + i02*nb02 + i03*nb03;
    ptrs_src[1*ne23 + i12 + i13*ne12] = (const char *) src1_as_f16 + i12*nb12 + i13*nb13;
    ptrs_dst[0*ne23 + i12 + i13*ne12] = (      char *)         dst + i12*nbd2 + i13*nbd3;
}

// Type traits for mapping ggml types to CUDA/cuBLAS types
template<ggml_type T>
struct batched_mul_mat_traits;

template<>
struct batched_mul_mat_traits<GGML_TYPE_F32> {
    using cuda_type = float;
    static inline const cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;
    static inline const cudaDataType_t data_type = CUDA_R_32F;
    static inline const ggml_type ggml_type_val = GGML_TYPE_F32;
    static inline const float alpha = 1.0f;
    static inline const float beta = 0.0f;
    static inline const void* get_alpha() { static const float val = alpha; return &val; }
    static inline const void* get_beta() { static const float val = beta; return &val; }
    static inline auto convert(ggml_type src_type) { return ggml_get_to_fp32_cuda(src_type); }
    static inline auto convert_nc(ggml_type src_type) { return ggml_get_to_fp32_nc_cuda(src_type); }
};

template<>
struct batched_mul_mat_traits<GGML_TYPE_BF16> {
    using cuda_type = nv_bfloat16;
    static inline const cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;
    static inline const cudaDataType_t data_type = CUDA_R_16BF;
    static inline const ggml_type ggml_type_val = GGML_TYPE_BF16;
    static inline const float alpha = 1.0f;
    static inline const float beta = 0.0f;
    static inline const void* get_alpha() { static const float val = alpha; return &val; }
    static inline const void* get_beta() { static const float val = beta; return &val; }
    static inline auto convert(ggml_type src_type) { return ggml_get_to_bf16_cuda(src_type); }
    static inline auto convert_nc(ggml_type src_type) { return ggml_get_to_bf16_nc_cuda(src_type); }
};

template<>
struct batched_mul_mat_traits<GGML_TYPE_F16> {
    using cuda_type = half;
    static inline const cublasComputeType_t compute_type = CUBLAS_COMPUTE_16F;
    static inline const cudaDataType_t data_type = CUDA_R_16F;
    static inline const ggml_type ggml_type_val = GGML_TYPE_F16;
    static inline const half alpha = 1.0;
    static inline const half beta = 0.0;
    static inline const void* get_alpha() { static const half val = alpha; return &val; }
    static inline const void* get_beta() { static const half val = beta; return &val; }
    static inline auto convert(ggml_type src_type) { return ggml_get_to_fp16_cuda(src_type); }
    static inline auto convert_nc(ggml_type src_type) { return ggml_get_to_fp16_nc_cuda(src_type); }
};

template<ggml_type compute_type>
static void ggml_cuda_mul_mat_cublas_impl(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    using traits = batched_mul_mat_traits<compute_type>;
    using cuda_t = typename traits::cuda_type;

    // F32 chunks can have padding between rows to preserve the original destination stride.
    GGML_ASSERT(ggml_is_contiguous(dst) ||
        (compute_type == GGML_TYPE_F32 && ggml_is_contiguous_1(dst)));

    GGML_TENSOR_BINARY_OP_LOCALS

    const int64_t ldc = nb1 / sizeof(float);
    const int64_t ne_dst = ggml_nelements(dst);
    cudaStream_t main_stream = ctx.stream();
    cublasHandle_t cublas_h = ctx.cublas_handle();

    const size_t src0_ts = ggml_type_size(src0->type);
    GGML_ASSERT(nb00 == src0_ts);
    int64_t s01 = nb01 / src0_ts;
    int64_t s02 = nb02 / src0_ts;
    int64_t s03 = nb03 / src0_ts;

    const size_t src1_ts = ggml_type_size(src1->type);
    GGML_ASSERT(nb10 == src1_ts);
    int64_t s11 = nb11 / src1_ts;
    int64_t s12 = nb12 / src1_ts;
    int64_t s13 = nb13 / src1_ts;

    float * dst_ddf = (float *) dst->data;

    const cuda_t * src0_ptr = nullptr;
    const cuda_t * src1_ptr = nullptr;

    ggml_cuda_pool_alloc<cuda_t> src0_alloc(ctx.pool());
    ggml_cuda_pool_alloc<cuda_t> src1_alloc(ctx.pool());

    bool is_src0_cont_2 = ggml_is_contiguous_2(src0);
    bool is_src1_cont_2 = ggml_is_contiguous_2(src1);

    if (src0->type == compute_type) {
        src0_ptr = (const cuda_t *) src0->data;
    } else {
        src0_alloc.alloc(ggml_nelements(src0));

        if (ggml_is_contiguously_allocated(src0)) {
            const auto convert_func = traits::convert(src0->type);
            GGML_ASSERT(convert_func != nullptr);
            convert_func(src0->data, src0_alloc.get(), ggml_nelements(src0), main_stream);
            const size_t src0_bs = ggml_blck_size(src0->type);
            s01 *= src0_bs;
            s02 *= src0_bs;
            s03 *= src0_bs;
        } else {
            const auto convert_func = traits::convert_nc(src0->type);
            GGML_ASSERT(convert_func != nullptr);
            convert_func(src0->data, src0_alloc.get(), ne00, ne01, ne02, ne03, s01, s02, s03, main_stream);
            s01 = ne00;
            s02 = ne01*s01;
            s03 = ne02*s02;
            is_src0_cont_2 = true;
        }
        src0_ptr = src0_alloc.get();
    }

    if (src1->type == compute_type) {
        src1_ptr = (const cuda_t *) src1->data;
    } else {
        src1_alloc.alloc(ggml_nelements(src1));

        if (ggml_is_contiguously_allocated(src1)) {
            const auto convert_func = traits::convert(src1->type);
            GGML_ASSERT(convert_func != nullptr);
            convert_func(src1->data, src1_alloc.get(), ggml_nelements(src1), main_stream);
            const size_t src1_bs = ggml_blck_size(src1->type);
            s11 *= src1_bs;
            s12 *= src1_bs;
            s13 *= src1_bs;
        } else {
            const auto convert_func = traits::convert_nc(src1->type);
            GGML_ASSERT(convert_func != nullptr);
            convert_func(src1->data, src1_alloc.get(), ne10, ne11, ne12, ne13, s11, s12, s13, main_stream);
            s11 = ne10;
            s12 = ne11*s11;
            s13 = ne12*s12;
            is_src1_cont_2 = true;
        }
        src1_ptr = src1_alloc.get();
    }

    ggml_cuda_pool_alloc<cuda_t> dst_temp(ctx.pool());
    char * dst_ptr;
    size_t nbd2 = dst->nb[2];
    size_t nbd3 = dst->nb[3];

    const bool f32_pedantic = compute_type == GGML_TYPE_F32 &&
        src0->type == GGML_TYPE_F32 && ggml_prec(dst->op_params[0]) == GGML_PREC_F32_PEDANTIC;

    cublasComputeType_t cu_compute_type = traits::compute_type;
    cudaDataType_t cu_data_type = traits::data_type;
    cudaDataType_t cu_data_type_a = traits::data_type;
    cudaDataType_t cu_data_type_b = traits::data_type;
    const void * alpha = traits::get_alpha();
    const void * beta = traits::get_beta();

    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    bool prefer_f32_output = false;
    if (compute_type == GGML_TYPE_F16) {
        prefer_f32_output = cc == GGML_CUDA_CC_VOLTA || GGML_CUDA_CC_IS_RDNA4(cc) || GGML_CUDA_CC_IS_CDNA(cc);
    } else if (compute_type == GGML_TYPE_BF16) {
        prefer_f32_output = !GGML_CUDA_CC_IS_RDNA3(cc) && !GGML_CUDA_CC_IS_RDNA4_M(cc) && !GGML_CUDA_CC_IS_CDNA(cc);
    }
    // local (measurement switch): GGML_CUDA_INK2_BLAS_F32OUT=1 asks F32 output (and F32 accumulation) for F16 / BF16 too
    static const bool ink2_f32out = getenv("GGML_CUDA_INK2_BLAS_F32OUT") != nullptr;
    if (ink2_f32out && (compute_type == GGML_TYPE_F16 || compute_type == GGML_TYPE_BF16)) {
        prefer_f32_output = true;
    }

    if (prefer_f32_output) {
        dst_ptr = (char *) dst_ddf;
        cu_compute_type = batched_mul_mat_traits<GGML_TYPE_F32>::compute_type;
        cu_data_type = batched_mul_mat_traits<GGML_TYPE_F32>::data_type;
        alpha = batched_mul_mat_traits<GGML_TYPE_F32>::get_alpha();
        beta = batched_mul_mat_traits<GGML_TYPE_F32>::get_beta();
    } else {
        if constexpr (compute_type == GGML_TYPE_F32) {
            dst_ptr = (char *) dst_ddf;  // Direct F32 output
        } else {
            dst_ptr = (char *) dst_temp.alloc(ne_dst);
            nbd2 /= sizeof(float) / sizeof(cuda_t);
            nbd3 /= sizeof(float) / sizeof(cuda_t);
        }
    }

    if (f32_pedantic) {
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) && CUDART_VERSION >= 11020
        cu_compute_type = CUBLAS_COMPUTE_32F_PEDANTIC;
#else
        cu_compute_type = CUBLAS_COMPUTE_32F;
#endif
    }

    const auto cu_gemm_algo = f32_pedantic ?
        CUBLAS_GEMM_DEFAULT : CUBLAS_GEMM_DEFAULT_TENSOR_OP;

    GGML_ASSERT(ne12 % ne02 == 0);
    GGML_ASSERT(ne13 % ne03 == 0);

    // broadcast factors
    const int64_t r2 = ne12/ne02;
    const int64_t r3 = ne13/ne03;

    // Theoretically cublasGemmStridedBatchedEx would always work, even for a single matrix.
    // However, for some old NVIDIA and AMD GPUs the strided/Ex GEMM is much slower,
    //     probably because the internal kernel selection logic is suboptimal.
    if (compute_type == GGML_TYPE_F32 && ne12 == 1 && ne13 == 1) {
        if (f32_pedantic) {
            CUBLAS_CHECK(
                cublasGemmEx(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
                        ne01, ne11, ne10,
                        alpha, src0_ptr, CUDA_R_32F, s01,
                               src1_ptr, CUDA_R_32F, s11,
                        beta,   dst_ptr, CUDA_R_32F, ldc,
                        cu_compute_type,
                        cu_gemm_algo));
        } else {
            CUBLAS_CHECK(
                cublasSgemm(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
                        ne01, ne11, ne10,
                        (const float *) alpha, (const float *) src0_ptr, s01,
                                               (const float *) src1_ptr, s11,
                        (const float *) beta,  (float       *)  dst_ptr, ldc));
        }
    } else if (ne12 == 1 && ne13 == 1) {
        CUBLAS_CHECK(
            cublasGemmEx(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
                    ne01, ne11, ne10,
                    alpha, src0_ptr, cu_data_type_a, s01,
                           src1_ptr, cu_data_type_b, s11,
                    beta,   dst_ptr, cu_data_type,   ldc,
                    cu_compute_type,
                    cu_gemm_algo));
    } else if (r2 == 1 && r3 == 1 && is_src0_cont_2 && is_src1_cont_2) {
        // with a [0, 2, 1, 3] perm. and ne02==1 the matrix strides need to be determined from dim 3:
        const int64_t sma = ne02 == 1 ? s03 : s02;
        const int64_t smb = ne12 == 1 ? s13 : s12;

        // there is no broadcast and src0, src1 are contiguous across dims 2, 3
        // use cublasGemmStridedBatchedEx
        CUBLAS_CHECK(
        cublasGemmStridedBatchedEx(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
                ne01, ne11, ne10,
                alpha, src0_ptr, cu_data_type_a, s01, sma,     // strideA
                       src1_ptr, cu_data_type_b, s11, smb,     // strideB
                beta,   dst_ptr, cu_data_type,   ldc, ne1*ldc, // strideC
                ne12*ne13,
                cu_compute_type,
                cu_gemm_algo));
    } else {
        // use cublasGemmBatchedEx
        const int64_t ne23 = ne12*ne13;

        ggml_cuda_pool_alloc<const void *> ptrs_src(ctx.pool(), 2*ne23);
        ggml_cuda_pool_alloc<      void *> ptrs_dst(ctx.pool(), 1*ne23);

        const size_t src_type_size = sizeof(cuda_t);

        const int threads_x = 16;
        const int threads_y = 16;
        const dim3 block_dims(threads_x, threads_y);

        const dim3 grid_dims(
            (ne13 + threads_x - 1) / threads_x,
            (ne12 + threads_y - 1) / threads_y
        );
        k_compute_batched_ptrs<<<grid_dims, block_dims, 0, main_stream>>>(
                src0_ptr, src1_ptr, dst_ptr,
                ptrs_src.get(), ptrs_dst.get(),
                ne12, ne13,
                ne23,
                s02*src_type_size, s03*src_type_size,
                s12*src_type_size, s13*src_type_size,
                nbd2, nbd3,
                r2, r3);

        CUDA_CHECK(cudaGetLastError());

        CUBLAS_CHECK(
        cublasGemmBatchedEx(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
                ne01, ne11, ne10,
                alpha, (const void **) (ptrs_src.get() + 0*ne23), cu_data_type_a, s01,
                       (const void **) (ptrs_src.get() + 1*ne23), cu_data_type_b, s11,
                beta,  (      void **) (ptrs_dst.get() + 0*ne23), cu_data_type,   ldc,
                ne23,
                cu_compute_type,
                cu_gemm_algo));
    }

    // Convert output back to F32 if needed
    if (cu_data_type != CUDA_R_32F) {
        const to_fp32_cuda_t to_fp32_cuda = ggml_get_to_fp32_cuda(traits::ggml_type_val);
        to_fp32_cuda(dst_temp.get(), dst_ddf, ne_dst, main_stream);
    }
}

static void ggml_cuda_mul_mat_cublas(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    const ggml_prec prec = ggml_prec_get_acc(dst);
    ggml_type compute_type = src0->type;
    if (ggml_is_quantized(compute_type) || compute_type == GGML_TYPE_F8_E4M3) {
        compute_type = fast_fp16_hardware_available(cc) ? GGML_TYPE_F16 : GGML_TYPE_F32;
        if (ggml_cuda_info().devices[ctx.device].prefer_fp32_acc) {
            compute_type = GGML_TYPE_BF16;
        }
    } else if (compute_type == GGML_TYPE_F16 && !fast_fp16_hardware_available(cc)) {
        compute_type = GGML_TYPE_F32;
    } else if (compute_type == GGML_TYPE_BF16 && !fast_bf16_hardware_available(cc)) {
        if (GGML_CUDA_CC_IS_AMD(cc) && src1->ne[1] > 32) {
            compute_type = GGML_TYPE_F32;
        }
        if (GGML_CUDA_CC_IS_NVIDIA(cc) && src1->ne[1] > (cc >= GGML_CUDA_CC_VOLTA ? 8 : 128)) {
            compute_type = GGML_TYPE_F32;
        }
    }
    // [TAG_GGML_PREC] any acc rank at least as strict as F32 (F32, F32_PEDANTIC) forces F32 compute
    const ggml_prec prec_acc = prec;
    // F16 is the only compute type that can not satisfy a request for BF16
    if (prec == GGML_PREC_BF16 && compute_type == GGML_TYPE_F16) {
        compute_type = fast_bf16_hardware_available(cc) ? GGML_TYPE_BF16 : GGML_TYPE_F32;
    } else if (prec_acc != GGML_PREC_UNDEFINED && prec_acc <= GGML_PREC_F32) {
        compute_type = GGML_TYPE_F32;
    }

    const char * env_c = getenv("GGML_CUDA_CUBLAS_COMPUTE_TYPE");
    if (env_c != nullptr) {
        std::string env_cpp = env_c;
        for (char & c : env_cpp) {
            c = std::tolower(c);
        }
        if (env_cpp == "f32" || env_cpp == "fp32") {
            compute_type = GGML_TYPE_F32;
        } else if (env_cpp == "f16" || env_cpp == "fp16") {
            compute_type = GGML_TYPE_F16;
        } else if (env_cpp == "bf16") {
            compute_type = GGML_TYPE_BF16;
        } else if (env_cpp != "auto") {
            GGML_LOG_WARN("%s: unknown value for GGML_CUDA_CUBLAS_COMPUTE_TYPE: %s", __func__, env_cpp.c_str());
        }
    }

    // a scoped pedantic request overrides the process-wide compute type, for F32 weights only
    if (src0->type == GGML_TYPE_F32 && prec_acc == GGML_PREC_F32_PEDANTIC) {
        compute_type = GGML_TYPE_F32;
    }

    constexpr size_t max_src0_convert_size = 512ull * 1024 * 1024;
    const size_t src0_f32_size = ggml_nelements(src0) * sizeof(float);

    if (compute_type == GGML_TYPE_F32 &&
            (src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_BF16) &&
            src0_f32_size > max_src0_convert_size) {
        const size_t f32_row_size = src0_f32_size / src0->ne[1];
        const int64_t rows_per_chunk = std::max<int64_t>(1, (int64_t) (max_src0_convert_size / f32_row_size));

        if (rows_per_chunk < src0->ne[1]) {
            ggml_tensor src0_chunk = *src0;
            ggml_tensor dst_chunk = *dst;

            for (int64_t i01 = 0; i01 < src0->ne[1]; i01 += rows_per_chunk) {
                src0_chunk.ne[1] = std::min(rows_per_chunk, src0->ne[1] - i01);
                src0_chunk.data = (char *) src0->data + i01*src0->nb[1];

                dst_chunk.ne[0] = src0_chunk.ne[1];
                dst_chunk.data = (char *) dst->data + i01*dst->nb[0];

                ggml_cuda_mul_mat_cublas_impl<GGML_TYPE_F32>(ctx, &src0_chunk, src1, &dst_chunk);
            }
            return;
        }
    }

    switch (compute_type) {
        case GGML_TYPE_F32:
            ggml_cuda_mul_mat_cublas_impl<GGML_TYPE_F32>(ctx, src0, src1, dst);
            break;
        case GGML_TYPE_BF16:
            ggml_cuda_mul_mat_cublas_impl<GGML_TYPE_BF16>(ctx, src0, src1, dst);
            break;
        case GGML_TYPE_F16:
            ggml_cuda_mul_mat_cublas_impl<GGML_TYPE_F16>(ctx, src0, src1, dst);
            break;
        default:
            GGML_ABORT("fatal error");
    }
}

static bool ggml_cuda_should_fuse_mul_mat(const ggml_tensor * ffn_up,
                                          const ggml_tensor * ffn_gate,
                                          const ggml_tensor * glu,
                                          const ggml_tensor * ffn_up_bias = nullptr,
                                          const ggml_tensor * ffn_gate_bias = nullptr,
                                          const ggml_tensor * ffn_up_scale = nullptr,
                                          const ggml_tensor * ffn_gate_scale = nullptr) {
    const bool has_bias = ffn_up_bias != nullptr || ffn_gate_bias != nullptr;
    const bool has_scale = ffn_up_scale != nullptr || ffn_gate_scale != nullptr;

    if (has_bias && (!ffn_up_bias || !ffn_gate_bias)) {
        return false;
    }
    if (has_scale && (!ffn_up_scale || !ffn_gate_scale)) {
        return false;
    }

    const bool is_mul_mat     = ffn_up->op == GGML_OP_MUL_MAT     && ffn_gate->op == GGML_OP_MUL_MAT     && glu->op == GGML_OP_GLU;
    const bool is_mul_mat_id  = ffn_up->op == GGML_OP_MUL_MAT_ID  && ffn_gate->op == GGML_OP_MUL_MAT_ID  && glu->op == GGML_OP_GLU;

    GGML_ASSERT(ffn_up && ffn_gate && glu);

    if (!is_mul_mat && !is_mul_mat_id) {
        return false;
    }

    const ggml_op expected_bias_op = is_mul_mat ? GGML_OP_ADD : GGML_OP_ADD_ID;
    const ggml_tensor * ffn_up_bias_src   = has_scale ? ffn_up_scale   : ffn_up;
    const ggml_tensor * ffn_gate_bias_src = has_scale ? ffn_gate_scale : ffn_gate;
    const ggml_tensor * ffn_up_out        = has_bias ? ffn_up_bias     : ffn_up_bias_src;
    const ggml_tensor * ffn_gate_out      = has_bias ? ffn_gate_bias   : ffn_gate_bias_src;

    if (glu->src[0] != ffn_gate_out || glu->src[1] != ffn_up_out) {
        return false;
    }

    if (has_scale) {
        if (ffn_up_scale->op != GGML_OP_MUL || ffn_gate_scale->op != GGML_OP_MUL) {
            return false;
        }
        const bool up_has_mm   = ffn_up_scale->src[0] == ffn_up || ffn_up_scale->src[1] == ffn_up;
        const bool gate_has_mm = ffn_gate_scale->src[0] == ffn_gate || ffn_gate_scale->src[1] == ffn_gate;
        if (!up_has_mm || !gate_has_mm) {
            return false;
        }
    }

    if (has_bias) {
        if (ffn_up_bias->op != expected_bias_op || ffn_gate_bias->op != expected_bias_op) {
            return false;
        }

        if (expected_bias_op == GGML_OP_ADD) {
            const bool up_has_mul   = ffn_up_bias->src[0] == ffn_up_bias_src || ffn_up_bias->src[1] == ffn_up_bias_src;
            const bool gate_has_mul = ffn_gate_bias->src[0] == ffn_gate_bias_src || ffn_gate_bias->src[1] == ffn_gate_bias_src;
            if (!up_has_mul || !gate_has_mul) {
                return false;
            }
        } else { // GGML_OP_ADD_ID
            if (ffn_up_bias->src[0] != ffn_up_bias_src || ffn_gate_bias->src[0] != ffn_gate_bias_src) {
                return false;
            }
            if (ffn_up_bias->src[2] != ffn_up->src[2] || ffn_gate_bias->src[2] != ffn_gate->src[2]) {
                return false;
            }
        }
    }

    if (ffn_up->src[0]->type != ffn_gate->src[0]->type || !ggml_are_same_shape(ffn_up->src[0], ffn_gate->src[0]) ||
        !ggml_are_same_stride(ffn_up->src[0], ffn_gate->src[0])) {
        return false;
    }

    if (ffn_up->src[1] != ffn_gate->src[1]) {
        return false;
    }

    if (is_mul_mat_id && ffn_up->src[2] != ffn_gate->src[2]) {
        return false;
    }

    static constexpr std::array<ggml_glu_op, 4> valid_glu_ops = { GGML_GLU_OP_SWIGLU, GGML_GLU_OP_GEGLU, GGML_GLU_OP_SWIGLU_OAI, GGML_GLU_OP_SWIGLU_CLAMP };

    if (std::find(valid_glu_ops.begin(), valid_glu_ops.end(), ggml_get_glu_op(glu)) == valid_glu_ops.end()) {
        return false;
    }

    if (const bool swapped = ggml_get_op_params_i32(glu, 1); swapped) {
        return false;
    }

    return true;
}

static bool ggml_cuda_should_fuse_mul_mat_vec_f(const ggml_tensor * tensor, const ggml_tensor * gate = nullptr) {
    ggml_tensor *       src0 = tensor->src[0];
    ggml_tensor *       src1 = tensor->src[1];
    const ggml_tensor * dst  = tensor;

    const bool is_mul_mat_id = tensor->op == GGML_OP_MUL_MAT_ID;

    bool use_mul_mat_vec_f =
        (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_BF16 ||
         src0->type == GGML_TYPE_F8_E4M3) &&
        src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32;

    const int cc        = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    use_mul_mat_vec_f = use_mul_mat_vec_f && ggml_cuda_should_use_mmvf(src0, cc, warp_size, is_mul_mat_id ? src1->ne[2] : src1->ne[1]);

    if (gate && !ggml_cuda_should_use_mmvf(gate->src[0], cc, warp_size, is_mul_mat_id ? src1->ne[2] : src1->ne[1])) {
        return false;
    }

    //we only support fusion for ncols_dst = 1
    if (tensor->op == GGML_OP_MUL_MAT && dst->ne[1] != 1) {
        return false;
    }

    if (tensor->op == GGML_OP_MUL_MAT_ID && dst->ne[2] != 1) {
        return false;
    }


    return use_mul_mat_vec_f;
}

static bool ggml_cuda_should_fuse_mul_mat_vec_q(const ggml_tensor * tensor) {
    ggml_tensor *       src0 = tensor->src[0];
    ggml_tensor *       src1 = tensor->src[1];
    const ggml_tensor * dst  = tensor;

    const bool bad_padding_clear = ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE &&
                                   ggml_nbytes(src0) != ggml_backend_buffer_get_alloc_size(src0->buffer, src0) &&
                                   src0->view_src;

    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    bool use_mul_mat_vec_q = ggml_cuda_should_use_mmvq(src0->type, cc, src1->ne[1]) && !bad_padding_clear &&
                             src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32;

    // fusion is not universally faster on Pascal
    if (cc < GGML_CUDA_CC_VOLTA) {
        return false;
    }
    //we only support fusion for ncols_dst = 1
    if (tensor->op == GGML_OP_MUL_MAT && dst->ne[1] != 1) {
        return false;
    }

    if (tensor->op == GGML_OP_MUL_MAT_ID && dst->ne[2] > get_mmvq_mmid_max_batch(src0->type, cc)) {
        return false;
    }

    // do not fuse the matmuls that few-row MMQ takes in ggml_cuda_mul_mat_id
    if (tensor->op == GGML_OP_MUL_MAT_ID && ggml_cuda_should_use_mmq_few(src0, src1, tensor->src[2], cc)) {
        return false;
    }

    return use_mul_mat_vec_q;
}

static bool ggml_cuda_match_shared_expert(const ggml_cgraph * graph, int routed_idx, int shared_idx) {
    if (routed_idx + 2 >= graph->n_nodes || shared_idx + 2 >= graph->n_nodes || shared_idx < routed_idx + 3) {
        return false;
    }
    const int nodes[] = { routed_idx, routed_idx + 1, routed_idx + 2, shared_idx, shared_idx + 1, shared_idx + 2 };
    const ggml_op ops[] = { GGML_OP_MUL_MAT_ID, GGML_OP_MUL_MAT_ID, GGML_OP_GLU,
                           GGML_OP_MUL_MAT, GGML_OP_MUL_MAT, GGML_OP_GLU };
    const int outputs[] = { routed_idx + 2, shared_idx + 2 };
    if (!ggml_can_fuse_subgraph_ext(graph, nodes, 6, ops, outputs, 2)) {
        return false;
    }

    const ggml_tensor * routed = graph->nodes[routed_idx + 2];
    const ggml_tensor * shared = graph->nodes[shared_idx + 2];
    const ggml_tensor * gate = routed->src[0];
    const ggml_tensor * up = routed->src[1];
    const ggml_tensor * shared_gate = shared->src[0];
    const ggml_tensor * shared_up = shared->src[1];
    const auto is_pair = [&](const ggml_tensor * a, const ggml_tensor * b, int idx) {
        return (a == graph->nodes[idx] && b == graph->nodes[idx + 1]) ||
               (b == graph->nodes[idx] && a == graph->nodes[idx + 1]);
    };
    // only batch-size independent checks here: graph_optimize must produce the same graph topology for every ubatch
    // size, otherwise ggml-alloc has to re-reserve (and the scheduler to synchronize) at runtime.
    // the MMVQ batch size check is done in ggml_cuda_try_fuse
    if (!is_pair(gate, up, routed_idx) || !is_pair(shared_gate, shared_up, shared_idx) ||
            !ggml_cuda_should_fuse_mul_mat(up, gate, routed) ||
            !ggml_cuda_should_fuse_mul_mat(shared_up, shared_gate, shared) ||
            !up->src[0]->buffer ||
            !ggml_is_quantized(up->src[0]->type)) {
        return false;
    }
    const ggml_tensor * input = up->src[1];
    const ggml_tensor * weight = up->src[0];
    const ggml_tensor * shared_weight = shared_up->src[0];
    if (input->op != GGML_OP_RESHAPE || input->src[0] != shared_up->src[1] ||
            input->ne[1] != 1 || input->ne[3] != 1 || !ggml_is_contiguous(input) ||
            !ggml_is_contiguous(shared_up->src[1]) || !ggml_is_matrix(shared_up->src[1]) ||
            weight->type != shared_weight->type || weight->ne[0] != shared_weight->ne[0] ||
            weight->ne[1] != shared_weight->ne[1] || weight->nb[1] != shared_weight->nb[1] || weight->ne[3] != 1 ||
            !ggml_is_matrix(shared_weight) || !ggml_is_contiguous(shared_weight) ||
            !ggml_is_contiguous(shared_gate->src[0]) || !ggml_is_contiguous(routed) || !ggml_is_contiguous(shared)) {
        return false;
    }
    if (shared_weight->op != GGML_OP_NONE || shared_gate->src[0]->op != GGML_OP_NONE ||
            ggml_get_glu_op(routed) != ggml_get_glu_op(shared) ||
            ggml_get_op_params_f32(routed, 3) != ggml_get_op_params_f32(shared, 3)) {
        return false;
    }
    return true;
}

// The weight scale that scale_node (MUL) applies to the output of mm_node (MUL_MAT), or nullptr.
// Shared by ggml_cuda_try_fuse and graph_optimize so that both accept the same patterns.
static const ggml_tensor * ggml_cuda_get_mul_mat_scale(const ggml_tensor * scale_node, const ggml_tensor * mm_node) {
    const bool scale_lhs_mm = scale_node->src[0] == mm_node;
    const bool scale_rhs_mm = scale_node->src[1] == mm_node;
    if (!scale_lhs_mm && !scale_rhs_mm) {
        return nullptr;
    }

    const ggml_tensor * scale = scale_lhs_mm ? scale_node->src[1] : scale_node->src[0];
    if ((mm_node->src[0]->type != GGML_TYPE_NVFP4 && mm_node->src[0]->type != GGML_TYPE_F8_E4M3) || scale_node->type != GGML_TYPE_F32 ||
            scale->type != GGML_TYPE_F32 || !ggml_is_contiguous(scale) || ggml_nelements(scale) != 1 ||
            !ggml_are_same_shape(scale_node, mm_node)) {
        return nullptr;
    }

    return scale;
}

// The same for MUL_MAT_ID with one scale per expert: RESHAPE, REPEAT and GET_ROWS (by the ids) of the scales, then the MUL.
static const ggml_tensor * ggml_cuda_get_mul_mat_id_scale(const ggml_tensor * reshape, const ggml_tensor * repeat, const ggml_tensor * getrows,
        const ggml_tensor * scale_node, const ggml_tensor * mm_node) {
    if (repeat->src[0] != reshape || getrows->src[0] != repeat || getrows->src[1] != mm_node->src[2]) {
        return nullptr;
    }
    if (!((scale_node->src[0] == mm_node && scale_node->src[1] == getrows) ||
            (scale_node->src[0] == getrows && scale_node->src[1] == mm_node))) {
        return nullptr;
    }

    const ggml_tensor * scale = reshape->src[0];
    if ((mm_node->src[0]->type != GGML_TYPE_NVFP4 && mm_node->src[0]->type != GGML_TYPE_F8_E4M3) || scale_node->type != GGML_TYPE_F32 ||
            scale->type != GGML_TYPE_F32 || !ggml_is_contiguous(scale) || ggml_nelements(scale) != mm_node->src[0]->ne[2] ||
            !ggml_are_same_shape(scale_node, mm_node)) {
        return nullptr;
    }

    return scale;
}

// Buffer- and batch-size independent, so graph_optimize produces the same graph topology for every ubatch size (see #29986).
// up_scale/gate_scale: the MUL nodes that apply the NVFP4 weight scales (per tensor or per expert) between the MUL_MATs and the GLU.
static bool ggml_cuda_mul_mat_q_fusion_matches(const ggml_tensor * up, const ggml_tensor * gate, const ggml_tensor * glu, const int device,
        const ggml_tensor * up_scale = nullptr, const ggml_tensor * gate_scale = nullptr) {
    const ggml_tensor * x_up = up->src[0];
    const ggml_tensor * y    = up->src[1];

    if ((up->op != GGML_OP_MUL_MAT && up->op != GGML_OP_MUL_MAT_ID) || !ggml_cuda_mmq_fusion_supported(x_up->type) || y->type != GGML_TYPE_F32) {
        return false;
    }

    // The fused kernel is only compiled for weights whose rows fill whole tiles (no fallback).
    if (ggml_cuda_mmq_needs_fallback(x_up->ne[1])) {
        return false;
    }

    if (!ggml_cuda_should_fuse_mul_mat(up, gate, glu, nullptr, nullptr, up_scale, gate_scale)) {
        return false;
    }

    // The fused kernel runs both MUL_MATs at one src1 precision, see ggml_cuda_mul_mat_q.
    if (ggml_prec_get_src(up, 1) != ggml_prec_get_src(gate, 1)) {
        return false;
    }

    // The fused write-back applies SWIGLU_OAI with the default alpha and limit of ggml_cuda_op_swiglu_oai_single.
    if (ggml_get_glu_op(glu) == GGML_GLU_OP_SWIGLU_OAI && (ggml_get_op_params_f32(glu, 2) != 1.702f || ggml_get_op_params_f32(glu, 3) != 7.0f)) {
        return false;
    }

    // NVIDIA only: the fused write-back relies on the MMA accumulator layout.
    return turing_mma_available(ggml_cuda_info().devices[device].cc);
}

static bool ggml_cuda_should_fuse_mul_mat_q(const ggml_tensor * up, const ggml_tensor * gate, const ggml_tensor * glu,
        const ggml_tensor * up_scale = nullptr, const ggml_tensor * gate_scale = nullptr) {
    // A hinted matmul can be fused, but its unfused path may be faster (e.g. FWHT for GGML_HINT_SRC0_IS_HADAMARD).
    if (ggml_get_op_params_i32(up, 1) != GGML_HINT_NONE || ggml_get_op_params_i32(gate, 1) != GGML_HINT_NONE) {
        return false;
    }

    // The padding of a compute-buffer weight that is a view cannot be cleared, see ggml_cuda_mul_mat.
    const auto bad_padding_clear = [](const ggml_tensor * x) {
        return ggml_backend_buffer_get_usage(x->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE &&
            ggml_nbytes(x) != ggml_backend_buffer_get_alloc_size(x->buffer, x) && x->view_src;
    };
    const int device = ggml_cuda_get_device();
    if (!ggml_cuda_mul_mat_q_fusion_matches(up, gate, glu, device, up_scale, gate_scale) || bad_padding_clear(up->src[0]) || bad_padding_clear(gate->src[0])) {
        return false;
    }

    // The batch size checks are done here, not in ggml_cuda_mul_mat_q_fusion_matches.
    const ggml_tensor * x_up = up->src[0];
    const ggml_tensor * y    = up->src[1];
    const int           cc   = ggml_cuda_info().devices[device].cc;

    // For MUL_MAT_ID: the MMVQ/MMQ choice of ggml_cuda_mul_mat_id, with the nominal tiles of all experts (the per-expert token counts are only known on the GPU).
    const bool    is_id     = up->op == GGML_OP_MUL_MAT_ID;
    const int64_t ncols     = is_id ? y->ne[2]    : y->ne[1];
    const int64_t nchannels = is_id ? x_up->ne[2] : y->ne[2]*y->ne[3];
    const bool    use_mmvq  = is_id ? ncols <= get_mmvq_mmid_max_batch(x_up->type, cc) : ggml_cuda_should_use_mmvq(x_up->type, cc, ncols);
    // do not fuse the matmuls that few-row MMQ takes in ggml_cuda_mul_mat; for MUL_MAT_ID the fused MMQ keeps the expert gate and up
    if (!is_id && ggml_cuda_should_use_mmq_few(x_up, y, nullptr, cc)) {
        return false;
    }
    if (use_mmvq || !ggml_cuda_should_use_mmq(x_up->type, cc, ncols, /*n_experts =*/ is_id ? x_up->ne[2] : 0)) {
        return false;
    }

    // The fused kernel needs one block per tile (no stream-k split), so only fuse if the tiling is efficient.
    return ggml_cuda_mmq_fusion_is_efficient(x_up->type, ggml_prec_get_src(up, 1), x_up->ne[1], ncols, nchannels, device);
}

static void ggml_cuda_mul_mat(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_TENSOR_BINARY_OP_LOCALS

    if (ggml_cuda_op_mul_mat_use_fwht(dst) && ggml_cuda_op_fwht(ctx, src1, dst)) {
        return;
    }

    // If src0 is a temporary compute buffer it may have some padding that needs to be cleared for mul_mat_vec_q or mul_mat_q.
    // But if src0 is also a view of another tensor then this cannot be done safely because it may overwrite valid tensor data.
    // Therefore, in such cases use cuBLAS.
    const bool bad_padding_clear = ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE
        && ggml_nbytes(src0) != ggml_backend_buffer_get_alloc_size(src0->buffer, src0) && src0->view_src;
    if (bad_padding_clear || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        ggml_cuda_mul_mat_cublas(ctx, src0, src1, dst);
        return;
    }

    const int cc        = ggml_cuda_info().devices[ctx.device].cc;
    const int warp_size = ggml_cuda_info().devices[ctx.device].warp_size;
    const bool f32_pedantic = src0->type == GGML_TYPE_F32 &&
        ggml_prec(dst->op_params[0]) == GGML_PREC_F32_PEDANTIC;

    if (ggml_cuda_should_use_mmvf(src0, cc, warp_size, ne11)) {
        // The custom vector kernel can be used over batched cuBLAS GEMM.
        // But this is only faster for GPUs without tensor cores or with a thin src0 matrix (particularly KQV in attention)
        ggml_cuda_mul_mat_vec_f(ctx, src0, src1, nullptr, dst);
        return;
    }
    // A transposed vector can still use MMVF (i.e. ne01 == 1)
    if (ne01 == 1 && ne11 > MMVF_MAX_BATCH_SIZE && ne2 == 1 && ne3 == 1
            && src0->type == GGML_TYPE_F32
            && ggml_is_contiguous(src0) && ggml_is_contiguous(src1) && ggml_is_contiguous(dst)
            && ggml_cuda_should_use_mmvf(src1, cc, warp_size, /*ne11 =*/ 1)) {
        ggml_tensor dst_vec = *dst;
        dst_vec.ne[0] = ne11;
        dst_vec.ne[1] = 1;
        dst_vec.nb[1] = dst_vec.nb[0]*ne11;
        dst_vec.nb[2] = dst_vec.nb[1];
        dst_vec.nb[3] = dst_vec.nb[1];
        ggml_cuda_mul_mat_vec_f(ctx, src1, src0, nullptr, &dst_vec);
        return;
    }
    if (!f32_pedantic && ggml_cuda_should_use_mmf(
            src0->type, cc, warp_size, src0->ne, src0->nb, ne11, /*mul_mat_id =*/ false)) {
        ggml_cuda_mul_mat_f(ctx, src0, src1, nullptr, dst);
        return;
    }
    // from a per-type number of src1 columns up to 24, few-row MMQ is faster than MMVQ and MMQ
    const bool use_mmq_few = ggml_cuda_should_use_mmq_few(src0, src1, nullptr, cc);
    if (!use_mmq_few && ggml_cuda_should_use_mmvq(src0->type, cc, ne11)) {
        ggml_cuda_mul_mat_vec_q(ctx, src0, src1, nullptr, dst);
        return;
    }
    if (use_mmq_few) {
        ggml_cuda_mul_mat_q_few(ctx, src0, src1, nullptr, dst);
        return;
    }
    if (ggml_cuda_should_use_mmq(src0->type, cc, ne11, /*n_experts =*/ 0)) {
        ggml_cuda_mul_mat_q(ctx, src0, src1, nullptr, dst);
        return;
    }
    if (src0->type == GGML_TYPE_F8_E4M3 && ggml_cuda_mul_mat_fp8(ctx, src0, src1, dst)) {
        return;
    }
    ggml_cuda_mul_mat_cublas(ctx, src0, src1, dst);
}

// returns true when ggml_cuda_mul_mat_id takes the fallback path that requires stream synchronization
// [TAG_MUL_MAT_ID_CUDA_GRAPHS]
static bool ggml_cuda_mul_mat_id_needs_sync(const ggml_tensor * dst, const int cc) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const bool f32_pedantic = src0->type == GGML_TYPE_F32 &&
        ggml_prec(dst->op_params[0]) == GGML_PREC_F32_PEDANTIC;

    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;

    if (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return true;
    }

    if (dst->ne[2] <= MMVQ_MAX_BATCH_SIZE) {
        if (ggml_is_quantized(src0->type)) {
            if (dst->ne[2] <= get_mmvq_mmid_max_batch(src0->type, cc)) {
                return false;
            }
        } else if (src0->type == GGML_TYPE_F8_E4M3 &&
                ggml_cuda_should_use_mmvf(src0, cc, warp_size, dst->ne[2])) {
            return false;
        } else if (src0->type != GGML_TYPE_F8_E4M3 && GGML_CUDA_CC_IS_AMD(cc) && src0->ne[0] % 2 == 0) {
            return false;
        }
    }

    if (ggml_cuda_should_use_mmq_few(src0, src1, dst->src[2], cc)) {
        return false;
    }

    if (ggml_cuda_should_use_mmq(src0->type, cc, src1->ne[2], /*n_experts=*/src0->ne[2])) {
        return false;
    }

    if (!f32_pedantic && ggml_cuda_should_use_mmf(src0->type, cc, warp_size, src0->ne, src0->nb, src1->ne[2], /*mul_mat_id=*/true)) {
        return false;
    }

    return true;
}

static void ggml_cuda_mul_mat_id(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * ids  = dst->src[2];

    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    GGML_TENSOR_BINARY_OP_LOCALS

    const int cc        = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const bool f32_pedantic = src0->type == GGML_TYPE_F32 &&
        ggml_prec(dst->op_params[0]) == GGML_PREC_F32_PEDANTIC;
    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;

    // [TAG_MUL_MAT_ID_CUDA_GRAPHS]
    if (src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
        // up to 24 tokens few-row MMQ is faster than MMQ, on DGX Spark also faster than MMVQ
        const bool use_mmq_few = ggml_cuda_should_use_mmq_few(src0, src1, ids, cc);

        static_assert(MMVQ_MAX_BATCH_SIZE == MMVF_MAX_BATCH_SIZE);
        if (!use_mmq_few && ne2 <= MMVQ_MAX_BATCH_SIZE) {
            if (ggml_is_quantized(src0->type)) {
                const int mmvq_mmid_max = get_mmvq_mmid_max_batch(src0->type, cc);
                if (ne2 <= mmvq_mmid_max) {
                    ggml_cuda_mul_mat_vec_q(ctx, src0, src1, ids, dst);
                    return;
                }
            } else if (src0->type == GGML_TYPE_F8_E4M3 &&
                    ggml_cuda_should_use_mmvf(src0, cc, warp_size, ne2)) {
                ggml_cuda_mul_mat_vec_f(ctx, src0, src1, ids, dst);
                return;
            } else {
                // MMVF reads src0 in pairs of values, so it needs an even number of columns
                if (src0->type != GGML_TYPE_F8_E4M3 && GGML_CUDA_CC_IS_AMD(cc) && ne00 % 2 == 0) {
                    ggml_cuda_mul_mat_vec_f(ctx, src0, src1, ids, dst);
                    return;
                }
            }
        }

        if (use_mmq_few) {
            ggml_cuda_mul_mat_q_few(ctx, src0, src1, ids, dst);
            return;
        }

        if (ggml_cuda_should_use_mmq(src0->type, cc, ne12, /*n_experts=*/ne02)) {
            ggml_cuda_mul_mat_q(ctx, src0, src1, ids, dst);
            return;
        }

        if (!f32_pedantic && ggml_cuda_should_use_mmf(src0->type, cc, warp_size, src0->ne, src0->nb, src1->ne[2], /*mul_mat_id=*/true)) {
            ggml_cuda_mul_mat_f(ctx, src0, src1, ids, dst);
            return;
        }
    }

    // note: this path should not be reached when recording CUDA graphs, because it requires stream synchronization
    GGML_ASSERT(ggml_cuda_mul_mat_id_needs_sync(dst, cc));
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(nb12 % nb11 == 0);
    GGML_ASSERT(nb2  % nb1  == 0);

    const ggml_type type_src1_sorted = (src0->type == GGML_TYPE_F16 && !fast_fp16_hardware_available(cc))
        || ggml_is_quantized(src0->type) || src0->type == GGML_TYPE_F8_E4M3 ? GGML_TYPE_F32 : src0->type;
    const ggml_type type_dst_sorted  = GGML_TYPE_F32;
    const size_t ts_src1_sorted = ggml_type_size(type_src1_sorted);
    const size_t ts_dst_sorted  = ggml_type_size(type_dst_sorted);

    const int64_t n_expert_used = ids->ne[0];
    const int64_t ne_get_rows = ne12 * n_expert_used;

    ggml_cuda_pool_alloc<int32_t> ids_buf_dev(ctx.pool(), 2*ne_get_rows);

    int32_t * ids_to_sorted   = ids_buf_dev.ptr + 0*ne_get_rows;
    int32_t * ids_from_sorted = ids_buf_dev.ptr + 1*ne_get_rows;

    std::vector<int32_t> tokens_per_expert(ne02);

    ggml_cuda_pool_alloc<char> src1_sorted(ctx.pool(), ne12*n_expert_used*ne10*ts_src1_sorted);
    ggml_cuda_pool_alloc<char>  dst_sorted(ctx.pool(), ne2 *n_expert_used* ne0*ts_dst_sorted);

    // sort the ids by expert on the device with the counting sort of the MMQ/MMF helper, else on the host
    const bool sort_on_device = ids->type == GGML_TYPE_I32 && ids->nb[0] == ggml_element_size(ids) &&
        ggml_cuda_mm_ids_device_sort_supported(ne02, ne12, n_expert_used);

    if (sort_on_device) {
        // only the device sort needs the compact map, the host sort builds the inverse map directly
        ggml_cuda_pool_alloc<int32_t> ids_dst_compact(ctx.pool(), ne_get_rows);
        ggml_cuda_pool_alloc<int32_t> expert_bounds_dev(ctx.pool(), ne02 + 1);

        const int si1  = ids->nb[1] / ggml_element_size(ids);
        const int sis1 = nb12 / nb11;
        GGML_ASSERT(sis1 > 0);

        ggml_cuda_launch_mm_ids_sort(ctx.pool(), (const int32_t *) ids->data, ids_to_sorted, ids_dst_compact.ptr, expert_bounds_dev.ptr,
            ne02, ne12, n_expert_used, ne11, si1, sis1, /*write_inverse =*/ false, stream);
        CUDA_CHECK(cudaGetLastError());

        // the GEMMs below need the per-expert token counts on the host
        std::vector<int32_t> expert_bounds_host(ne02 + 1);
        CUDA_CHECK(cudaMemcpyAsync(expert_bounds_host.data(), expert_bounds_dev.ptr,
            (ne02 + 1)*sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // fails for out of range ids
        GGML_ASSERT(expert_bounds_host[ne02] == ne_get_rows);
        GGML_ASSERT(expert_bounds_host[0] == 0);

        for (int64_t i02 = 0; i02 < ne02; ++i02) {
            tokens_per_expert[i02] = expert_bounds_host[i02 + 1] - expert_bounds_host[i02];
        }

        ggml_cuda_launch_mm_ids_invert(ids_dst_compact.ptr, ids_from_sorted, ne_get_rows, stream);
        CUDA_CHECK(cudaGetLastError());
    } else {
        std::vector<int32_t> ids_to_sorted_host;
        ids_to_sorted_host.reserve(2*ne_get_rows);
        std::vector<int32_t> ids_from_sorted_host(ne_get_rows);

        std::vector<char> ids_host(ggml_nbytes(ids));
        CUDA_CHECK(cudaMemcpyAsync(ids_host.data(), ids->data, ggml_nbytes(ids), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        for (int64_t i02 = 0; i02 < ne02; ++i02) {
            for (int64_t i12 = 0; i12 < ne12; ++i12) {
                for (int64_t iex = 0; iex < n_expert_used; ++iex) {
                    const int32_t expert_to_use = *(const int32_t *)(ids_host.data() + i12*ids->nb[1] + iex*ids->nb[0]);
                    // GGML_ASSERT, not assert: plain assert compiles out in a release build
                    GGML_ASSERT(expert_to_use >= 0 && expert_to_use < ne02);
                    // no break: a token that repeats an expert gets one row per slot, as in the device sort
                    if (expert_to_use == i02) {
                        ids_from_sorted_host[i12*n_expert_used + iex] = ids_to_sorted_host.size();
                        ids_to_sorted_host.push_back(i12*ne11 + iex % ne11);
                        tokens_per_expert[i02]++;
                    }
                }
            }
        }
        GGML_ASSERT(ids_to_sorted_host.size() == size_t(ne_get_rows));

        ids_to_sorted_host.insert(ids_to_sorted_host.end(), ids_from_sorted_host.begin(), ids_from_sorted_host.end());

        // ids_to_sorted and ids_from_sorted are adjacent in ids_buf_dev: one copy as before
        CUDA_CHECK(cudaMemcpyAsync(ids_buf_dev.ptr, ids_to_sorted_host.data(), 2*ne_get_rows*sizeof(int32_t), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    get_rows_cuda(src1->data, src1->type, ids_to_sorted, src1_sorted.ptr, type_src1_sorted,
        ne10, nb11, nb12, nb13,
        ne_get_rows, 1, 1, sizeof(int32_t), ne_get_rows*sizeof(int32_t), ne_get_rows*sizeof(int32_t),
        ne10*ts_src1_sorted, ne_get_rows*ne10*ts_src1_sorted, ne_get_rows*ne10*ts_src1_sorted, stream);
    CUDA_CHECK(cudaGetLastError());

    char * src1_data_cur = (char *) src1_sorted.ptr;
    char *  dst_data_cur = (char *)  dst_sorted.ptr;
    for (int64_t i02 = 0; i02 < ne02; ++i02) {
        if (tokens_per_expert[i02] == 0) {
            continue;
        }

        ggml_tensor src0_slice = *src0;
        src0_slice.ne[2]    = 1;
        src0_slice.nb[3]    = src0_slice.nb[2];
        src0_slice.op       = GGML_OP_VIEW;
        src0_slice.view_src = dst->src[0]; // non-const pointer to src0
        src0_slice.data     = (char *) src0->data + i02*nb02;

        ggml_tensor src1_slice;
        memset(&src1_slice, 0, sizeof(src1_slice));
        src1_slice.buffer = src1->buffer;
        src1_slice.type   = type_src1_sorted;
        src1_slice.ne[0]  = ne10;
        src1_slice.ne[1]  = tokens_per_expert[i02];
        src1_slice.ne[2]  = 1;
        src1_slice.ne[3]  = 1;
        src1_slice.nb[0]  = ts_src1_sorted;
        src1_slice.nb[1]  = src1_slice.ne[0] * src1_slice.nb[0];
        src1_slice.nb[2]  = src1_slice.ne[1] * src1_slice.nb[1];
        src1_slice.nb[3]  = src1_slice.ne[2] * src1_slice.nb[2];
        src1_slice.data   = src1_data_cur;

        ggml_tensor dst_slice;
        memset(&dst_slice, 0, sizeof(dst_slice));
        memcpy(dst_slice.op_params, dst->op_params, sizeof(dst_slice.op_params));
        dst_slice.op     = GGML_OP_MUL_MAT; // ggml_prec_get_acc/ggml_prec_get_src check the op
        dst_slice.buffer = dst->buffer;
        dst_slice.type   = type_dst_sorted;
        dst_slice.ne[0]  = ne0;
        dst_slice.ne[1]  = tokens_per_expert[i02];
        dst_slice.ne[2]  = 1;
        dst_slice.ne[3]  = 1;
        dst_slice.nb[0]  = ts_dst_sorted;
        dst_slice.nb[1]  = dst_slice.ne[0] * dst_slice.nb[0];
        dst_slice.nb[2]  = dst_slice.ne[1] * dst_slice.nb[1];
        dst_slice.nb[3]  = dst_slice.ne[2] * dst_slice.nb[2];
        dst_slice.data   = dst_data_cur;
        // [TAG_GGML_PREC] carry acc (0) and src precision (2, 3), not the unsliced op hint (1)
        dst_slice.op_params[0] = dst->op_params[0];
        dst_slice.op_params[2] = dst->op_params[2];
        dst_slice.op_params[3] = dst->op_params[3];

        ggml_cuda_mul_mat(ctx, &src0_slice, &src1_slice, &dst_slice);
        CUDA_CHECK(cudaGetLastError());

        src1_data_cur += src1_slice.nb[2];
        dst_data_cur  +=  dst_slice.nb[2];
    }

    get_rows_cuda(dst_sorted.ptr, type_dst_sorted, ids_from_sorted, dst->data, dst->type,
        ne0, ne0*ts_dst_sorted, ne_get_rows*ne0*ts_dst_sorted, ne_get_rows*ne0*ts_dst_sorted,
        ne_get_rows, 1, 1, sizeof(int32_t), ne_get_rows*sizeof(int32_t), ne_get_rows*sizeof(int32_t),
        nb1, nb2, nb3, stream);
}

static bool ggml_cuda_compute_forward(ggml_backend_cuda_context & ctx, struct ggml_tensor * dst) {
    switch (dst->op) {
        case GGML_OP_ARGMAX:
            ggml_cuda_argmax(ctx, dst);
            break;
        case GGML_OP_COUNT_EQUAL:
            ggml_cuda_count_equal(ctx, dst);
            break;
        case GGML_OP_REPEAT:
            ggml_cuda_op_repeat(ctx, dst);
            break;
        case GGML_OP_REPEAT_BACK:
            ggml_cuda_op_repeat_back(ctx, dst);
            break;
        case GGML_OP_GET_ROWS:
            ggml_cuda_op_get_rows(ctx, dst);
            break;
        case GGML_OP_GET_ROWS_BACK:
            ggml_cuda_op_get_rows_back(ctx, dst);
            break;
        case GGML_OP_SET_ROWS:
            ggml_cuda_op_set_rows(ctx, dst);
            break;
        case GGML_OP_SET:
            ggml_cuda_op_set(ctx, dst);
            break;
        case GGML_OP_DUP:
            ggml_cuda_dup(ctx, dst);
            break;
        case GGML_OP_CPY:
            ggml_cuda_cpy(ctx, dst->src[0], dst->src[1]);
            break;
        case GGML_OP_CONT:
            ggml_cuda_dup(ctx, dst);
            break;
        case GGML_OP_ADD:
            ggml_cuda_op_add(ctx, dst);
            break;
        case GGML_OP_ADD_ID:
            ggml_cuda_op_add_id(ctx, dst);
            break;
        case GGML_OP_SUB:
            ggml_cuda_op_sub(ctx, dst);
            break;
        case GGML_OP_ACC:
            ggml_cuda_op_acc(ctx, dst);
            break;
        case GGML_OP_MUL:
            ggml_cuda_op_mul(ctx, dst);
            break;
        case GGML_OP_DIV:
            ggml_cuda_op_div(ctx, dst);
            break;
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(dst)) {
                case GGML_UNARY_OP_ABS:
                    ggml_cuda_op_abs(ctx, dst);
                    break;
                case GGML_UNARY_OP_SGN:
                    ggml_cuda_op_sgn(ctx, dst);
                    break;
                case GGML_UNARY_OP_NEG:
                    ggml_cuda_op_neg(ctx, dst);
                    break;
                case GGML_UNARY_OP_STEP:
                    ggml_cuda_op_step(ctx, dst);
                    break;
                case GGML_UNARY_OP_GELU:
                    ggml_cuda_op_gelu(ctx, dst);
                    break;
                case GGML_UNARY_OP_SILU:
                    ggml_cuda_op_silu(ctx, dst);
                    break;
                case GGML_UNARY_OP_GELU_ERF:
                    ggml_cuda_op_gelu_erf(ctx, dst);
                    break;
                case GGML_UNARY_OP_GELU_QUICK:
                    ggml_cuda_op_gelu_quick(ctx, dst);
                    break;
                case GGML_UNARY_OP_TANH:
                    ggml_cuda_op_tanh(ctx, dst);
                    break;
                case GGML_UNARY_OP_RELU:
                    ggml_cuda_op_relu(ctx, dst);
                    break;
                case GGML_UNARY_OP_SIGMOID:
                    ggml_cuda_op_sigmoid(ctx, dst);
                    break;
                case GGML_UNARY_OP_HARDSIGMOID:
                    ggml_cuda_op_hardsigmoid(ctx, dst);
                    break;
                case GGML_UNARY_OP_HARDSWISH:
                    ggml_cuda_op_hardswish(ctx, dst);
                    break;
                case GGML_UNARY_OP_EXP:
                    ggml_cuda_op_exp(ctx, dst);
                    break;
                case GGML_UNARY_OP_ELU:
                    ggml_cuda_op_elu(ctx, dst);
                    break;
                case GGML_UNARY_OP_XIELU:
                    ggml_cuda_op_xielu(ctx, dst);
                    break;
                case GGML_UNARY_OP_FLOOR:
                    ggml_cuda_op_floor(ctx, dst);
                    break;
                case GGML_UNARY_OP_CEIL:
                    ggml_cuda_op_ceil(ctx, dst);
                    break;
                case GGML_UNARY_OP_ROUND:
                    ggml_cuda_op_round(ctx, dst);
                    break;
                case GGML_UNARY_OP_TRUNC:
                    ggml_cuda_op_trunc(ctx, dst);
                    break;
                case GGML_UNARY_OP_EXPM1:
                    ggml_cuda_op_expm1(ctx, dst);
                    break;
                case GGML_UNARY_OP_SOFTPLUS:
                    ggml_cuda_op_softplus(ctx, dst);
                    break;
                default:
                    return false;
            }
            break;
        case GGML_OP_GLU:
            switch (ggml_get_glu_op(dst)) {
                case GGML_GLU_OP_REGLU:
                    ggml_cuda_op_reglu(ctx, dst);
                    break;
                case GGML_GLU_OP_GEGLU:
                    ggml_cuda_op_geglu(ctx, dst);
                    break;
                case GGML_GLU_OP_SWIGLU:
                    ggml_cuda_op_swiglu(ctx, dst);
                    break;
                case GGML_GLU_OP_SWIGLU_OAI:
                    ggml_cuda_op_swiglu_oai(ctx, dst);
                    break;
                case GGML_GLU_OP_GEGLU_ERF:
                    ggml_cuda_op_geglu_erf(ctx, dst);
                    break;
                case GGML_GLU_OP_GEGLU_QUICK:
                    ggml_cuda_op_geglu_quick(ctx, dst);
                    break;
                case GGML_GLU_OP_SWIGLU_CLAMP:
                    ggml_cuda_op_swiglu_clamp(ctx, dst);
                    break;
                default:
                    return false;
            }
            break;
        case GGML_OP_NORM:
            ggml_cuda_op_norm(ctx, dst);
            break;
        case GGML_OP_GROUP_NORM:
            ggml_cuda_op_group_norm(ctx, dst);
            break;
        case GGML_OP_L2_NORM:
            ggml_cuda_op_l2_norm(ctx, dst);
            break;
        case GGML_OP_CONCAT:
            ggml_cuda_op_concat(ctx, dst);
            break;
        case GGML_OP_UPSCALE:
            ggml_cuda_op_upscale(ctx, dst);
            break;
        case GGML_OP_PAD:
            ggml_cuda_op_pad(ctx, dst);
            break;
        case GGML_OP_PAD_REFLECT_1D:
            ggml_cuda_op_pad_reflect_1d(ctx, dst);
            break;
        case GGML_OP_ARANGE:
            ggml_cuda_op_arange(ctx, dst);
            break;
        case GGML_OP_TIMESTEP_EMBEDDING:
            ggml_cuda_op_timestep_embedding(ctx, dst);
            break;
        case GGML_OP_LEAKY_RELU:
            ggml_cuda_op_leaky_relu(ctx, dst);
            break;
        case GGML_OP_SILU_BACK:
            ggml_cuda_op_silu_back(ctx, dst);
            break;
        case GGML_OP_RMS_NORM:
            ggml_cuda_op_rms_norm(ctx, dst);
            break;
        case GGML_OP_RMS_NORM_BACK:
            ggml_cuda_op_rms_norm_back(ctx, dst);
            break;
        case GGML_OP_MUL_MAT:
            ggml_cuda_mul_mat(ctx, dst->src[0], dst->src[1], dst);
            break;
        case GGML_OP_MUL_MAT_ID:
            ggml_cuda_mul_mat_id(ctx, dst);
            break;
        case GGML_OP_OUT_PROD:
            ggml_cuda_out_prod(ctx, dst);
            break;
        case GGML_OP_SCALE:
            ggml_cuda_op_scale(ctx, dst);
            break;
        case GGML_OP_SQR:
            ggml_cuda_op_sqr(ctx, dst);
            break;
        case GGML_OP_SQRT:
            ggml_cuda_op_sqrt(ctx, dst);
            break;
        case GGML_OP_SIN:
            ggml_cuda_op_sin(ctx, dst);
            break;
        case GGML_OP_COS:
            ggml_cuda_op_cos(ctx, dst);
            break;
        case GGML_OP_CLAMP:
            ggml_cuda_op_clamp(ctx, dst);
            break;
        case GGML_OP_LOG:
            ggml_cuda_op_log(ctx, dst);
            break;
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
                break;
        case GGML_OP_DIAG:
            ggml_cuda_op_diag(ctx, dst);
            break;
        case GGML_OP_SOFT_MAX:
            ggml_cuda_op_soft_max(ctx, dst);
            break;
        case GGML_OP_SOFT_MAX_BACK:
            ggml_cuda_op_soft_max_back(ctx, dst);
            break;
        case GGML_OP_ROPE:
            ggml_cuda_op_rope(ctx, dst);
            break;
        case GGML_OP_ROPE_BACK:
            ggml_cuda_op_rope_back(ctx, dst);
            break;
        case GGML_OP_ROLL:
            ggml_cuda_op_roll(ctx, dst);
            break;
        case GGML_OP_IM2COL:
            ggml_cuda_op_im2col(ctx, dst);
            break;
        case GGML_OP_IM2COL_3D:
            ggml_cuda_op_im2col_3d(ctx, dst);
            break;
        case GGML_OP_CONV_2D:
            ggml_cuda_op_conv2d(ctx, dst);
            break;
        case GGML_OP_CONV_3D:
            ggml_cuda_op_conv3d(ctx, dst);
            break;
        case GGML_OP_CONV_2D_DW:
            ggml_cuda_op_conv2d_dw(ctx, dst);
            break;
        case GGML_OP_CONV_TRANSPOSE_2D:
            ggml_cuda_conv_2d_transpose_p0(ctx, dst);
            break;
        case GGML_OP_CONV_TRANSPOSE_1D:
            ggml_cuda_op_conv_transpose_1d(ctx,dst);
            break;
        case GGML_OP_COL2IM_1D:
            ggml_cuda_op_col2im_1d(ctx, dst);
            break;
        case GGML_OP_POOL_2D:
            ggml_cuda_op_pool2d(ctx, dst);
            break;
        case GGML_OP_POOL_1D:
            ggml_cuda_op_pool1d(ctx, dst);
            break;
        case GGML_OP_SUM:
            ggml_cuda_op_sum(ctx, dst);
            break;
        case GGML_OP_CUMSUM:
            ggml_cuda_op_cumsum(ctx, dst);
            break;
        case GGML_OP_SUM_ROWS:
            ggml_cuda_op_sum_rows(ctx, dst);
            break;
        case GGML_OP_MEAN:
            ggml_cuda_op_mean(ctx, dst);
            break;
        case GGML_OP_SSM_CONV:
            ggml_cuda_op_ssm_conv(ctx, dst);
            break;
        case GGML_OP_SSM_SCAN:
            ggml_cuda_op_ssm_scan(ctx, dst);
            break;
        case GGML_OP_TOP_K:
            ggml_cuda_op_top_k(ctx, dst);
            break;
        case GGML_OP_ARGSORT:
            ggml_cuda_op_argsort(ctx, dst);
            break;
        case GGML_OP_FLASH_ATTN_EXT:
            ggml_cuda_flash_attn_ext(ctx, dst);
            break;
        case GGML_OP_FLASH_ATTN_EXT_BANDED:
            ggml_cuda_flash_attn_ext_banded(ctx, dst);
            break;
        case GGML_OP_CROSS_ENTROPY_LOSS:
            ggml_cuda_cross_entropy_loss(ctx, dst);
            break;
        case GGML_OP_TRI:
            ggml_cuda_op_tri(ctx, dst);
            break;
        case GGML_OP_RWKV_WKV6:
            ggml_cuda_op_rwkv_wkv6(ctx, dst);
            break;
        case GGML_OP_GATED_LINEAR_ATTN:
            ggml_cuda_op_gated_linear_attn(ctx, dst);
            break;
        case GGML_OP_GATED_DELTA_NET:
            ggml_cuda_op_gated_delta_net(ctx, dst);
            break;
        case GGML_OP_DSV4_HC_COMB:
            ggml_cuda_op_dsv4_hc_comb(ctx, dst);
            break;
        case GGML_OP_DSV4_HC_PRE:
            ggml_cuda_op_dsv4_hc_pre(ctx, dst);
            break;
        case GGML_OP_DSV4_HC_POST:
            ggml_cuda_op_dsv4_hc_post(ctx, dst);
            break;
        case GGML_OP_RWKV_WKV7:
            ggml_cuda_op_rwkv_wkv7(ctx, dst);
            break;
        case GGML_OP_CROSS_ENTROPY_LOSS_BACK:
            ggml_cuda_cross_entropy_loss_back(ctx, dst);
            break;
        case GGML_OP_OPT_STEP_ADAMW:
            ggml_cuda_opt_step_adamw(ctx, dst);
            break;
        case GGML_OP_OPT_STEP_SGD:
            ggml_cuda_opt_step_sgd(ctx, dst);
            break;
        case GGML_OP_SOLVE_TRI:
            ggml_cuda_op_solve_tri(ctx, dst);
            break;
        case GGML_OP_FILL:
            ggml_cuda_op_fill(ctx, dst);
            break;
        case GGML_OP_SLEEP:
            ggml_cuda_op_sleep(ctx, dst);
            break;
        case GGML_OP_LIGHTNING_INDEXER:
            ggml_cuda_lightning_indexer(ctx, dst);
            break;
        case GGML_OP_TOP_K_MASK:
            ggml_cuda_op_top_k_mask(ctx, dst);
            break;
        default:
            return false;
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        GGML_LOG_ERROR("%s: %s failed\n", __func__, ggml_op_desc(dst));
        CUDA_CHECK(err);
    }

    return true;
}

////////////////////////////////////////////////////////////////////////////////

// backend

static const char * ggml_backend_cuda_get_name(ggml_backend_t backend) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;

    return cuda_ctx->name.c_str();
}

static void ggml_backend_cuda_free(ggml_backend_t backend) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;

    delete cuda_ctx;
    delete backend;
}

static void ggml_backend_cuda_set_tensor_async(ggml_backend_t backend, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    GGML_ASSERT(buf->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device) && "unsupported buffer type");

    ggml_cuda_ink2_tl(cuda_ctx, "aset", (long long) size);
    CUDA_CHECK(cudaMemcpyAsync((char *) tensor->data + offset, data, size, cudaMemcpyHostToDevice, cuda_ctx->stream()));
}

static void ggml_backend_cuda_get_tensor_async(ggml_backend_t backend, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    GGML_ASSERT(buf->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device) && "unsupported buffer type");

    ggml_cuda_ink2_tl(cuda_ctx, "aget", (long long) size);
    CUDA_CHECK(cudaMemcpyAsync(data, (const char *) tensor->data + offset, size, cudaMemcpyDeviceToHost, cuda_ctx->stream()));
}

static void ggml_backend_cuda_set_tensor_2d_async(ggml_backend_t backend, struct ggml_tensor * tensor, const void * data,
        size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    GGML_ASSERT(buf->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device) && "unsupported buffer type");

    CUDA_CHECK(cudaMemcpy2DAsync(
        (char *) tensor->data + offset, stride_tensor, data, stride_data, size, n_copies, cudaMemcpyHostToDevice, cuda_ctx->stream()));
}

static void ggml_backend_cuda_get_tensor_2d_async(ggml_backend_t backend, const struct ggml_tensor * tensor, void * data,
        size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    GGML_ASSERT(buf->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device) && "unsupported buffer type");

    CUDA_CHECK(cudaMemcpy2DAsync(
        data, stride_data, (const char *) tensor->data + offset, stride_tensor, size, n_copies, cudaMemcpyDeviceToHost, cuda_ctx->stream()));
}

static bool ggml_backend_cuda_cpy_tensor_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, const ggml_tensor * src, ggml_tensor * dst) {
    ggml_backend_buffer_t buf_src = src->view_src ? src->view_src->buffer : src->buffer;
    ggml_backend_buffer_t buf_dst = dst->view_src ? dst->view_src->buffer : dst->buffer;

    if (!ggml_backend_is_cuda(backend_src) || !ggml_backend_is_cuda(backend_dst)) {
        return false;
    }

    if (!ggml_backend_buffer_is_cuda(buf_src) || !ggml_backend_buffer_is_cuda(buf_dst)) {
        return false;
    }

    // device -> device copy
    ggml_backend_cuda_context * cuda_ctx_src = (ggml_backend_cuda_context *) backend_src->context;
    ggml_backend_cuda_context * cuda_ctx_dst = (ggml_backend_cuda_context *) backend_dst->context;

    ggml_backend_cuda_buffer_context * buf_ctx_src = (ggml_backend_cuda_buffer_context *) buf_src->context;
    ggml_backend_cuda_buffer_context * buf_ctx_dst = (ggml_backend_cuda_buffer_context *) buf_dst->context;

    if (cuda_ctx_src->device != buf_ctx_src->device || cuda_ctx_dst->device != buf_ctx_dst->device) {
#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: backend and buffer devices do not match\n", __func__);
#endif // NDEBUG
        return false;
    }

    if (backend_src != backend_dst) {
        // copy on src stream
        // compare the backing physical devices: distinct virtual devices may share one physical GPU,
        // in which case a same-device copy (not a peer copy) is required
        const int src_physical = ggml_cuda_get_physical_device(cuda_ctx_src->device);
        const int dst_physical = ggml_cuda_get_physical_device(cuda_ctx_dst->device);
        if (src_physical == dst_physical) {
            CUDA_CHECK(cudaMemcpyAsync(dst->data, src->data, ggml_nbytes(dst), cudaMemcpyDeviceToDevice, cuda_ctx_src->stream()));
        } else {
#ifdef GGML_CUDA_NO_PEER_COPY
            return false;
#else
            CUDA_CHECK(cudaMemcpyPeerAsync(dst->data, dst_physical, src->data, src_physical, ggml_nbytes(dst), cuda_ctx_src->stream()));
#endif // GGML_CUDA_NO_PEER_COPY
        }

        // record event on src stream after the copy
        if (!cuda_ctx_src->copy_event) {
            ggml_cuda_set_device(cuda_ctx_src->device);
            CUDA_CHECK(cudaEventCreateWithFlags(&cuda_ctx_src->copy_event, cudaEventDisableTiming));
        }

        CUDA_CHECK(cudaEventRecord(cuda_ctx_src->copy_event, cuda_ctx_src->stream()));

        // wait on dst stream for the copy to complete
        CUDA_CHECK(cudaStreamWaitEvent(cuda_ctx_dst->stream(), cuda_ctx_src->copy_event, 0));
    } else {
        // src and dst are on the same backend
        CUDA_CHECK(cudaMemcpyAsync(dst->data, src->data, ggml_nbytes(dst), cudaMemcpyDeviceToDevice, cuda_ctx_src->stream()));
    }
    return true;
}

static void ggml_backend_cuda_synchronize(ggml_backend_t backend) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;

    ggml_cuda_ink2_tl(cuda_ctx, "sync");
    CUDA_CHECK(cudaStreamSynchronize(cuda_ctx->stream()));
    ggml_cuda_ink2_tl(cuda_ctx, "sync_end");

    GGML_UNUSED(backend);
}

static bool ggml_cuda_is_view_or_noop(const ggml_tensor * t) {
    return ggml_is_empty(t) || t->op == GGML_OP_RESHAPE || t->op == GGML_OP_TRANSPOSE ||
           t->op == GGML_OP_VIEW || t->op == GGML_OP_PERMUTE || t->op == GGML_OP_NONE;
}

#ifdef USE_CUDA_GRAPH
static bool ggml_cuda_graph_debug_enabled() {
    static const bool enabled = ggml_cuda_env_flag("GGML_CUDA_GRAPH_DEBUG");
    return enabled;
}

static bool ggml_cuda_graph_check_compability(ggml_cgraph * cgraph) {

    bool use_cuda_graph = true;
    // Loop over nodes in GGML graph to obtain info needed for CUDA graph

    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * node = cgraph->nodes[i];

        if (ggml_cuda_is_view_or_noop(node)) {
            continue;
        }

        // [TAG_MUL_MAT_ID_CUDA_GRAPHS]
        if (node->op == GGML_OP_MUL_MAT_ID) {
            const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
            if (ggml_cuda_mul_mat_id_needs_sync(node, cc)) {
                // the mul_mat_id fallback path synchronizes the stream, so we cannot use CUDA graphs
                // ref: https://github.com/ggml-org/llama.cpp/pull/18958
                use_cuda_graph = false;
                if (ggml_cuda_graph_debug_enabled()) {
                    GGML_LOG_DEBUG("%s: disabling CUDA graphs due to unsupported node type\n", __func__);
                }
            }
        }

        if (!use_cuda_graph) {
            break;
        }
    }

    return use_cuda_graph;
}

// the key identifies both the split (its first node) and the shapes it was called with.
// a captured cuda graph hard-codes the shapes, so a caller that alternates shapes - a
// speculative verify batch, for example - needs a separate instance per shape. with a
// single key per split, every shape change resets the warmup and no graph is ever used.
//
// this stays O(1) on purpose: walking every node undoes the point of a cuda graph, which is
// to not touch per-node data on the hot path. the first and last node carry the batch
// dimension, which is what changes when a verify batch changes size. a shape this does not
// separate just shares an entry and re-captures, exactly as before, so it can only help.
static uint64_t ggml_cuda_graph_get_key(ggml_cgraph * cgraph) {
    if (cgraph->n_nodes == 0) {
        return 0;
    }

    uint64_t key = (uint64_t) (uintptr_t) cgraph->nodes[0];

    auto mix = [&key](uint64_t v) {
        key = (key ^ v) * 0x100000001b3ull;
    };

    mix(cgraph->n_nodes);

    for (int d = 0; d < GGML_MAX_DIMS; d++) {
        mix(cgraph->nodes[0]->ne[d]);
        mix(cgraph->nodes[cgraph->n_nodes - 1]->ne[d]);
    }

    return key;
}

static bool ggml_cuda_graph_update_required(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph) {
    bool res = false;

    const uint64_t graph_key = ggml_cuda_graph_get_key(cgraph);
    ggml_cuda_graph * graph = cuda_ctx->cuda_graph(graph_key);

    if (cgraph->uid != 0 &&
        cgraph->uid == graph->uid) {
        if (ggml_cuda_graph_debug_enabled()) {
            GGML_LOG_DEBUG("CUDA Graph id %zu reused\n", cgraph->uid);
        }
        GGML_ASSERT((int)graph->node_props.size() == cgraph->n_nodes);
        return false;
    }

    graph->uid = cgraph->uid;

    // Check if the graph size has changed
    if ((int)graph->node_props.size() != cgraph->n_nodes) {
        res = true;
        graph->node_props.resize(cgraph->n_nodes);
    }

    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_cuda_graph::node_properties prop = {};
        memcpy(&prop.node, cgraph->nodes[i], sizeof(ggml_tensor));

        for (int j = 0; j < GGML_MAX_SRC; ++j) {
            if (cgraph->nodes[i]->src[j]) {
                prop.node_src_data_ptrs[j] = cgraph->nodes[i]->src[j]->data;
                memcpy(prop.node_src_ne[j], cgraph->nodes[i]->src[j]->ne, sizeof(prop.node_src_ne[j]));
                memcpy(prop.node_src_nb[j], cgraph->nodes[i]->src[j]->nb, sizeof(prop.node_src_nb[j]));
            }
        }

        if (res || memcmp(&graph->node_props[i], &prop, sizeof(prop)) != 0) {
            graph->node_props[i] = prop;
            res = true;
        }
    }

    return res;
}

#ifdef USE_CUDA_GRAPH_CLUSTER_CHECK
static std::vector<ggml_cuda_graph_cluster> ggml_cuda_graph_get_clusters(cudaGraph_t cuda_graph, int device) {
    std::vector<ggml_cuda_graph_cluster> clusters;
    static const bool runtime_affected = [] {
        int version = 0;
        CUDA_CHECK(cudaRuntimeGetVersion(&version));
        // TODO: upper bound once fixed in the CUDA runtime
        return version >= 11080;
    }();
    if (!runtime_affected || ggml_cuda_info().devices[device].cc < GGML_CUDA_CC_HOPPER) {
        return clusters;
    }
    size_t n_nodes = 0;
    CUDA_CHECK(cudaGraphGetNodes(cuda_graph, nullptr, &n_nodes));
    std::vector<cudaGraphNode_t> nodes(n_nodes);
    CUDA_CHECK(cudaGraphGetNodes(cuda_graph, nodes.data(), &n_nodes));
    for (size_t i = 0; i < n_nodes; ++i) {
        cudaGraphNodeType type;
        CUDA_CHECK(cudaGraphNodeGetType(nodes[i], &type));
        if (type != cudaGraphNodeTypeKernel) {
            continue;
        }
        cudaLaunchAttributeValue val = {};
        CUDA_CHECK(cudaGraphKernelNodeGetAttribute(nodes[i], cudaLaunchAttributeClusterDimension, &val));
        if (val.clusterDim.x * val.clusterDim.y * val.clusterDim.z > 1) {
            clusters.push_back({ i, val.clusterDim.x, val.clusterDim.y, val.clusterDim.z });
        }
    }
    return clusters;
}
#endif // USE_CUDA_GRAPH_CLUSTER_CHECK

static void ggml_cuda_graph_update_executable(ggml_cuda_graph_instance & instance, cudaGraph_t cuda_graph, int device) {
#ifdef USE_CUDA_GRAPH_CLUSTER_CHECK
    std::vector<ggml_cuda_graph_cluster> clusters = ggml_cuda_graph_get_clusters(cuda_graph, device);
    if (instance.clusters != clusters) {
        if (ggml_cuda_graph_debug_enabled()) {
            GGML_LOG_DEBUG("%s: cluster dims changed, re-instantiating CUDA graph\n", __func__);
        }
        CUDA_CHECK(cudaGraphExecDestroy(instance));
        CUDA_CHECK(cudaGraphInstantiate(&instance.handle, cuda_graph, nullptr, nullptr, 0));
        instance.clusters = std::move(clusters);
        return;
    }
#else
    GGML_UNUSED(device);
#endif // USE_CUDA_GRAPH_CLUSTER_CHECK

#if CUDART_VERSION >= 12000
    cudaGraphExecUpdateResultInfo result_info;
    cudaError_t stat = cudaGraphExecUpdate(instance, cuda_graph, &result_info);
#else
    cudaGraphNode_t errorNode;
    cudaGraphExecUpdateResult result_info;
    cudaError_t stat = cudaGraphExecUpdate(instance, cuda_graph, &errorNode, &result_info);
#endif // CUDART_VERSION >= 12000

    if (stat == cudaErrorGraphExecUpdateFailure) {
        if (ggml_cuda_graph_debug_enabled()) {
            GGML_LOG_DEBUG("%s: CUDA graph update failed\n", __func__);
        }

        // The pre-existing graph exec cannot be updated due to violated constraints
        // so instead clear error and re-instantiate
        (void)cudaGetLastError();
        CUDA_CHECK(cudaGraphExecDestroy(instance));
        instance.handle = nullptr;
        CUDA_CHECK(cudaGraphInstantiate(&instance.handle, cuda_graph, nullptr, nullptr, 0));
    } else {
        GGML_ASSERT(stat == cudaSuccess);
    }
}

// number of ggml nodes per CUDA graph chunk, 0 captures the whole ggml graph as one CUDA graph
static int ggml_cuda_graph_chunk_size() {
    static const int chunk_size = [] {
        const char * env = getenv("GGML_CUDA_GRAPH_CHUNK_SIZE");
        return env != nullptr ? atoi(env) : 512;
    }();
    return chunk_size;
}

// end the capture of chunk `idx`, update or create its executable and launch it
static void ggml_cuda_graph_launch_chunk(ggml_backend_cuda_context * cuda_ctx, ggml_cuda_graph * graph, size_t idx) {
    cudaGraph_t cuda_graph = nullptr;
    CUDA_CHECK(cudaStreamEndCapture(cuda_ctx->stream(), &cuda_graph));

    if (idx < graph->instances.size()) {
        ggml_cuda_graph_update_executable(graph->instances[idx], cuda_graph, cuda_ctx->device);
    } else {
        GGML_ASSERT(idx == graph->instances.size());
        graph->instances.emplace_back();
        CUDA_CHECK(cudaGraphInstantiate(&graph->instances[idx].handle, cuda_graph, nullptr, nullptr, 0));
#ifdef USE_CUDA_GRAPH_CLUSTER_CHECK
        graph->instances[idx].clusters = ggml_cuda_graph_get_clusters(cuda_graph, cuda_ctx->device);
#endif // USE_CUDA_GRAPH_CLUSTER_CHECK
    }
    CUDA_CHECK(cudaGraphDestroy(cuda_graph));

    CUDA_CHECK(cudaGraphLaunch(graph->instances[idx], cuda_ctx->stream()));
}
#endif // USE_CUDA_GRAPH

static bool ggml_cuda_should_fuse_rope_set_rows(const ggml_tensor * rope,
                                                const ggml_tensor * view,
                                                const ggml_tensor * set_rows) {

    if (rope->op != GGML_OP_ROPE || view->op != GGML_OP_VIEW || set_rows->op != GGML_OP_SET_ROWS) {
        return false;
    }
    // ne3 not tested
    if (rope->src[0]->ne[3] != 1) {
        return false;
    }

    if (set_rows->type != GGML_TYPE_F32 && set_rows->type != GGML_TYPE_F16 && set_rows->type != GGML_TYPE_BF16) {
        return false;
    }

    // BF16 rope fuses into an F16 or BF16 cache, other ropes not into a BF16 cache
    if (rope->type == GGML_TYPE_BF16 ? set_rows->type == GGML_TYPE_F32 : set_rows->type == GGML_TYPE_BF16) {
        return false;
    }

    if (set_rows->src[1]->type != GGML_TYPE_I64) {
        return false;
    }

    // The view should flatten two dims of rope into one dim
    if (!ggml_is_contiguous(view) || view->ne[0] != rope->ne[0] * rope->ne[1]) {
        return false;
    }

    // Only norm/neox/multi shaders have the fusion code
    const int mode = ((const int32_t *) rope->op_params)[2];
    if (mode != GGML_ROPE_TYPE_NORMAL && mode != GGML_ROPE_TYPE_NEOX &&
        mode != GGML_ROPE_TYPE_MROPE && mode != GGML_ROPE_TYPE_IMROPE) {
        return false;
    }

    return true;
}

static bool ggml_cuda_should_fuse_rms_norm_mul_rope(const ggml_tensor * rms_norm,
                                                    const ggml_tensor * mul,
                                                    const ggml_tensor * rope) {
    if (rms_norm->op != GGML_OP_RMS_NORM || mul->op != GGML_OP_MUL || rope->op != GGML_OP_ROPE) {
        return false;
    }

    if (rms_norm->src[0]->type != GGML_TYPE_F32 || rms_norm->type != GGML_TYPE_F32 ||
        mul->src[0]->type != GGML_TYPE_F32 || mul->src[1]->type != GGML_TYPE_F32 ||
        mul->type != GGML_TYPE_F32 || rope->type != GGML_TYPE_F32) {
        return false;
    }

    if (rope->src[0] != mul) {
        return false;
    }

    //if rms norm is the B operand, then we don't handle broadcast
    if (rms_norm == mul->src[1] && !ggml_are_same_shape(mul->src[0], rms_norm)) {
        return false;
    }

    if (!ggml_are_same_shape(rms_norm, mul)) {
        return false;
    }

    //rms_norm kernel assumes contiguous rows
    if (!ggml_is_contiguous_rows(rms_norm->src[0]) ||
        !ggml_is_contiguous_rows(mul->src[0]) || !ggml_is_contiguous_rows(mul->src[1])) {
        return false;
    }

    // the fused kernel handles the norm/neox rope modes only
    const int mode = ((const int32_t *) rope->op_params)[2];
    if (mode != GGML_ROPE_TYPE_NORMAL && mode != GGML_ROPE_TYPE_NEOX) {
        return false;
    }

    const int n_dims = ((const int32_t *) rope->op_params)[1];
    if (n_dims % 2 != 0 || rope->src[0]->ne[0] % 2 != 0) {
        return false;
    }

    // ggml_rope_set_offset is not yet supported in the fused kernel
    const int n_offs = ((const int32_t *) rope->op_params)[15];
    if (n_offs != 0) {
        return false;
    }

    return true;
}

// match gated_delta_net + the strided cpy that scatters its state snapshots into the cache
// (slot i -> rollback group i, slot 0 newest), so the kernel can write them and skip the cpy.
static int ggml_cuda_try_gdn_cache_fusion(
        const ggml_cgraph * cgraph, int node_idx, ggml_cuda_gated_delta_net_fused_cache & fused_state_cpy) {
    const ggml_tensor * gdn = cgraph->nodes[node_idx];
    // the kernel skips the snapshot tail, so the gdn output must not be a graph output
    if (gdn->op != GGML_OP_GATED_DELTA_NET || gdn->type != GGML_TYPE_F32 ||
        (gdn->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        return 0;
    }

    const ggml_tensor * src_v     = gdn->src[2];
    const int64_t       S_v       = src_v->ne[0];
    const int64_t       H         = src_v->ne[1];
    const int64_t       n_tokens  = src_v->ne[2];
    const int64_t       n_seqs    = src_v->ne[3];
    const int64_t       D         = S_v * S_v * H;
    const int64_t       K         = ggml_get_op_params_i32(gdn, 0); // snapshot slot count
    const int64_t       n_written = std::min<int64_t>(n_tokens, K); // newest n_written slots are written

    // snapshot tail starts right after the attention scores
    const size_t tail_off = ggml_row_size(GGML_TYPE_F32, S_v * H * n_tokens * n_seqs);

    // snapshot cpy is the first real node after the gdn (skip views/no-ops)
    const ggml_tensor * cpy  = nullptr;
    int                 skip = 0;
    for (int j = node_idx + 1; j < cgraph->n_nodes && cpy == nullptr; ++j) {
        const ggml_tensor * n = cgraph->nodes[j];
        if (ggml_cuda_is_view_or_noop(n)) {
            continue;
        }
        if (n->op != GGML_OP_CPY || (n->flags & GGML_TENSOR_FLAG_OUTPUT)) {
            return 0;
        }
        cpy  = n;
        skip = j - node_idx;
    }
    if (cpy == nullptr) {
        return 0;
    }

    const ggml_tensor * src = cpy->src[0]; // view of the gdn snapshot tail
    const ggml_tensor * dst = cpy->src[1]; // cache view the kernel writes to

    // src must be this gdn's snapshot tail (contiguous, at the tail offset)
    if (src->op != GGML_OP_VIEW || src->view_src != gdn || src->view_offs != tail_off ||
        !ggml_is_contiguous(src)) {
        return 0;
    }

    // dst is the [D, n_seqs, n_written] cache view; require nb[1] == D (the per-seq stride the kernel
    // assumes). ggml_cpy pins src to the same element count.
    const std::array<int64_t, GGML_MAX_DIMS> expected_ne = { D, n_seqs, n_written, 1 };
    if (dst->op != GGML_OP_VIEW || dst->type != GGML_TYPE_F32 || dst->data == nullptr ||
        !std::equal(expected_ne.begin(), expected_ne.end(), dst->ne) ||
        dst->nb[0] != ggml_type_size(GGML_TYPE_F32) || dst->nb[1] != (size_t) ggml_row_size(GGML_TYPE_F32, D)) {
        return 0;
    }

    fused_state_cpy.data        = (float *) dst->data; // rollback group 0 (newest)
    fused_state_cpy.slot_stride = K > 1 ? (int64_t) (dst->nb[2] / sizeof(float)) : 0;
    return skip;
}

// match ssm_scan + the strided cpy that scatters its state snapshots into the cache, so the kernel writes them and skips the cpy
static int ggml_cuda_try_ssm_scan_cache_fusion(
        const ggml_cgraph * cgraph, int node_idx, ggml_cuda_ssm_scan_fused_cache & fused_state_cpy) {
    const ggml_tensor * ssm = cgraph->nodes[node_idx];
    // the kernel skips the snapshot tail, so the scan output must not be a graph output
    if (ssm->op != GGML_OP_SSM_SCAN || ssm->type != GGML_TYPE_F32 || (ssm->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        return 0;
    }

    const int64_t K = ggml_get_op_params_i32(ssm, 0); // snapshot slot count

    const ggml_tensor * s = ssm->src[0];
    const ggml_tensor * x = ssm->src[1];
    const ggml_tensor * A = ssm->src[3];

    const int64_t d_state = s->ne[0];
    const int64_t D       = d_state * s->ne[1] * x->ne[1]; // d_state * head_dim * n_head
    const int64_t n_tok   = x->ne[2];
    const int64_t n_seqs  = x->ne[3];

    // only the mamba-2 kernels (group scan and SSD) write to the cache; mamba-1 still uses the cpy
    if (A->nb[1] != sizeof(float) || (d_state != 96 && d_state != 128 && d_state != 256)) {
        return 0;
    }

    // the scan reads its input rows from the cache (picked by ids), so with more than one seq a seq can read a row that another seq writes in the same launch
    if (n_seqs != 1) {
        return 0;
    }

    const int64_t n_written = std::min<int64_t>(n_tok, K);
    const size_t  tail_off  = ggml_row_size(GGML_TYPE_F32, ggml_nelements(x));

    // snapshot cpy is the first real node after the scan (skip views/no-ops)
    const ggml_tensor * cpy  = nullptr;
    int                 skip = 0;
    for (int j = node_idx + 1; j < cgraph->n_nodes && cpy == nullptr; ++j) {
        const ggml_tensor * n = cgraph->nodes[j];
        if (ggml_cuda_is_view_or_noop(n)) {
            continue;
        }
        if (n->op != GGML_OP_CPY || (n->flags & GGML_TENSOR_FLAG_OUTPUT)) {
            return 0;
        }
        cpy  = n;
        skip = j - node_idx;
    }
    if (cpy == nullptr) {
        return 0;
    }

    const ggml_tensor * src = cpy->src[0]; // view of the scan snapshot tail
    const ggml_tensor * dst = cpy->src[1]; // cache view the kernel writes to

    // src must be this scan's snapshot tail (contiguous, at the tail offset)
    if (src->op != GGML_OP_VIEW || src->view_src != ssm || src->view_offs != tail_off ||
        !ggml_is_contiguous(src)) {
        return 0;
    }

    // dst is the [D, n_seqs, n_written] cache view; require nb[1] == D, the per-seq stride the kernel takes from src0->nb[3]
    const std::array<int64_t, GGML_MAX_DIMS> expected_ne = { D, n_seqs, n_written, 1 };
    if (dst->op != GGML_OP_VIEW || dst->type != GGML_TYPE_F32 || dst->data == nullptr ||
        !std::equal(expected_ne.begin(), expected_ne.end(), dst->ne) ||
        dst->nb[0] != ggml_type_size(GGML_TYPE_F32) || dst->nb[1] != (size_t) ggml_row_size(GGML_TYPE_F32, D)) {
        return 0;
    }

    fused_state_cpy.data        = (float *) dst->data; // rollback slot 0 (newest)
    fused_state_cpy.slot_stride = K > 1 ? (int64_t) (dst->nb[2] / sizeof(float)) : 0;
    return skip;
}

// expert group selection from build_moe_ffn (n_expert_groups > 1)
// src0/src1 index into this table, -1 is the (biased) probs, -2 is not checked
static const struct {
    ggml_op op;
    int     src0;
    int     src1;
} topk_moe_group_pattern[] = {
    { GGML_OP_RESHAPE,  -1, -2 }, //  0: [n_exp_per_group, n_expert_groups, n_tokens]
    { GGML_OP_RESHAPE,   0, -2 }, //  1: [1, n_exp_per_group, n_expert_groups, n_tokens]
    { GGML_OP_ARGSORT,   0, -2 }, //  2
    { GGML_OP_VIEW,      2, -2 }, //  3: top-2 per group
    { GGML_OP_GET_ROWS,  1,  3 }, //  4
    { GGML_OP_RESHAPE,   4, -2 }, //  5
    { GGML_OP_SUM_ROWS,  5, -2 }, //  6: group scores
    { GGML_OP_RESHAPE,   6, -2 }, //  7
    { GGML_OP_ARGSORT,   7, -2 }, //  8
    { GGML_OP_VIEW,      8, -2 }, //  9: top n_group_used groups
    { GGML_OP_GET_ROWS,  0,  9 }, // 10
    { GGML_OP_FILL,      0, -2 }, // 11
    { GGML_OP_SET_ROWS, 10,  9 }, // 12: unused groups -> -inf
    { GGML_OP_RESHAPE,  12, -2 }, // 13: [n_expert, n_tokens]
};

static bool ggml_cuda_topk_moe_group_fusion(const struct ggml_cgraph * cgraph, int node_idx, const ggml_tensor * selection, ggml_cuda_topk_moe_args & args) {
    const int n_ops = (int) std::size(topk_moe_group_pattern);
    if (node_idx + n_ops > cgraph->n_nodes) {
        return false;
    }

    ggml_tensor ** nodes = cgraph->nodes + node_idx;
    auto src_ok = [&](const ggml_tensor * src, int idx) {
        return idx == -2 || src == (idx == -1 ? selection : nodes[idx]);
    };

    for (int i = 0; i < n_ops; i++) {
        const auto & p = topk_moe_group_pattern[i];
        if (nodes[i]->op != p.op || !src_ok(nodes[i]->src[0], p.src0) || !src_ok(nodes[i]->src[1], p.src1)) {
            return false;
        }
    }

    const int n_groups     = nodes[0]->ne[1];
    const int n_group_used = nodes[9]->ne[0];

    if (nodes[3]->ne[0] != 2 || nodes[12]->src[2] != nodes[11] || ggml_get_op_params_f32(nodes[11], 0) != -INFINITY ||
            n_groups > WARP_SIZE) {
        return false;
    }

    args.n_groups     = n_groups;
    args.n_group_used = n_group_used;
    return true;
}

static bool ggml_cuda_topk_moe_fusion(const struct ggml_cgraph * cgraph, int node_idx, ggml_cuda_topk_moe_args & args) {
    args.sigmoid         = false;
    args.sqrt_softplus   = false;
    args.softmax         = false;
    args.delayed_softmax = false;
    args.prob_bias       = false;
    args.norm            = false;
    args.n_groups        = 0;
    args.n_group_used    = 0;

    const int      n_nodes = cgraph->n_nodes;
    ggml_tensor ** nodes   = cgraph->nodes;

    if (nodes[node_idx]->op == GGML_OP_SOFT_MAX) {
        args.softmax = true;
    }

    if (nodes[node_idx]->op == GGML_OP_UNARY) {
        const ggml_unary_op unary_op = ggml_get_unary_op(nodes[node_idx]);
        if (unary_op == GGML_UNARY_OP_SIGMOID) {
            args.sigmoid = true;
        } else if (unary_op == GGML_UNARY_OP_SOFTPLUS && node_idx + 1 < n_nodes &&
                   nodes[node_idx + 1]->op == GGML_OP_SQRT && nodes[node_idx + 1]->src[0] == nodes[node_idx]) {
            // sqrt(softplus(x)) scoring (DeepSeek-V4)
            args.sqrt_softplus = true;
            node_idx++;
        } else {
            return false;
        }
    }

    if (nodes[node_idx]->op == GGML_OP_ARGSORT) {
        args.delayed_softmax = true;
    }

    node_idx++;

    if (args.sigmoid || args.sqrt_softplus || args.softmax) {
        // SOFTMAX -> RESHAPE
        if (node_idx >= n_nodes || nodes[node_idx]->op != GGML_OP_RESHAPE ||
                nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
            return false;
        }
        ggml_tensor * probs_reshaped = nodes[node_idx];
        node_idx++;

        if (node_idx >= n_nodes) {
            return false;
        }

        // src of bias add is the unreshaped probs (-2 instead of -1)
        const ggml_tensor * selection = nodes[node_idx - 2];
        if (nodes[node_idx]->op == GGML_OP_ADD && nodes[node_idx]->src[0] == selection) {
            args.prob_bias = true;
            selection = nodes[node_idx];
            node_idx++;
        }

        if (ggml_cuda_topk_moe_group_fusion(cgraph, node_idx, selection, args)) {
            node_idx += (int) std::size(topk_moe_group_pattern);
            selection = nodes[node_idx - 1];
        }

        // RESHAPE/ADD/group mask -> ARGSORT
        if (node_idx >= n_nodes || nodes[node_idx]->op != GGML_OP_ARGSORT || nodes[node_idx]->src[0] != selection) {
            return false;
        }

        node_idx++;

        // ARGSORT-> VIEW
        if (node_idx >= n_nodes || nodes[node_idx]->op != GGML_OP_VIEW ||
                nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
            return false;
        }

        // kept groups must hold all used experts
        if (args.n_groups > 0 && nodes[node_idx]->ne[0] > args.n_group_used * (probs_reshaped->ne[1] / args.n_groups)) {
            return false;
        }
        node_idx++;

        if (node_idx >= n_nodes || nodes[node_idx]->op != GGML_OP_GET_ROWS) {
            return false;
        }

        // GET_ROWS
        if (nodes[node_idx]->src[0] != probs_reshaped || nodes[node_idx]->src[1] != nodes[node_idx - 1]) {
            return false;
        }
        node_idx++;
    } else if (args.delayed_softmax) {
        if (node_idx - 2 < 0) {
            return false;
        }
        ggml_tensor * probs_reshaped = nodes[node_idx - 2];

        // VIEW->ARGSORT
        if (node_idx >= n_nodes || nodes[node_idx]->op != GGML_OP_VIEW ||
            nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
            return false;
        }
        node_idx++;

        // GET_ROWS
        if (node_idx >= n_nodes || nodes[node_idx]->src[1] != nodes[node_idx - 1] ||
                nodes[node_idx]->src[0] != probs_reshaped) {
            return false;
        }
        node_idx++;

        static const std::vector<ggml_op> remaining_ops = { GGML_OP_RESHAPE, GGML_OP_SOFT_MAX, GGML_OP_RESHAPE };

        for (const ggml_op op : remaining_ops) {
            if (node_idx >= n_nodes || nodes[node_idx]->op != op || nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
                return false;
            }
            node_idx++;
        }
    }

    // At this point we can check for norm + scale. Everything is now at least valid till the norm
    if (node_idx >= n_nodes) {
        return true;
    }

    if (nodes[node_idx]->op == GGML_OP_RESHAPE) {
        //check RESHAPE->SUM_ROWS->CLAMP->DIV->RESHAPE
        static const std::vector<ggml_op> norm_ops = { GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_CLAMP };

        args.norm = true;
        for (const ggml_op op : norm_ops) {
            if (nodes[node_idx]->op == op && nodes[node_idx]->src[0] == nodes[node_idx - 1]) {
                node_idx++;
            } else {
                args.norm = false;
                return true;
            }
        }

        // DIV <- CLAMP, RESHAPE
        if (nodes[node_idx]->op != GGML_OP_DIV || nodes[node_idx]->src[1] != nodes[node_idx - 1] ||
            nodes[node_idx]->src[0] != nodes[node_idx - 3]) {
            args.norm = false;
            return true;
        }
        node_idx++;

        if (nodes[node_idx]->op != GGML_OP_RESHAPE || nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
            args.norm = false;
            return true;
        }

        node_idx++;
    }

    if (nodes[node_idx]->op == GGML_OP_SCALE && nodes[node_idx]->src[0] == nodes[node_idx - 1]) {
        args.scale = true;
    }

    return true;
}

// returns whether the write (out) nodes overwrite the read nodes in operation
static bool ggml_cuda_check_fusion_memory_ranges(const ggml_cgraph * cgraph,
                                                 const int           node_idx,
                                                 const int           node_count,
                                                 const int *         out_nodes,
                                                 const int           out_count,
                                                 const bool          is_topk_moe = false) {
    auto nodes_overlap = [&](const ggml_tensor * a, const ggml_tensor * b) {
        const int64_t a_start = (int64_t) a->data;
        const int64_t a_end   = a_start + ggml_backend_buft_get_alloc_size(a->buffer->buft, a);

        const int64_t b_start = (int64_t) b->data;
        const int64_t b_end   = b_start + ggml_backend_buft_get_alloc_size(b->buffer->buft, b);

        if ((b_start <= a_start && a_start < b_end) || (a_start <= b_start && b_start < a_end)) {
            return true;
        }

        return false;
    };

    bool is_ok = true;
    // one block reads all logits before it writes, so logits may alias the out nodes
    const ggml_tensor * logits_may_alias = nullptr;
    if (is_topk_moe && ggml_nrows(cgraph->nodes[node_idx]) <= TOPK_MOE_ROWS_PER_BLOCK) {
        logits_may_alias = cgraph->nodes[node_idx]->src[0];
    }

    for (int i = 0; i < out_count; ++i) {
        const ggml_tensor * dst = cgraph->nodes[out_nodes[i]];

        for (int j = node_idx; j < node_idx + node_count; ++j) {
            // Loop over all srcs of all nodes in the fusion. If the src overlaps
            // the destination and the src is not an intermediate node that's being
            // elided, then disable fusion.

            for (int src_idx = 0; src_idx < GGML_MAX_SRC; ++src_idx) {
                const ggml_tensor * src = cgraph->nodes[j]->src[src_idx];

                if (!src || src->op == GGML_OP_NONE || src == logits_may_alias) {
                    continue;
                }

                if (nodes_overlap(dst, src)) {
                    bool found = false;

                    for (int k = node_idx; k < j; ++k) {
                        if (cgraph->nodes[k] == src) {
                            found = true;
                            break;
                        }
                    }

                    if (!found) {
                        is_ok = false;
                        break;
                    }
                }
            }
        }
    }

    return is_ok;
}

// The long form spans 2*k + 1 nodes. ggml_can_fuse_subgraph() accepts at most
// 31 nodes, so k <= 15; larger values use the per-operation path.
static constexpr int MOE_WEIGHTED_REDUCTION_MAX_EXPERTS = 15;

struct ggml_cuda_moe_weighted_reduction_match {
    const ggml_tensor * experts      = nullptr;
    const ggml_tensor * expert_scale = nullptr;
    const ggml_tensor * weights      = nullptr;
    ggml_tensor *       dst          = nullptr;
    int                 node_count   = 0;
};

static bool ggml_cuda_match_moe_weighted_reduction(
        const ggml_cgraph * cgraph,
        int node_idx,
        ggml_cuda_moe_weighted_reduction_match & match) {
    const ggml_tensor * first = cgraph->nodes[node_idx];
    if (first->op != GGML_OP_MUL || first->type != GGML_TYPE_F32 || !ggml_is_contiguous(first)) {
        return false;
    }

    auto split_mul = [](const ggml_tensor * mul, const ggml_tensor *& full, const ggml_tensor *& broadcast) {
        auto is_weights = [mul](const ggml_tensor * tensor) {
            return tensor && tensor->type == GGML_TYPE_F32 && ggml_is_contiguous(tensor) && tensor->ne[0] == 1 &&
                tensor->ne[1] == mul->ne[1] && tensor->ne[2] == mul->ne[2] && tensor->ne[3] == mul->ne[3];
        };
        auto is_experts = [mul](const ggml_tensor * tensor) {
            return tensor && tensor->type == GGML_TYPE_F32 && ggml_is_contiguous(tensor) &&
                ggml_are_same_shape(tensor, mul);
        };

        if (is_experts(mul->src[0]) && is_weights(mul->src[1])) {
            full      = mul->src[0];
            broadcast = mul->src[1];
            return true;
        }
        if (is_experts(mul->src[1]) && is_weights(mul->src[0])) {
            full      = mul->src[1];
            broadcast = mul->src[0];
            return true;
        }
        return false;
    };

    const ggml_tensor * weighted     = first;
    const ggml_tensor * experts      = nullptr;
    const ggml_tensor * expert_scale = nullptr;
    const ggml_tensor * weights      = nullptr;
    int                 mul_count    = 1;

    // Match both structural forms:
    //   (experts * expert_scale) * router_weight
    //   experts * router_weight
    // The matcher does not depend on the model or quantization type.
    if (node_idx + 1 < cgraph->n_nodes) {
        const ggml_tensor * second = cgraph->nodes[node_idx + 1];
        const ggml_tensor * scaled = nullptr;
        const ggml_tensor * route  = nullptr;
        const ggml_tensor * raw    = nullptr;
        const ggml_tensor * scale  = nullptr;
        if (second->op == GGML_OP_MUL && second->type == GGML_TYPE_F32 && ggml_is_contiguous(second) &&
                split_mul(second, scaled, route) && scaled == first && split_mul(first, raw, scale)) {
            weighted     = second;
            experts      = raw;
            expert_scale = scale;
            weights      = route;
            mul_count    = 2;
        }
    }

    if (experts == nullptr && !split_mul(first, experts, weights)) {
        return false;
    }

    const int     n_expert_used = (int) weighted->ne[1];
    const int64_t n_tokens      = weighted->ne[2] * weighted->ne[3];
    if (n_expert_used < 2 || n_expert_used > MOE_WEIGHTED_REDUCTION_MAX_EXPERTS) {
        return false;
    }

    const int node_count = 2 * n_expert_used + mul_count - 1;
    if (node_idx + node_count > cgraph->n_nodes) {
        return false;
    }

    std::vector<ggml_op> ops(node_count, GGML_OP_VIEW);
    ops[0] = GGML_OP_MUL;
    if (mul_count == 2) {
        ops[1] = GGML_OP_MUL;
    }
    std::vector<const ggml_tensor *> views;
    views.reserve(n_expert_used);
    const ggml_tensor * previous = nullptr;
    int n_adds = 0;
    for (int offset = mul_count; offset < node_count; ++offset) {
        const ggml_tensor * candidate = cgraph->nodes[node_idx + offset];
        ops[offset] = candidate->op;

        if (candidate->op == GGML_OP_VIEW) {
            const int expert = (int) views.size();
            if (expert >= n_expert_used || candidate->src[0] != weighted || candidate->view_src != weighted ||
                    candidate->type != GGML_TYPE_F32 || candidate->ne[0] != weighted->ne[0] ||
                    candidate->ne[1] != n_tokens || candidate->ne[2] != 1 || candidate->ne[3] != 1 ||
                    candidate->nb[0] != weighted->nb[0] || candidate->nb[1] != weighted->nb[2] ||
                    candidate->view_offs != (size_t) expert * weighted->nb[1]) {
                return false;
            }
            views.push_back(candidate);
            continue;
        }

        if (candidate->op != GGML_OP_ADD || views.size() < 2 || n_adds + 1 >= (int) views.size()) {
            return false;
        }
        const ggml_tensor * lhs = n_adds == 0 ? views[0] : previous;
        const ggml_tensor * rhs = views[n_adds + 1];
        if (candidate->src[0] != lhs || candidate->src[1] != rhs || candidate->type != GGML_TYPE_F32) {
            return false;
        }
        previous = candidate;
        ++n_adds;
    }

    if ((int) views.size() != n_expert_used || n_adds != n_expert_used - 1 || previous == nullptr) {
        return false;
    }
    if (!ggml_is_contiguous(previous) || previous->ne[0] != weighted->ne[0] ||
            previous->ne[1] != n_tokens || previous->ne[2] != 1 || previous->ne[3] != 1) {
        return false;
    }

    const int output_idx = node_idx + node_count - 1;
    if (!ggml_can_fuse_subgraph(cgraph, node_idx, node_count, ops.data(), &output_idx, 1)) {
        return false;
    }

    match.experts      = experts;
    match.expert_scale = expert_scale;
    match.weights      = weights;
    match.dst          = cgraph->nodes[output_idx];
    match.node_count   = node_count;
    return true;
}


static bool ggml_cuda_can_fuse(const struct ggml_cgraph *                cgraph,
                               int                                       node_idx,
                               std::initializer_list<enum ggml_op>       ops,
                               std::initializer_list<enum ggml_unary_op> unary_ops) {
#ifndef NDEBUG
    const size_t num_unary = std::count(ops.begin(), ops.end(), GGML_OP_UNARY);
    GGML_ASSERT(unary_ops.size() == num_unary);
#endif

    const auto is_equal = [](const std::initializer_list<enum ggml_op> & list1,
                             const std::initializer_list<enum ggml_op> & list2) {
        return std::equal(list1.begin(), list1.end(), list2.begin(), list2.end());
    };

    std::initializer_list<enum ggml_op> mul_mat_bias_glu_ops    = { GGML_OP_MUL_MAT,    GGML_OP_ADD,    GGML_OP_MUL_MAT,    GGML_OP_ADD,    GGML_OP_GLU };
    std::initializer_list<enum ggml_op> mul_mat_id_bias_glu_ops = { GGML_OP_MUL_MAT_ID, GGML_OP_ADD_ID, GGML_OP_MUL_MAT_ID, GGML_OP_ADD_ID, GGML_OP_GLU };

    std::initializer_list<enum ggml_op> mul_mat_id_glu_ops = { GGML_OP_MUL_MAT_ID, GGML_OP_MUL_MAT_ID, GGML_OP_GLU };
    std::initializer_list<enum ggml_op> mul_mat_glu_ops    = { GGML_OP_MUL_MAT,    GGML_OP_MUL_MAT,    GGML_OP_GLU };

    if ((is_equal(mul_mat_bias_glu_ops, ops) || is_equal(mul_mat_id_bias_glu_ops, ops)) &&
        ggml_can_fuse_subgraph(cgraph, node_idx, ops, { node_idx + 4 })) {
        const ggml_tensor * ffn_gate      = cgraph->nodes[node_idx];
        const ggml_tensor * ffn_gate_bias = cgraph->nodes[node_idx + 1];
        const ggml_tensor * ffn_up        = cgraph->nodes[node_idx + 2];
        const ggml_tensor * ffn_up_bias   = cgraph->nodes[node_idx + 3];
        const ggml_tensor * glu           = cgraph->nodes[node_idx + 4];

        if (ggml_cuda_should_fuse_mul_mat(ffn_up, ffn_gate, glu, ffn_up_bias, ffn_gate_bias)) {
            int out_nodes[] = { node_idx + 4 };
            return ggml_cuda_check_fusion_memory_ranges(cgraph, node_idx, (int)ops.size(), out_nodes, 1);
        }
    }

    if ((is_equal(mul_mat_id_glu_ops, ops) || is_equal(mul_mat_glu_ops, ops)) &&
        ggml_can_fuse_subgraph(cgraph, node_idx, ops, { node_idx + 2 })) {
        const ggml_tensor * ffn_gate = cgraph->nodes[node_idx];
        const ggml_tensor * ffn_up   = cgraph->nodes[node_idx + 1];
        const ggml_tensor * glu      = cgraph->nodes[node_idx + 2];

        if (ggml_cuda_should_fuse_mul_mat(ffn_up, ffn_gate, glu)) {
            int out_nodes[] = { node_idx + 2 };
            return ggml_cuda_check_fusion_memory_ranges(cgraph, node_idx, (int)ops.size(), out_nodes, 1);
        }
    }

    std::initializer_list<enum ggml_op> rms_norm_mul_rope_ops          = { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ROPE };
    std::initializer_list<enum ggml_op> rms_norm_mul_rope_set_rows_ops = { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ROPE, GGML_OP_VIEW, GGML_OP_SET_ROWS };

    if (is_equal(rms_norm_mul_rope_set_rows_ops, ops) && ggml_can_fuse_subgraph(cgraph, node_idx, ops, { node_idx + 4 })) {
        const ggml_tensor * rms_norm = cgraph->nodes[node_idx];
        const ggml_tensor * mul      = cgraph->nodes[node_idx + 1];
        const ggml_tensor * rope     = cgraph->nodes[node_idx + 2];
        const ggml_tensor * view     = cgraph->nodes[node_idx + 3];
        const ggml_tensor * set_rows = cgraph->nodes[node_idx + 4];

        if (ggml_check_edges(cgraph, node_idx, {{1, 0, 0}, {2, 0, 1}, {3, 0, 2}, {4, 0, 3}}) &&
            ggml_cuda_should_fuse_rms_norm_mul_rope(rms_norm, mul, rope) &&
            ggml_cuda_should_fuse_rope_set_rows(rope, view, set_rows)) {
            int out_nodes[] = { node_idx + 4 };
            return ggml_cuda_check_fusion_memory_ranges(cgraph, node_idx, (int)ops.size(), out_nodes, 1);
        }
    }

    if (is_equal(rms_norm_mul_rope_ops, ops) && ggml_can_fuse(cgraph, node_idx, ops)) {
        const ggml_tensor * rms_norm = cgraph->nodes[node_idx];
        const ggml_tensor * mul      = cgraph->nodes[node_idx + 1];
        const ggml_tensor * rope     = cgraph->nodes[node_idx + 2];

        if (ggml_cuda_should_fuse_rms_norm_mul_rope(rms_norm, mul, rope)) {
            int out_nodes[] = { node_idx + 2 };
            return ggml_cuda_check_fusion_memory_ranges(cgraph, node_idx, (int)ops.size(), out_nodes, 1);
        }
        return false;
    }

    std::initializer_list<enum ggml_op> rope_set_rows_ops = { GGML_OP_ROPE, GGML_OP_VIEW, GGML_OP_SET_ROWS };

    if (is_equal(rope_set_rows_ops, ops) && ggml_can_fuse_subgraph(cgraph, node_idx, ops, { node_idx + 2 })) {
        const ggml_tensor * rope     = cgraph->nodes[node_idx];
        const ggml_tensor * view     = cgraph->nodes[node_idx + 1];
        const ggml_tensor * set_rows = cgraph->nodes[node_idx + 2];

        if (ggml_cuda_should_fuse_rope_set_rows(rope, view, set_rows)) {
            int out_nodes[] = { node_idx + 2 };
            return ggml_cuda_check_fusion_memory_ranges(cgraph, node_idx, (int)ops.size(), out_nodes, 1);
        }
    }

    // scale_out*sigmoid(scale_in*post) as the hc_post weights (qwen4exp hc_combine)
    std::initializer_list<enum ggml_op> hc_post_gate_ops = { GGML_OP_SCALE, GGML_OP_UNARY, GGML_OP_SCALE, GGML_OP_DSV4_HC_POST };

    if (is_equal(hc_post_gate_ops, ops) && ggml_can_fuse_subgraph(cgraph, node_idx, ops, { node_idx + 3 })) {
        const ggml_tensor * scale_in  = cgraph->nodes[node_idx];
        const ggml_tensor * sigmoid   = cgraph->nodes[node_idx + 1];
        const ggml_tensor * scale_out = cgraph->nodes[node_idx + 2];

        // the chain must feed hc_post->src[2]; a bias on either scale is not handled
        if (ggml_check_edges(cgraph, node_idx, {{1, 0, 0}, {2, 0, 1}, {3, 2, 2}}) &&
            ggml_get_unary_op(sigmoid) == GGML_UNARY_OP_SIGMOID &&
            ggml_get_op_params_f32(scale_in, 1) == 0.0f && ggml_get_op_params_f32(scale_out, 1) == 0.0f &&
            scale_in->src[0]->type == GGML_TYPE_F32 && scale_out->type == GGML_TYPE_F32 &&
            ggml_are_same_shape(scale_in->src[0], scale_out)) {
            int out_nodes[] = { node_idx + 3 };
            return ggml_cuda_check_fusion_memory_ranges(cgraph, node_idx, (int)ops.size(), out_nodes, 1);
        }
        return false;
    }

    if (!ggml_can_fuse(cgraph, node_idx, ops)) {
        return false;
    }

    if ((ops.size() == 2 || ops.size() == 3) && ops.begin()[0] == GGML_OP_NORM && ops.begin()[1] == GGML_OP_MUL) {
        const ggml_tensor * norm   = cgraph->nodes[node_idx];
        const ggml_tensor * mul    = cgraph->nodes[node_idx + 1];
        const ggml_tensor * weight = mul->src[0] == norm ? mul->src[1] : mul->src[0];
        if (norm->src[0]->type != GGML_TYPE_F32 || norm->type != GGML_TYPE_F32 || mul->type != GGML_TYPE_F32 ||
            weight->type != GGML_TYPE_F32 || !ggml_is_vector(weight) || weight->ne[0] != norm->ne[0] ||
            !ggml_is_contiguous(weight)) {
            return false;
        }
        if (ops.size() == 3) {
            const ggml_tensor * add  = cgraph->nodes[node_idx + 2];
            const ggml_tensor * bias = add->src[0] == mul ? add->src[1] : add->src[0];
            if (ops.begin()[2] != GGML_OP_ADD || add->type != GGML_TYPE_F32 || bias->type != GGML_TYPE_F32 ||
                !ggml_is_vector(bias) || bias->ne[0] != norm->ne[0] || !ggml_is_contiguous(bias)) {
                return false;
            }
        }
        return true;
    }

    if ((ops.size() == 2 || ops.size() == 3) && ops.begin()[0] == GGML_OP_RMS_NORM && ops.begin()[1] == GGML_OP_MUL) {
        const ggml_tensor *rms_norm = cgraph->nodes[node_idx];
        const ggml_tensor *mul      = cgraph->nodes[node_idx+1];
        const ggml_tensor *add      = nullptr;

        if (ops.size() == 3 && ops.begin()[2] == GGML_OP_ADD) {
            add = cgraph->nodes[node_idx+2];
        }

        // the mul operand is a norm weight, which model files store as F32;
        // the add operand is an F32 bias or a residual with the type of x
        const ggml_type type = rms_norm->type;
        if ((type != GGML_TYPE_F32 && type != GGML_TYPE_BF16) || rms_norm->src[0]->type != type) {
            return false;
        }

        const ggml_tensor * mul_src = mul->src[0] == rms_norm ? mul->src[1] : mul->src[0];
        if (mul_src->type != GGML_TYPE_F32 || mul->type != type) {
            return false;
        }

        if (add) {
            const ggml_tensor * add_src = add->src[0] == mul ? add->src[1] : add->src[0];
            if ((add_src->type != GGML_TYPE_F32 && add_src->type != type) || add->type != type) {
                return false;
            }
        }

        //if rms norm is the B operand, then we don't handle broadcast
        if (rms_norm == mul->src[1] && !ggml_are_same_shape(mul->src[0], rms_norm)) {
            return false;
        }

        //rms_norm kernel assumes contiguous rows
        if (!ggml_is_contiguous_rows(mul->src[0]) || !ggml_is_contiguous_rows(mul->src[1])) {
            return false;
        }

        if (add && (!ggml_is_contiguous(add->src[0]) || !ggml_is_contiguous_rows(add->src[1]))) {
            return false;
        }

        return true;
    }

    if (ops.size() == 2 && ops.begin()[0] == GGML_OP_RMS_NORM && ops.begin()[1] == GGML_OP_SCALE) {
        const ggml_tensor * rms_norm = cgraph->nodes[node_idx];
        const ggml_tensor * scale    = cgraph->nodes[node_idx+1];

        if (rms_norm->src[0]->type != GGML_TYPE_F32 || rms_norm->type != GGML_TYPE_F32) {
            return false;
        }

        float bias;
        memcpy(&bias, (const float *) scale->op_params + 1, sizeof(float));

        return bias == 0.0f && scale->type == GGML_TYPE_F32;
    }

    if (ops.size() == 2 && ops.begin()[0] == GGML_OP_SSM_CONV && ops.begin()[1] == GGML_OP_UNARY
     && unary_ops.size() == 1 && unary_ops.begin()[0] == GGML_UNARY_OP_SILU) {
        const ggml_tensor * ssm_conv = cgraph->nodes[node_idx];
        const ggml_tensor * silu     = cgraph->nodes[node_idx+1];
        if (ggml_get_unary_op(silu) != unary_ops.begin()[0]) {
            return false;
        }

        if (ssm_conv->type != GGML_TYPE_F32 || silu->type != GGML_TYPE_F32) {
            return false;
        }

        return true;
    }

    if (ops.size() == 3 && ops.begin()[0] == GGML_OP_SSM_CONV && ops.begin()[1] == GGML_OP_ADD
     && ops.begin()[2] == GGML_OP_UNARY && unary_ops.size() == 1 && unary_ops.begin()[0] == GGML_UNARY_OP_SILU) {
        const ggml_tensor * ssm_conv = cgraph->nodes[node_idx];
        const ggml_tensor * add      = cgraph->nodes[node_idx+1];
        const ggml_tensor * silu     = cgraph->nodes[node_idx+2];
        if (ggml_get_unary_op(silu) != unary_ops.begin()[0]) {
            return false;
        }

        if (ssm_conv->type != GGML_TYPE_F32 || add->type != GGML_TYPE_F32 || silu->type != GGML_TYPE_F32) {
            return false;
        }

        // ADD must consume ssm_conv's output and broadcast a 1-D channel-wise bias.
        const ggml_tensor * bias = (add->src[0] == ssm_conv) ? add->src[1] : add->src[0];
        if (bias->type != GGML_TYPE_F32 || !ggml_is_contiguous(bias)) {
            return false;
        }
        if (ggml_nelements(bias) != ssm_conv->ne[0] || bias->ne[0] != ssm_conv->ne[0]) {
            return false;
        }

        return true;
    }

    if (ops.size() == 2 && ops.begin()[0] == GGML_OP_UNARY && ops.begin()[1] == GGML_OP_MUL
     && unary_ops.size() == 1 && (unary_ops.begin()[0] == GGML_UNARY_OP_SILU || unary_ops.begin()[0] == GGML_UNARY_OP_SIGMOID || unary_ops.begin()[0] == GGML_UNARY_OP_SOFTPLUS)) {
        const ggml_tensor * unary = cgraph->nodes[node_idx];
        const ggml_tensor * mul   = cgraph->nodes[node_idx+1];

        if (ggml_get_unary_op(unary) != unary_ops.begin()[0]) {
            return false;
        }

        if (unary->type != GGML_TYPE_F32 && unary->type != GGML_TYPE_F16 && unary->type != GGML_TYPE_BF16) {
            return false;
        }

        if (unary->type != mul->type) {
            return false;
        }

        const ggml_tensor * other = (mul->src[0] == unary) ? mul->src[1] : mul->src[0];
        if (other->type != unary->type) {
            return false;
        }
        if (!ggml_is_contiguous_1(other) || !ggml_is_contiguous_1(unary->src[0]) || !ggml_are_same_shape(other, unary)) {
            return false;
        }

        return true;
    }

    if (ops.size() == 2 && ops.begin()[0] == GGML_OP_UNARY && ops.begin()[1] == GGML_OP_SQR
     && unary_ops.size() == 1 && unary_ops.begin()[0] == GGML_UNARY_OP_RELU) {
        const ggml_tensor * unary = cgraph->nodes[node_idx];
        const ggml_tensor * sqr   = cgraph->nodes[node_idx+1];

        if (ggml_get_unary_op(unary) != GGML_UNARY_OP_RELU) {
            return false;
        }

        if (unary->type != GGML_TYPE_F32 && unary->type != GGML_TYPE_F16) {
            return false;
        }

        if (unary->type != sqr->type) {
            return false;
        }

        if (!ggml_is_contiguous(unary->src[0])) {
            return false;
        }

        return true;
    }

    if (ops.size() == 3 && ops.begin()[0] == GGML_OP_SCALE && ops.begin()[1] == GGML_OP_UNARY && ops.begin()[2] == GGML_OP_SCALE
     && unary_ops.size() == 1 && unary_ops.begin()[0] == GGML_UNARY_OP_TANH) {
        const ggml_tensor *scale  = cgraph->nodes[node_idx];
        const ggml_tensor *tanh   = cgraph->nodes[node_idx+1];
        const ggml_tensor *scale2 = cgraph->nodes[node_idx+2];

        // the fused softcap kernel is F32 only
        if (scale->src[0]->type != GGML_TYPE_F32 || scale->type != GGML_TYPE_F32) {
            return false;
        }

        // TODO: make the fused softcap kernel stride aware, it reads src and writes dst as dense
        if (!ggml_is_contiguous(scale->src[0]) || !ggml_is_contiguous(scale2)) {
            return false;
        }

        if (ggml_get_unary_op(tanh) != GGML_UNARY_OP_TANH) {
            return false;
        }

        // Check for bias
        if (ggml_get_op_params_f32(scale, 1) != 0.0f || ggml_get_op_params_f32(scale2, 1) != 0.0f) {
            return false;
        }

        return true;
    }

    return false;
}

// ---------------------------------------------------------------------------------------------------------------------
// local (tuned on RDNA3): fusions of the small-op patterns of the Inkling 2 graph, kernels in inkling2-fused.cu, taken on
// RDNA3 GPUs only (ggml_cuda_ink2_arch_ok). GGML_CUDA_INK2_NO_FUSE=1 disables them. Each matcher returns the number of nodes of the pattern (0 = no match);
// the last node of the pattern is its output.
// ---------------------------------------------------------------------------------------------------------------------

static bool ggml_cuda_ink2_fusion_enabled() {
    static const bool enabled = !ggml_cuda_env_flag("GGML_CUDA_INK2_NO_FUSE");
    return enabled;
}

// the short-convolution (sconv, sconv_in, sconv_rb, sconv_full, dconv) and relative-bias patterns also match batches of
// several sequences (several slots of a server decoding at once: x {.., C, n_seqs}, the bias and mask {.., n_stream});
// GGML_CUDA_INK2_MULTISEQ=0 restores the one-sequence matchers
static bool ggml_cuda_ink2_multiseq_enabled() {
    static const bool enabled = []() { const char * e = getenv("GGML_CUDA_INK2_MULTISEQ"); return !(e && e[0] == '0'); }();
    return enabled;
}

// GGML_CUDA_INK2_Q8_GROUP_MULTI=0: the groups only take products of the same activations (see graph_optimize)
static bool ggml_cuda_ink2_q8_group_multi_enabled() {
    static const bool enabled = []() { const char * e = getenv("GGML_CUDA_INK2_Q8_GROUP_MULTI"); return !(e && e[0] == '0'); }();
    return enabled;
}

// GGML_CUDA_INK2_Q8_GROUP=0: no grouping of the Q8_0 products that read the same activations (graph_optimize + graph_compute)
static bool ggml_cuda_ink2_q8_group_enabled() {
    static const bool enabled = []() { const char * e = getenv("GGML_CUDA_INK2_Q8_GROUP"); return !(e && e[0] == '0'); }();
    return enabled;
}

// GGML_CUDA_INK2_FUSE_MASK: 1 = L2 clip, 2 = depth mix, 4 = loop gate, 8 = short conv tail, 16 = SITU activation,
// 32 = MoE weight normalisation, 64 = short conv input and state store, 4096 = first maximum of a row,
// 8192 = SITU activation written by the FP16 expert kernel of prompt batches, 16384 = the down projection in the same call
// (needs 8192) (default: all)
static int ggml_cuda_ink2_fusion_mask() {
    static const int mask = []() {
        const char * e = getenv("GGML_CUDA_INK2_FUSE_MASK");
        return e ? atoi(e) : 0x7fffffff;
    }();
    return mask;
}

// GGML_CUDA_INK2_DEBUG=2: print the number of fused launches of each kind at exit
static void ggml_cuda_ink2_count(const int kind) {
    static const bool enabled = []() { const char * e = getenv("GGML_CUDA_INK2_DEBUG"); return e && atoi(e) >= 2; }();
    static int64_t counts[3] = { 0, 0, 0 };
    if (!enabled) {
        return;
    }
    if (counts[0] + counts[1] + counts[2] == 0) {
        atexit([]() { fprintf(stderr, "ink2 fused launches: clip %lld, gate %lld, mix %lld\n", (long long) counts[0], (long long) counts[1], (long long) counts[2]); });
    }
    counts[kind]++;
}

// whether a and b are the same data: same view root and offset
static bool ggml_cuda_ink2_same_data(const ggml_tensor * a, const ggml_tensor * b) {
    const ggml_tensor * ra = a->view_src ? a->view_src : a;
    const ggml_tensor * rb = b->view_src ? b->view_src : b;
    return ra == rb && a->view_offs == b->view_offs;
}

static bool ggml_cuda_ink2_f32_contiguous(const ggml_tensor * t) {
    return t != nullptr && t->type == GGML_TYPE_F32 && ggml_is_contiguous(t);
}

struct ggml_cuda_ink2_clip_match {
    const ggml_tensor * x;
    float threshold;
    float eps;
    const ggml_tensor * scale = nullptr; // per-channel weight multiplied first (clip_scaled)
};

// x -> SQR, SUM_ROWS, SQRT, SCALE(1, eps), FILL(threshold), DIV, CLAMP(0, 1), MUL(x, .)
static int ggml_cuda_ink2_match_clip(const ggml_cgraph * cgraph, const int i, ggml_cuda_ink2_clip_match & m) {
    static const ggml_op ops[] = { GGML_OP_SQR, GGML_OP_SUM_ROWS, GGML_OP_SQRT, GGML_OP_SCALE, GGML_OP_FILL, GGML_OP_DIV, GGML_OP_CLAMP, GGML_OP_MUL };
    constexpr int n = 8;
    if (!(ggml_cuda_ink2_fusion_mask() & 1) || cgraph->nodes[i]->op != GGML_OP_SQR || i + n > cgraph->n_nodes) {
        return 0;
    }
    const int out = i + n - 1;
    if (!ggml_can_fuse_subgraph(cgraph, i, n, ops, &out, 1)) {
        return 0;
    }
    ggml_tensor ** nd = cgraph->nodes + i;
    const ggml_tensor * x = nd[0]->src[0];
    if (nd[1]->src[0] != nd[0] || nd[2]->src[0] != nd[1] || nd[3]->src[0] != nd[2] || nd[4]->src[0] != nd[3] ||
            nd[5]->src[0] != nd[4] || nd[5]->src[1] != nd[3] || nd[6]->src[0] != nd[5]) {
        return 0;
    }
    if (!((nd[7]->src[0] == x && nd[7]->src[1] == nd[6]) || (nd[7]->src[1] == x && nd[7]->src[0] == nd[6]))) {
        return 0;
    }
    if (!ggml_cuda_ink2_f32_contiguous(x) || !ggml_cuda_ink2_f32_contiguous(nd[7]) || !ggml_are_same_shape(x, nd[7])) {
        return 0;
    }
    if (nd[1]->ne[0] != 1 || nd[1]->ne[1] != x->ne[1] || nd[1]->ne[2] != x->ne[2] || nd[1]->ne[3] != x->ne[3]) {
        return 0;
    }
    if (ggml_get_op_params_f32(nd[3], 0) != 1.0f || ggml_get_op_params_f32(nd[6], 0) != 0.0f || ggml_get_op_params_f32(nd[6], 1) != 1.0f) {
        return 0;
    }
    m.x         = x;
    m.eps       = ggml_get_op_params_f32(nd[3], 1);
    m.threshold = ggml_get_op_params_f32(nd[4], 0);
    return n;
}

// MUL(x, scale) followed by the clip of its result, scale a weight {d, n_head} broadcast over the tokens (the q and r clips of
// inkling2.cpp): 9 nodes, one kernel. GGML_CUDA_INK2_CLIP_SCALED=0 keeps the MUL apart.
static int ggml_cuda_ink2_match_clip_scaled(const ggml_cgraph * cgraph, const int i, ggml_cuda_ink2_clip_match & m) {
    static const bool enabled = []() { const char * e = getenv("GGML_CUDA_INK2_CLIP_SCALED"); return !(e && e[0] == '0'); }();
    static const ggml_op ops[] = { GGML_OP_MUL, GGML_OP_SQR, GGML_OP_SUM_ROWS, GGML_OP_SQRT, GGML_OP_SCALE, GGML_OP_FILL, GGML_OP_DIV, GGML_OP_CLAMP, GGML_OP_MUL };
    constexpr int n = 9;
    if (!enabled || !(ggml_cuda_ink2_fusion_mask() & 1) || cgraph->nodes[i]->op != GGML_OP_MUL || i + n > cgraph->n_nodes ||
            cgraph->nodes[i + 1]->op != GGML_OP_SQR) {
        return 0;
    }
    const ggml_tensor * mul = cgraph->nodes[i];
    if (ggml_cuda_ink2_match_clip(cgraph, i + 1, m) != n - 1 || m.x != mul) {
        return 0;
    }
    const int out = i + n - 1;
    if (!ggml_can_fuse_subgraph(cgraph, i, n, ops, &out, 1)) {
        return 0;
    }
    const ggml_tensor * x     = mul->src[0];
    const ggml_tensor * scale = mul->src[1];
    if (!ggml_cuda_ink2_f32_contiguous(x) || !ggml_cuda_ink2_f32_contiguous(scale) || !ggml_are_same_shape(x, mul) ||
            !scale->buffer || ggml_backend_buffer_get_usage(scale->buffer) != GGML_BACKEND_BUFFER_USAGE_WEIGHTS ||
            scale->ne[0] != x->ne[0] || scale->ne[2] != 1 || scale->ne[3] != 1 ||
            (scale->ne[1] != x->ne[1] && scale->ne[1] != 1) || x->ne[3] != 1) {
        return 0;
    }
    m.x     = x;
    m.scale = scale;
    return n;
}

struct ggml_cuda_ink2_mix_match {
    const ggml_tensor * states[GGML_CUDA_INK2_MIX_MAX_STATES];
    int                 n_states;
    const ggml_tensor * query;
    float               eps;
    float               scale;
    bool                rows_first; // the output is the permuted copy [n_embd, n_tokens, n_q] (inkling2.cpp, mix_rows)
};

// CONCAT x n (dim 1), RMS_NORM, TRANSPOSE, CONT, [VIEW of the query], REPEAT, MUL_MAT, SOFT_MAX, MUL_MAT [, PERMUTE(0, 2, 1, 3), CONT]
static int ggml_cuda_ink2_match_mix(const ggml_cgraph * cgraph, const int i, ggml_cuda_ink2_mix_match & m) {
    if (!(ggml_cuda_ink2_fusion_mask() & 2) || cgraph->nodes[i]->op != GGML_OP_CONCAT) {
        return 0;
    }
    ggml_op ops[24];
    int     n = 0;

    int n_concat = 0;
    while (i + n < cgraph->n_nodes && cgraph->nodes[i + n]->op == GGML_OP_CONCAT && n_concat < GGML_CUDA_INK2_MIX_MAX_STATES - 1) {
        const ggml_tensor * c = cgraph->nodes[i + n];
        if (ggml_get_op_params_i32(c, 0) != 1 || (n_concat > 0 && c->src[1] != cgraph->nodes[i + n - 1])) {
            break;
        }
        ops[n++] = GGML_OP_CONCAT;
        n_concat++;
    }
    if (n_concat == 0 || i + n + 7 > cgraph->n_nodes) {
        return 0;
    }
    const int i_rms = i + n;
    ops[n++] = GGML_OP_RMS_NORM;
    ops[n++] = GGML_OP_TRANSPOSE;
    ops[n++] = GGML_OP_CONT;
    if (cgraph->nodes[i + n]->op == GGML_OP_VIEW) {
        ops[n++] = GGML_OP_VIEW;
    }
    const int i_rep = i + n;
    ops[n++] = GGML_OP_REPEAT;
    ops[n++] = GGML_OP_MUL_MAT;
    ops[n++] = GGML_OP_SOFT_MAX;
    ops[n++] = GGML_OP_MUL_MAT;
    if (i + n > cgraph->n_nodes) {
        return 0;
    }
    const int i_mm = i + n - 1;
    // optional permuted copy of the result (rows of a multi-token mix made contiguous)
    bool rows_first = false;
    if (i + n + 2 <= cgraph->n_nodes && cgraph->nodes[i + n]->op == GGML_OP_PERMUTE && cgraph->nodes[i + n]->src[0] == cgraph->nodes[i_mm] &&
            cgraph->nodes[i + n + 1]->op == GGML_OP_CONT && cgraph->nodes[i + n + 1]->src[0] == cgraph->nodes[i + n] &&
            ggml_get_op_params_i32(cgraph->nodes[i + n], 0) == 0 && ggml_get_op_params_i32(cgraph->nodes[i + n], 1) == 2 &&
            ggml_get_op_params_i32(cgraph->nodes[i + n], 2) == 1 && ggml_get_op_params_i32(cgraph->nodes[i + n], 3) == 3) {
        ops[n++] = GGML_OP_PERMUTE;
        ops[n++] = GGML_OP_CONT;
        rows_first = true;
    }
    const int out = i + n - 1;
    if (!ggml_can_fuse_subgraph(cgraph, i, n, ops, &out, 1)) {
        return 0;
    }

    const ggml_tensor * rms    = cgraph->nodes[i_rms];
    const ggml_tensor * transp = cgraph->nodes[i_rms + 1];
    const ggml_tensor * cont   = cgraph->nodes[i_rms + 2];
    const ggml_tensor * rep    = cgraph->nodes[i_rep];
    const ggml_tensor * scores = cgraph->nodes[i_rep + 1];
    const ggml_tensor * smax   = cgraph->nodes[i_rep + 2];
    const ggml_tensor * mm     = cgraph->nodes[i_mm];
    const ggml_tensor * dst    = cgraph->nodes[out];

    if (rms->src[0] != cgraph->nodes[i_rms - 1] || transp->src[0] != rms || cont->src[0] != transp ||
            scores->src[0] != rms || scores->src[1] != rep || smax->src[0] != scores || smax->src[1] != nullptr ||
            mm->src[0] != cont || mm->src[1] != smax) {
        return 0;
    }
    if (i_rep > i_rms + 3 && rep->src[0] != cgraph->nodes[i_rms + 3]) {
        return 0;
    }

    const ggml_tensor * query  = rep->src[0];
    const int64_t       n_embd = rms->ne[0];
    const int64_t       n_tok  = rms->ne[2];
    if (!ggml_cuda_ink2_f32_contiguous(query) || !ggml_cuda_ink2_f32_contiguous(dst) || rms->type != GGML_TYPE_F32 || rms->ne[3] != 1) {
        return 0;
    }
    const int64_t n_q = ggml_nelements(query)/query->ne[0];
    if (query->ne[0] != n_embd || !ggml_cuda_ink2_depth_mix_supported(n_concat + 1, (int) n_q) ||
            rep->ne[0] != n_embd || rep->ne[1] != n_q || rep->ne[2] != n_tok || rep->ne[3] != 1 ||
            mm->ne[0] != n_embd || mm->ne[1] != n_q || mm->ne[2] != n_tok || mm->ne[3] != 1 ||
            dst->ne[0] != n_embd || dst->ne[1] != (rows_first ? n_tok : n_q) || dst->ne[2] != (rows_first ? n_q : n_tok) || dst->ne[3] != 1 ||
            rms->ne[1] != n_concat + 1) {
        return 0;
    }
    if (ggml_get_op_params_f32(smax, 1) != 0.0f) {
        return 0;
    }

    // C_1 = concat(b_1, cur), C_k = concat(b_k, C_{k-1})
    m.n_states = 0;
    m.states[m.n_states++] = cgraph->nodes[i]->src[1];
    for (int k = 0; k < n_concat; ++k) {
        m.states[m.n_states++] = cgraph->nodes[i + k]->src[0];
    }
    for (int k = 0; k < m.n_states; ++k) {
        const ggml_tensor * s = m.states[k];
        if (!ggml_cuda_ink2_f32_contiguous(s) || s->ne[0] != n_embd || s->ne[1] != 1 || s->ne[2] != n_tok || s->ne[3] != 1) {
            return 0;
        }
    }
    m.query      = query;
    m.eps        = ggml_get_op_params_f32(rms, 0);
    m.scale      = ggml_get_op_params_f32(smax, 0);
    m.rows_first = rows_first;
    return n;
}

struct ggml_cuda_ink2_gate_match {
    const ggml_tensor * cur;
    const ggml_tensor * msg;
    const ggml_tensor * down;
    const ggml_tensor * gain;
    const ggml_tensor * forget;
    float               eps;
};

// CONCAT(cur, msg), RMS_NORM, RESHAPE, MUL_MAT(D), MUL_MAT(F), TANH, SCALE(-1, 1), MUL(., cur), MUL_MAT(G), TANH, MUL(., msg), ADD
static int ggml_cuda_ink2_match_gate(const ggml_cgraph * cgraph, const int i, ggml_cuda_ink2_gate_match & m, const bool large = false) {
    static const ggml_op ops[] = { GGML_OP_CONCAT, GGML_OP_RMS_NORM, GGML_OP_RESHAPE, GGML_OP_MUL_MAT, GGML_OP_MUL_MAT, GGML_OP_UNARY,
                                   GGML_OP_SCALE, GGML_OP_MUL, GGML_OP_MUL_MAT, GGML_OP_UNARY, GGML_OP_MUL, GGML_OP_ADD };
    constexpr int n = 12;
    if (!(ggml_cuda_ink2_fusion_mask() & 4) || cgraph->nodes[i]->op != GGML_OP_CONCAT || i + n > cgraph->n_nodes) {
        return 0;
    }
    const int out = i + n - 1;
    if (!ggml_can_fuse_subgraph(cgraph, i, n, ops, &out, 1)) {
        return 0;
    }
    ggml_tensor ** nd = cgraph->nodes + i;
    const ggml_tensor * concat = nd[0];
    const ggml_tensor * mm_d = nd[3];
    const ggml_tensor * mm_f = nd[4];
    const ggml_tensor * mul_c = nd[7];
    const ggml_tensor * mm_g = nd[8];
    const ggml_tensor * mul_m = nd[10];
    const ggml_tensor * add  = nd[11];

    if (ggml_get_op_params_i32(concat, 0) != 1 || nd[1]->src[0] != concat || nd[2]->src[0] != nd[1] ||
            mm_d->src[1] != nd[2] || mm_f->src[1] != mm_d || nd[5]->src[0] != mm_f || nd[6]->src[0] != nd[5] ||
            mm_g->src[1] != mm_d || nd[9]->src[0] != mm_g) {
        return 0;
    }
    if (ggml_get_unary_op(nd[5]) != GGML_UNARY_OP_TANH || ggml_get_unary_op(nd[9]) != GGML_UNARY_OP_TANH ||
            ggml_get_op_params_f32(nd[6], 0) != -1.0f || ggml_get_op_params_f32(nd[6], 1) != 1.0f) {
        return 0;
    }
    const ggml_tensor * cur_in = mul_c->src[0] == nd[6] ? mul_c->src[1] : (mul_c->src[1] == nd[6] ? mul_c->src[0] : nullptr);
    const ggml_tensor * msg_in = mul_m->src[0] == nd[9] ? mul_m->src[1] : (mul_m->src[1] == nd[9] ? mul_m->src[0] : nullptr);
    if (!cur_in || !msg_in || !ggml_cuda_ink2_same_data(cur_in, concat->src[0]) || !ggml_cuda_ink2_same_data(msg_in, concat->src[1])) {
        return 0;
    }
    if (!((add->src[0] == mul_c && add->src[1] == mul_m) || (add->src[0] == mul_m && add->src[1] == mul_c))) {
        return 0;
    }

    const ggml_tensor * down   = mm_d->src[0];
    const ggml_tensor * forget = mm_f->src[0];
    const ggml_tensor * gain   = mm_g->src[0];
    const int64_t n_embd = add->ne[0];
    const int64_t n_rank = down ? down->ne[1] : 0;
    for (const ggml_tensor * t : { cur_in, msg_in, add }) {
        if (!ggml_cuda_ink2_f32_contiguous(t)) {
            return 0;
        }
    }
    // weights: all F32, or all BF16 (lossless storage of the Inkling 2 Mini gates, multi-token kernels only)
    if (!down || !forget || !gain) {
        return 0;
    }
    const bool w_bf16 = down->type == GGML_TYPE_BF16;
    for (const ggml_tensor * t : { down, forget, gain }) {
        if (t->type != (w_bf16 ? GGML_TYPE_BF16 : GGML_TYPE_F32) || !ggml_is_contiguous(t)) {
            return 0;
        }
    }
    if (w_bf16 && !ggml_cuda_ink2_loop_gate_bf16_supported((int) n_embd, (int) n_rank)) {
        return 0;
    }
    if (n_rank < 1 || n_rank > 128 || down->ne[0] != 2*n_embd || gain->ne[0] != n_rank || gain->ne[1] != n_embd ||
            forget->ne[0] != n_rank || forget->ne[1] != n_embd || ggml_nelements(cur_in) != ggml_nelements(add) ||
            ggml_nelements(msg_in) != ggml_nelements(add) || add->ne[2] != 1 || add->ne[3] != 1 || concat->ne[1] != 2) {
        return 0;
    }
    // the h kernel reads cur and msg once per rank: good for decode, slower than the generic matmuls for a prefill batch
    // (large: the prompt-batch matcher below, which keeps the products generic and fuses only the tail)
    if ((add->ne[1] > 16) != large) {
        return 0;
    }
    m.cur    = cur_in;
    m.msg    = msg_in;
    m.down   = down;
    m.gain   = gain;
    m.forget = forget;
    m.eps    = ggml_get_op_params_f32(nd[1], 0);
    return n;
}

// loop gate of a prompt batch (more than 16 tokens): the three products stay generic GEMMs, the tail is one kernel.
// Starts at the TANH of F h (node 5 of the gate pattern): TANH, SCALE, MUL(cur), MUL_MAT(G, h), TANH, MUL(msg), ADD = 7 nodes;
// the product by G is computed first by the fused step. Mask 2048.
static int ggml_cuda_ink2_match_gate_tail(const ggml_cgraph * cgraph, const int j, ggml_cuda_ink2_gate_match & m) {
    if (!(ggml_cuda_ink2_fusion_mask() & 2048) || j < 5 || cgraph->nodes[j]->op != GGML_OP_UNARY ||
            cgraph->nodes[j - 5]->op != GGML_OP_CONCAT) {
        return 0;
    }
    if (ggml_cuda_ink2_match_gate(cgraph, j - 5, m, true) != 12) {
        return 0;
    }
    return 7;
}

struct ggml_cuda_ink2_sconv_match {
    const ggml_tensor * x;
    const ggml_tensor * w;
    const ggml_tensor * cur;
    const ggml_tensor * res;
    float               scale;
};

// SSM_CONV(x, w), RESHAPE, ADD(cur, .), SCALE(s, 0) [, ADD(res, .)]
static int ggml_cuda_ink2_match_sconv(const ggml_cgraph * cgraph, const int i, ggml_cuda_ink2_sconv_match & m) {
    static const ggml_op ops[] = { GGML_OP_SSM_CONV, GGML_OP_RESHAPE, GGML_OP_ADD, GGML_OP_SCALE, GGML_OP_ADD };
    if (!(ggml_cuda_ink2_fusion_mask() & 8) || cgraph->nodes[i]->op != GGML_OP_SSM_CONV || i + 4 > cgraph->n_nodes) {
        return 0;
    }
    int n = 5;
    int out = i + 4;
    if (i + 5 > cgraph->n_nodes || !ggml_can_fuse_subgraph(cgraph, i, 5, ops, &out, 1)) {
        n   = 4;
        out = i + 3;
        if (!ggml_can_fuse_subgraph(cgraph, i, 4, ops, &out, 1)) {
            return 0;
        }
    }
    ggml_tensor ** nd = cgraph->nodes + i;
    const ggml_tensor * conv = nd[0];
    const ggml_tensor * resh = nd[1];
    const ggml_tensor * add  = nd[2];
    const ggml_tensor * scl  = nd[3];
    if (resh->src[0] != conv || scl->src[0] != add || ggml_get_op_params_f32(scl, 1) != 0.0f) {
        return 0;
    }
    const ggml_tensor * cur = add->src[0] == resh ? add->src[1] : (add->src[1] == resh ? add->src[0] : nullptr);
    const ggml_tensor * res = nullptr;
    if (n == 5) {
        const ggml_tensor * add2 = nd[4];
        res = add2->src[0] == scl ? add2->src[1] : (add2->src[1] == scl ? add2->src[0] : nullptr);
        if (!ggml_cuda_ink2_f32_contiguous(res) || !ggml_are_same_shape(res, add2)) {
            return 0;
        }
    }
    const ggml_tensor * x = conv->src[0];
    const ggml_tensor * w = conv->src[1];
    const ggml_tensor * dst = nd[n - 1];
    for (const ggml_tensor * t : { x, w, cur, dst }) {
        if (!ggml_cuda_ink2_f32_contiguous(t)) {
            return 0;
        }
    }
    // x {n_conv - 1 + n_seq_tokens, n_channels, n_seqs}, conv {n_channels, n_seq_tokens, n_seqs} (one sequence unless multiseq)
    const int64_t n_seqs = x->ne[2];
    if ((n_seqs != 1 && !ggml_cuda_ink2_multiseq_enabled()) || x->ne[3] != 1 || conv->ne[2] != n_seqs || conv->ne[3] != 1 || w->ne[1] != x->ne[1] ||
            conv->ne[0] != x->ne[1] || conv->ne[1] != x->ne[0] - w->ne[0] + 1 ||
            ggml_nelements(cur) != ggml_nelements(conv) || ggml_nelements(dst) != ggml_nelements(conv) || !ggml_are_same_shape(cur, add)) {
        return 0;
    }
    m.x     = x;
    m.w     = w;
    m.cur   = cur;
    m.res   = res;
    m.scale = ggml_get_op_params_f32(scl, 0);
    return n;
}

struct ggml_cuda_ink2_situ_match {
    const ggml_tensor * gate;
    const ggml_tensor * up;
    float               beta;
    float               lb;
};

// SCALE(gate, 1/beta), TANH, SCALE(beta), SIGMOID(gate), MUL, [VIEW up], SCALE(up, 1/lb), TANH, SCALE(lb), MUL
static int ggml_cuda_ink2_match_situ(const ggml_cgraph * cgraph, const int i, ggml_cuda_ink2_situ_match & m) {
    if (!(ggml_cuda_ink2_fusion_mask() & 16) || cgraph->nodes[i]->op != GGML_OP_SCALE || i + 9 > cgraph->n_nodes) {
        return 0;
    }
    ggml_op ops[10] = { GGML_OP_SCALE, GGML_OP_UNARY, GGML_OP_SCALE, GGML_OP_UNARY, GGML_OP_MUL };
    int n = 5;
    if (cgraph->nodes[i + n]->op == GGML_OP_VIEW) {
        ops[n++] = GGML_OP_VIEW;
    }
    const int i_up = i + n;
    ops[n++] = GGML_OP_SCALE;
    ops[n++] = GGML_OP_UNARY;
    ops[n++] = GGML_OP_SCALE;
    ops[n++] = GGML_OP_MUL;
    if (i + n > cgraph->n_nodes) {
        return 0;
    }
    const int out = i + n - 1;
    // two halves: the gate half ends in MUL (i + 4), read once by the final MUL; the up half ends in the final MUL.
    // The VIEW of up between them is not part of either check (its source, the gate_up projection, is an input).
    {
        const int out_gate = i + 4;
        static const ggml_op ops_up[] = { GGML_OP_SCALE, GGML_OP_UNARY, GGML_OP_SCALE, GGML_OP_MUL };
        if (!ggml_can_fuse_subgraph(cgraph, i, 5, ops, &out_gate, 1) || ggml_node_get_use_count(cgraph, out_gate) != 1 ||
                (cgraph->nodes[out_gate]->flags & GGML_TENSOR_FLAG_OUTPUT) ||
                !ggml_can_fuse_subgraph(cgraph, i_up, 4, ops_up, &out, 1)) {
            return 0;
        }
    }
    ggml_tensor ** nd = cgraph->nodes + i;
    const ggml_tensor * gate = nd[0]->src[0];
    const ggml_tensor * sc_c = cgraph->nodes[i_up];
    const ggml_tensor * up   = sc_c->src[0];
    const ggml_tensor * dst  = cgraph->nodes[out];
    if (nd[1]->src[0] != nd[0] || nd[2]->src[0] != nd[1] || nd[3]->src[0] != gate ||
            !((nd[4]->src[0] == nd[2] && nd[4]->src[1] == nd[3]) || (nd[4]->src[1] == nd[2] && nd[4]->src[0] == nd[3])) ||
            cgraph->nodes[i_up + 1]->src[0] != sc_c || cgraph->nodes[i_up + 2]->src[0] != cgraph->nodes[i_up + 1] ||
            !((dst->src[0] == nd[4] && dst->src[1] == cgraph->nodes[i_up + 2]) || (dst->src[1] == nd[4] && dst->src[0] == cgraph->nodes[i_up + 2]))) {
        return 0;
    }
    if (i_up > i + 5 && up != cgraph->nodes[i + 5]) {
        return 0;
    }
    if (ggml_get_unary_op(nd[1]) != GGML_UNARY_OP_TANH || ggml_get_unary_op(nd[3]) != GGML_UNARY_OP_SIGMOID ||
            ggml_get_unary_op(cgraph->nodes[i_up + 1]) != GGML_UNARY_OP_TANH) {
        return 0;
    }
    const float inv_beta = ggml_get_op_params_f32(nd[0], 0);
    const float beta     = ggml_get_op_params_f32(nd[2], 0);
    const float inv_lb   = ggml_get_op_params_f32(sc_c, 0);
    const float lb       = ggml_get_op_params_f32(cgraph->nodes[i_up + 2], 0);
    if (ggml_get_op_params_f32(nd[0], 1) != 0.0f || ggml_get_op_params_f32(nd[2], 1) != 0.0f || ggml_get_op_params_f32(sc_c, 1) != 0.0f ||
            ggml_get_op_params_f32(cgraph->nodes[i_up + 2], 1) != 0.0f || !(beta > 0.0f) || !(lb > 0.0f) ||
            fabsf(inv_beta*beta - 1.0f) > 1e-5f || fabsf(inv_lb*lb - 1.0f) > 1e-5f) {
        return 0;
    }
    if (gate->type != GGML_TYPE_F32 || up->type != GGML_TYPE_F32 || !ggml_cuda_ink2_f32_contiguous(dst) ||
            !ggml_are_same_shape(gate, dst) || !ggml_are_same_shape(up, dst) || gate->ne[3] != 1 ||
            gate->nb[0] != sizeof(float) || up->nb[0] != sizeof(float) || gate->nb[1] != up->nb[1] || gate->nb[1] % sizeof(float) != 0 ||
            gate->nb[2] != gate->nb[1]*(size_t) gate->ne[1] || up->nb[2] != up->nb[1]*(size_t) up->ne[1]) {
        return 0;
    }
    m.gate = gate;
    m.up   = up;
    m.beta = beta;
    m.lb   = lb;
    return n;
}

struct ggml_cuda_ink2_moe_situ_match {
    const ggml_tensor * mm;      // MUL_MAT_ID of the merged gate_up projection
    const ggml_tensor * x_scale; // per-expert scale of the projection (folded MUL), or nullptr
    ggml_cuda_ink2_situ_match situ;
};

// MUL_MAT_ID(gate_up) [RESHAPE, REPEAT, GET_ROWS, MUL (per-expert scale)], VIEW(gate), then the SITU pattern on the two halves of
// that projection: the FP16 expert kernel writes the activation directly (ggml_cuda_ink2_moe_f16 with situ). Structural match only;
// whether the kernel takes the shape is ggml_cuda_ink2_moe_f16_supported (graph_optimize and compute ask the same question).
static int ggml_cuda_ink2_match_moe_situ(const ggml_cgraph * cgraph, const int i, ggml_cuda_ink2_moe_situ_match & m) {
    if (!(ggml_cuda_ink2_fusion_mask() & 8192) || cgraph->nodes[i]->op != GGML_OP_MUL_MAT_ID) {
        return 0;
    }
    ggml_tensor ** nd = cgraph->nodes + i;
    const ggml_tensor * mm = nd[0];
    const ggml_type     wt = mm->src[0]->type;
    if ((wt != GGML_TYPE_NVFP4 && wt != GGML_TYPE_Q5_K && wt != GGML_TYPE_Q8_0) || mm->type != GGML_TYPE_F32 ||
            (mm->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        return 0;
    }
    const ggml_tensor * gu      = mm;
    const ggml_tensor * x_scale = nullptr;
    int j = i + 1;
    if (i + 5 < cgraph->n_nodes && nd[1]->op == GGML_OP_RESHAPE && nd[2]->op == GGML_OP_REPEAT && nd[3]->op == GGML_OP_GET_ROWS &&
            nd[4]->op == GGML_OP_MUL) {
        x_scale = ggml_cuda_get_mul_mat_id_scale(nd[1], nd[2], nd[3], nd[4], mm);
        if (x_scale) {
            // the product is read by the scale MUL only, the repeated scale by its GET_ROWS only
            if (ggml_node_get_use_count(cgraph, i) != 1 || ggml_node_get_use_count(cgraph, i + 2) != 1 ||
                    ggml_node_get_use_count(cgraph, i + 3) != 1 || (nd[4]->flags & GGML_TENSOR_FLAG_OUTPUT)) {
                return 0;
            }
            gu = nd[4];
            j  = i + 5;
        }
    }
    // the gate view, then the SITU pattern
    if (j + 1 >= cgraph->n_nodes || cgraph->nodes[j]->op != GGML_OP_VIEW) {
        return 0;
    }
    const int n_situ = ggml_cuda_ink2_match_situ(cgraph, j + 1, m.situ);
    if (n_situ == 0) {
        return 0;
    }
    // gu is read by the two views only; gate = rows 0..n_ff-1 and up = rows n_ff..2 n_ff-1 of every gu row
    const int i_gu = (int) (gu == mm ? i : i + 4);
    const ggml_tensor * gate = m.situ.gate;
    const ggml_tensor * up   = m.situ.up;
    const int64_t n_ff = gu->ne[0]/2;
    if (ggml_node_get_use_count(cgraph, i_gu) != 2 || gate != cgraph->nodes[j] || gate->view_src != gu || up->view_src != gu ||
            gate->view_offs != 0 || up->view_offs != (size_t) n_ff*sizeof(float) || gu->ne[0] != 2*n_ff ||
            gate->ne[0] != n_ff || up->ne[0] != n_ff || gate->ne[1] != gu->ne[1] || gate->ne[2] != gu->ne[2] ||
            gate->nb[1] != gu->nb[1] || gate->nb[2] != gu->nb[2] || up->nb[1] != gu->nb[1] || up->nb[2] != gu->nb[2] ||
            !ggml_is_contiguous(gu)) {
        return 0;
    }
    m.mm      = mm;
    m.x_scale = x_scale;
    return j + 1 + n_situ - i;
}

// the down projection right after a moe_situ pattern (activation at node i_act): MUL_MAT_ID(down weights, activation, the same
// ids) [RESHAPE, REPEAT, GET_ROWS, MUL of its per-expert scale], the activation read by it only. GGML_CUDA_INK2_MOE_FFN=0 (or
// FUSE_MASK without 16384) keeps the two products apart. Returns the node count (0: no match).
struct ggml_cuda_ink2_moe_down_match {
    const ggml_tensor * mm;
    const ggml_tensor * x_scale;
};

static int ggml_cuda_ink2_match_moe_down(const ggml_cgraph * cgraph, const int i_act, const ggml_tensor * ids, ggml_cuda_ink2_moe_down_match & d) {
    static const bool enabled = []() { const char * e = getenv("GGML_CUDA_INK2_MOE_FFN"); return !e || atoi(e) != 0; }();
    const int j = i_act + 1;
    if (!enabled || !(ggml_cuda_ink2_fusion_mask() & 16384) || j >= cgraph->n_nodes) {
        return 0;
    }
    const ggml_tensor * act = cgraph->nodes[i_act];
    const ggml_tensor * mm  = cgraph->nodes[j];
    if (mm->op != GGML_OP_MUL_MAT_ID || mm->src[1] != act || mm->src[2] != ids || mm->type != GGML_TYPE_F32 ||
            ggml_node_get_use_count(cgraph, i_act) != 1 || (act->flags & GGML_TENSOR_FLAG_OUTPUT) || (mm->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        return 0;
    }
    const ggml_type wt = mm->src[0]->type;
    if (wt != GGML_TYPE_NVFP4 && wt != GGML_TYPE_Q5_K && wt != GGML_TYPE_Q8_0) {
        return 0;
    }
    d.mm      = mm;
    d.x_scale = nullptr;
    ggml_tensor * const * nd = cgraph->nodes + j;
    if (j + 4 < cgraph->n_nodes && nd[1]->op == GGML_OP_RESHAPE && nd[2]->op == GGML_OP_REPEAT && nd[3]->op == GGML_OP_GET_ROWS &&
            nd[4]->op == GGML_OP_MUL) {
        const ggml_tensor * s = ggml_cuda_get_mul_mat_id_scale(nd[1], nd[2], nd[3], nd[4], mm);
        if (s) {
            if (ggml_node_get_use_count(cgraph, j) != 1 || ggml_node_get_use_count(cgraph, j + 2) != 1 ||
                    ggml_node_get_use_count(cgraph, j + 3) != 1 || (nd[4]->flags & GGML_TENSOR_FLAG_OUTPUT)) {
                return 0;
            }
            d.x_scale = s;
            return 5;
        }
    }
    return 1;
}

struct ggml_cuda_ink2_wnorm_match {
    const ggml_tensor * w;
    float               eps;
    float               scale;
};

// (weights reshaped to 2d) SUM_ROWS, SCALE(1, eps), DIV(w, .), RESHAPE, SCALE(scale, 0)
static int ggml_cuda_ink2_match_wnorm(const ggml_cgraph * cgraph, const int i, ggml_cuda_ink2_wnorm_match & m) {
    static const ggml_op ops[] = { GGML_OP_SUM_ROWS, GGML_OP_SCALE, GGML_OP_DIV, GGML_OP_RESHAPE, GGML_OP_SCALE };
    constexpr int n = 5;
    if (!(ggml_cuda_ink2_fusion_mask() & 32) || cgraph->nodes[i]->op != GGML_OP_SUM_ROWS || i + n > cgraph->n_nodes) {
        return 0;
    }
    const int out = i + n - 1;
    if (!ggml_can_fuse_subgraph(cgraph, i, n, ops, &out, 1)) {
        return 0;
    }
    ggml_tensor ** nd = cgraph->nodes + i;
    const ggml_tensor * w = nd[0]->src[0];
    if (nd[1]->src[0] != nd[0] || nd[2]->src[0] != w || nd[2]->src[1] != nd[1] || nd[3]->src[0] != nd[2] || nd[4]->src[0] != nd[3]) {
        return 0;
    }
    if (ggml_get_op_params_f32(nd[1], 0) != 1.0f || ggml_get_op_params_f32(nd[4], 1) != 0.0f) {
        return 0;
    }
    if (!ggml_cuda_ink2_f32_contiguous(w) || !ggml_cuda_ink2_f32_contiguous(nd[4]) || w->ne[2] != 1 || w->ne[3] != 1 ||
            w->ne[0] > 128 || nd[0]->ne[0] != 1 || nd[0]->ne[1] != w->ne[1] || ggml_nelements(nd[4]) != ggml_nelements(w) ||
            nd[4]->ne[0] != 1 || nd[4]->ne[1] != w->ne[0] || nd[4]->ne[2] != w->ne[1]) {
        return 0;
    }
    m.w     = w;
    m.eps   = ggml_get_op_params_f32(nd[1], 1);
    m.scale = ggml_get_op_params_f32(nd[4], 0);
    return n;
}

struct ggml_cuda_ink2_sconv_in_match {
    const ggml_tensor * past;
    const ggml_tensor * cur_t;
    ggml_tensor       * state;
};

// CONCAT(past, cur^T, dim 0), VIEW(last rows of the concat), VIEW(state slot), CPY(view of the concat -> view of the state)
static int ggml_cuda_ink2_match_sconv_in(const ggml_cgraph * cgraph, const int i, ggml_cuda_ink2_sconv_in_match & m) {
    constexpr int n = 4;
    if (!(ggml_cuda_ink2_fusion_mask() & 64) || cgraph->nodes[i]->op != GGML_OP_CONCAT || i + n > cgraph->n_nodes) {
        return 0;
    }
    ggml_tensor ** nd = cgraph->nodes + i;
    ggml_tensor * x   = nd[0];
    ggml_tensor * cpy = nd[3];
    if (ggml_get_op_params_i32(x, 0) != 0 || nd[1]->op != GGML_OP_VIEW || nd[2]->op != GGML_OP_VIEW || cpy->op != GGML_OP_CPY) {
        return 0;
    }
    // the CPY reads a view of x and writes a view of the state, in either node order
    ggml_tensor * vx = cpy->src[0];
    ggml_tensor * vs = cpy->src[1];
    if (!((vx == nd[1] && vs == nd[2]) || (vx == nd[2] && vs == nd[1])) || vx->view_src != x || vs->view_src == nullptr || vs->view_src == x) {
        return 0;
    }
    const ggml_tensor * past  = x->src[0];
    const ggml_tensor * cur_t = x->src[1];
    const int64_t n_past = past->ne[0];
    const int64_t n_tok  = cur_t->ne[0];
    const int64_t n_ch   = x->ne[1];
    // several sequences (multiseq): x {n_past + n_tok, C, n_seqs}, the state one row of n_past*C values per sequence
    const int64_t n_seqs = x->ne[2];
    if (n_seqs != 1 && !ggml_cuda_ink2_multiseq_enabled()) {
        return 0;
    }
    const bool vs_ok = n_seqs == 1 ? ggml_nelements(vs) == n_past*n_ch && ggml_is_contiguous(vs) :
        vs->ne[0] == n_past*n_ch && vs->ne[1] == n_seqs && vs->ne[2] == 1 && vs->ne[3] == 1 && vs->nb[0] == sizeof(float) &&
        vs->nb[1] % sizeof(float) == 0 && vs->nb[1] >= (size_t) (n_past*n_ch)*sizeof(float);
    if (x->type != GGML_TYPE_F32 || past->type != GGML_TYPE_F32 || cur_t->type != GGML_TYPE_F32 || vs->type != GGML_TYPE_F32 ||
            !ggml_is_contiguous(x) || x->ne[3] != 1 || past->ne[2] != n_seqs || cur_t->ne[2] != n_seqs || past->ne[3] != 1 || cur_t->ne[3] != 1 ||
            x->ne[0] != n_past + n_tok || past->ne[1] != n_ch || cur_t->ne[1] != n_ch ||
            vx->ne[0] != n_past || vx->ne[1] != n_ch || vx->ne[2] != n_seqs || vx->ne[3] != 1 || !vs_ok ||
            vx->view_offs != (size_t) n_tok*sizeof(float) || vx->nb[1] != x->nb[1] || (n_seqs > 1 && vx->nb[2] != x->nb[2])) {
        return 0;
    }
    for (const ggml_tensor * t : { past, cur_t }) {
        if (t->nb[0] % sizeof(float) != 0 || t->nb[1] % sizeof(float) != 0 || t->nb[2] % sizeof(float) != 0) {
            return 0;
        }
    }
    // the CPY result is the state write: nothing else may read it as a value
    m.past  = past;
    m.cur_t = cur_t;
    m.state = vs;
    return n;
}

struct ggml_cuda_ink2_sconv_rb_match {
    const ggml_tensor * past;
    const ggml_tensor * cur_t;
    const ggml_tensor * slots;
    const ggml_tensor * slot_order;
    int                 n_win;
};

// rollback form of the conv input (build_conv_state_store with n_rs_seq > 0):
// CONCAT(past, cur^T, dim 0), VIEW(windows of the concat), VIEW(slot order), VIEW(slots), SET_ROWS(slots, windows, slot order)
static int ggml_cuda_ink2_match_sconv_rb(const ggml_cgraph * cgraph, const int i, ggml_cuda_ink2_sconv_rb_match & m) {
    constexpr int n = 5;
    if (!(ggml_cuda_ink2_fusion_mask() & 256) || cgraph->nodes[i]->op != GGML_OP_CONCAT || i + n > cgraph->n_nodes) {
        return 0;
    }
    ggml_tensor ** nd = cgraph->nodes + i;
    ggml_tensor * x   = nd[0];
    ggml_tensor * set = nd[4];
    if (ggml_get_op_params_i32(x, 0) != 0 || nd[1]->op != GGML_OP_VIEW || nd[2]->op != GGML_OP_VIEW || nd[3]->op != GGML_OP_VIEW ||
            set->op != GGML_OP_SET_ROWS) {
        return 0;
    }
    // ggml_set_rows(a, b, c) stores its sources as (b, c, a)
    const ggml_tensor * windows = set->src[0];
    const ggml_tensor * order   = set->src[1];
    const ggml_tensor * slots   = set->src[2];
    const auto inside = [&](const ggml_tensor * t) { return t == nd[1] || t == nd[2] || t == nd[3]; };
    if (!slots || !windows || !order || !inside(slots) || !inside(windows) || !inside(order) || slots == windows || slots == order ||
            windows == order || windows->view_src != x || slots->view_src == nullptr || slots->view_src == x ||
            order->view_src == nullptr || order->view_src == x) {
        return 0;
    }
    const ggml_tensor * past  = x->src[0];
    const ggml_tensor * cur_t = x->src[1];
    const int64_t n_past = past->ne[0];
    const int64_t n_tok  = cur_t->ne[0];
    const int64_t n_ch   = x->ne[1];
    const int64_t n_win  = windows->ne[1];
    // several sequences (multiseq): x {n_past + n_tok, C, n_seqs}, windows and slots {.., n_seqs}, the same slot order for all
    const int64_t n_seqs = x->ne[2];
    if (n_seqs != 1 && !ggml_cuda_ink2_multiseq_enabled()) {
        return 0;
    }
    if (x->type != GGML_TYPE_F32 || past->type != GGML_TYPE_F32 || cur_t->type != GGML_TYPE_F32 || slots->type != GGML_TYPE_F32 ||
            order->type != GGML_TYPE_I32 || !ggml_is_contiguous(x) || x->ne[3] != 1 || past->ne[2] != n_seqs || cur_t->ne[2] != n_seqs ||
            past->ne[3] != 1 || cur_t->ne[3] != 1 ||
            x->ne[0] != n_past + n_tok || past->ne[1] != n_ch || cur_t->ne[1] != n_ch || n_win < 1 || n_win > n_tok ||
            windows->ne[0] != n_past || windows->ne[2] != n_ch || windows->ne[3] != n_seqs ||
            windows->nb[0] != sizeof(float) || windows->nb[1] != sizeof(float) || windows->nb[2] != x->nb[1] ||
            (n_seqs > 1 && windows->nb[3] != x->nb[2]) ||
            windows->view_offs != (size_t) (n_tok - (n_win - 1))*sizeof(float) ||
            slots->ne[0] != n_past || slots->ne[1] < n_win || slots->ne[2] != n_ch || slots->ne[3] != n_seqs ||
            order->ne[0] != n_win || order->nb[0] != sizeof(int32_t) || ggml_nelements(order) != n_win) {
        return 0;
    }
    for (const ggml_tensor * t : { past, cur_t, slots }) {
        if (t->nb[0] % sizeof(float) != 0 || t->nb[1] % sizeof(float) != 0 || t->nb[2] % sizeof(float) != 0 || t->nb[3] % sizeof(float) != 0) {
            return 0;
        }
    }
    m.past       = past;
    m.cur_t      = cur_t;
    m.slots      = slots;
    m.slot_order = order;
    m.n_win      = (int) n_win;
    return n;
}

struct ggml_cuda_ink2_relbias_match {
    const ggml_tensor * bias;
    const ggml_tensor * R;
    const ggml_tensor * sinks;
    const ggml_tensor * rel_pos;
    const ggml_tensor * mask;
};

// dense relative bias of one attention call (inkling2.cpp, dense path, one chunk):
// PAD(bias), MUL_MAT(., R), RESHAPE(sinks), SUB, PERMUTE, CONT, GET_ROWS(., rel_pos), RESHAPE, PERMUTE, CONT, CPY(mask -> f32), ADD,
// CPY(-> f16)
static int ggml_cuda_ink2_match_relbias(const ggml_cgraph * cgraph, const int i, ggml_cuda_ink2_relbias_match & m) {
    static const ggml_op ops[] = { GGML_OP_PAD, GGML_OP_MUL_MAT, GGML_OP_RESHAPE, GGML_OP_SUB, GGML_OP_PERMUTE, GGML_OP_CONT, GGML_OP_GET_ROWS,
                                   GGML_OP_RESHAPE, GGML_OP_PERMUTE, GGML_OP_CONT, GGML_OP_CPY, GGML_OP_ADD, GGML_OP_CPY };
    constexpr int n = 13;
    if (!(ggml_cuda_ink2_fusion_mask() & 128) || cgraph->nodes[i]->op != GGML_OP_PAD || i + n > cgraph->n_nodes) {
        return 0;
    }
    ggml_tensor ** nd = cgraph->nodes + i;
    for (int k = 0; k < n; ++k) {
        if (nd[k]->op != ops[k] || (nd[k]->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            return 0;
        }
        if (k < n - 1) {
            // an intermediate result: read once, by the next step of the chain (a cast also lists itself as a source)
            const int uses = ggml_node_get_use_count(cgraph, i + k);
            const int self = nd[k]->src[1] == nd[k] ? 1 : 0;
            if ((nd[k]->flags & GGML_TENSOR_FLAG_OUTPUT) || (uses != 1 && uses != 1 + self)) {
                return 0;
            }
        }
    }
    if (nd[1]->src[0] != nd[0] || nd[3]->src[0] != nd[1] || nd[3]->src[1] != nd[2] || nd[4]->src[0] != nd[3] || nd[5]->src[0] != nd[4] ||
            nd[6]->src[0] != nd[5] || nd[7]->src[0] != nd[6] || nd[8]->src[0] != nd[7] || nd[9]->src[0] != nd[8] ||
            nd[11]->src[0] != nd[9] || nd[11]->src[1] != nd[10] || nd[12]->src[0] != nd[11]) {
        return 0;
    }
    const ggml_tensor * bias    = nd[0]->src[0];
    const ggml_tensor * R       = nd[1]->src[1];
    const ggml_tensor * sinks   = nd[2]->src[0];
    const ggml_tensor * rel_pos = nd[6]->src[1];
    const ggml_tensor * mask    = nd[10]->src[0];
    const ggml_tensor * dst     = nd[12];
    // PAD: one zero row after the last distance
    for (int k = 0; k < 8; ++k) {
        if (ggml_get_op_params_i32(nd[0], k) != (k == 3 ? 1 : 0)) {
            return 0;
        }
    }
    if (ggml_get_op_params_i32(nd[0], 8) != 0) {
        return 0;
    }
    const int64_t n_rel  = bias->ne[0];
    const int64_t extent = bias->ne[1];
    const int64_t n_head = R->ne[1];
    const int64_t n_tok  = R->ne[2];
    const int64_t n_kv   = rel_pos->ne[0];
    // only for small batches (decode, verify batches of a drafter), where the launches dominate: on a 512-token batch one
    // 16-term dot per (key, token, head) is slower than the table product followed by a gather (measured: +66 ms per ubatch)
    if (n_tok > 16) {
        return 0;
    }
    if (!ggml_cuda_ink2_f32_contiguous(bias) || !ggml_cuda_ink2_f32_contiguous(R) || !ggml_cuda_ink2_f32_contiguous(sinks) ||
            rel_pos->type != GGML_TYPE_I32 || !ggml_is_contiguous(rel_pos) || mask->type != GGML_TYPE_F16 || !ggml_is_contiguous(mask) ||
            dst->type != GGML_TYPE_F16 || !ggml_is_contiguous(dst) || nd[10]->type != GGML_TYPE_F32) {
        return 0;
    }
    // several streams (multiseq): the gathered bias is {n_kv, nr, n_head, n_stream} and the mask {n_kv, nr, 1, n_stream}, with
    // token t = s nr + j as row j of stream s (rel_pos, R and the sinks stay indexed by t)
    const int64_t n_stream = nd[9]->ne[3];
    const int64_t nr       = nd[9]->ne[1];
    if (n_stream < 1 || (n_stream != 1 && !ggml_cuda_ink2_multiseq_enabled()) || nr*n_stream != n_tok) {
        return 0;
    }
    if (bias->ne[2] != 1 || bias->ne[3] != 1 || R->ne[0] != n_rel || R->ne[3] != 1 || n_rel > 64 ||
            nd[0]->ne[0] != n_rel || nd[0]->ne[1] != extent + 1 ||
            nd[1]->ne[0] != extent + 1 || nd[1]->ne[1] != n_head || nd[1]->ne[2] != n_tok ||
            ggml_nelements(sinks) != n_head*n_tok || nd[2]->ne[0] != 1 || nd[2]->ne[1] != n_head || nd[2]->ne[2] != n_tok ||
            nd[5]->ne[0] != n_head || nd[5]->ne[1] != extent + 1 || nd[5]->ne[2] != n_tok || nd[5]->ne[3] != 1 ||
            rel_pos->ne[1] != n_tok || rel_pos->ne[2] != 1 || rel_pos->ne[3] != 1 ||
            nd[6]->ne[0] != n_head || nd[6]->ne[1] != n_kv || nd[6]->ne[2] != n_tok || nd[6]->ne[3] != 1 ||
            nd[7]->ne[0] != n_head || nd[7]->ne[1] != n_kv || nd[7]->ne[2] != nr || nd[7]->ne[3] != n_stream ||
            nd[9]->ne[0] != n_kv || nd[9]->ne[2] != n_head ||
            mask->ne[0] != n_kv || mask->ne[1] != nr || mask->ne[2] != 1 || mask->ne[3] != n_stream ||
            !ggml_are_same_shape(dst, nd[9]) || n_kv*n_tok*n_head >= (int64_t(1) << 31)) {
        return 0;
    }
    // the two permutes: (1, 0, 2, 3) of the table and (2, 0, 1, 3) of the gathered rows
    if (ggml_get_op_params_i32(nd[4], 0) != 1 || ggml_get_op_params_i32(nd[4], 1) != 0 || ggml_get_op_params_i32(nd[4], 2) != 2 ||
            ggml_get_op_params_i32(nd[8], 0) != 2 || ggml_get_op_params_i32(nd[8], 1) != 0 || ggml_get_op_params_i32(nd[8], 2) != 1) {
        return 0;
    }
    m.bias    = bias;
    m.R       = R;
    m.sinks   = sinks;
    m.rel_pos = rel_pos;
    m.mask    = mask;
    return n;
}

struct ggml_cuda_ink2_sconv_full_match {
    bool                          rb;
    ggml_cuda_ink2_sconv_rb_match in_rb;
    ggml_cuda_ink2_sconv_in_match in;
    ggml_cuda_ink2_sconv_match    tail;
};

// a whole short convolution: its input and state store (sconv_rb or sconv_in) immediately followed by its tail (sconv) on the
// same x, read by nothing else (inkling2.cpp expands the convolution right after its state store). Mask 1024.
static int ggml_cuda_ink2_match_sconv_full(const ggml_cgraph * cgraph, const int i, ggml_cuda_ink2_sconv_full_match & m) {
    if (!(ggml_cuda_ink2_fusion_mask() & 1024) || cgraph->nodes[i]->op != GGML_OP_CONCAT) {
        return 0;
    }
    int n1 = ggml_cuda_ink2_match_sconv_rb(cgraph, i, m.in_rb);
    m.rb = n1 > 0;
    if (!m.rb) {
        n1 = ggml_cuda_ink2_match_sconv_in(cgraph, i, m.in);
    }
    if (n1 == 0 || i + n1 >= cgraph->n_nodes || cgraph->nodes[i + n1]->op != GGML_OP_SSM_CONV) {
        return 0;
    }
    const int n2 = ggml_cuda_ink2_match_sconv(cgraph, i + n1, m.tail);
    const ggml_tensor * x = cgraph->nodes[i];
    if (n2 == 0 || m.tail.x != x || ggml_node_get_use_count(cgraph, i) != 2 || (x->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        return 0;
    }
    // the tail's cur is the convolution input itself (cur^T is its transpose)
    const ggml_tensor * cur_t = m.rb ? m.in_rb.cur_t : m.in.cur_t;
    const ggml_tensor * cur_root  = cur_t->view_src ? cur_t->view_src : cur_t;
    const ggml_tensor * tail_root = m.tail.cur->view_src ? m.tail.cur->view_src : m.tail.cur;
    if (cur_root != tail_root || m.tail.w->ne[0] != (m.rb ? m.in_rb.past : m.in.past)->ne[0] + 1) {
        return 0;
    }
    return n1 + n2;
}

struct ggml_cuda_ink2_dconv_match {
    const ggml_tensor * x;
    const ggml_tensor * w;
    const ggml_tensor * D;   // the dynamic taps viewed as {n_conv, n_heads, n_tokens}
    const ggml_tensor * cur;
    int                 n_heads;
    float               scale;
};

// dynamic short convolution of the full-attention layers (inkling2.cpp, convolve with dynamic taps):
// SSM_CONV(x, w), RESHAPE, then per lag VIEW(x), TRANSPOSE, CONT, RESHAPE, [RESHAPE of the taps], VIEW(taps), MUL, RESHAPE, ADD,
// and ADD(cur, .), SCALE: 37 nodes for 4 taps, 15 kernels -> 1. The nodes are found by following the data flow; every node of
// the range must belong to the pattern and be read only inside it. Mask 512.
static int ggml_cuda_ink2_match_dconv(const ggml_cgraph * cgraph, const int i, ggml_cuda_ink2_dconv_match & m) {
    if (!(ggml_cuda_ink2_fusion_mask() & 512) || cgraph->nodes[i]->op != GGML_OP_SSM_CONV) {
        return 0;
    }
    const int i_max = std::min(cgraph->n_nodes, i + 64);
    const ggml_tensor * conv = cgraph->nodes[i];
    const ggml_tensor * x    = conv->src[0];
    const ggml_tensor * w    = conv->src[1];
    // several sequences (multiseq): x {n_x, C, n_seqs}; n_tok is per sequence, n_all over the batch
    const int64_t n_seqs = x->ne[2];
    if (!ggml_cuda_ink2_f32_contiguous(x) || !ggml_cuda_ink2_f32_contiguous(w) || (n_seqs != 1 && !ggml_cuda_ink2_multiseq_enabled()) ||
            x->ne[3] != 1 || w->ne[1] != x->ne[1]) {
        return 0;
    }
    const int64_t n_conv = w->ne[0];
    const int64_t n_ch   = x->ne[1];
    const int64_t n_tok  = x->ne[0] - n_conv + 1;
    const int64_t n_all  = n_tok*n_seqs;
    if (n_conv < 1 || n_conv > 8 || n_tok < 1 || conv->ne[0] != n_ch || conv->ne[1] != n_tok || conv->ne[2] != n_seqs) {
        return 0;
    }
    // the first consumer of t with this op in (i, i_max), -1 if none
    const auto consumer = [&](const ggml_tensor * t, const ggml_op op) {
        for (int k = i + 1; k < i_max; ++k) {
            const ggml_tensor * n = cgraph->nodes[k];
            if (n->op == op && (n->src[0] == t || n->src[1] == t)) {
                return k;
            }
        }
        return -1;
    };
    const ggml_tensor * set[64];
    int n_set = 0;
    set[n_set++] = conv;

    const int k_out = consumer(conv, GGML_OP_RESHAPE);
    if (k_out < 0 || cgraph->nodes[k_out]->ne[0] != n_ch || cgraph->nodes[k_out]->ne[1] != n_all) {
        return 0;
    }
    const ggml_tensor * out    = cgraph->nodes[k_out];
    const ggml_tensor * d_resh = nullptr;
    set[n_set++] = out;
    for (int64_t lag = 0; lag < n_conv; ++lag) {
        const int k_add = consumer(out, GGML_OP_ADD);
        if (k_add < 0) {
            return 0;
        }
        const ggml_tensor * add  = cgraph->nodes[k_add];
        const ggml_tensor * term = add->src[0] == out ? add->src[1] : add->src[0];
        if (!term || term->op != GGML_OP_RESHAPE || term->src[0]->op != GGML_OP_MUL) {
            return 0;
        }
        const ggml_tensor * mul = term->src[0];
        const ggml_tensor * a   = mul->src[0];
        const ggml_tensor * tap = mul->src[1];
        if (!a || !tap || a->op != GGML_OP_RESHAPE || a->src[0]->op != GGML_OP_CONT || a->src[0]->src[0]->op != GGML_OP_TRANSPOSE ||
                a->src[0]->src[0]->src[0]->op != GGML_OP_VIEW || tap->op != GGML_OP_VIEW) {
            return 0;
        }
        const ggml_tensor * cont   = a->src[0];
        const ggml_tensor * transp = cont->src[0];
        const ggml_tensor * window = transp->src[0];
        // window: rows n_conv - 1 - lag .. of x, all channels (of every sequence)
        if (window->view_src != x || window->view_offs != (size_t) (n_conv - 1 - lag)*sizeof(float) || window->ne[0] != n_tok ||
                window->ne[1] != n_ch || window->nb[1] != x->nb[1] || window->ne[2] != n_seqs || window->ne[3] != 1 ||
                (n_seqs > 1 && window->nb[2] != x->nb[2]) || a->ne[2] != n_all || a->ne[0]*a->ne[1] != n_ch) {
            return 0;
        }
        // tap: element lag of the taps {n_conv, n_heads, n_tokens}
        const ggml_tensor * dr = tap->src[0];
        if (dr == nullptr || (d_resh && dr != d_resh) || !ggml_cuda_ink2_f32_contiguous(dr) || dr->ne[0] != n_conv || dr->ne[2] != n_all ||
                dr->ne[3] != 1 || tap->ne[0] != 1 || tap->ne[1] != dr->ne[1] || tap->ne[2] != n_all || tap->nb[1] != dr->nb[1] ||
                tap->nb[2] != dr->nb[2] || tap->view_offs != dr->view_offs + (size_t) lag*sizeof(float) || a->ne[1] != dr->ne[1]) {
            return 0;
        }
        if (!d_resh) {
            d_resh = dr;
            set[n_set++] = dr;
        }
        set[n_set++] = window;
        set[n_set++] = transp;
        set[n_set++] = cont;
        set[n_set++] = a;
        set[n_set++] = tap;
        set[n_set++] = mul;
        set[n_set++] = term;
        set[n_set++] = add;
        out = add;
    }
    const int k_fin = consumer(out, GGML_OP_ADD);
    if (k_fin < 0) {
        return 0;
    }
    const ggml_tensor * fin = cgraph->nodes[k_fin];
    const ggml_tensor * cur = fin->src[0] == out ? fin->src[1] : fin->src[0];
    const int k_sc = consumer(fin, GGML_OP_SCALE);
    if (k_sc < 0 || !ggml_cuda_ink2_f32_contiguous(cur) || ggml_nelements(cur) != n_ch*n_all ||
            ggml_get_op_params_f32(cgraph->nodes[k_sc], 1) != 0.0f || !ggml_cuda_ink2_f32_contiguous(cgraph->nodes[k_sc]) ||
            ggml_nelements(cgraph->nodes[k_sc]) != n_ch*n_all) {
        return 0;
    }
    set[n_set++] = fin;
    set[n_set++] = cgraph->nodes[k_sc];
    const auto in_set = [&](const ggml_tensor * t) {
        for (int s = 0; s < n_set; ++s) {
            if (set[s] == t) {
                return true;
            }
        }
        return false;
    };
    // the range [i, k_sc] is exactly the pattern (only the reshape of the taps may sit before it), read only inside
    int  n_in = 0;
    bool d_in = false;
    for (int k = i; k <= k_sc; ++k) {
        const ggml_tensor * n = cgraph->nodes[k];
        if (!in_set(n) || (n->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            return 0;
        }
        n_in++;
        d_in = d_in || n == d_resh;
        if (k == k_sc) {
            break;
        }
        if (n->flags & GGML_TENSOR_FLAG_OUTPUT) {
            return 0;
        }
        int uses = 0;
        for (int q = k + 1; q <= k_sc; ++q) {
            for (int s = 0; s < GGML_MAX_SRC; ++s) {
                uses += cgraph->nodes[q]->src[s] == n ? 1 : 0;
            }
        }
        if (uses != ggml_node_get_use_count(cgraph, k)) {
            return 0;
        }
    }
    if (n_in != n_set - (d_in ? 0 : 1)) {
        return 0;
    }
    m.x       = x;
    m.w       = w;
    m.D       = d_resh;
    m.cur     = cur;
    m.n_heads = (int) d_resh->ne[1];
    m.scale   = ggml_get_op_params_f32(cgraph->nodes[k_sc], 0);
    return k_sc - i + 1;
}

// The first index of the maximum of a row, as the DSpark drafting graph selects its tokens (independent of the argmax tie
// order): CONT(row view), POOL_1D(max over the row), REPEAT, SUB(repeat, row), STEP, SCALE(-1, 1), MUL(., tie), ARGMAX, where tie
// = SCALE(ARANGE(0, n, 1), -1, n) is the strictly decreasing rank n - k (checked structurally). One kernel instead of eight.
// Mask bit 4096.
static int ggml_cuda_ink2_match_first_max(const ggml_cgraph * cgraph, const int i, const ggml_tensor *& row) {
    static const ggml_op ops[] = { GGML_OP_CONT, GGML_OP_POOL_1D, GGML_OP_REPEAT, GGML_OP_SUB, GGML_OP_UNARY, GGML_OP_SCALE, GGML_OP_MUL, GGML_OP_ARGMAX };
    constexpr int n = 8;
    if (!(ggml_cuda_ink2_fusion_mask() & 4096) || cgraph->nodes[i]->op != GGML_OP_CONT || i + n > cgraph->n_nodes ||
            cgraph->nodes[i + 1]->op != GGML_OP_POOL_1D) {
        return 0;
    }
    const int out = i + n - 1;
    if (!ggml_can_fuse_subgraph(cgraph, i, n, ops, &out, 1)) {
        return 0;
    }
    ggml_tensor ** nd = cgraph->nodes + i;
    const ggml_tensor * v   = nd[0];
    const ggml_tensor * src = v->src[0];
    const int64_t       len = v->ne[0];
    if (!ggml_cuda_ink2_f32_contiguous(v) || ggml_nrows(v) != 1 || src == nullptr || src->type != GGML_TYPE_F32 || src->nb[0] != sizeof(float) ||
            src->ne[0] != len || ggml_nrows(src) != 1 || len >= INT_MAX) {
        return 0;
    }
    const int32_t * pp = (const int32_t *) nd[1]->op_params;
    if (nd[1]->src[0] != v || pp[0] != GGML_OP_POOL_MAX || pp[1] != len || pp[2] != 1 || pp[3] != 0 || ggml_nelements(nd[1]) != 1) {
        return 0;
    }
    if (nd[2]->src[0] != nd[1] || !ggml_are_same_shape(nd[2], v) || nd[3]->src[0] != nd[2] || nd[3]->src[1] != v ||
            nd[4]->src[0] != nd[3] || ggml_get_unary_op(nd[4]) != GGML_UNARY_OP_STEP || nd[5]->src[0] != nd[4] ||
            ggml_get_op_params_f32(nd[5], 0) != -1.0f || ggml_get_op_params_f32(nd[5], 1) != 1.0f) {
        return 0;
    }
    const ggml_tensor * tie = nd[6]->src[0] == nd[5] ? nd[6]->src[1] : (nd[6]->src[1] == nd[5] ? nd[6]->src[0] : nullptr);
    if (tie == nullptr || !ggml_are_same_shape(tie, v) || tie->op != GGML_OP_SCALE || ggml_get_op_params_f32(tie, 0) != -1.0f ||
            ggml_get_op_params_f32(tie, 1) != (float) len || tie->src[0] == nullptr || tie->src[0]->op != GGML_OP_ARANGE ||
            ggml_get_op_params_f32(tie->src[0], 0) != 0.0f || ggml_get_op_params_f32(tie->src[0], 1) != (float) len ||
            ggml_get_op_params_f32(tie->src[0], 2) != 1.0f) {
        return 0;
    }
    if (nd[7]->src[0] != nd[6] || nd[7]->type != GGML_TYPE_I32 || ggml_nelements(nd[7]) != 1) {
        return 0;
    }
    row = src;
    return n;
}

// number of nodes of the Inkling 2 pattern starting at i, 0 if none
// (the expert product + SITU pattern starting at a MUL_MAT_ID is handled apart, see graph_optimize)
// post-projection row ops of an attention layer (kind 16 "post"): a run of L2 clips (scaled or not) and RMS norms of rows,
// each optionally followed by its product with a per-token scalar (MUL) and by the K/V cache store of its rows (SET_ROWS of a
// view of it), with only views in between; inkling2.cpp orders the r, q, k and v ops of a layer that way (the clip of r and
// of q, the k clip, the v norm, the log-scaled positions of the full-attention layers, the two stores). One launch for what
// were 4 to 8. GGML_CUDA_INK2_POST=0 disables it.
static bool ggml_cuda_ink2_post_enabled() {
    static const bool enabled = []() { const char * e = getenv("GGML_CUDA_INK2_POST"); return !(e && e[0] == '0'); }();
    return enabled;
}

struct ggml_cuda_ink2_post_match {
    struct item {
        int                 kind      = 0;       // 0 = L2 clip, 1 = RMS norm
        int                 node0     = -1;      // first node of the op
        const ggml_tensor * x         = nullptr;
        const ggml_tensor * scale     = nullptr;
        ggml_tensor       * out       = nullptr;
        float               threshold = 0.0f;
        float               eps       = 0.0f;
        ggml_tensor       * mul       = nullptr; // MUL of out by a per-token scalar
        ggml_tensor       * store     = nullptr; // SET_ROWS of out
    } it[GGML_CUDA_INK2_POST_MAX];
    int n        = 0;
    int launches = 0; // launches of the separate ops
};

static int ggml_cuda_ink2_match_post(const ggml_cgraph * cgraph, const int i, ggml_cuda_ink2_post_match & m) {
    const ggml_op op0 = cgraph->nodes[i]->op;
    if (!ggml_cuda_ink2_post_enabled() || !(ggml_cuda_ink2_fusion_mask() & 1) || (op0 != GGML_OP_MUL && op0 != GGML_OP_SQR)) {
        return 0;
    }
    m.n        = 0;
    m.launches = 0;
    // rows [d, heads, tokens] of small batches only (decode and verify): in prompt batches the run would keep its buffers
    // alive longer for a negligible gain
    const auto rows_ok = [](const ggml_tensor * t) {
        return ggml_cuda_ink2_f32_contiguous(t) && t->ne[3] == 1 && t->ne[0] <= 4*WARP_SIZE && t->ne[2] <= 16;
    };
    const auto produced = [&](const ggml_tensor * t) {
        const ggml_tensor * r = t->view_src ? t->view_src : t;
        for (int k = 0; k < m.n; ++k) {
            if (r == m.it[k].out || r == m.it[k].mul || r == m.it[k].store || (m.it[k].store && r == m.it[k].store->view_src)) {
                return true;
            }
        }
        return false;
    };
    int j    = i;
    int last = -1;
    while (j < cgraph->n_nodes) {
        ggml_tensor * node = cgraph->nodes[j];
        if (node->op == GGML_OP_RESHAPE || node->op == GGML_OP_VIEW) {
            j++;
            continue;
        }
        ggml_cuda_ink2_clip_match cm;
        int nc = 0;
        if (m.n < GGML_CUDA_INK2_POST_MAX && node->op == GGML_OP_MUL) {
            nc = ggml_cuda_ink2_match_clip_scaled(cgraph, j, cm);
        } else if (m.n < GGML_CUDA_INK2_POST_MAX && node->op == GGML_OP_SQR) {
            nc = ggml_cuda_ink2_match_clip(cgraph, j, cm);
        }
        if (nc > 0) {
            ggml_tensor * out = cgraph->nodes[j + nc - 1];
            if (!rows_ok(cm.x) || !rows_ok(out) || produced(cm.x)) {
                break;
            }
            auto & it = m.it[m.n++];
            it = {};
            it.kind = 0; it.node0 = j; it.x = cm.x; it.scale = cm.scale; it.out = out; it.threshold = cm.threshold; it.eps = cm.eps;
            m.launches++;
            j += nc;
            last = j - 1;
            continue;
        }
        if (m.n < GGML_CUDA_INK2_POST_MAX && node->op == GGML_OP_RMS_NORM) {
            const ggml_tensor * x = node->src[0];
            // the generic op takes its 32-thread float4 kernel for these rows (aligned 128-element rows)
            if (!ggml_cuda_ink2_f32_contiguous(x) || !ggml_cuda_ink2_f32_contiguous(node) || x->ne[0] != 4*WARP_SIZE || x->ne[3] != 1 || x->ne[2] > 16 ||
                    !ggml_are_same_shape(x, node) || produced(x)) {
                break;
            }
            auto & it = m.it[m.n++];
            it = {};
            it.kind = 1; it.node0 = j; it.x = x; it.out = node; it.eps = ggml_get_op_params_f32(node, 0);
            m.launches++;
            j++;
            last = j - 1;
            continue;
        }
        if (node->op == GGML_OP_MUL && m.n > 0) {
            // the product of the last op's rows by one scalar per token
            auto & it = m.it[m.n - 1];
            const ggml_tensor * s = node->src[1];
            if (node->src[0] != it.out || it.mul || it.store || !ggml_cuda_ink2_f32_contiguous(s) || s->ne[0] != 1 || s->ne[1] != 1 ||
                    s->ne[2] != it.out->ne[2] || s->ne[3] != 1 || !ggml_cuda_ink2_f32_contiguous(node) || !ggml_are_same_shape(node, it.out) ||
                    produced(s)) {
                break;
            }
            it.mul = node;
            m.launches++;
            j++;
            last = j - 1;
            continue;
        }
        if (node->op == GGML_OP_SET_ROWS && m.n > 0) {
            // the store of the rows of one op, all heads of a token in one cache row
            const ggml_tensor * src = node->src[0];
            const ggml_tensor * idx = node->src[1];
            int k = m.n - 1;
            while (k >= 0 && !(src->view_src == m.it[k].out && src->view_offs == 0)) {
                k--;
            }
            if (k < 0) {
                break;
            }
            auto & it = m.it[k];
            const ggml_tensor * o = it.out;
            if (it.mul || it.store || !ggml_is_contiguous(src) || src->type != GGML_TYPE_F32 || src->ne[0] != o->ne[0]*o->ne[1] ||
                    src->ne[1] != o->ne[2] || src->ne[2] != 1 || src->ne[3] != 1 || idx->type != GGML_TYPE_I64 || !ggml_is_contiguous(idx) ||
                    idx->ne[0] != src->ne[1] || idx->ne[1] != 1 || idx->ne[2] != 1 || node->ne[0] != src->ne[0] || node->ne[2] != 1 ||
                    node->ne[3] != 1 || !(node->type == GGML_TYPE_F16 || (node->type == GGML_TYPE_Q8_0 && o->ne[0] == 4*WARP_SIZE)) ||
                    produced(idx)) {
                break;
            }
            it.store = node;
            m.launches++;
            j++;
            last = j - 1;
            continue;
        }
        break;
    }
    if (m.launches < 2 || last < i) {
        return 0;
    }
    return last - i + 1;
}

static int ggml_cuda_ink2_match_sconv_pair(const ggml_cgraph * cgraph, int i, ggml_cuda_ink2_sconv_full_match & m1,
        ggml_cuda_ink2_sconv_full_match & m2, int & n1, int & j2, int & n2);

// 2 or 3 loop gates (kind 1) with only views between them, of the same shape, none reading what an earlier one computes (the
// q, k and v controllers of a layer application, which inkling2.cpp puts one after the other): one launch of each gate kernel
// for all of them. Returns the node count from i, the gates in g[0..ng) and their outputs in out[]. GGML_CUDA_INK2_GATE_N=0:
// one gate per launch.
static int ggml_cuda_ink2_match_gate_n(const ggml_cgraph * cgraph, const int i, ggml_cuda_ink2_gate_match * g, ggml_tensor ** out, int & ng) {
    static const bool enabled = []() { const char * e = getenv("GGML_CUDA_INK2_GATE_N"); return !(e && e[0] == '0'); }();
    ng = 0;
    if (!enabled) {
        return 0;
    }
    int j    = i;
    int last = -1;
    while (ng < GGML_CUDA_INK2_GATE_N_MAX && j < cgraph->n_nodes) {
        if (ng > 0 && ggml_cuda_is_view_or_noop(cgraph->nodes[j])) {
            j++;
            continue;
        }
        if (cgraph->nodes[j]->op != GGML_OP_CONCAT) {
            break;
        }
        ggml_cuda_ink2_gate_match m;
        const int n = ggml_cuda_ink2_match_gate(cgraph, j, m);
        if (n == 0) {
            break;
        }
        ggml_tensor * dst = cgraph->nodes[j + n - 1];
        if (!ggml_cuda_ink2_loop_gate_n_ok(m.cur, m.msg, m.down, m.gain, m.forget, dst)) {
            break;
        }
        if (ng > 0 && (dst->ne[0] != out[0]->ne[0] || ggml_nelements(dst) != ggml_nelements(out[0]) || m.down->ne[1] != g[0].down->ne[1] ||
                m.down->type != g[0].down->type || m.eps != g[0].eps)) {
            break;
        }
        // independent of the earlier gates of the run
        bool dep = false;
        for (const ggml_tensor * t : { m.cur, m.msg }) {
            const ggml_tensor * r = t->view_src ? t->view_src : t;
            for (int k = i; k < j && !dep; ++k) {
                dep = cgraph->nodes[k] == t || cgraph->nodes[k] == r;
            }
        }
        if (dep) {
            break;
        }
        g[ng]   = m;
        out[ng] = dst;
        ng++;
        j   += n;
        last = j - 1;
    }
    if (ng < 2) {
        ng = 0;
        return 0;
    }
    return last - i + 1;
}

static int ggml_cuda_ink2_match_any(const ggml_cgraph * cgraph, const int i) {
    const ggml_op op = cgraph->nodes[i]->op;
    if (op == GGML_OP_MUL || op == GGML_OP_SQR) {
        ggml_cuda_ink2_post_match pm;
        if (const int n = ggml_cuda_ink2_match_post(cgraph, i, pm)) {
            return n;
        }
    }
    if (op == GGML_OP_CONT) {
        const ggml_tensor * row = nullptr;
        return ggml_cuda_ink2_match_first_max(cgraph, i, row);
    }
    if (op == GGML_OP_PAD) {
        ggml_cuda_ink2_relbias_match m;
        return ggml_cuda_ink2_match_relbias(cgraph, i, m);
    }
    if (op == GGML_OP_SQR) {
        ggml_cuda_ink2_clip_match m;
        return ggml_cuda_ink2_match_clip(cgraph, i, m);
    }
    if (op == GGML_OP_MUL) {
        ggml_cuda_ink2_clip_match m;
        return ggml_cuda_ink2_match_clip_scaled(cgraph, i, m);
    }
    if (op == GGML_OP_CONCAT) {
        ggml_cuda_ink2_sconv_full_match f;
        ggml_cuda_ink2_sconv_full_match f2;
        int n1 = 0;
        int j2 = 0;
        int n2 = 0;
        if (const int n = ggml_cuda_ink2_match_sconv_pair(cgraph, i, f, f2, n1, j2, n2)) {
            return n;
        }
        if (const int n = ggml_cuda_ink2_match_sconv_full(cgraph, i, f)) {
            return n;
        }
        ggml_cuda_ink2_sconv_in_match c;
        if (const int n = ggml_cuda_ink2_match_sconv_in(cgraph, i, c)) {
            return n;
        }
        ggml_cuda_ink2_sconv_rb_match rb;
        if (const int n = ggml_cuda_ink2_match_sconv_rb(cgraph, i, rb)) {
            return n;
        }
        ggml_cuda_ink2_gate_match gn[GGML_CUDA_INK2_GATE_N_MAX];
        ggml_tensor * gout[GGML_CUDA_INK2_GATE_N_MAX];
        int ng = 0;
        if (const int n = ggml_cuda_ink2_match_gate_n(cgraph, i, gn, gout, ng)) {
            return n;
        }
        ggml_cuda_ink2_gate_match g;
        if (const int n = ggml_cuda_ink2_match_gate(cgraph, i, g)) {
            return n;
        }
        ggml_cuda_ink2_mix_match m;
        return ggml_cuda_ink2_match_mix(cgraph, i, m);
    }
    if (op == GGML_OP_SSM_CONV) {
        ggml_cuda_ink2_sconv_match m;
        if (const int n = ggml_cuda_ink2_match_sconv(cgraph, i, m)) {
            return n;
        }
        ggml_cuda_ink2_dconv_match d;
        return ggml_cuda_ink2_match_dconv(cgraph, i, d);
    }
    if (op == GGML_OP_SCALE) {
        ggml_cuda_ink2_situ_match m;
        return ggml_cuda_ink2_match_situ(cgraph, i, m);
    }
    if (op == GGML_OP_SUM_ROWS) {
        ggml_cuda_ink2_wnorm_match m;
        return ggml_cuda_ink2_match_wnorm(cgraph, i, m);
    }
    if (op == GGML_OP_UNARY) {
        ggml_cuda_ink2_gate_match m;
        return ggml_cuda_ink2_match_gate_tail(cgraph, i, m);
    }
    return 0;
}

// GGML_CUDA_INK2_VERIFY=1 (with GGML_CUDA_DISABLE_GRAPHS=1): compute every fused pattern with the generic ops first, then with
// the fused kernel, and track the normalized squared error between the two outputs; the maxima are printed at exit
static bool ggml_cuda_ink2_verify_enabled() {
    static const bool enabled = ggml_cuda_env_flag("GGML_CUDA_INK2_VERIFY");
    return enabled;
}

template <typename F>
static void ggml_cuda_ink2_run(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, const int i, const int n, const int kind, F && fused,
        const bool inplace_ok = false) {
    // TIMING EXPERIMENT ONLY (wrong output): GGML_CUDA_INK2_SKIP_KIND=<bits> does not run the fused kernels of these kinds
#if defined(GGML_CUDA_INK2_COST_TOOLS)
    static const int skip_kinds = []() { const char * e = getenv("GGML_CUDA_INK2_SKIP_KIND"); return e ? atoi(e) : 0; }();
#else
    static const int skip_kinds = 0;
#endif
    if (skip_kinds & (1 << kind)) {
        return;
    }
    if (!ggml_cuda_ink2_verify_enabled()) {
        fused();
        return;
    }
    static const char * names[17] = { "clip", "gate", "mix", "sconv", "situ", "wnorm", "sconv_in", "relbias", "sconv_rb", "dconv", "sconv_full", "gate_tail",
                                      "first_max", "moe_situ", "moe_ffn", "gate_pp", "post" };
    static double  max_nmse = 0.0;
    static int64_t count    = 0;
    static int64_t skipped  = 0;
    static int64_t n_nan    = 0; // comparisons with a NaN (a non-finite value on either side): max nmse is then NaN for good
    static int     kind_s   = 0;
    if (count + skipped == 0) {
        kind_s = kind;
        atexit([]() {
            fprintf(stderr, "ink2 verify %s: %lld patterns compared, max nmse %.3g, %lld not comparable (output aliases an input)%s\n",
                names[kind_s], (long long) count, n_nan ? NAN : max_nmse, (long long) skipped,
                n_nan ? (", NaN in " + std::to_string(n_nan) + " comparisons").c_str() : "");
        });
    }
    ggml_tensor * dst = cgraph->nodes[i + n - 1];
    // the generic ops would overwrite an input that the fused kernel still has to read
    for (int k = i; k < i + n; ++k) {
        for (int j = 0; j < GGML_MAX_SRC; ++j) {
            const ggml_tensor * src = cgraph->nodes[k]->src[j];
            if (src && src->data == dst->data && src != dst && !inplace_ok) {
                bool inside = false;
                for (int q = i; q < i + n; ++q) {
                    inside = inside || cgraph->nodes[q] == src;
                }
                if (!inside) {
                    skipped++;
                    fused();
                    return;
                }
            }
        }
    }
    const size_t  nb  = ggml_nbytes(dst);
    const bool    f16 = dst->type == GGML_TYPE_F16;
    std::vector<float> ref(nb/sizeof(float) + 1);
    std::vector<float> got(nb/sizeof(float) + 1);

    for (int k = i; k < i + n; ++k) {
        ggml_tensor * node = cgraph->nodes[k];
        if (ggml_cuda_is_view_or_noop(node)) {
            continue;
        }
        const bool ok = ggml_cuda_compute_forward(*cuda_ctx, node);
        GGML_ASSERT(ok);
    }
    CUDA_CHECK(cudaStreamSynchronize(cuda_ctx->stream()));
    CUDA_CHECK(cudaMemcpy(ref.data(), dst->data, nb, cudaMemcpyDeviceToHost));

    fused();
    CUDA_CHECK(cudaStreamSynchronize(cuda_ctx->stream()));
    CUDA_CHECK(cudaMemcpy(got.data(), dst->data, nb, cudaMemcpyDeviceToHost));
    // GGML_CUDA_INK2_VERIFY_POISON=1 (test of the check itself): the host copy of the first fused output of each kind gets a
    // NaN (the device data is untouched, so every later comparison is finite): the kind must report "max nmse nan" at exit
    static const bool poison = ggml_cuda_env_flag("GGML_CUDA_INK2_VERIFY_POISON");
    if (poison && count == 0) {
        memset(got.data(), 0xFF, f16 ? sizeof(ggml_fp16_t) : sizeof(float));
    }

    double num = 0.0;
    double den = 0.0;
    if (f16) {
        // half floats, with infinities (masked cells) that must match exactly
        const ggml_fp16_t * r16 = (const ggml_fp16_t *) ref.data();
        const ggml_fp16_t * g16 = (const ggml_fp16_t *) got.data();
        for (size_t j = 0; j < nb/sizeof(ggml_fp16_t); ++j) {
            const double r = GGML_FP16_TO_FP32(r16[j]);
            const double g = GGML_FP16_TO_FP32(g16[j]);
            if (r == g) {
                den += std::isfinite(r) ? r*r : 0.0;
                continue;
            }
            if (!std::isfinite(r) || !std::isfinite(g)) {
                num = NAN;
                break;
            }
            num += (r - g)*(r - g);
            den += r*r;
        }
        if (den == 0.0 && num == 0.0) {
            den = 1.0;
        }
    } else
    for (size_t j = 0; j < nb/sizeof(float); ++j) {
        const double d = (double) ref[j] - (double) got[j];
        num += d*d;
        den += (double) ref[j]*(double) ref[j];
    }
    const double nmse = num/(den + 1e-30);
    if (std::isnan(nmse)) {
        n_nan++;
    } else if (nmse > max_nmse) {
        max_nmse = nmse;
    }
    count++;
    // GGML_CUDA_INK2_VERIFY_EVERY=n: also print the running result every n comparisons (a server is stopped without its exit
    // handlers), with the number of them on batches of several sequences (conv input) or streams (bias)
    static int64_t n_multi = 0;
    n_multi += cgraph->nodes[i]->ne[2] > 1 || dst->ne[3] > 1;
    static const int64_t every = []() { const char * e = getenv("GGML_CUDA_INK2_VERIFY_EVERY"); return int64_t(e ? atoll(e) : 0); }();
    if (every > 0 && count % every == 0) {
        fprintf(stderr, "ink2 verify %s: %lld patterns compared (%lld of several sequences), max nmse %.3g, %lld not comparable, NaN in %lld\n",
            names[kind_s], (long long) count, (long long) n_multi, max_nmse, (long long) skipped, (long long) n_nan);
    }
}

// consecutive Q8_0 products of the same activations (grouped by graph_optimize) in one launch; returns the nodes to skip
static int ggml_cuda_ink2_try_q8_group(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, const int i) {
    if (!ggml_cuda_ink2_q8_group_enabled() || !ggml_cuda_ink2_arch_ok(cuda_ctx->device)) {
        return 0;
    }
    ggml_tensor * a = cgraph->nodes[i];
    int k = 1;
    // adjacent products: the activations of each are computed before the first (other activations: see graph_optimize)
    const bool same = !ggml_cuda_ink2_q8_group_multi_enabled();
    while (k < GGML_CUDA_INK2_Q8_GROUP_MAX && i + k < cgraph->n_nodes && ggml_cuda_ink2_mmv_q8_f32_group_ok(a, cgraph->nodes[i + k], same)) {
        k++;
    }
    if (k < 2) {
        return 0;
    }
    // every output is written while the activations are read: no overlap with them or with each other
    const auto overlap = [](const ggml_tensor * x, const ggml_tensor * y) {
        const char * x0 = (const char *) x->data;
        const char * y0 = (const char *) y->data;
        return x0 < y0 + ggml_nbytes(y) && y0 < x0 + ggml_nbytes(x);
    };
    for (int q = 0; q < k; ++q) {
        for (int r = 0; r < k; ++r) {
            if (overlap(cgraph->nodes[i + q], cgraph->nodes[i + r]->src[1])) {
                return 0;
            }
        }
        for (int r = 0; r < q; ++r) {
            if (overlap(cgraph->nodes[i + q], cgraph->nodes[i + r])) {
                return 0;
            }
        }
    }
    // TIMING EXPERIMENT ONLY: GGML_CUDA_INK2_SKIP_MMVQ=q8_0 does not run them
    static const char * skip_type = getenv("GGML_CUDA_INK2_SKIP_MMVQ");
    if (skip_type && strcmp(skip_type, "q8_0") == 0) {
        return k - 1;
    }
    if (ggml_cuda_ink2_verify_enabled()) {
        // reference: each product alone, then the group; max normalized squared error over the outputs, printed at exit
        static double  max_nmse = 0.0;
        static int64_t count    = 0;
        if (count == 0) {
            atexit([]() { fprintf(stderr, "ink2 verify q8group: %lld groups compared, max nmse %.3g\n", (long long) count, max_nmse); });
        }
        std::vector<std::vector<float>> ref(k);
        for (int q = 0; q < k; ++q) {
            ggml_tensor * node = cgraph->nodes[i + q];
            GGML_ASSERT(ggml_cuda_compute_forward(*cuda_ctx, node));
            CUDA_CHECK(cudaStreamSynchronize(cuda_ctx->stream()));
            ref[q].resize(ggml_nelements(node));
            CUDA_CHECK(cudaMemcpy(ref[q].data(), node->data, ggml_nbytes(node), cudaMemcpyDeviceToHost));
        }
        ggml_cuda_op_ink2_mmv_q8_f32_group(*cuda_ctx, cgraph->nodes + i, k);
        CUDA_CHECK(cudaStreamSynchronize(cuda_ctx->stream()));
        for (int q = 0; q < k; ++q) {
            ggml_tensor * node = cgraph->nodes[i + q];
            std::vector<float> got(ggml_nelements(node));
            CUDA_CHECK(cudaMemcpy(got.data(), node->data, ggml_nbytes(node), cudaMemcpyDeviceToHost));
            double num = 0.0;
            double den = 0.0;
            for (size_t e = 0; e < got.size(); ++e) {
                const double d = (double) ref[q][e] - (double) got[e];
                num += d*d;
                den += (double) ref[q][e]*(double) ref[q][e];
            }
            const double nmse = num/(den + 1e-30);
            // a NaN stays (a later finite comparison must not overwrite it)
            if (!(nmse <= max_nmse) && !std::isnan(max_nmse)) {
                max_nmse = nmse;
            }
        }
        count++;
        return k - 1;
    }
    ggml_cuda_op_ink2_mmv_q8_f32_group(*cuda_ctx, cgraph->nodes + i, k);
    return k - 1;
}

// the MoE weight normalisation (wnorm) right after a fused top-k pattern whose weights are `weights` (views between): it
// becomes the epilogue of the top-k kernel (kind "router", same arithmetic). Returns the nodes from j to its end, 0 = none.
// GGML_CUDA_INK2_ROUTER=0 keeps it a launch of its own.
static int ggml_cuda_ink2_match_router_norm(const ggml_cgraph * cgraph, const int j0, const ggml_tensor * weights, ggml_cuda_ink2_wnorm_match & m) {
    static const bool enabled = []() { const char * e = getenv("GGML_CUDA_INK2_ROUTER"); return !(e && e[0] == '0'); }();
    if (!enabled || !ggml_cuda_ink2_fusion_enabled()) {
        return 0;
    }
    int j = j0;
    while (j < cgraph->n_nodes && (cgraph->nodes[j]->op == GGML_OP_RESHAPE || cgraph->nodes[j]->op == GGML_OP_VIEW)) {
        j++;
    }
    if (j >= cgraph->n_nodes || cgraph->nodes[j]->op != GGML_OP_SUM_ROWS) {
        return 0;
    }
    const int n = ggml_cuda_ink2_match_wnorm(cgraph, j, m);
    if (n == 0 || !(m.w == weights || (m.w->view_src == weights && m.w->view_offs == 0)) || !ggml_is_contiguous(weights) ||
            ggml_nelements(m.w) != ggml_nelements(weights) || weights->ne[1] > WARP_SIZE) {
        return 0;
    }
    for (int k = j0; k < j; ++k) {
        // only views of the weights in between
        if (cgraph->nodes[k]->view_src != weights) {
            return 0;
        }
    }
    return j + n - j0;
}

static const ggml_tensor * ggml_cuda_ink2_root(const ggml_tensor * t) {
    while (t->view_src) {
        t = t->view_src;
    }
    return t;
}

static bool ggml_cuda_ink2_overlap(const ggml_tensor * a, const ggml_tensor * b) {
    const char * a0 = (const char *) a->data;
    const char * b0 = (const char *) b->data;
    return a0 < b0 + ggml_nbytes(b) && b0 < a0 + ggml_nbytes(a);
}

// the buffer checks of a matched sconv_full pattern (nodes [i, i + n)) and the description of its launch
static bool ggml_cuda_ink2_sconv_full_check(const ggml_cgraph * cgraph, const int i, const int n, const ggml_cuda_ink2_sconv_full_match & m,
        ggml_cuda_ink2_sconv_desc & d) {
    const auto root    = ggml_cuda_ink2_root;
    const auto overlap = ggml_cuda_ink2_overlap;
    ggml_tensor       * dst     = cgraph->nodes[i + n - 1];
    const ggml_tensor * past    = m.rb ? m.in_rb.past : m.in.past;
    const ggml_tensor * cur_t   = m.rb ? m.in_rb.cur_t : m.in.cur_t;
    ggml_tensor       * state   = m.rb ? nullptr : m.in.state;
    const ggml_tensor * st_root = root(m.rb ? m.in_rb.slots : (const ggml_tensor *) m.in.state);
    const ggml_tensor * p_root  = root(past);
    const ggml_tensor * c_root  = root(cur_t);
    const bool dst_res = m.tail.res && dst->data == m.tail.res->data;
    bool ok = !overlap(dst, p_root) && !overlap(dst, c_root) && !overlap(dst, st_root) && !overlap(dst, m.tail.w) &&
        !overlap(c_root, st_root) && !overlap(p_root, st_root) && (dst_res || !m.tail.res || !overlap(dst, m.tail.res));
    if (m.rb) {
        ok = ok && !overlap(dst, root(m.in_rb.slot_order)) && !overlap(root(m.in_rb.slot_order), st_root);
    }
    d = { past, cur_t, m.rb ? m.in_rb.slots : nullptr, m.rb ? m.in_rb.slot_order : nullptr, m.rb ? m.in_rb.n_win : 0, state,
          m.tail.w, m.tail.cur, m.tail.res, dst, m.tail.scale };
    return ok;
}

// GGML_CUDA_INK2_SCONV_PAIR=0: each short convolution in its own launch
static bool ggml_cuda_ink2_sconv_pair_enabled() {
    static const bool enabled = []() { const char * e = getenv("GGML_CUDA_INK2_SCONV_PAIR"); return !(e && e[0] == '0'); }();
    return enabled;
}

// two sconv_full patterns with only views between them, the second reading nothing the first computes (the K and V
// convolutions of a sliding-window layer): returns the node count of both, with the first at i (n1 nodes), the second at j2
static int ggml_cuda_ink2_match_sconv_pair(const ggml_cgraph * cgraph, const int i, ggml_cuda_ink2_sconv_full_match & m1,
        ggml_cuda_ink2_sconv_full_match & m2, int & n1, int & j2, int & n2) {
    if (!ggml_cuda_ink2_sconv_pair_enabled()) {
        return 0;
    }
    n1 = ggml_cuda_ink2_match_sconv_full(cgraph, i, m1);
    if (n1 == 0) {
        return 0;
    }
    int j = i + n1;
    while (j < cgraph->n_nodes && ggml_cuda_is_view_or_noop(cgraph->nodes[j])) {
        j++;
    }
    if (j >= cgraph->n_nodes || cgraph->nodes[j]->op != GGML_OP_CONCAT) {
        return 0;
    }
    n2 = ggml_cuda_ink2_match_sconv_full(cgraph, j, m2);
    if (n2 == 0 || m1.rb != m2.rb) {
        return 0;
    }
    // small batches only (decode and verify): in prompt batches the pair would keep the buffers of the first alive longer
    const ggml_tensor * ct = m1.rb ? m1.in_rb.cur_t : m1.in.cur_t;
    if (ct->ne[0] > 16) {
        return 0;
    }
    const ggml_tensor * ins[] = { m2.rb ? m2.in_rb.past : m2.in.past, m2.rb ? m2.in_rb.cur_t : m2.in.cur_t, m2.tail.w, m2.tail.cur,
                                  m2.tail.res, m2.rb ? m2.in_rb.slot_order : nullptr };
    for (const ggml_tensor * t : ins) {
        if (!t) {
            continue;
        }
        const ggml_tensor * r = ggml_cuda_ink2_root(t);
        for (int k = i; k < i + n1; ++k) {
            if (cgraph->nodes[k] == t || cgraph->nodes[k] == r) {
                return 0;
            }
        }
    }
    j2 = j;
    return j + n2 - i;
}

// the outputs of each convolution of a pair overlap no input of the other, and their state stores are disjoint
static bool ggml_cuda_ink2_sconv_pair_disjoint(const ggml_cuda_ink2_sconv_desc & a, const ggml_cuda_ink2_sconv_desc & b) {
    const auto root    = ggml_cuda_ink2_root;
    const auto overlap = ggml_cuda_ink2_overlap;
    for (int k = 0; k < 2; ++k) {
        const ggml_cuda_ink2_sconv_desc & x = k ? b : a; // writer
        const ggml_cuda_ink2_sconv_desc & y = k ? a : b; // reader
        const ggml_tensor * st = root(x.slots ? x.slots : x.state);
        for (const ggml_tensor * t : { y.past, y.cur_t, y.w, y.cur, y.res, y.slot_order }) {
            if (t && (overlap(x.dst, root(t)) || overlap(st, root(t)))) {
                return false;
            }
        }
    }
    if (overlap(a.dst, b.dst)) {
        return false;
    }
    if (a.slots && b.slots) {
        // both in the slots of one state: the same slot stride, and the channel ranges of the two inside a slot row apart
        if (root(a.slots) != root(b.slots)) {
            return !overlap(root(a.slots), root(b.slots));
        }
        if (a.slots->nb[1] != b.slots->nb[1] || a.slots->nb[3] != b.slots->nb[3]) {
            return false;
        }
        const size_t ra = a.slots->nb[1];
        const size_t a0 = a.slots->view_offs % ra;
        const size_t b0 = b.slots->view_offs % ra;
        const size_t a1 = a0 + (a.slots->ne[2] - 1)*a.slots->nb[2] + (a.slots->ne[0] - 1)*a.slots->nb[0] + sizeof(float);
        const size_t b1 = b0 + (b.slots->ne[2] - 1)*b.slots->nb[2] + (b.slots->ne[0] - 1)*b.slots->nb[0] + sizeof(float);
        return a1 <= ra && b1 <= ra && (a1 <= b0 || b1 <= a0);
    }
    if (a.state && b.state) {
        // plain form: rows of n_past*C values (one per sequence, nb[1] apart), at two channel offsets of one state row
        if (root(a.state) != root(b.state)) {
            return !overlap(root(a.state), root(b.state));
        }
        const int64_t ns = a.state->ne[1];
        if (b.state->ne[1] != ns || a.state->nb[0] != sizeof(float) || b.state->nb[0] != sizeof(float)) {
            return false;
        }
        const size_t a0 = a.state->view_offs;
        const size_t b0 = b.state->view_offs;
        const size_t a1 = a0 + a.state->ne[0]*sizeof(float);
        const size_t b1 = b0 + b.state->ne[0]*sizeof(float);
        if (ns > 1 && (a.state->nb[1] != b.state->nb[1] || a1 - a0 > a.state->nb[1])) {
            return false;
        }
        if (ns > 1) {
            const size_t r = a.state->nb[1];
            if (a0 / r != b0 / r || a1 > (a0 / r + 1)*r || b1 > (a0 / r + 1)*r) {
                return false;
            }
        }
        return a1 <= b0 || b1 <= a0;
    }
    return false;
}

// launch of a matched "post" run (see ggml_cuda_ink2_match_post); false when the buffers do not allow it (then the ops run
// one by one). GGML_CUDA_INK2_CMP=post (with GGML_CUDA_DISABLE_GRAPHS=1) first runs the previous kernels (the fused clip,
// the generic RMS norm, MUL and SET_ROWS) and compares every output byte for byte with the fused launch.
static bool ggml_cuda_ink2_post_run(ggml_backend_cuda_context * cuda_ctx, const ggml_cuda_ink2_post_match & pm, const bool dry) {
    const auto overlap = [](const void * a, size_t na, const void * b, size_t nb) {
        const char * a0 = (const char *) a;
        const char * b0 = (const char *) b;
        return a0 < b0 + nb && b0 < a0 + na;
    };
    struct range { const void * p; size_t n; int item; };
    struct ranges {
        range r[4*GGML_CUDA_INK2_POST_MAX]; // per op: x, scale, the scalars, the row indices
        int   n = 0;
        void push_back(const range & x) { r[n++] = x; }
        size_t size() const { return (size_t) n; }
        const range & operator[](const size_t k) const { return r[k]; }
        const range * begin() const { return r; }
        const range * end() const { return r + n; }
    };
    ranges ins;
    ranges outs;
    for (int k = 0; k < pm.n; ++k) {
        const auto & it = pm.it[k];
        if (it.kind == 1 && (((uintptr_t) it.x->data % 16) != 0 || ((uintptr_t) it.out->data % 16) != 0)) {
            return false;
        }
        ins.push_back({ it.x->data, ggml_nbytes(it.x), k });
        if (it.scale) {
            ins.push_back({ it.scale->data, ggml_nbytes(it.scale), k });
        }
        outs.push_back({ it.out->data, ggml_nbytes(it.out), k });
        if (it.mul) {
            ins.push_back({ it.mul->src[1]->data, ggml_nbytes(it.mul->src[1]), k });
            outs.push_back({ it.mul->data, ggml_nbytes(it.mul), k });
        }
        if (it.store) {
            ins.push_back({ it.store->src[1]->data, ggml_nbytes(it.store->src[1]), k });
            outs.push_back({ it.store->data, ggml_nbytes(it.store), k });
        }
    }
    for (size_t a = 0; a < outs.size(); ++a) {
        for (const range & r : ins) {
            // a row is read whole before it is written: an op may write its own input in place, nothing else may overlap
            const bool same = r.item == outs[a].item && r.p == outs[a].p && r.n == outs[a].n && r.p == pm.it[r.item].x->data;
            if (!same && overlap(outs[a].p, outs[a].n, r.p, r.n)) {
                return false;
            }
        }
        for (size_t b = 0; b < a; ++b) {
            if (overlap(outs[a].p, outs[a].n, outs[b].p, outs[b].n)) {
                return false;
            }
        }
    }
    if (dry) {
        return true;
    }
    ggml_cuda_ink2_post_args args = {};
    args.n = pm.n;
    for (int k = 0; k < pm.n; ++k) {
        const auto & it = pm.it[k];
        auto & t = args.t[k];
        t.x            = (const float *) it.x->data;
        t.scale        = it.scale ? (const float *) it.scale->data : nullptr;
        t.dst          = (float *) it.out->data;
        t.tok          = it.mul ? (const float *) it.mul->src[1]->data : nullptr;
        t.dst_tok      = it.mul ? (float *) it.mul->data : nullptr;
        t.kv           = it.store ? (char *) it.store->data : nullptr;
        t.idx          = it.store ? (const int64_t *) it.store->src[1]->data : nullptr;
        t.kv_nb1       = it.store ? (int64_t) it.store->nb[1] : 0;
        t.kv_q8        = it.store && it.store->type == GGML_TYPE_Q8_0;
        t.d            = (int) it.out->ne[0];
        t.rows_per_tok = (int) it.out->ne[1];
        t.n_rows       = (int) (it.out->ne[1]*it.out->ne[2]);
        t.n_scale_rows = it.scale ? (int) (ggml_nelements(it.scale)/it.out->ne[0]) : 1;
        t.kind         = it.kind;
        t.threshold    = it.threshold;
        t.eps          = it.eps;
    }
    static const bool cmp = []() { const char * e = getenv("GGML_CUDA_INK2_CMP"); return e && strstr(e, "post") != nullptr; }();
    if (!cmp) {
        ggml_cuda_op_ink2_post(*cuda_ctx, args);
        return true;
    }
    // reference: the previous kernels, then the outputs (and the stored cache rows) copied to the host
    cudaStream_t stream = cuda_ctx->stream();
    for (int k = 0; k < pm.n; ++k) {
        const auto & it = pm.it[k];
        if (it.kind == 0) {
            ggml_cuda_op_ink2_l2_clip(*cuda_ctx, it.x, it.out, it.threshold, it.eps, it.scale);
        } else {
            GGML_ASSERT(ggml_cuda_compute_forward(*cuda_ctx, it.out));
        }
        if (it.mul) {
            GGML_ASSERT(ggml_cuda_compute_forward(*cuda_ctx, it.mul));
        }
        if (it.store) {
            GGML_ASSERT(ggml_cuda_compute_forward(*cuda_ctx, it.store));
        }
    }
    const auto snapshot = [&](std::vector<std::vector<uint8_t>> & v) {
        CUDA_CHECK(cudaStreamSynchronize(stream));
        v.clear();
        for (int k = 0; k < pm.n; ++k) {
            const auto & it = pm.it[k];
            for (const ggml_tensor * t : { (const ggml_tensor *) it.out, (const ggml_tensor *) it.mul }) {
                if (t) {
                    v.emplace_back(ggml_nbytes(t));
                    CUDA_CHECK(cudaMemcpy(v.back().data(), t->data, ggml_nbytes(t), cudaMemcpyDeviceToHost));
                }
            }
            if (it.store) {
                std::vector<int64_t> idx(it.store->src[1]->ne[0]);
                CUDA_CHECK(cudaMemcpy(idx.data(), it.store->src[1]->data, idx.size()*sizeof(int64_t), cudaMemcpyDeviceToHost));
                const size_t rs = ggml_row_size(it.store->type, it.store->ne[0]);
                for (int64_t r : idx) {
                    v.emplace_back(rs);
                    CUDA_CHECK(cudaMemcpy(v.back().data(), (const char *) it.store->data + r*it.store->nb[1], rs, cudaMemcpyDeviceToHost));
                }
            }
        }
    };
    std::vector<std::vector<uint8_t>> ref;
    std::vector<std::vector<uint8_t>> got;
    snapshot(ref);
    ggml_cuda_op_ink2_post(*cuda_ctx, args);
    snapshot(got);
    static int64_t calls = 0, diff_calls = 0, diff_bytes = 0;
    if (calls == 0) {
        atexit([]() { fprintf(stderr, "ink2 cmp post: %lld calls, %lld with differences, %lld differing bytes\n", (long long) calls,
            (long long) diff_calls, (long long) diff_bytes); });
    }
    int64_t d = 0;
    for (size_t a = 0; a < ref.size(); ++a) {
        for (size_t b = 0; b < ref[a].size(); ++b) {
            d += ref[a][b] != got[a][b];
        }
    }
    calls++;
    diff_calls += d > 0;
    diff_bytes += d;
    if ((d > 0 && diff_calls <= 5) || calls % 2000 == 0) {
        fprintf(stderr, "ink2 cmp post: %lld calls, %lld with differences, %lld differing bytes (this call: %lld, %d ops)\n",
            (long long) calls, (long long) diff_calls, (long long) diff_bytes, (long long) d, pm.n);
    }
    return true;
}

static int ggml_cuda_ink2_try_fuse(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, const int i) {
    const ggml_op op = cgraph->nodes[i]->op;
    if ((op == GGML_OP_MUL || op == GGML_OP_SQR) && ggml_cuda_ink2_post_enabled()) {
        ggml_cuda_ink2_post_match pm;
        const int n = ggml_cuda_ink2_match_post(cgraph, i, pm);
        if (n > 0 && ggml_cuda_ink2_post_run(cuda_ctx, pm, true)) {
            ggml_cuda_ink2_run(cuda_ctx, cgraph, i, n, 16, [&]() { ggml_cuda_ink2_post_run(cuda_ctx, pm, false); });
            return n - 1;
        }
    }
    if (op == GGML_OP_MUL_MAT) {
        return ggml_cuda_ink2_try_q8_group(cuda_ctx, cgraph, i);
    }
    if (op == GGML_OP_MUL_MAT_ID) {
        ggml_cuda_ink2_moe_situ_match m;
        const int n   = ggml_cuda_ink2_match_moe_situ(cgraph, i, m);
        const int out = i + n - 1;
        // only the activation is written, from the weights and the activations: no overlap with any input
        if (n > 0 && ggml_cuda_ink2_moe_f16_supported(cuda_ctx->device, m.mm->src[0], m.mm->src[1], m.mm->src[2], cgraph->nodes[out],
                    m.x_scale, true) &&
                ggml_cuda_check_fusion_memory_ranges(cgraph, i, n, &out, 1)) {
            const float bl[2] = { m.situ.beta, m.situ.lb };
            // with the down projection of the same experts (kind 14 "moe_ffn"): the activation stays in FP16 inside the call
            ggml_cuda_ink2_moe_down_match d;
            const int nd     = ggml_cuda_ink2_match_moe_down(cgraph, out, m.mm->src[2], d);
            const int out_dn = out + nd;
            if (nd > 0 && ggml_cuda_ink2_moe_f16_supported(cuda_ctx->device, d.mm->src[0], cgraph->nodes[out], d.mm->src[2],
                        cgraph->nodes[out_dn], d.x_scale, false) &&
                    ggml_cuda_ink2_moe_ffn_f16_ok(m.mm->src[0], d.mm->src[0]) &&
                    ggml_cuda_check_fusion_memory_ranges(cgraph, i, n + nd, &out_dn, 1)) {
                ggml_cuda_ink2_run(cuda_ctx, cgraph, i, n + nd, 14, [&]() {
                    GGML_ASSERT(ggml_cuda_ink2_moe_ffn_f16(*cuda_ctx, m.mm->src[0], m.mm->src[1], m.mm->src[2], m.x_scale, bl,
                        cgraph->nodes[out], d.mm->src[0], cgraph->nodes[out_dn], d.x_scale));
                });
                return n + nd - 1;
            }
            ggml_cuda_ink2_run(cuda_ctx, cgraph, i, n, 13, [&]() {
                GGML_ASSERT(ggml_cuda_ink2_moe_f16(*cuda_ctx, m.mm->src[0], m.mm->src[1], m.mm->src[2], cgraph->nodes[out], m.x_scale, bl));
            });
            return n - 1;
        }
        return 0;
    }
    if (op == GGML_OP_CONT) {
        const ggml_tensor * row = nullptr;
        const int n   = ggml_cuda_ink2_match_first_max(cgraph, i, row);
        const int out = i + n - 1;
        // one block reads the whole row before it writes the index: no overlap allowed anyway
        if (n > 0 && ggml_cuda_check_fusion_memory_ranges(cgraph, i, n, &out, 1)) {
            ggml_cuda_ink2_run(cuda_ctx, cgraph, i, n, 12, [&]() { ggml_cuda_op_ink2_first_max(*cuda_ctx, row, cgraph->nodes[out]); });
            return n - 1;
        }
        return 0;
    }
    if (op == GGML_OP_UNARY) {
        ggml_cuda_ink2_gate_match m;
        const int n = ggml_cuda_ink2_match_gate_tail(cgraph, i, m);
        if (n > 0) {
            ggml_tensor * f   = cgraph->nodes[i - 1]; // F h
            ggml_tensor * g   = cgraph->nodes[i + 3]; // G h, computed here
            ggml_tensor * dst = cgraph->nodes[i + n - 1];
            // elementwise: dst may be exactly one of the inputs, never a partial overlap; g must not overlap the others
            const auto overlap = [](const ggml_tensor * a, const ggml_tensor * b) {
                const char * a0 = (const char *) a->data;
                const char * b0 = (const char *) b->data;
                return a0 < b0 + ggml_nbytes(b) && b0 < a0 + ggml_nbytes(a);
            };
            bool ok = true;
            for (const ggml_tensor * t : { (const ggml_tensor *) f, (const ggml_tensor *) g, m.cur, m.msg }) {
                ok = ok && (t->data == dst->data || !overlap(t, dst));
            }
            for (const ggml_tensor * t : { (const ggml_tensor *) f, m.cur, m.msg, (const ggml_tensor *) g->src[1] }) {
                ok = ok && !overlap(g, t);
            }
            if (ok) {
                ggml_cuda_ink2_run(cuda_ctx, cgraph, i, n, 11, [&]() {
                    GGML_ASSERT(ggml_cuda_compute_forward(*cuda_ctx, g));
                    ggml_cuda_op_ink2_gate_tail(*cuda_ctx, f, g, m.cur, m.msg, dst);
                });
                return n - 1;
            }
        }
        return 0;
    }
    if (op == GGML_OP_PAD) {
        ggml_cuda_ink2_relbias_match m;
        const int n = ggml_cuda_ink2_match_relbias(cgraph, i, m);
        if (n > 0) {
            // only the last node is written: it must not overlap any of the five inputs
            const ggml_tensor * dst = cgraph->nodes[i + n - 1];
            const auto overlap = [](const ggml_tensor * a, const ggml_tensor * b) {
                const char * a0 = (const char *) a->data;
                const char * b0 = (const char *) b->data;
                return a0 < b0 + ggml_nbytes(b) && b0 < a0 + ggml_nbytes(a);
            };
            static const bool relbias_log = getenv("GGML_CUDA_INK2_RELBIAS_LOG") != nullptr;
            if (relbias_log) {
                static std::map<int64_t, int> seen;
                const int64_t key = (m.rel_pos->ne[0]/4096)*100000 + m.rel_pos->ne[1];
                if (seen[key]++ == 0) {
                    fprintf(stderr, "ink2 relbias: n_kv %lld, n_tok %lld, mask %lld x %lld\n", (long long) m.rel_pos->ne[0],
                        (long long) m.rel_pos->ne[1], (long long) m.mask->ne[0], (long long) m.mask->ne[1]);
                }
            }
            if (!overlap(dst, m.bias) && !overlap(dst, m.R) && !overlap(dst, m.sinks) && !overlap(dst, m.rel_pos) && !overlap(dst, m.mask)) {
                ggml_cuda_ink2_run(cuda_ctx, cgraph, i, n, 7, [&]() {
                    ggml_cuda_op_ink2_rel_bias(*cuda_ctx, m.bias, m.R, m.sinks, m.rel_pos, m.mask, cgraph->nodes[i + n - 1]); });
                return n - 1;
            }
        }
        return 0;
    }
    if (op == GGML_OP_MUL) {
        ggml_cuda_ink2_clip_match m;
        const int n = ggml_cuda_ink2_match_clip_scaled(cgraph, i, m);
        const int out = i + n - 1;
        // the kernel reads a row (and the weights) before it writes it: the output may be x itself, nothing else
        if (n > 0 && (cgraph->nodes[out]->data == m.x->data || ggml_cuda_check_fusion_memory_ranges(cgraph, i, n, &out, 1))) {
            ggml_cuda_ink2_run(cuda_ctx, cgraph, i, n, 0, [&]() { ggml_cuda_op_ink2_l2_clip(*cuda_ctx, m.x, cgraph->nodes[out], m.threshold, m.eps, m.scale); });
            return n - 1;
        }
        return 0;
    }
    if (op == GGML_OP_SQR) {
        ggml_cuda_ink2_clip_match m;
        const int n = ggml_cuda_ink2_match_clip(cgraph, i, m);
        const int out = i + n - 1;
        // the kernel reads a row before it writes it, so the output may be x itself; any other overlap is refused
        if (n > 0 && (cgraph->nodes[out]->data == m.x->data || ggml_cuda_check_fusion_memory_ranges(cgraph, i, n, &out, 1))) {
            ggml_cuda_ink2_run(cuda_ctx, cgraph, i, n, 0, [&]() { ggml_cuda_op_ink2_l2_clip(*cuda_ctx, m.x, cgraph->nodes[out], m.threshold, m.eps); });
            ggml_cuda_ink2_count(0);
            return n - 1;
        }
        return 0;
    }
    if (op == GGML_OP_SSM_CONV) {
        ggml_cuda_ink2_sconv_match m;
        const int n = ggml_cuda_ink2_match_sconv(cgraph, i, m);
        const int out = i + n - 1;
        if (n > 0) {
            // the kernel reads cur[i] and res[i] before it writes dst[i]: dst may be exactly cur or res, nothing else
            const void * d = cgraph->nodes[out]->data;
            if (d != m.x->data && (d == m.cur->data || (m.res && d == m.res->data) || ggml_cuda_check_fusion_memory_ranges(cgraph, i, n, &out, 1))) {
                ggml_cuda_ink2_run(cuda_ctx, cgraph, i, n, 3, [&]() { ggml_cuda_op_ink2_sconv_res(*cuda_ctx, m.x, m.w, m.cur, m.res, cgraph->nodes[out], m.scale); });
                return n - 1;
            }
            return 0;
        }
        ggml_cuda_ink2_dconv_match d;
        const int nd = ggml_cuda_ink2_match_dconv(cgraph, i, d);
        if (nd > 0) {
            const int o = i + nd - 1;
            // the kernel reads cur[i] before it writes dst[i]: dst may be exactly cur, never x or the taps
            const void * dd = cgraph->nodes[o]->data;
            const bool alias_ok = dd == d.cur->data && dd != d.x->data && dd != d.D->data;
            if (alias_ok || ggml_cuda_check_fusion_memory_ranges(cgraph, i, nd, &o, 1)) {
                ggml_cuda_ink2_run(cuda_ctx, cgraph, i, nd, 9, [&]() { ggml_cuda_op_ink2_dconv(*cuda_ctx, d.x, d.w, d.D, d.cur, cgraph->nodes[o], d.n_heads, d.scale); });
                return nd - 1;
            }
        }
        return 0;
    }
    if (op == GGML_OP_SCALE) {
        ggml_cuda_ink2_situ_match m;
        const int n = ggml_cuda_ink2_match_situ(cgraph, i, m);
        const int out = i + n - 1;
        if (n > 0 && ggml_cuda_check_fusion_memory_ranges(cgraph, i, n, &out, 1)) {
            ggml_cuda_ink2_run(cuda_ctx, cgraph, i, n, 4, [&]() { ggml_cuda_op_ink2_situ(*cuda_ctx, m.gate, m.up, cgraph->nodes[out], m.beta, m.lb); });
            return n - 1;
        }
        return 0;
    }
    if (op == GGML_OP_SUM_ROWS) {
        ggml_cuda_ink2_wnorm_match m;
        const int n = ggml_cuda_ink2_match_wnorm(cgraph, i, m);
        const int out = i + n - 1;
        // the kernel reads a token's weights before it writes them: the output may be w itself
        if (n > 0 && (cgraph->nodes[out]->data == m.w->data || ggml_cuda_check_fusion_memory_ranges(cgraph, i, n, &out, 1))) {
            ggml_cuda_ink2_run(cuda_ctx, cgraph, i, n, 5, [&]() { ggml_cuda_op_ink2_moe_wnorm(*cuda_ctx, m.w, cgraph->nodes[out], m.eps, m.scale); });
            return n - 1;
        }
        return 0;
    }
    if (op == GGML_OP_CONCAT) {
        {
            // two adjacent independent short convolutions (the K and V convolutions of a sliding-window layer) in one launch
            ggml_cuda_ink2_sconv_full_match m1;
            ggml_cuda_ink2_sconv_full_match m2;
            int n1 = 0;
            int j2 = 0;
            int n2 = 0;
            const int n = ggml_cuda_ink2_match_sconv_pair(cgraph, i, m1, m2, n1, j2, n2);
            ggml_cuda_ink2_sconv_desc d1;
            ggml_cuda_ink2_sconv_desc d2;
            if (n > 0 && ggml_cuda_ink2_sconv_full_check(cgraph, i, n1, m1, d1) && ggml_cuda_ink2_sconv_full_check(cgraph, j2, n2, m2, d2) &&
                    ggml_cuda_ink2_sconv_pair_disjoint(d1, d2)) {
                ggml_cuda_ink2_run(cuda_ctx, cgraph, i, n, 10, [&]() {
                    if (!ggml_cuda_op_ink2_sconv_full2(*cuda_ctx, d1, d2)) {
                        for (const ggml_cuda_ink2_sconv_desc * d : { &d1, &d2 }) {
                            ggml_cuda_op_ink2_sconv_full(*cuda_ctx, d->past, d->cur_t, d->slots, d->slot_order, d->n_win, d->state, d->w,
                                d->cur, d->res, d->dst, d->scale);
                        }
                    }
                });
                return n - 1;
            }
        }
        {
            ggml_cuda_ink2_sconv_full_match m;
            const int n = ggml_cuda_ink2_match_sconv_full(cgraph, i, m);
            ggml_cuda_ink2_sconv_desc d;
            if (n > 0 && ggml_cuda_ink2_sconv_full_check(cgraph, i, n, m, d)) {
                ggml_cuda_ink2_run(cuda_ctx, cgraph, i, n, 10, [&]() {
                    ggml_cuda_op_ink2_sconv_full(*cuda_ctx, d.past, d.cur_t, d.slots, d.slot_order, d.n_win, d.state, d.w, d.cur, d.res,
                        d.dst, d.scale); });
                return n - 1;
            }
        }
        {
            ggml_cuda_ink2_sconv_in_match m;
            const int n = ggml_cuda_ink2_match_sconv_in(cgraph, i, m);
            if (n > 0) {
                // two outputs: the concat (node i) and the state slot written by the CPY; inputs must not overlap either
                ggml_tensor * x = cgraph->nodes[i];
                const auto overlap = [](const ggml_tensor * a, const ggml_tensor * b) {
                    const char * a0 = (const char *) a->data;
                    const char * b0 = (const char *) b->data;
                    return a0 < b0 + ggml_nbytes(b) && b0 < a0 + ggml_nbytes(a);
                };
                // the views report their own extent: use the root of the state for the overlap with x
                const ggml_tensor * st_root = m.state->view_src ? m.state->view_src : m.state;
                const ggml_tensor * past_root = m.past->view_src ? m.past->view_src : m.past;
                const ggml_tensor * cur_root = m.cur_t->view_src ? m.cur_t->view_src : m.cur_t;
                if (!overlap(x, st_root) && !overlap(x, past_root) && !overlap(x, cur_root) && !overlap(cur_root, st_root) && !overlap(past_root, st_root)) {
                    ggml_cuda_ink2_run(cuda_ctx, cgraph, i, n, 6, [&]() { ggml_cuda_op_ink2_sconv_in(*cuda_ctx, m.past, m.cur_t, x, m.state); });
                    return n - 1;
                }
            }
        }
        {
            ggml_cuda_ink2_sconv_rb_match m;
            const int n = ggml_cuda_ink2_match_sconv_rb(cgraph, i, m);
            if (n > 0) {
                // two outputs: the concat (node i) and the slots of the state; neither may overlap an input or each other
                ggml_tensor * x = cgraph->nodes[i];
                const auto root = [](const ggml_tensor * t) {
                    while (t->view_src) {
                        t = t->view_src;
                    }
                    return t;
                };
                const auto overlap = [](const ggml_tensor * a, const ggml_tensor * b) {
                    const char * a0 = (const char *) a->data;
                    const char * b0 = (const char *) b->data;
                    return a0 < b0 + ggml_nbytes(b) && b0 < a0 + ggml_nbytes(a);
                };
                const ggml_tensor * st_root   = root(m.slots);
                const ggml_tensor * past_root = root(m.past);
                const ggml_tensor * cur_root  = root(m.cur_t);
                const ggml_tensor * ord_root  = root(m.slot_order);
                // past is itself a view of the state history (another tensor than the slots written here)
                if (!overlap(x, st_root) && !overlap(x, past_root) && !overlap(x, cur_root) && !overlap(x, ord_root) &&
                        !overlap(cur_root, st_root) && !overlap(ord_root, st_root) && !overlap(past_root, st_root)) {
                    ggml_cuda_ink2_run(cuda_ctx, cgraph, i, n, 8, [&]() {
                        ggml_cuda_op_ink2_sconv_in_rb(*cuda_ctx, m.past, m.cur_t, x, m.slots, m.slot_order, m.n_win); }, true);
                    return n - 1;
                }
            }
        }
        {
            // the whole loop gate of a prompt batch (kind 15 "gate_pp", BF16 weights): when it does not qualify, the products
            // run as generic GEMMs and the tail is fused at its TANH (gate_tail)
            ggml_cuda_ink2_gate_match m;
            const int n = ggml_cuda_ink2_match_gate(cgraph, i, m, true);
            ggml_tensor * dst = n > 0 ? cgraph->nodes[i + n - 1] : nullptr;
            if (n > 0 && ggml_cuda_ink2_gate_pp_supported(cuda_ctx->device, m.cur, m.msg, m.down, m.gain, m.forget, dst)) {
                // the inputs are read before the same index of dst is written: dst may be cur or msg exactly, no partial overlap
                const auto overlap = [](const ggml_tensor * a, const ggml_tensor * b) {
                    const char * a0 = (const char *) a->data;
                    const char * b0 = (const char *) b->data;
                    return a0 < b0 + ggml_nbytes(b) && b0 < a0 + ggml_nbytes(a);
                };
                bool ok = true;
                for (const ggml_tensor * t : { m.cur, m.msg }) {
                    ok = ok && (t->data == dst->data || !overlap(t, dst));
                }
                for (const ggml_tensor * t : { m.down, m.gain, m.forget }) {
                    ok = ok && !overlap(t, dst);
                }
                if (ok) {
                    ggml_cuda_ink2_run(cuda_ctx, cgraph, i, n, 15, [&]() {
                        ggml_cuda_op_ink2_gate_pp(*cuda_ctx, m.cur, m.msg, m.down, m.gain, m.forget, dst, m.eps); });
                    return n - 1;
                }
            }
        }
        {
            ggml_cuda_ink2_gate_match g[GGML_CUDA_INK2_GATE_N_MAX];
            ggml_tensor * outs[GGML_CUDA_INK2_GATE_N_MAX];
            int ng = 0;
            const int n = ggml_cuda_ink2_match_gate_n(cgraph, i, g, outs, ng);
            // every gate reads its inputs at an index before it writes that index of its output: an output may be its own cur or
            // msg exactly; no other overlap between an output and any input or output of the run
            bool ok = n > 0;
            for (int a = 0; a < ng && ok; ++a) {
                for (int b = 0; b < ng && ok; ++b) {
                    for (const ggml_tensor * t : { g[b].cur, g[b].msg }) {
                        ok = ok && ((a == b && t->data == outs[a]->data && ggml_nbytes(t) == ggml_nbytes(outs[a])) || !ggml_cuda_ink2_overlap(t, outs[a]));
                    }
                    for (const ggml_tensor * t : { g[b].down, g[b].gain, g[b].forget }) {
                        ok = ok && !ggml_cuda_ink2_overlap(t, outs[a]);
                    }
                    ok = ok && (a == b || !ggml_cuda_ink2_overlap(outs[a], outs[b]));
                }
            }
            if (ok) {
                ggml_cuda_ink2_run(cuda_ctx, cgraph, i, n, 1, [&]() {
                    const ggml_tensor * c[GGML_CUDA_INK2_GATE_N_MAX];
                    const ggml_tensor * mg[GGML_CUDA_INK2_GATE_N_MAX];
                    const ggml_tensor * d[GGML_CUDA_INK2_GATE_N_MAX];
                    const ggml_tensor * gn[GGML_CUDA_INK2_GATE_N_MAX];
                    const ggml_tensor * fg[GGML_CUDA_INK2_GATE_N_MAX];
                    for (int k = 0; k < ng; ++k) {
                        c[k] = g[k].cur; mg[k] = g[k].msg; d[k] = g[k].down; gn[k] = g[k].gain; fg[k] = g[k].forget;
                    }
                    // GGML_CUDA_INK2_CMP=gaten: the gates one by one first (outputs saved), then together, compared byte for byte
                    // (only when no output is its own input)
                    static const bool cmp = []() { const char * e = getenv("GGML_CUDA_INK2_CMP"); return e && strstr(e, "gaten"); }();
                    bool cmp_ok = cmp;
                    for (int k = 0; k < ng; ++k) {
                        cmp_ok = cmp_ok && outs[k]->data != g[k].cur->data && outs[k]->data != g[k].msg->data;
                    }
                    std::vector<std::vector<uint8_t>> ref(ng);
                    if (cmp_ok) {
                        for (int k = 0; k < ng; ++k) {
                            ggml_cuda_op_ink2_loop_gate(*cuda_ctx, g[k].cur, g[k].msg, g[k].down, g[k].gain, g[k].forget, outs[k], g[k].eps);
                        }
                        CUDA_CHECK(cudaStreamSynchronize(cuda_ctx->stream()));
                        for (int k = 0; k < ng; ++k) {
                            ref[k].resize(ggml_nbytes(outs[k]));
                            CUDA_CHECK(cudaMemcpy(ref[k].data(), outs[k]->data, ref[k].size(), cudaMemcpyDeviceToHost));
                        }
                    }
                    ggml_cuda_op_ink2_loop_gate_n(*cuda_ctx, c, mg, d, gn, fg, outs, ng, g[0].eps);
                    if (cmp_ok) {
                        CUDA_CHECK(cudaStreamSynchronize(cuda_ctx->stream()));
                        static int64_t calls = 0, diff = 0;
                        if (calls == 0) {
                            atexit([]() { fprintf(stderr, "ink2 cmp gaten: %lld calls, %lld with differences\n", (long long) calls, (long long) diff); });
                        }
                        bool same = true;
                        for (int k = 0; k < ng; ++k) {
                            std::vector<uint8_t> got(ref[k].size());
                            CUDA_CHECK(cudaMemcpy(got.data(), outs[k]->data, got.size(), cudaMemcpyDeviceToHost));
                            same = same && got == ref[k];
                        }
                        calls++;
                        diff += !same;
                    }
                });
                return n - 1;
            }
        }
        {
            ggml_cuda_ink2_gate_match m;
            const int n = ggml_cuda_ink2_match_gate(cgraph, i, m);
            const int out = i + n - 1;
            if (n > 0 && ggml_cuda_check_fusion_memory_ranges(cgraph, i, n, &out, 1)) {
                ggml_cuda_ink2_run(cuda_ctx, cgraph, i, n, 1, [&]() { ggml_cuda_op_ink2_loop_gate(*cuda_ctx, m.cur, m.msg, m.down, m.gain, m.forget, cgraph->nodes[out], m.eps); });
                ggml_cuda_ink2_count(1);
                return n - 1;
            }
        }
        {
            ggml_cuda_ink2_mix_match m;
            const int n = ggml_cuda_ink2_match_mix(cgraph, i, m);
            const int out = i + n - 1;
            const bool mem_ok = n > 0 && ggml_cuda_check_fusion_memory_ranges(cgraph, i, n, &out, 1);
            static const bool debug = ggml_cuda_env_flag("GGML_CUDA_INK2_DEBUG");
            static int n_log = 0;
            if (debug && n_log < 8 && ggml_get_op_params_i32(cgraph->nodes[i], 0) == 1) {
                std::string s;
                for (int k = i; k < std::min(i + 12, cgraph->n_nodes); ++k) {
                    s += ggml_op_name(cgraph->nodes[k]->op);
                    s += ' ';
                }
                GGML_LOG_INFO("%s: mix at %d: n = %d, mem_ok = %d: %s\n", __func__, i, n, mem_ok ? 1 : 0, s.c_str());
                n_log++;
            }
            if (mem_ok) {
                ggml_cuda_ink2_run(cuda_ctx, cgraph, i, n, 2, [&]() { ggml_cuda_op_ink2_depth_mix(*cuda_ctx, m.states, m.n_states, m.query, cgraph->nodes[out], m.eps, m.scale, m.rows_first); });
                ggml_cuda_ink2_count(2);
                return n - 1;
            }
        }
    }
    return 0;
}

struct ggml_cuda_ink2_trace_data {
    std::map<std::string, int>    hist;
    std::map<std::string, double> time_us;
    std::string cur;
    int64_t     t_last = 0;
    bool        timed  = false;
};

static ggml_cuda_ink2_trace_data & ggml_cuda_ink2_trace_state() {
    static ggml_cuda_ink2_trace_data st;
    return st;
}

// which graphs the trace looks at: GGML_CUDA_INK2_TRACE_NODES=min,max nodes (default 5000 and up: the target model's graphs;
// e.g. 200,400 selects the drafting graph of the DSpark drafter)
static bool ggml_cuda_ink2_trace_graph(const ggml_cgraph * cgraph) {
    static int lo = 5000;
    static int hi = INT_MAX;
    static const bool init = []() {
        const char * e = getenv("GGML_CUDA_INK2_TRACE_NODES");
        if (e) {
            sscanf(e, "%d,%d", &lo, &hi);
        }
        return true;
    }();
    GGML_UNUSED(init);
    return cgraph->n_nodes >= lo && cgraph->n_nodes <= hi;
}

// GGML_CUDA_INK2_TRACE=1: histogram of what the last large graph launched (fused groups and single ops), printed at exit
static void ggml_cuda_ink2_trace(const ggml_cgraph * cgraph, const int i, const int nodes_to_skip) {
    static const bool enabled = ggml_cuda_env_flag("GGML_CUDA_INK2_TRACE");
    if (!enabled || !ggml_cuda_ink2_trace_graph(cgraph)) {
        return;
    }
    auto & hist = ggml_cuda_ink2_trace_state().hist;
    auto & time = ggml_cuda_ink2_trace_state().time_us;
    static int last_i = INT_MAX;
    static bool registered = false;
    if (!registered) {
        registered = true;
        atexit([]() {
            auto & st = ggml_cuda_ink2_trace_state();
            std::vector<std::pair<double, std::string>> v;
            int    total   = 0;
            double total_t = 0.0;
            for (const auto & kv : st.hist) {
                v.push_back({ st.timed ? st.time_us[kv.first] : (double) kv.second, kv.first });
                total   += kv.second;
                total_t += st.time_us[kv.first];
            }
            std::sort(v.rbegin(), v.rend());
            fprintf(stderr, "ink2 trace: %d launches in the last large graph, %.0f us timed\n", total, total_t);
            for (const auto & e : v) {
                fprintf(stderr, "ink2 trace: %5d  %8.0f us  %6.1f us each  %s\n", st.hist[e.second], st.time_us[e.second],
                    st.time_us[e.second]/st.hist[e.second], e.second.c_str());
            }
        });
    }
    if (i < last_i) {
        // GGML_CUDA_INK2_TRACE_AT=n: also print the histogram of the n-th traced graph when the next one starts (a server is
        // usually killed, its exit handlers do not run)
        static int n_graphs = 0;
        static const int print_at = []() { const char * e = getenv("GGML_CUDA_INK2_TRACE_AT"); return e ? atoi(e) : 0; }();
        if (++n_graphs == print_at + 1 && print_at > 0) {
            int total = 0;
            for (const auto & kv : hist) {
                total += kv.second;
            }
            fprintf(stderr, "ink2 trace: graph %d: %d launches\n", print_at, total);
            for (const auto & kv : hist) {
                fprintf(stderr, "ink2 trace: %5d  %8.0f us  %s\n", kv.second, time[kv.first], kv.first.c_str());
            }
            fflush(stderr);
        }
        hist.clear(); // a new graph
        time.clear();
    }
    last_i = i;
    const ggml_tensor * node = cgraph->nodes[i];
    std::string key = node->op == GGML_OP_UNARY ? ggml_unary_op_name(ggml_get_unary_op(node)) : ggml_op_name(node->op);
    if (nodes_to_skip > 0) {
        int real = 0;
        for (int k = i; k <= i + nodes_to_skip; ++k) {
            real += ggml_cuda_is_view_or_noop(cgraph->nodes[k]) ? 0 : 1;
        }
        key = "FUSED " + key + " x" + std::to_string(real);
    }
    if (node->op == GGML_OP_MUL_MAT || node->op == GGML_OP_MUL_MAT_ID) {
        key += std::string(" ") + ggml_type_name(node->src[0]->type);
    }
    if (node->op == GGML_OP_FLASH_ATTN_EXT && node->src[1]) {
        // the key count of the call (the full-attention layers and the sliding-window ring apart)
        key += std::string(" ") + ggml_type_name(node->src[1]->type) + " kv " + std::to_string(node->src[1]->ne[1]);
    }
    hist[key]++;
    ggml_cuda_ink2_trace_state().cur = key;

    // GGML_CUDA_INK2_SEQ=start,count: the launches of that node range, in execution order, for the second large graph
    static const char * seq = getenv("GGML_CUDA_INK2_SEQ");
    static int seq_start = 0;
    static int seq_count = 0;
    static int seq_graph = 0;
    if (seq && seq_count == 0) {
        sscanf(seq, "%d,%d", &seq_start, &seq_count);
    }
    if (seq) {
        static int prev = INT_MAX;
        if (i < prev) {
            seq_graph++;
        }
        prev = i;
        if (seq_graph == 2 && i >= seq_start && i < seq_start + seq_count) {
            const ggml_tensor * out = cgraph->nodes[i + nodes_to_skip];
            std::string srcs;
            for (int j = 0; j < GGML_MAX_SRC && node->src[j]; ++j) {
                const ggml_tensor * t = node->src[j];
                srcs += std::string(" ") + (t->name[0] ? std::string(t->name).substr(0, 28) : std::string("?")) + ":" + ggml_type_name(t->type) +
                    "[" + std::to_string(t->ne[0]) + "," + std::to_string(t->ne[1]) + "," + std::to_string(t->ne[2]) + "]";
            }
            fprintf(stderr, "ink2 seq %5d %-28s -> %s[%lld,%lld,%lld,%lld] %.24s <-%s\n", i, key.c_str(), ggml_type_name(out->type),
                (long long) out->ne[0], (long long) out->ne[1], (long long) out->ne[2], (long long) out->ne[3], out->name, srcs.c_str());
        }
    }
}

// GGML_CUDA_INK2_TRACE=3 (with GGML_CUDA_DISABLE_GRAPHS=1): every launch of the traced graphs is timed on the device instead of
// with a synchronization per launch: a one-thread kernel writes the GPU wall clock into a slot before each launch (stream order:
// it starts once the previous launch has finished). A spin kernel queued at the start of each traced graph
// (GGML_CUDA_INK2_TRACE_SPIN_US, default 40000) holds the GPU while the host queues the graph, so the launches then run back to
// back; the cost of a stamp itself (two stamps in a row, measured per graph) is subtracted. A delta is the GPU time of a launch
// plus the gap before the next one. Averages per graph size (node count) over every traced graph; printed at exit, or once
// GGML_CUDA_INK2_TRACE_AT graphs were resolved (servers are killed, exit handlers do not run). Stamps use stream events nowhere:
// on this runtime an event record splits the command submission and inflates every delta.
static __global__ void ggml_cuda_ink2_spin_kernel(const long long ticks) {
#if defined(GGML_USE_HIP)
    const long long t0 = wall_clock64();
    while (wall_clock64() - t0 < ticks) {}
#else
    const long long t0 = clock64();
    while (clock64() - t0 < ticks) {}
#endif
}

static __global__ void ggml_cuda_ink2_stamp_kernel(unsigned long long * buf, const int k) {
#if defined(GGML_USE_HIP)
    buf[k] = wall_clock64();
#else
    buf[k] = clock64();
#endif
}

struct ggml_cuda_ink2_evt_group {
    int    graphs   = 0;
    double total_us = 0.0;
    double stamp_us = 0.0;
    std::map<std::string, double> sum_us;
    std::map<std::string, long>   count;
};

struct ggml_cuda_ink2_evt_data {
    static constexpr int n_slots = 16384;
    unsigned long long * dbuf = nullptr;
    std::vector<unsigned long long> hbuf;
    std::vector<std::string> keys;   // keys[k] = launch timed from stamp k to stamp k + 1
    int    n_used   = 0;
    int    n_nodes  = 0;
    int    last_i   = INT_MAX;
    int    resolved = 0;
    double us_per_tick = 0.01;
    std::map<int, ggml_cuda_ink2_evt_group> groups;
};

static void ggml_cuda_ink2_evt_print(const std::map<int, ggml_cuda_ink2_evt_group> & groups) {
    for (const auto & g : groups) {
        if (g.second.graphs <= 0) {
            continue;
        }
        const double n = g.second.graphs;
        fprintf(stderr, "ink2 evt: graph of %d nodes: %d graphs, %.1f us per graph (stamp cost %.2f us, subtracted)\n", g.first,
            g.second.graphs, g.second.total_us / n, g.second.stamp_us / n);
        std::vector<std::pair<double, std::string>> v;
        for (const auto & kv : g.second.sum_us) {
            v.push_back({ kv.second, kv.first });
        }
        std::sort(v.rbegin(), v.rend());
        for (const auto & e : v) {
            const long c = g.second.count.at(e.second);
            fprintf(stderr, "ink2 evt: %7.1f per graph %9.1f us per graph %7.2f us each  %s\n", c / n, e.first / n, e.first / c,
                e.second.c_str());
        }
    }
    fflush(stderr);
}

static void ggml_cuda_ink2_evt_tick(ggml_backend_cuda_context * cuda_ctx, const ggml_cgraph * cgraph, const int i) {
    static ggml_cuda_ink2_evt_data st;
    static const int print_at = []() { const char * e = getenv("GGML_CUDA_INK2_TRACE_AT"); return e ? atoi(e) : 0; }();
    cudaStream_t stream = cuda_ctx->stream();
    if (st.dbuf == nullptr) {
        CUDA_CHECK(cudaMalloc((void **) &st.dbuf, st.n_slots * sizeof(unsigned long long)));
        st.hbuf.resize(st.n_slots);
        st.keys.resize(st.n_slots);
        int rate_khz = 100000;
#if defined(GGML_USE_HIP)
        if (cudaDeviceGetAttribute(&rate_khz, hipDeviceAttributeWallClockRate, ggml_cuda_get_device()) != cudaSuccess || rate_khz <= 0) {
            rate_khz = 100000;
        }
#else
        rate_khz = 2000000; // clock64 counts SM cycles, ~2 GHz
#endif
        st.us_per_tick = 1000.0 / rate_khz;
        fprintf(stderr, "ink2 evt: device clock %d kHz\n", rate_khz);
        atexit([]() { ggml_cuda_ink2_evt_print(st.groups); });
    }
    if (i < st.last_i) {
        // a new graph: close and resolve the previous one, then hold the GPU while this one is queued
        if (st.n_used > 2 && st.n_used < st.n_slots) {
            st.keys[st.n_used - 1] = ggml_cuda_ink2_trace_state().cur;
            ggml_cuda_ink2_stamp_kernel<<<1, 1, 0, stream>>>(st.dbuf, st.n_used);
            CUDA_CHECK(cudaMemcpyAsync(st.hbuf.data(), st.dbuf, (st.n_used + 1) * sizeof(unsigned long long), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            auto & g = st.groups[st.n_nodes];
            // the cost of a stamp: the smaller of two stamps in a row and the shortest delta of the graph
            double stamp = (double) (st.hbuf[1] - st.hbuf[0]) * st.us_per_tick;
            for (int k = 2; k < st.n_used; ++k) {
                stamp = std::min(stamp, (double) (st.hbuf[k + 1] - st.hbuf[k]) * st.us_per_tick);
            }
            for (int k = 2; k < st.n_used; ++k) {
                const double us = (double) (st.hbuf[k + 1] - st.hbuf[k]) * st.us_per_tick - stamp;
                g.sum_us[st.keys[k]] += us;
                g.count[st.keys[k]]  += 1;
                g.total_us           += us;
            }
            g.stamp_us += stamp;
            g.graphs++;
            if (++st.resolved == print_at && print_at > 0) {
                ggml_cuda_ink2_evt_print(st.groups);
            }
        }
        st.n_used  = 0;
        st.n_nodes = cgraph->n_nodes;
        static const long long spin_ticks = []() {
            const char * e = getenv("GGML_CUDA_INK2_TRACE_SPIN_US");
            const long long us = e ? atoll(e) : 40000;
            return (long long) (us / st.us_per_tick);
        }();
        ggml_cuda_ink2_spin_kernel<<<1, 1, 0, stream>>>(spin_ticks);
        // two stamps in a row: the cost of one stamp, subtracted from every delta
        ggml_cuda_ink2_stamp_kernel<<<1, 1, 0, stream>>>(st.dbuf, 0);
        ggml_cuda_ink2_stamp_kernel<<<1, 1, 0, stream>>>(st.dbuf, 1);
        st.n_used = 2;
    } else if (st.n_used > 2) {
        st.keys[st.n_used - 1] = ggml_cuda_ink2_trace_state().cur;
    }
    st.last_i = i;
    if (st.n_used < st.n_slots - 1) {
        ggml_cuda_ink2_stamp_kernel<<<1, 1, 0, stream>>>(st.dbuf, st.n_used);
        st.n_used++;
    }
}
// GGML_CUDA_INK2_TRACE=2: called before a node is dispatched; the time since the previous call belongs to the previous launch
static void ggml_cuda_ink2_trace_tick(ggml_backend_cuda_context * cuda_ctx, const ggml_cgraph * cgraph, const int i) {
    static const int level = []() { const char * e = getenv("GGML_CUDA_INK2_TRACE"); return e ? atoi(e) : 0; }();
    if (level < 2 || !ggml_cuda_ink2_trace_graph(cgraph)) {
        return;
    }
    if (level >= 3) {
        ggml_cuda_ink2_evt_tick(cuda_ctx, cgraph, i);
        return;
    }
    auto & st = ggml_cuda_ink2_trace_state();
    CUDA_CHECK(cudaStreamSynchronize(cuda_ctx->stream()));
    const int64_t now = ggml_time_us();
    if (!st.cur.empty()) {
        st.time_us[st.cur] += (double) (now - st.t_last);
    }
    st.cur.clear();
    st.t_last = ggml_time_us();
    st.timed  = true;
}

// TIMING EXPERIMENT ONLY (wrong output): GGML_CUDA_INK2_SKIP_OP="KEY;KEY" does not run the single nodes of a large graph whose
// trace key (see GGML_CUDA_INK2_TRACE) is listed; the speed difference is what that kind of node really costs
// (only in builds with GGML_CUDA_INK2_COST_TOOLS)
static bool ggml_cuda_ink2_skip_op(const ggml_cgraph * cgraph, const ggml_tensor * node) {
#if defined(GGML_CUDA_INK2_COST_TOOLS)
    static const std::string keys = []() { const char * e = getenv("GGML_CUDA_INK2_SKIP_OP"); return std::string(e ? e : ""); }();
#else
    static const std::string keys;
#endif
    if (keys.empty() || cgraph->n_nodes < 5000) {
        return false;
    }
    std::string key = node->op == GGML_OP_UNARY ? ggml_unary_op_name(ggml_get_unary_op(node)) : ggml_op_name(node->op);
    if (node->op == GGML_OP_MUL_MAT || node->op == GGML_OP_MUL_MAT_ID) {
        key += std::string(" ") + ggml_type_name(node->src[0]->type);
    }
    return (";" + keys + ";").find(";" + key + ";") != std::string::npos;
}

// try and fuse nodes and return the number of nodes to skip
static int ggml_cuda_try_fuse(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, int i) {

    if (ggml_cuda_fusion_disabled()) {
        return 0;
    }

    if (ggml_cuda_ink2_fusion_enabled() && ggml_cuda_ink2_arch_ok(cuda_ctx->device)) {
        if (const int n_skip = ggml_cuda_ink2_try_fuse(cuda_ctx, cgraph, i)) {
            return n_skip;
        }
    }

    ggml_tensor * node = cgraph->nodes[i];

    if (node->op == GGML_OP_MUL_MAT_ID && cuda_ctx->stream_context().concurrent_events.empty() &&
            ggml_cuda_match_shared_expert(cgraph, i, i + 3) &&
            ggml_cuda_should_fuse_mul_mat_vec_q(cgraph->nodes[i + 2]->src[1])) {
        const int outputs[] = { i + 2, i + 5 };
        if (ggml_cuda_check_fusion_memory_ranges(cgraph, i, 6, outputs, 2)) {
            ggml_tensor * routed = cgraph->nodes[i + 2];
            ggml_tensor * shared = cgraph->nodes[i + 5];
            const ggml_tensor * up = routed->src[1];
            ggml_cuda_mm_fusion_args_host fusion{};
            fusion.gate = routed->src[0]->src[0];
            fusion.glu_op = ggml_get_glu_op(routed);
            fusion.glu_limit = ggml_get_op_params_f32(routed, 3);
            fusion.shared_up = shared->src[1]->src[0];
            fusion.shared_gate = shared->src[0]->src[0];
            fusion.shared_dst = shared;
            ggml_cuda_mul_mat_vec_q(*cuda_ctx, up->src[0], up->src[1], up->src[2], routed, &fusion);
            return 5;
        }
    }

    if (node->op == GGML_OP_MUL) {
        ggml_cuda_moe_weighted_reduction_match match;
        if (ggml_cuda_match_moe_weighted_reduction(cgraph, i, match)) {
            const int output_idx = i + match.node_count - 1;
            if (ggml_cuda_check_fusion_memory_ranges(cgraph, i, match.node_count, &output_idx, 1)) {
                ggml_cuda_op_moe_weighted_reduction(
                    *cuda_ctx, match.experts, match.expert_scale, match.weights, match.dst);
                return match.node_count - 1;
            }
        }
    }

    // gated_delta_net -> cpy: scatter recurrent-state snapshots into the cache
    if (node->op == GGML_OP_GATED_DELTA_NET) {
        ggml_cuda_gated_delta_net_fused_cache fused_state_cpy;
        const int nodes_to_skip = ggml_cuda_try_gdn_cache_fusion(cgraph, i, fused_state_cpy);
        if (nodes_to_skip > 0) {
#ifdef GGML_CUDA_DEBUG
            GGML_LOG_INFO("%s: fused gated_delta_net snapshot copies for %s (skipped %d nodes)\n",
                          __func__, node->name, nodes_to_skip);
#endif
            ggml_cuda_op_gated_delta_net_fused_cache(*cuda_ctx, node, fused_state_cpy);
            return nodes_to_skip;
        }
    }

    // ssm_scan -> cpy: scatter recurrent-state snapshots into the cache
    if (node->op == GGML_OP_SSM_SCAN) {
        ggml_cuda_ssm_scan_fused_cache fused_state_cpy;
        const int nodes_to_skip = ggml_cuda_try_ssm_scan_cache_fusion(cgraph, i, fused_state_cpy);
        if (nodes_to_skip > 0) {
#ifdef GGML_CUDA_DEBUG
            GGML_LOG_INFO("%s: fused ssm_scan snapshot copies for %s (skipped %d nodes)\n",
                          __func__, node->name, nodes_to_skip);
#endif
            ggml_cuda_op_ssm_scan_fused_cache(*cuda_ctx, node, fused_state_cpy);
            return nodes_to_skip;
        }
    }

    //topk-moe
    if (cgraph->nodes[i]->op == GGML_OP_UNARY || cgraph->nodes[i]->op == GGML_OP_SOFT_MAX ||
            cgraph->nodes[i]->op == GGML_OP_ARGSORT) {
        ggml_cuda_topk_moe_args args;
        const bool              can_fuse = ggml_cuda_topk_moe_fusion(cgraph, i, args);
        std::vector<ggml_op>    ops;
        ops.reserve(13);  // max ops; avoids gcc -Wstringop-overflow false positive

        if (can_fuse) {
            const ggml_tensor * logits  = node->src[0];
            ggml_tensor *       weights = nullptr;
            ggml_tensor *       ids     = nullptr;
            const ggml_tensor * bias    = nullptr;
            const ggml_tensor * clamp   = nullptr;
            const ggml_tensor * scale   = nullptr;

            if (!args.delayed_softmax) {
                int out_nodes[2];  // nodes which can't be elided

                if (args.sigmoid) {
                    ops.insert(ops.end(), { GGML_OP_UNARY });
                } else if (args.sqrt_softplus) {
                    ops.insert(ops.end(), { GGML_OP_UNARY, GGML_OP_SQRT });
                } else {
                    ops.insert(ops.end(), { GGML_OP_SOFT_MAX });
                }
                ops.push_back(GGML_OP_RESHAPE);
                if (args.prob_bias) {
                    ops.push_back(GGML_OP_ADD);
                    bias = cgraph->nodes[i + ops.size() - 1]->src[1];
                }
                if (args.n_groups > 0) {
                    for (const auto & p : topk_moe_group_pattern) {
                        ops.push_back(p.op);
                    }
                }
                ops.insert(ops.end(), { GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS });
                out_nodes[0] = i + ops.size() - 2;
                ids = cgraph->nodes[out_nodes[0]];

                if (args.norm) {
                    ops.insert(ops.end(),
                               { GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_CLAMP, GGML_OP_DIV, GGML_OP_RESHAPE });
                    clamp = cgraph->nodes[i + ops.size() - 3];
                }
                if (args.scale) {
                    ops.insert(ops.end(), { GGML_OP_SCALE });
                    scale = cgraph->nodes[i + ops.size() - 1];
                }

                weights      = cgraph->nodes[i + ops.size() - 1];
                out_nodes[1] = i + ops.size() - 1;

                if (ggml_can_fuse_subgraph(cgraph, i, ops.size(), ops.data(), out_nodes, 2) &&
                        ggml_cuda_should_use_topk_moe(node, logits, weights, ids) &&
                        ggml_cuda_check_fusion_memory_ranges(cgraph, i, ops.size(), out_nodes, 2, /*is_topk_moe=*/true)) {
                    // local (Inkling 2): the weight normalisation that follows, in the same kernel
                    ggml_cuda_ink2_wnorm_match wm;
                    const int n_norm = !clamp && ggml_cuda_ink2_arch_ok(cuda_ctx->device) ?
                        ggml_cuda_ink2_match_router_norm(cgraph, i + (int) ops.size(), weights, wm) : 0;
                    if (n_norm > 0) {
                        ggml_tensor * nd = cgraph->nodes[i + ops.size() + n_norm - 1];
                        // the normalised weights may replace the weights (the same lane writes both), nothing else
                        bool ok = !ggml_cuda_ink2_overlap(nd, logits) && !ggml_cuda_ink2_overlap(nd, ids) &&
                            (!bias || !ggml_cuda_ink2_overlap(nd, bias)) && (nd->data == weights->data || !ggml_cuda_ink2_overlap(nd, weights));
                        if (ok) {
                            ggml_cuda_ink2_run(cuda_ctx, cgraph, i, (int) ops.size() + n_norm, 5, [&]() {
                                // GGML_CUDA_INK2_CMP=router: the previous two launches first, the normalised weights and the ids
                                // compared byte for byte
                                static const bool cmp = []() { const char * e = getenv("GGML_CUDA_INK2_CMP"); return e && strstr(e, "router"); }();
                                std::vector<uint8_t> r0, r1, g0, g1;
                                if (cmp) {
                                    ggml_cuda_op_topk_moe(*cuda_ctx, logits, weights, ids, clamp, scale, bias, args);
                                    ggml_cuda_op_ink2_moe_wnorm(*cuda_ctx, wm.w, nd, wm.eps, wm.scale);
                                    CUDA_CHECK(cudaStreamSynchronize(cuda_ctx->stream()));
                                    r0.resize(ggml_nbytes(nd)); r1.resize(ggml_nbytes(ids));
                                    CUDA_CHECK(cudaMemcpy(r0.data(), nd->data, r0.size(), cudaMemcpyDeviceToHost));
                                    CUDA_CHECK(cudaMemcpy(r1.data(), ids->data, r1.size(), cudaMemcpyDeviceToHost));
                                }
                                ggml_cuda_op_topk_moe(*cuda_ctx, logits, weights, ids, clamp, scale, bias, args, nd, wm.eps, wm.scale);
                                if (cmp) {
                                    CUDA_CHECK(cudaStreamSynchronize(cuda_ctx->stream()));
                                    g0.resize(r0.size()); g1.resize(r1.size());
                                    CUDA_CHECK(cudaMemcpy(g0.data(), nd->data, g0.size(), cudaMemcpyDeviceToHost));
                                    CUDA_CHECK(cudaMemcpy(g1.data(), ids->data, g1.size(), cudaMemcpyDeviceToHost));
                                    static int64_t calls = 0, diff = 0;
                                    if (calls == 0) {
                                        atexit([]() { fprintf(stderr, "ink2 cmp router: %lld calls, %lld with differences\n", (long long) calls, (long long) diff); });
                                    }
                                    calls++;
                                    diff += r0 != g0 || r1 != g1;
                                }
                            });
                            return ops.size() + n_norm - 1;
                        }
                    }
                    ggml_cuda_op_topk_moe(*cuda_ctx, logits, weights, ids, clamp, scale, bias, args);
                    return ops.size() - 1;
                }
            } else if (!args.norm && !args.prob_bias) {
                //special case gpt-oss, no norm, no bias.
                ops.insert(ops.end(), { GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS, GGML_OP_RESHAPE,
                                        GGML_OP_SOFT_MAX, GGML_OP_RESHAPE });
                weights                     = cgraph->nodes[i + 5];
                ids                         = cgraph->nodes[i + 1];
                const ggml_tensor * softmax = cgraph->nodes[i + 4];

                int out_nodes[2] = { i + 1, i + 5 };
                if (ggml_can_fuse_subgraph(cgraph, i, ops.size(), ops.data(), out_nodes, 2) &&
                        ggml_cuda_should_use_topk_moe(softmax, logits, weights, ids) &&
                        ggml_cuda_check_fusion_memory_ranges(cgraph, i, ops.size(), out_nodes, 2, /*is_topk_moe=*/true)) {
                    ggml_cuda_op_topk_moe(*cuda_ctx, logits, weights, ids, clamp, scale, bias, args);
                    return ops.size() - 1;
                }
            }
        }
    }

    //RoPE + view + set-rows
    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_ROPE, GGML_OP_VIEW, GGML_OP_SET_ROWS }, {})) {
        ggml_tensor * rope     = cgraph->nodes[i];
        ggml_tensor * set_rows = cgraph->nodes[i + 2];

        ggml_cuda_op_rope_fused(*cuda_ctx, rope, set_rows);
        return 2;
    }

    // Snake activation: y = x + sin(a*x)^2 * inv_b
    // Naive 5-op decomposition emitted by frontends: mul -> sin -> sqr -> mul -> add
    if (ggml_can_fuse_subgraph(cgraph, i,
            { GGML_OP_MUL, GGML_OP_SIN, GGML_OP_SQR, GGML_OP_MUL, GGML_OP_ADD },
            { i + 4 })) {
        const ggml_tensor * mul0 = cgraph->nodes[i];
        const ggml_tensor * sqr  = cgraph->nodes[i + 2];
        const ggml_tensor * mul1 = cgraph->nodes[i + 3];
        ggml_tensor *       add  = cgraph->nodes[i + 4];

        // x carries the full activation shape, a is the broadcast operand
        const ggml_tensor * x = ggml_are_same_shape(mul0, mul0->src[0]) ? mul0->src[0] : mul0->src[1];
        const ggml_tensor * a = (x == mul0->src[0]) ? mul0->src[1] : mul0->src[0];

        // mul1 reads sqr and inv_b in either operand order
        const ggml_tensor * inv_b = (mul1->src[0] == sqr) ? mul1->src[1] : mul1->src[0];

        // closure check: the trailing add must read the same x as the leading mul
        const ggml_tensor * x_in_add = (add->src[0] == mul1) ? add->src[1] : add->src[0];

        // Kernel iterates over total = T * C, so x and add must be 2D and
        // a / inv_b must collapse to [1, C, 1, 1]. Higher dims are not handled.
        const bool dim_ok   = (x->ne[2]   == 1 && x->ne[3]   == 1) &&
                              (add->ne[2] == 1 && add->ne[3] == 1) &&
                              (a->ne[2]   == 1 && a->ne[3]   == 1);
        const bool shape_ok = ggml_are_same_shape(a, inv_b) && a->ne[0] == 1 && a->ne[1] == x->ne[1];

        // x is in the supported whitelist and every chain intermediate shares
        // x's type. launch_snake reads a and inv_b as const float *, so they
        // stay F32.
        const ggml_tensor * sin1 = cgraph->nodes[i + 1];
        const bool types_ok = (x->type == GGML_TYPE_F32 || x->type == GGML_TYPE_F16 || x->type == GGML_TYPE_BF16) &&
                              (a->type    == GGML_TYPE_F32) && (inv_b->type == GGML_TYPE_F32) &&
                              (mul0->type == x->type) && (sin1->type  == x->type) &&
                              (sqr->type  == x->type) && (mul1->type  == x->type) &&
                              (add->type  == x->type);

        // kernel reads x[idx] and a[c] / inv_b[c] linearly, so every operand is contiguous
        const bool contig_ok = ggml_is_contiguous(x) && ggml_is_contiguous(add) &&
                               ggml_is_contiguous(a) && ggml_is_contiguous(inv_b);

        if (types_ok && shape_ok && dim_ok && contig_ok && x_in_add == x) {
            ggml_cuda_op_snake_fused(*cuda_ctx, x, a, inv_b, add);
            return 4;
        }
    }

    // multi-(add or mul)
    if (node->op == GGML_OP_ADD || node->op == GGML_OP_MUL) {
        int     n_fuse = 0;
        ggml_op ops[8];
        std::fill(ops, ops + 8, node->op);

        for (; n_fuse <= 6; ++n_fuse) {
            if (!ggml_can_fuse(cgraph, i + n_fuse, ops + n_fuse, 2)) {
                break;
            }
            if (cgraph->nodes[i + n_fuse] != cgraph->nodes[i + n_fuse + 1]->src[0]) {
                break;
            }
            if (!ggml_are_same_layout(cgraph->nodes[i + n_fuse]->src[1], cgraph->nodes[i + n_fuse + 1]->src[1])) {
                break;
            }
        }

        n_fuse++;

        if (n_fuse > 1) {
            ggml_tensor fused_node;
            memcpy(&fused_node, node, sizeof(ggml_tensor));
            for (int j = 0; j < n_fuse - 1; ++j) {
                fused_node.src[j + 2] = cgraph->nodes[i + j + 1]->src[1];
            }
            fused_node.data = cgraph->nodes[i + n_fuse - 1]->data;
            if (node->op == GGML_OP_ADD) {
                ggml_cuda_op_fused_add(*cuda_ctx, &fused_node, n_fuse);
            } else {
                ggml_cuda_op_fused_mul(*cuda_ctx, &fused_node, n_fuse);
            }
            return n_fuse - 1;
        }
    }

    bool fused_mul_mat_vec = false;
    int  fused_node_count  = 0;

    auto get_bias_tensor = [](const ggml_tensor * bias_node, const ggml_tensor * mul_node, ggml_op op_bias) -> const ggml_tensor * {
        if (op_bias == GGML_OP_ADD) {
            if (bias_node->src[0] == mul_node) {
                return bias_node->src[1];
            }
            if (bias_node->src[1] == mul_node) {
                return bias_node->src[0];
            }
            return nullptr;
        }
        GGML_ASSERT(op_bias == GGML_OP_ADD_ID);
        GGML_ASSERT(bias_node->src[0] == mul_node);
        return bias_node->src[1];
    };

    // gate + glu + up, with optional scale/bias on both lanes.
    for (ggml_op op : { GGML_OP_MUL_MAT, GGML_OP_MUL_MAT_ID }) {
        const ggml_op bias_op = op == GGML_OP_MUL_MAT ? GGML_OP_ADD : GGML_OP_ADD_ID;

        if (op == GGML_OP_MUL_MAT) {
            for (const bool with_bias : { false, true }) {
                const int gate_idx       = i;
                const int gate_scale_idx = i + 1;
                const int gate_bias_idx  = with_bias ? i + 2 : -1;
                const int up_idx         = with_bias ? i + 3 : i + 2;
                const int up_scale_idx   = up_idx + 1;
                const int up_bias_idx    = with_bias ? up_idx + 2 : -1;
                const int glu_idx        = with_bias ? up_idx + 3 : up_idx + 2;

                const int out_nodes[] = { glu_idx };
                ggml_op ops[7];
                if (with_bias) {
                    ops[0] = op;
                    ops[1] = GGML_OP_MUL;
                    ops[2] = bias_op;
                    ops[3] = op;
                    ops[4] = GGML_OP_MUL;
                    ops[5] = bias_op;
                    ops[6] = GGML_OP_GLU;
                } else {
                    ops[0] = op;
                    ops[1] = GGML_OP_MUL;
                    ops[2] = op;
                    ops[3] = GGML_OP_MUL;
                    ops[4] = GGML_OP_GLU;
                }
                const int n_ops = with_bias ? 7 : 5;

                if (!ggml_can_fuse_subgraph(cgraph, i, n_ops, ops, out_nodes, 1) ||
                        !ggml_cuda_check_fusion_memory_ranges(cgraph, i, n_ops, out_nodes, 1)) {
                    continue;
                }

                ggml_tensor * gate_n       = cgraph->nodes[gate_idx];
                ggml_tensor * gate_scale_n = cgraph->nodes[gate_scale_idx];
                ggml_tensor * gate_out_n   = with_bias ? cgraph->nodes[gate_bias_idx] : gate_scale_n;
                ggml_tensor * up_n         = cgraph->nodes[up_idx];
                ggml_tensor * up_scale_n   = cgraph->nodes[up_scale_idx];
                ggml_tensor * up_out_n     = with_bias ? cgraph->nodes[up_bias_idx] : up_scale_n;
                const ggml_tensor * glu = cgraph->nodes[glu_idx];

                if (!ggml_cuda_should_fuse_mul_mat(up_n, gate_n, glu,
                        with_bias ? up_out_n : nullptr, with_bias ? gate_out_n : nullptr, up_scale_n, gate_scale_n)) {
                    continue;
                }

                const ggml_tensor * gate_scale = ggml_cuda_get_mul_mat_scale(gate_scale_n, gate_n);
                const ggml_tensor * up_scale   = ggml_cuda_get_mul_mat_scale(up_scale_n, up_n);
                if (!gate_scale || !up_scale) {
                    continue;
                }

                const ggml_tensor * up_bias   = with_bias ? get_bias_tensor(up_out_n, up_scale_n, bias_op) : nullptr;
                const ggml_tensor * gate_bias = with_bias ? get_bias_tensor(gate_out_n, gate_scale_n, bias_op) : nullptr;
                if (with_bias && (!ggml_are_same_shape(gate_out_n->src[0], gate_out_n->src[1]) ||
                        !ggml_are_same_shape(up_out_n->src[0], up_out_n->src[1]))) {
                    continue;
                }

                const ggml_tensor * src0 = up_n->src[0];
                const ggml_tensor * src1 = up_n->src[1];
                const ggml_tensor * ids  = up_n->src[2];

                ggml_cuda_mm_fusion_args_host fusion_data{};
                fusion_data.gate       = gate_n->src[0];
                fusion_data.x_bias     = up_bias;
                fusion_data.gate_bias  = gate_bias;
                fusion_data.x_scale    = up_scale;
                fusion_data.gate_scale = gate_scale;
                fusion_data.glu_op     = ggml_get_glu_op(glu);
                fusion_data.glu_limit  = ggml_get_op_params_f32(glu, 3);

                if (ggml_cuda_should_fuse_mul_mat_vec_f(up_n, gate_n)) {
                    ggml_cuda_mul_mat_vec_f(*cuda_ctx, src0, src1, ids, cgraph->nodes[glu_idx], &fusion_data);
                    fused_mul_mat_vec = true;
                    fused_node_count  = n_ops;
                    break;
                }

                if (ggml_cuda_should_fuse_mul_mat_vec_q(up_n)) {
                    ggml_cuda_mul_mat_vec_q(*cuda_ctx, src0, src1, ids, cgraph->nodes[glu_idx], &fusion_data);
                    fused_mul_mat_vec = true;
                    fused_node_count  = n_ops;
                    break;
                }

                if (!with_bias && ggml_cuda_should_fuse_mul_mat_q(up_n, gate_n, glu, up_scale_n, gate_scale_n)) {
                    fusion_data.prec_src1 = ggml_prec_get_src(up_n, 1);
                    ggml_cuda_mul_mat_q(*cuda_ctx, src0, src1, ids, cgraph->nodes[glu_idx], &fusion_data);
                    fused_mul_mat_vec = true;
                    fused_node_count  = n_ops;
                    break;
                }
            }

            if (fused_mul_mat_vec) {
                break;
            }
        } else {
            for (const bool with_bias : { false, true }) {
                const int gate_idx       = i;
                const int gate_scale_idx = i + 4;
                const int gate_bias_idx  = with_bias ? i + 5 : -1;
                const int up_idx         = with_bias ? i + 6 : i + 5;
                const int up_scale_idx   = up_idx + 4;
                const int up_bias_idx    = with_bias ? up_idx + 5 : -1;
                const int glu_idx        = with_bias ? up_idx + 6 : up_idx + 5;

                const int out_nodes[] = { glu_idx };
                ggml_op ops[13];
                if (with_bias) {
                    ops[0]  = op;
                    ops[1]  = GGML_OP_RESHAPE;
                    ops[2]  = GGML_OP_REPEAT;
                    ops[3]  = GGML_OP_GET_ROWS;
                    ops[4]  = GGML_OP_MUL;
                    ops[5]  = bias_op;
                    ops[6]  = op;
                    ops[7]  = GGML_OP_RESHAPE;
                    ops[8]  = GGML_OP_REPEAT;
                    ops[9]  = GGML_OP_GET_ROWS;
                    ops[10] = GGML_OP_MUL;
                    ops[11] = bias_op;
                    ops[12] = GGML_OP_GLU;
                } else {
                    ops[0]  = op;
                    ops[1]  = GGML_OP_RESHAPE;
                    ops[2]  = GGML_OP_REPEAT;
                    ops[3]  = GGML_OP_GET_ROWS;
                    ops[4]  = GGML_OP_MUL;
                    ops[5]  = op;
                    ops[6]  = GGML_OP_RESHAPE;
                    ops[7]  = GGML_OP_REPEAT;
                    ops[8]  = GGML_OP_GET_ROWS;
                    ops[9]  = GGML_OP_MUL;
                    ops[10] = GGML_OP_GLU;
                }
                const int n_ops = with_bias ? 13 : 11;

                if (!ggml_can_fuse_subgraph(cgraph, i, n_ops, ops, out_nodes, 1) ||
                        !ggml_cuda_check_fusion_memory_ranges(cgraph, i, n_ops, out_nodes, 1)) {
                    continue;
                }

                ggml_tensor * gate_n       = cgraph->nodes[gate_idx];
                ggml_tensor * gate_scale_n = cgraph->nodes[gate_scale_idx];
                ggml_tensor * gate_out_n   = with_bias ? cgraph->nodes[gate_bias_idx] : gate_scale_n;
                ggml_tensor * up_n         = cgraph->nodes[up_idx];
                ggml_tensor * up_scale_n   = cgraph->nodes[up_scale_idx];
                ggml_tensor * up_out_n     = with_bias ? cgraph->nodes[up_bias_idx] : up_scale_n;
                const ggml_tensor * glu = cgraph->nodes[glu_idx];

                if (!ggml_cuda_should_fuse_mul_mat(up_n, gate_n, glu,
                        with_bias ? up_out_n : nullptr, with_bias ? gate_out_n : nullptr, up_scale_n, gate_scale_n)) {
                    continue;
                }

                const ggml_tensor * gate_scale = ggml_cuda_get_mul_mat_id_scale(cgraph->nodes[gate_idx + 1], cgraph->nodes[gate_idx + 2],
                        cgraph->nodes[gate_idx + 3], gate_scale_n, gate_n);
                const ggml_tensor * up_scale = ggml_cuda_get_mul_mat_id_scale(cgraph->nodes[up_idx + 1], cgraph->nodes[up_idx + 2],
                        cgraph->nodes[up_idx + 3], up_scale_n, up_n);
                if (!gate_scale || !up_scale) {
                    continue;
                }

                const ggml_tensor * up_bias   = with_bias ? get_bias_tensor(up_out_n, up_scale_n, bias_op) : nullptr;
                const ggml_tensor * gate_bias = with_bias ? get_bias_tensor(gate_out_n, gate_scale_n, bias_op) : nullptr;

                const ggml_tensor * src0 = up_n->src[0];
                const ggml_tensor * src1 = up_n->src[1];
                const ggml_tensor * ids  = up_n->src[2];

                ggml_cuda_mm_fusion_args_host fusion_data{};
                fusion_data.gate       = gate_n->src[0];
                fusion_data.x_bias     = up_bias;
                fusion_data.gate_bias  = gate_bias;
                fusion_data.x_scale    = up_scale;
                fusion_data.gate_scale = gate_scale;
                fusion_data.glu_op     = ggml_get_glu_op(glu);
                fusion_data.glu_limit  = ggml_get_op_params_f32(glu, 3);

                if (ggml_cuda_should_fuse_mul_mat_vec_f(up_n, gate_n)) {
                    ggml_cuda_mul_mat_vec_f(*cuda_ctx, src0, src1, ids, cgraph->nodes[glu_idx], &fusion_data);
                    fused_mul_mat_vec = true;
                    fused_node_count  = n_ops;
                    break;
                }

                if (ggml_cuda_should_fuse_mul_mat_vec_q(up_n)) {
                    ggml_cuda_mul_mat_vec_q(*cuda_ctx, src0, src1, ids, cgraph->nodes[glu_idx], &fusion_data);
                    fused_mul_mat_vec = true;
                    fused_node_count  = n_ops;
                    break;
                }

                if (!with_bias && ggml_cuda_should_fuse_mul_mat_q(up_n, gate_n, glu, up_scale_n, gate_scale_n)) {
                    fusion_data.prec_src1 = ggml_prec_get_src(up_n, 1);
                    ggml_cuda_mul_mat_q(*cuda_ctx, src0, src1, ids, cgraph->nodes[glu_idx], &fusion_data);
                    fused_mul_mat_vec = true;
                    fused_node_count  = n_ops;
                    break;
                }
            }

            if (fused_mul_mat_vec) {
                break;
            }
        }

        if (ggml_cuda_can_fuse(cgraph, i, { op, bias_op, op, bias_op, GGML_OP_GLU }, {})) {
            ggml_tensor * glu         = cgraph->nodes[i + 4];
            ggml_tensor * gate_bias_n = glu->src[0];
            ggml_tensor * up_bias_n   = glu->src[1];

            //we don't assume the order for {gate, up}. Instead infer it from the bias tensor
            ggml_tensor * gate_n = nullptr;
            ggml_tensor * up_n   = nullptr;

            if (gate_bias_n->src[0] == cgraph->nodes[i] || gate_bias_n->src[1] == cgraph->nodes[i]) {
                gate_n = cgraph->nodes[i];
                up_n   = cgraph->nodes[i + 2];
            } else if (gate_bias_n->src[0] == cgraph->nodes[i + 2] || gate_bias_n->src[1] == cgraph->nodes[i + 2]) {
                gate_n = cgraph->nodes[i + 2];
                up_n   = cgraph->nodes[i];
            } else {
                continue;
            }

            const ggml_tensor * up_bias_tensor   = get_bias_tensor(up_bias_n, up_n, bias_op);
            const ggml_tensor * gate_bias_tensor = get_bias_tensor(gate_bias_n, gate_n, bias_op);

            if (!up_bias_tensor || !gate_bias_tensor) {
                continue;
            }

            // we don't support repeating adds
            if (bias_op == GGML_OP_ADD && (!ggml_are_same_shape(gate_bias_n->src[0], gate_bias_n->src[1]) ||
                                           !ggml_are_same_shape(up_bias_n->src[0], up_bias_n->src[1]))) {
                continue;
            }

            const ggml_tensor * src0 = up_n->src[0];
            const ggml_tensor * src1 = up_n->src[1];
            const ggml_tensor * ids  = up_n->src[2];

            if (ggml_cuda_should_fuse_mul_mat_vec_f(up_n, gate_n)) {
                ggml_cuda_mm_fusion_args_host fusion_data{};
                fusion_data.gate      = gate_n->src[0];
                fusion_data.x_bias    = up_bias_tensor;
                fusion_data.gate_bias = gate_bias_tensor;
                fusion_data.glu_op    = ggml_get_glu_op(glu);
                fusion_data.glu_limit = ggml_get_op_params_f32(glu, 3);

                ggml_cuda_mul_mat_vec_f(*cuda_ctx, src0, src1, ids, glu, &fusion_data);
                fused_mul_mat_vec = true;
                fused_node_count  = 5;
                break;
            }

            if (ggml_cuda_should_fuse_mul_mat_vec_q(up_n)) {
                ggml_cuda_mm_fusion_args_host fusion_data{};
                fusion_data.gate      = gate_n->src[0];
                fusion_data.x_bias    = up_bias_tensor;
                fusion_data.gate_bias = gate_bias_tensor;
                fusion_data.glu_op    = ggml_get_glu_op(glu);
                fusion_data.glu_limit = ggml_get_op_params_f32(glu, 3);

                ggml_cuda_mul_mat_vec_q(*cuda_ctx, src0, src1, ids, glu, &fusion_data);
                fused_mul_mat_vec = true;
                fused_node_count  = 5;
                break;
            }
        } else if (ggml_cuda_can_fuse(cgraph, i, { op, op, GGML_OP_GLU }, {})) {
            ggml_tensor * glu  = cgraph->nodes[i + 2];
            ggml_tensor * gate = glu->src[0];
            ggml_tensor * up   = glu->src[1];

            bool ok = (gate == cgraph->nodes[i] && up == cgraph->nodes[i + 1]) ||
                      (gate == cgraph->nodes[i + 1] && up == cgraph->nodes[i]);

            if (!ok) {
                continue;
            }

            const ggml_tensor * src0 = up->src[0];
            const ggml_tensor * src1 = up->src[1];
            const ggml_tensor * ids  = up->src[2];

            if (ggml_cuda_should_fuse_mul_mat_vec_f(up, gate)) {
                ggml_cuda_mm_fusion_args_host fusion_data{};
                fusion_data.gate      = gate->src[0];
                fusion_data.glu_op    = ggml_get_glu_op(glu);
                fusion_data.glu_limit = ggml_get_op_params_f32(glu, 3);

                ggml_cuda_mul_mat_vec_f(*cuda_ctx, src0, src1, ids, glu, &fusion_data);
                fused_mul_mat_vec = true;
                fused_node_count  = 3;
                break;
            }

            if (ggml_cuda_should_fuse_mul_mat_vec_q(up)) {
                ggml_cuda_mm_fusion_args_host fusion_data{};
                fusion_data.gate      = gate->src[0];
                fusion_data.glu_op    = ggml_get_glu_op(glu);
                fusion_data.glu_limit = ggml_get_op_params_f32(glu, 3);

                ggml_cuda_mul_mat_vec_q(*cuda_ctx, src0, src1, ids, glu, &fusion_data);
                fused_mul_mat_vec = true;
                fused_node_count  = 3;
                break;
            }

            if (ggml_cuda_should_fuse_mul_mat_q(up, gate, glu)) {
                ggml_cuda_mm_fusion_args_host fusion_data{};
                fusion_data.gate      = gate->src[0];
                fusion_data.glu_op    = ggml_get_glu_op(glu);
                fusion_data.glu_limit = ggml_get_op_params_f32(glu, 3);
                fusion_data.prec_src1 = ggml_prec_get_src(up, 1);

                ggml_cuda_mul_mat_q(*cuda_ctx, src0, src1, ids, glu, &fusion_data);
                fused_mul_mat_vec = true;
                fused_node_count  = 3;
                break;
            }
        }
    }

    if (fused_mul_mat_vec) {
        return fused_node_count - 1;
    }

    fused_mul_mat_vec = false;
    fused_node_count  = 0;

    // mul_mat + scale + optional bias
    for (ggml_op op : { GGML_OP_MUL_MAT, GGML_OP_MUL_MAT_ID }) {
        const ggml_op bias_op = op == GGML_OP_MUL_MAT ? GGML_OP_ADD : GGML_OP_ADD_ID;

        for (const bool with_bias : { false, true }) {
            const int n_ops = op == GGML_OP_MUL_MAT ? (with_bias ? 3 : 2) : (with_bias ? 6 : 5);
            const int out_nodes[] = { i + n_ops - 1 };
            ggml_op ops[6];
            if (op == GGML_OP_MUL_MAT) {
                if (with_bias) {
                    ops[0] = op;
                    ops[1] = GGML_OP_MUL;
                    ops[2] = bias_op;
                } else {
                    ops[0] = op;
                    ops[1] = GGML_OP_MUL;
                }
            } else {
                if (with_bias) {
                    ops[0] = op;
                    ops[1] = GGML_OP_RESHAPE;
                    ops[2] = GGML_OP_REPEAT;
                    ops[3] = GGML_OP_GET_ROWS;
                    ops[4] = GGML_OP_MUL;
                    ops[5] = bias_op;
                } else {
                    ops[0] = op;
                    ops[1] = GGML_OP_RESHAPE;
                    ops[2] = GGML_OP_REPEAT;
                    ops[3] = GGML_OP_GET_ROWS;
                    ops[4] = GGML_OP_MUL;
                }
            }

            if (!ggml_can_fuse_subgraph(cgraph, i, n_ops, ops, out_nodes, 1) ||
                    !ggml_cuda_check_fusion_memory_ranges(cgraph, i, n_ops, out_nodes, 1)) {
                continue;
            }

            ggml_tensor * mm_node    = cgraph->nodes[i];
            ggml_tensor * scale_node = op == GGML_OP_MUL_MAT ? cgraph->nodes[i + 1] : cgraph->nodes[i + 4];
            ggml_tensor * out_node   = with_bias ? cgraph->nodes[i + n_ops - 1] : scale_node;

            const ggml_tensor * scale = nullptr;
            if (op == GGML_OP_MUL_MAT) {
                scale = ggml_cuda_get_mul_mat_scale(scale_node, mm_node);
            } else {
                scale = ggml_cuda_get_mul_mat_id_scale(cgraph->nodes[i + 1], cgraph->nodes[i + 2], cgraph->nodes[i + 3], scale_node, mm_node);
            }
            if (!scale) {
                continue;
            }

            const ggml_tensor * bias = with_bias ? get_bias_tensor(out_node, scale_node, bias_op) : nullptr;
            if (with_bias && !bias) {
                continue;
            }
            if (with_bias && bias_op == GGML_OP_ADD && !ggml_are_same_shape(out_node->src[0], out_node->src[1])) {
                continue;
            }
            if (with_bias && bias_op == GGML_OP_ADD_ID && out_node->src[2] != mm_node->src[2]) {
                continue;
            }

            const ggml_tensor * src0 = mm_node->src[0];
            const ggml_tensor * src1 = mm_node->src[1];
            const ggml_tensor * ids  = mm_node->src[2];

            ggml_cuda_mm_fusion_args_host fusion_data{};
            fusion_data.x_bias  = bias;
            fusion_data.x_scale = scale;

            if (ggml_cuda_should_fuse_mul_mat_vec_f(mm_node)) {
                ggml_cuda_mul_mat_vec_f(*cuda_ctx, src0, src1, ids, out_node, &fusion_data);
                fused_mul_mat_vec = true;
                fused_node_count  = n_ops;
                break;
            }

            if (op == GGML_OP_MUL_MAT && !with_bias && src0->type == GGML_TYPE_F8_E4M3 &&
                    ggml_get_op_params_i32(mm_node, 3) != GGML_PREC_F32 &&
                    ggml_cuda_mul_mat_fp8(*cuda_ctx, src0, src1, scale_node, scale)) {
                return 1;
            }

            if (!with_bias && src0->type == GGML_TYPE_NVFP4 && src1->type == GGML_TYPE_F32 && mm_node->type == GGML_TYPE_F32) {
                const int cc = ggml_cuda_info().devices[cuda_ctx->device].cc;
                const bool is_id = op == GGML_OP_MUL_MAT_ID;
                const int64_t ncols = is_id ? src1->ne[2] : src1->ne[1];
                const bool use_mmvq = is_id ? ncols <= get_mmvq_mmid_max_batch(src0->type, cc) :
                                            ggml_cuda_should_use_mmvq(src0->type, cc, ncols);
                const bool bad_padding_clear = ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE &&
                                               ggml_nbytes(src0) != ggml_backend_buffer_get_alloc_size(src0->buffer, src0) &&
                                               src0->view_src;
                if (!use_mmvq && !bad_padding_clear &&
                        ggml_cuda_should_use_mmq(src0->type, cc, ncols, /*n_experts =*/ is_id ? src0->ne[2] : 0)) {
                    fusion_data.prec_src1 = ggml_prec_get_src(mm_node, 1);
                    ggml_cuda_mul_mat_q(*cuda_ctx, src0, src1, ids, out_node, &fusion_data);
                    return n_ops - 1;
                }
            }

            if (ggml_cuda_should_fuse_mul_mat_vec_q(mm_node)) {
                ggml_cuda_mul_mat_vec_q(*cuda_ctx, src0, src1, ids, out_node, &fusion_data);
                fused_mul_mat_vec = true;
                fused_node_count  = n_ops;
                break;
            }
        }
        if (fused_mul_mat_vec) {
            break;
        }
    }

    if (fused_mul_mat_vec) {
        return fused_node_count - 1;
    }

    // mul_mat + add
    for (ggml_op op : { GGML_OP_MUL_MAT, GGML_OP_MUL_MAT_ID }) {
        const ggml_op bias_op = op == GGML_OP_MUL_MAT ? GGML_OP_ADD : GGML_OP_ADD_ID;

        if (!ggml_can_fuse(cgraph, i, { op, bias_op })) {
            continue;
        }

        ggml_tensor * mm_node   = cgraph->nodes[i];
        ggml_tensor * bias_node = cgraph->nodes[i + 1];

        ggml_tensor * bias_tensor = nullptr;
        if (bias_op == GGML_OP_ADD) {
            if (bias_node->src[0] == mm_node) {
                bias_tensor = bias_node->src[1];
            } else if (bias_node->src[1] == mm_node) {
                bias_tensor = bias_node->src[0];
            } else {
                continue;
            }
        } else {
            if (bias_node->src[0] != mm_node) {
                continue;
            }
            bias_tensor = bias_node->src[1];
        }

        const ggml_tensor * src0 = mm_node->src[0];
        const ggml_tensor * src1 = mm_node->src[1];
        const ggml_tensor * ids  = mm_node->src[2];

        if (bias_op == GGML_OP_ADD_ID && bias_node->src[2] != ids) {
            continue;
        }

        if (bias_op == GGML_OP_ADD && !ggml_are_same_shape(bias_node->src[0], bias_node->src[1])) {
            continue;
        }

        ggml_cuda_mm_fusion_args_host fusion_data{};
        fusion_data.x_bias = bias_tensor;

        if (ggml_cuda_should_fuse_mul_mat_vec_f(mm_node)) {
            ggml_cuda_mul_mat_vec_f(*cuda_ctx, src0, src1, ids, bias_node, &fusion_data);
            fused_mul_mat_vec = true;
            fused_node_count  = 2;
            break;
        }

        if (ggml_cuda_should_fuse_mul_mat_vec_q(mm_node)) {
            ggml_cuda_mul_mat_vec_q(*cuda_ctx, src0, src1, ids, bias_node, &fusion_data);
            fused_mul_mat_vec = true;
            fused_node_count  = 2;
            break;
        }
    }

    if (fused_mul_mat_vec) {
        return fused_node_count - 1;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ROPE, GGML_OP_VIEW, GGML_OP_SET_ROWS }, {})) {
        ggml_cuda_op_rms_norm_mul_rope_fused(*cuda_ctx, node, cgraph->nodes[i + 1], cgraph->nodes[i + 2], cgraph->nodes[i + 4]);
        return 4;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ROPE }, {})) {
        ggml_cuda_op_rms_norm_mul_rope_fused(*cuda_ctx, node, cgraph->nodes[i + 1], cgraph->nodes[i + 2], nullptr);
        return 2;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_NORM, GGML_OP_MUL, GGML_OP_ADD }, {})) {
        ggml_cuda_op_norm(*cuda_ctx, node, cgraph->nodes[i + 1], cgraph->nodes[i + 2]);
        return 2;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_NORM, GGML_OP_MUL }, {})) {
        ggml_cuda_op_norm(*cuda_ctx, node, cgraph->nodes[i + 1]);
        return 1;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ADD }, {})) {
        ggml_cuda_op_rms_norm_fused_add(*cuda_ctx, node, cgraph->nodes[i + 1], cgraph->nodes[i + 2]);
        return 2;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL }, {})) {
        ggml_cuda_op_rms_norm_fused(*cuda_ctx, node, cgraph->nodes[i + 1]);
        return 1;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_SCALE }, {})) {
        ggml_cuda_op_rms_norm_scale_fused(*cuda_ctx, node, cgraph->nodes[i + 1]);
        return 1;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_SSM_CONV, GGML_OP_ADD, GGML_OP_UNARY }, { GGML_UNARY_OP_SILU })) {
        ggml_cuda_op_ssm_conv(*cuda_ctx, node, cgraph->nodes[i + 1], cgraph->nodes[i + 2]);
        return 2;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_SSM_CONV, GGML_OP_UNARY }, { GGML_UNARY_OP_SILU })) {
        ggml_cuda_op_ssm_conv(*cuda_ctx, node, /*bias_add_node=*/ nullptr, cgraph->nodes[i + 1]);
        return 1;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_UNARY, GGML_OP_MUL }, { GGML_UNARY_OP_SILU }) ||
        ggml_cuda_can_fuse(cgraph, i, { GGML_OP_UNARY, GGML_OP_MUL }, { GGML_UNARY_OP_SIGMOID }) ||
        ggml_cuda_can_fuse(cgraph, i, { GGML_OP_UNARY, GGML_OP_MUL }, { GGML_UNARY_OP_SOFTPLUS })) {
        ggml_cuda_op_unary_mul(*cuda_ctx, node, cgraph->nodes[i + 1]);
        return 1;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_UNARY, GGML_OP_SQR }, { GGML_UNARY_OP_RELU })) {
        ggml_cuda_op_relu_sqr(*cuda_ctx, node, cgraph->nodes[i + 1]);
        return 1;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_SCALE, GGML_OP_UNARY, GGML_OP_SCALE, GGML_OP_DSV4_HC_POST }, { GGML_UNARY_OP_SIGMOID })) {
        ggml_cuda_op_dsv4_hc_post(*cuda_ctx, cgraph->nodes[i + 3], node, cgraph->nodes[i + 2]);
        return 3;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_SCALE, GGML_OP_UNARY, GGML_OP_SCALE }, { GGML_UNARY_OP_TANH })) {
        ggml_cuda_op_softcap(*cuda_ctx, cgraph->nodes[i + 2], node);
        return 2;
    }

    return 0;
}

// local diagnostic: GGML_CUDA_MEM_TRACE=1 (with GGML_CUDA_DISABLE_GRAPHS=1) synchronizes after every node and prints the nodes
// after which the free device memory dropped by more than 8 MiB: memory the runtime takes at a kernel launch (scratch, code)
static void ggml_cuda_mem_trace(ggml_backend_cuda_context * cuda_ctx, const ggml_tensor * node, const int n_fused) {
    static const bool on = getenv("GGML_CUDA_MEM_TRACE") != nullptr;
    if (!on) {
        return;
    }
    static size_t free_prev = 0;
    CUDA_CHECK(cudaStreamSynchronize(cuda_ctx->stream()));
    size_t free_b = 0, total_b = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));
    if (free_prev && free_b + (size_t(8) << 20) < free_prev) {
        fprintf(stderr, "mem trace: free %.1f -> %.1f MiB after %s (%s, %d fused) ne %lld %lld %lld\n", free_prev/1048576.0, free_b/1048576.0,
            node->name, ggml_op_name(node->op), n_fused, (long long) node->ne[0], (long long) node->ne[1], (long long) node->ne[2]);
    }
    free_prev = free_b;
}

static void ggml_cuda_graph_evaluate_and_capture(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, const bool use_cuda_graph, const bool cuda_graph_update_required, uint64_t graph_key) {
    bool graph_evaluated_or_captured = false;

    // flag used to determine whether it is an integrated_gpu
    const bool integrated            = ggml_cuda_info().devices[cuda_ctx->device].integrated;

    ggml_cuda_stream_context & stream_ctx = cuda_ctx->stream_context();
    bool                         is_concurrent_event_active = false;
    ggml_cuda_concurrent_event * concurrent_event           = nullptr;
    bool                         should_launch_concurrent_events = false;

#ifdef USE_CUDA_GRAPH
    ggml_cuda_graph * graph      = use_cuda_graph ? cuda_ctx->cuda_graph(graph_key) : nullptr;
    const int         chunk_size = ggml_cuda_graph_chunk_size();
    int               chunk_end  = chunk_size;
    size_t            n_chunks   = 0;
#endif // USE_CUDA_GRAPH

    const auto try_launch_concurrent_event = [&](const ggml_tensor * node) {
        if (stream_ctx.concurrent_events.find(node) != stream_ctx.concurrent_events.end()) {
            concurrent_event = &stream_ctx.concurrent_events[node];

            is_concurrent_event_active = true;

            GGML_LOG_DEBUG("Launching %d streams at %s\n", concurrent_event->n_streams, node->name);

            cudaStream_t main_stream = cuda_ctx->stream();  // this should be stream 0
            GGML_ASSERT(cuda_ctx->curr_stream_no == 0);
            CUDA_CHECK(cudaEventRecord(concurrent_event->fork_event, main_stream));

            for (int i = 1; i <= concurrent_event->n_streams; ++i) {
                cudaStream_t stream = cuda_ctx->stream(cuda_ctx->device, i);
                CUDA_CHECK(cudaStreamWaitEvent(stream, concurrent_event->fork_event));
            }
        }
    };

    while (!graph_evaluated_or_captured) {
        // Only perform the graph execution if CUDA graphs are not enabled, or we are capturing the graph.
        // With the use of CUDA graphs, the execution will be performed by the graph launch.
        if (!use_cuda_graph || cuda_graph_update_required) {
            [[maybe_unused]] int prev_i = 0;

            if (stream_ctx.concurrent_events.size() > 0) {
                should_launch_concurrent_events = true;
                for (const auto & [tensor, event] : stream_ctx.concurrent_events) {
                    should_launch_concurrent_events = should_launch_concurrent_events && event.is_valid();
                }
            }

            if (should_launch_concurrent_events) {
                // Restore original node order within each concurrent region to enable fusion within streams

                std::unordered_map<const ggml_tensor *, int> node_to_idx;
                node_to_idx.reserve(cgraph->n_nodes);
                for (int i = 0; i < cgraph->n_nodes; ++i) {
                    node_to_idx[cgraph->nodes[i]] = i;
                }

                for (auto & [fork_node, event] : stream_ctx.concurrent_events) {
                    // Find positions of all nodes from this event in the current graph
                    std::vector<int> positions;
                    positions.reserve(event.original_order.size());

                    bool all_found = true;
                    for (const ggml_tensor * orig_node : event.original_order) {
                        auto it = node_to_idx.find(orig_node);
                        if (it != node_to_idx.end()) {
                            positions.push_back(it->second);
                        } else {
                            all_found = false;
                            break;
                        }
                    }

                    if (!all_found || positions.size() != event.original_order.size()) {
                        continue;
                    }

                    // Sort positions to get contiguous range
                    std::vector<int> sorted_positions = positions;
                    std::sort(sorted_positions.begin(), sorted_positions.end());

                    bool is_contiguous = true;
                    for (size_t i = 1; i < sorted_positions.size(); ++i) {
                        if (sorted_positions[i] != sorted_positions[i-1] + 1) {
                            is_contiguous = false;
                            break;
                        }
                    }

                    if (!is_contiguous) {
                        continue;
                    }

                    // Restore original order at the sorted positions
                    int start_pos = sorted_positions[0];
                    for (size_t i = 0; i < event.original_order.size(); ++i) {
                        cgraph->nodes[start_pos + i] = const_cast<ggml_tensor *>(event.original_order[i]);
                    }
                }
            } else {
                stream_ctx.concurrent_events.clear();
            }

            for (int i = 0; i < cgraph->n_nodes; i++) {
                ggml_tensor * node = cgraph->nodes[i];
#ifdef USE_CUDA_GRAPH
                // fused nodes are skipped together, so a cut never splits a fusion
                // a concurrent region must join before its capture can end
                if (use_cuda_graph && chunk_size > 0 && i >= chunk_end && !is_concurrent_event_active) {
                    ggml_cuda_graph_launch_chunk(cuda_ctx, graph, n_chunks++);
                    CUDA_CHECK(cudaStreamBeginCapture(cuda_ctx->stream(), cudaStreamCaptureModeRelaxed));
                    chunk_end = i + chunk_size;
                }
#endif // USE_CUDA_GRAPH
                if (is_concurrent_event_active) {
                    GGML_ASSERT(concurrent_event);

                    if (node == concurrent_event->join_node) {
                        cuda_ctx->curr_stream_no = 0;
                        for (int i = 1; i <= concurrent_event->n_streams; ++i) {
                            // Wait on join events of forked streams in the main stream
                            CUDA_CHECK(cudaEventRecord(concurrent_event->join_events[i - 1],
                                                       cuda_ctx->stream(cuda_ctx->device, i)));
                            CUDA_CHECK(cudaStreamWaitEvent(cuda_ctx->stream(), concurrent_event->join_events[i - 1]));
                        }

                        is_concurrent_event_active = false;
                        concurrent_event           = nullptr;
                    } else {
                        GGML_ASSERT (concurrent_event->stream_mapping.find(node) != concurrent_event->stream_mapping.end());
                        cuda_ctx->curr_stream_no = concurrent_event->stream_mapping[node];
                        GGML_LOG_DEBUG("Setting stream no to %d for node %s\n", cuda_ctx->curr_stream_no, node->name);
                    }
                } else if (i - prev_i > 1) {
                    //the previous node was fused
                    const ggml_tensor * prev_node = cgraph->nodes[i - 1];
                    try_launch_concurrent_event(prev_node);

                    if (is_concurrent_event_active) {
                        cuda_ctx->curr_stream_no = concurrent_event->stream_mapping[node];
                        GGML_LOG_DEBUG("Setting stream no to %d for node %s\n", cuda_ctx->curr_stream_no, node->name);
                    }
                }

                prev_i = i;

                if (ggml_cuda_is_view_or_noop(node)) {
                    continue;
                }

                if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
                    continue;
                }

                ggml_cuda_ink2_trace_tick(cuda_ctx, cgraph, i);
                int nodes_to_skip = ggml_cuda_try_fuse(cuda_ctx, cgraph, i);
                ggml_cuda_ink2_trace(cgraph, i, nodes_to_skip);

                if (nodes_to_skip != 0) {
#ifdef GGML_CUDA_DEBUG
                    const int last_fused = i + nodes_to_skip;
                    GGML_LOG_INFO("nodes_fused: %d, first: %s (%s), last: %s (%s)\n",
                            nodes_to_skip + 1, ggml_op_name(node->op), node->name,
                            ggml_op_name(cgraph->nodes[last_fused]->op), cgraph->nodes[last_fused]->name);
#endif
                    ggml_cuda_mem_trace(cuda_ctx, node, nodes_to_skip + 1);
                    i += nodes_to_skip;
                    continue;
                }
#ifndef NDEBUG
                // On integrated GPUs (APUs, e.g. RDNA3.5) the scheduler may place a
                // node's output on the host-visible buffer, which the compute path
                // handles. Allow that here, mirroring the src-tensor check below.
                assert(node->buffer->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device) ||
                       (integrated && ggml_backend_buft_is_cuda_host(node->buffer->buft)));
                for (int j = 0; j < GGML_MAX_SRC; j++) {
                    if (node->src[j] != nullptr) {
                        assert(node->src[j]->buffer);
                        assert(node->src[j]->buffer->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device) ||
                               (integrated && ggml_backend_buft_is_cuda_host(node->src[j]->buffer->buft)));
                    }
                }
#else
                GGML_UNUSED(integrated);
#endif  // NDEBUG

                if (ggml_cuda_ink2_skip_op(cgraph, node)) {
                    continue;
                }

                bool ok = ggml_cuda_compute_forward(*cuda_ctx, node);
                if (!ok) {
                    GGML_LOG_ERROR("%s: op not supported %s (%s)\n", __func__, node->name, ggml_op_name(node->op));
                }
                GGML_ASSERT(ok);
                ggml_cuda_mem_trace(cuda_ctx, node, 1);

                if (!is_concurrent_event_active) {
                    try_launch_concurrent_event(node);
               }
            }
        }

#ifdef USE_CUDA_GRAPH
        if (use_cuda_graph && cuda_graph_update_required) { // End CUDA graph capture
            ggml_cuda_graph_launch_chunk(cuda_ctx, graph, n_chunks++);

            // drop the chunks left over from a longer graph
            for (size_t i = n_chunks; i < graph->instances.size(); ++i) {
                CUDA_CHECK(cudaGraphExecDestroy(graph->instances[i]));
            }
            graph->instances.resize(n_chunks);
            graph_evaluated_or_captured = true; // CUDA graph has been captured

            std::lock_guard<std::mutex> lock(ggml_cuda_lock);
            if (ggml_cuda_lock_counter.fetch_sub(1, std::memory_order_relaxed) == 1) {
                ggml_cuda_lock_cv.notify_all();
            }
        } else {
            graph_evaluated_or_captured = true; // ggml graph has been directly evaluated
        }
    }

    if (use_cuda_graph && !cuda_graph_update_required) {
        // the chunks were launched during capture, otherwise launch them here
        for (cudaGraphExec_t instance : graph->instances) {
            CUDA_CHECK(cudaGraphLaunch(instance, cuda_ctx->stream()));
        }
#else
        GGML_UNUSED(graph_key);
        graph_evaluated_or_captured = true;
#endif  // USE_CUDA_GRAPH
    }
}

#ifdef USE_CUDA_GRAPH
static bool ggml_cuda_graph_set_enabled(ggml_backend_cuda_context * cuda_ctx, uint64_t graph_key) {
    ggml_cuda_graph * graph = cuda_ctx->cuda_graph(graph_key);

    if (graph->instances.empty()) {
        if (ggml_cuda_info().devices[cuda_ctx->device].cc < GGML_CUDA_CC_VOLTA) {
            if (!graph->disable_due_to_gpu_arch && ggml_cuda_graph_debug_enabled()) {
                GGML_LOG_DEBUG("%s: disabling CUDA graphs due to GPU architecture\n", __func__);
            }
            graph->disable_due_to_gpu_arch = true;
        }
    }

    return graph->is_enabled();
}
#endif // USE_CUDA_GRAPH

static enum ggml_status ggml_backend_cuda_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;

    ggml_cuda_set_device(cuda_ctx->device);

    bool use_cuda_graph             = false;
    bool cuda_graph_update_required = false;
    uint64_t graph_key = 0;

#ifdef USE_CUDA_GRAPH
    graph_key = ggml_cuda_graph_get_key(cgraph);

    ggml_cuda_graph_set_enabled(cuda_ctx, graph_key);

    ggml_cuda_graph * graph = cuda_ctx->cuda_graph(graph_key);
    if (graph->is_enabled()) {
        // graph->uid is set only for a compatible graph (see ggml_cuda_graph_update_required), so a reused graph needs no new check
        const bool graph_reused     = cgraph->uid != 0 && cgraph->uid == graph->uid;
        const bool graph_compatible = graph_reused || ggml_cuda_graph_check_compability(cgraph);
        if (graph_compatible) {
            const bool properties_changed = ggml_cuda_graph_update_required(cuda_ctx, cgraph);

            if (!graph->warmup_complete) {
                // Warmup: need at least 2 calls with no property change on the 2nd call
                if (!properties_changed) {
                    graph->warmup_complete = true;
                    if (ggml_cuda_graph_debug_enabled()) {
                        GGML_LOG_DEBUG("%s: CUDA graph warmup complete\n", __func__);
                    }
                    use_cuda_graph = true;
                    cuda_graph_update_required = true;
                }
                // else: properties changed or first call - execute directly (use_cuda_graph stays false)
            } else {
                // Post-warmup: normal CUDA graph operation
                if (properties_changed) {
                    // Properties changed - reset warmup, execute directly until stable again
                    graph->warmup_complete = false;
                    if (ggml_cuda_graph_debug_enabled()) {
                        GGML_LOG_DEBUG("%s: CUDA graph warmup reset\n", __func__);
                    }
                } else {
                    use_cuda_graph = true;
                    cuda_graph_update_required = graph->instances.empty();
                }
            }
        }
    }
#endif // USE_CUDA_GRAPH

    if (use_cuda_graph && cuda_graph_update_required) {
        // Start CUDA graph capture
        {
            std::lock_guard<std::mutex> lock(ggml_cuda_lock);
            ggml_cuda_lock_counter.fetch_add(1, std::memory_order_relaxed);
        }

        CUDA_CHECK(cudaStreamBeginCapture(cuda_ctx->stream(), cudaStreamCaptureModeRelaxed));
    }

    // local: GGML_CUDA_INK2_TIME=1 splits the time of a large graph between the host (enqueueing the kernels) and what the
    // GPU still has to run once everything is queued; averages printed at exit
    static const bool ink2_time = ggml_cuda_env_flag("GGML_CUDA_INK2_TIME");
    static const int ink2_time_min = []() { const char * e = getenv("GGML_CUDA_INK2_TIME_MIN"); return e ? atoi(e) : 5000; }();
    if (ink2_time && cgraph->n_nodes >= ink2_time_min && !use_cuda_graph) {
        static double  t_enqueue = 0.0;
        static double  t_tail    = 0.0;
        static int64_t n_graphs  = 0;
        static double  last[3]   = { 0.0, 0.0, 0.0 }; // of the last large graph: enqueue, tail, host time since the previous one
        static int64_t t_prev_end = 0;
        if (n_graphs == 0) {
            atexit([]() {
                fprintf(stderr, "ink2 time: %lld large graphs, host enqueue %.0f us, GPU tail after enqueue %.0f us per graph\n",
                    (long long) n_graphs, t_enqueue/n_graphs, t_tail/n_graphs);
                fprintf(stderr, "ink2 time: last large graph: enqueue %.0f us, GPU tail %.0f us, host gap before it %.0f us\n",
                    last[0], last[1], last[2]);
            });
        }
        CUDA_CHECK(cudaStreamSynchronize(cuda_ctx->stream()));
        const int64_t t0 = ggml_time_us();
        ggml_cuda_graph_evaluate_and_capture(cuda_ctx, cgraph, use_cuda_graph, cuda_graph_update_required, graph_key);
        const int64_t t1 = ggml_time_us();
        CUDA_CHECK(cudaStreamSynchronize(cuda_ctx->stream()));
        const int64_t t2 = ggml_time_us();
        t_enqueue += (double) (t1 - t0);
        t_tail    += (double) (t2 - t1);
        last[0] = (double) (t1 - t0);
        last[1] = (double) (t2 - t1);
        last[2] = t_prev_end ? (double) (t0 - t_prev_end) : 0.0;
        t_prev_end = t2;
        n_graphs++;
        // GGML_CUDA_INK2_TIME=2: one line per large graph
        static const bool each = []() { const char * e = getenv("GGML_CUDA_INK2_TIME"); return e && atoi(e) >= 2; }();
        if (each) {
            fprintf(stderr, "ink2 time: graph %lld (%d nodes): enqueue %.0f us, GPU tail %.0f us, total %.0f us, host gap before %.0f us\n",
                (long long) n_graphs, cgraph->n_nodes, last[0], last[1], last[0] + last[1], last[2]);
        }
        return GGML_STATUS_SUCCESS;
    }

    ggml_cuda_ink2_tl(cuda_ctx, "graph", cgraph->n_nodes, use_cuda_graph ? 1 : 0);
    ggml_cuda_graph_evaluate_and_capture(cuda_ctx, cgraph, use_cuda_graph, cuda_graph_update_required, graph_key);
    ggml_cuda_ink2_tl(cuda_ctx, "graph_end", cgraph->n_nodes);
    {
        char what[48];
        snprintf(what, sizeof(what), "after graph of %d nodes", cgraph->n_nodes);
        ggml_backend_ink2_vidmem_log(what, 0);
    }

    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_cuda_event_record(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;

    ggml_cuda_ink2_tl(cuda_ctx, "ev_record");
    CUDA_CHECK(cudaEventRecord((cudaEvent_t)event->context, cuda_ctx->stream()));
}

static void ggml_backend_cuda_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;

    if (ggml_backend_is_cuda(backend)) {
        ggml_cuda_ink2_tl(cuda_ctx, "ev_wait");
        CUDA_CHECK(cudaStreamWaitEvent(cuda_ctx->stream(), (cudaEvent_t)event->context, 0));
    } else {
#if 0
        // untested
        auto wait_fn = [](void * user_data) {
            ggml_backend_event_t event = (ggml_backend_event_t)user_data;
            ggml_backend_event_synchronize(event);
        };

        CUDA_CHECK(cudaLaunchHostFunc(cuda_ctx->stream(), wait_fn, event));
#endif
        GGML_ABORT("fatal error");
    }
}

static void ggml_backend_cuda_graph_optimize(ggml_backend_t backend, ggml_cgraph * cgraph, ggml_backend_graph_optimize_params * params) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;

    const bool disable_fusion = ggml_cuda_fusion_disabled();

    auto add_alloc_deps = [&](size_t start, size_t last_node) {

        for (size_t i = start; i < last_node; ++i) {
            params->add_alloc_dep(params->user_data, cgraph->nodes[i], cgraph->nodes[last_node]);

            for (int j = 0; j < GGML_MAX_SRC; ++j) {
                if (cgraph->nodes[i]->src[j]) {
                    params->add_alloc_dep(params->user_data, cgraph->nodes[i]->src[j], cgraph->nodes[last_node]);
                }
            }
        }
    };

    if (!disable_fusion && ggml_cuda_ink2_fusion_enabled() && ggml_cuda_ink2_q8_group_enabled() &&
            ggml_cuda_ink2_arch_ok(cuda_ctx->device)) {
        // local: the Q8_0 products of a few columns that read the same activations (the Q, K, V, gate and r projections of
        // an Inkling 2 layer) are moved next to the first one, so that graph_compute runs them in one launch. A product only
        // needs its activations (computed before the first one) and its weights, so moving it earlier keeps a valid order.
        const auto weight = [](const ggml_tensor * t) {
            return t->buffer && !t->view_src && ggml_backend_buffer_get_usage(t->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS;
        };
        int n_groups = 0;
        // local: products of other activations of the same shape join a group too, when those activations are computed before
        // the group's first product (nothing between them computes them; e.g. the K, V and Q-side products of a layer whose
        // three inputs have their own norms). GGML_CUDA_INK2_Q8_GROUP_MULTI=0: the same activations only.
        const bool multi = ggml_cuda_ink2_q8_group_multi_enabled();
        for (int i = 0; i < cgraph->n_nodes; ++i) {
            ggml_tensor * a = cgraph->nodes[i];
            if (a->op != GGML_OP_MUL_MAT || !ggml_cuda_ink2_mmv_q8_f32_supported(a->src[0], a->src[1], a) || !weight(a->src[0])) {
                continue;
            }
            int k = 1;
            for (int j = i + 1; j < cgraph->n_nodes && j < i + 512 && k < GGML_CUDA_INK2_Q8_GROUP_MAX; ++j) {
                ggml_tensor * b = cgraph->nodes[j];
                if (b->src[1] == a->src[1] && ggml_cuda_ink2_mmv_q8_f32_group_ok(a, b) && weight(b->src[0])) {
                    std::rotate(cgraph->nodes + i + k, cgraph->nodes + j, cgraph->nodes + j + 1);
                    k++;
                } else if (multi && b->src[1] != a->src[1] && b->op == GGML_OP_MUL_MAT &&
                        ggml_cuda_ink2_mmv_q8_f32_group_ok(a, b, false) && weight(b->src[0])) {
                    const ggml_tensor * y  = b->src[1];
                    const ggml_tensor * yr = y->view_src ? y->view_src : y;
                    bool ready = true;
                    for (int q = i; q < j && ready; ++q) {
                        // neither computed nor written in place (an op whose result is a view of them) in between
                        const ggml_tensor * nq = cgraph->nodes[q];
                        ready = nq != yr && (nq != y || ggml_cuda_is_view_or_noop(y)) && !(nq->view_src == yr && !ggml_cuda_is_view_or_noop(nq));
                    }
                    if (ready) {
                        std::rotate(cgraph->nodes + i + k, cgraph->nodes + j, cgraph->nodes + j + 1);
                        k++;
                    }
                }
            }
            if (k > 1 && ggml_cuda_ink2_mmv_q8_f32_group_ok(a, cgraph->nodes[i + 1], !multi)) {
                // the activations stay alive until the last output is allocated
                for (int q = i; q < i + k - 1; ++q) {
                    params->add_alloc_dep(params->user_data, a->src[1], cgraph->nodes[i + k - 1]);
                    params->add_alloc_dep(params->user_data, cgraph->nodes[q], cgraph->nodes[i + k - 1]);
                    params->add_alloc_dep(params->user_data, cgraph->nodes[q + 1]->src[1], cgraph->nodes[i + k - 1]);
                }
                n_groups++;
                i += k - 1;
            }
        }
        if (ggml_cuda_env_flag("GGML_CUDA_INK2_DEBUG")) {
            GGML_LOG_INFO("%s: %d groups of Q8_0 products in %d nodes\n", __func__, n_groups, cgraph->n_nodes);
        }
    }

    if (!disable_fusion && ggml_cuda_ink2_fusion_enabled() && ggml_cuda_ink2_arch_ok(cuda_ctx->device)) {
        // local: keep the inputs of the fused Inkling 2 patterns alive until their output is allocated
        static const bool debug = ggml_cuda_env_flag("GGML_CUDA_INK2_DEBUG");
        int n_match[3] = { 0, 0, 0 }; // clip, gate, mix
        int n_dump = 0;
        for (int i = 0; i < cgraph->n_nodes; ++i) {
            if (cgraph->nodes[i]->op == GGML_OP_MUL_MAT_ID) {
                // expert product + SITU in the FP16 expert kernel: it reads only the activations and the ids (and weights), and
                // writes only the activation; the intermediates are never written, so only those two stay alive until the
                // output is allocated. A batch the kernel does not take keeps the plain SITU pattern (matched at its SCALE).
                ggml_cuda_ink2_moe_situ_match ms;
                const int n = ggml_cuda_ink2_match_moe_situ(cgraph, i, ms);
                if (n > 0 && ggml_cuda_ink2_moe_f16_supported(cuda_ctx->device, ms.mm->src[0], ms.mm->src[1], ms.mm->src[2],
                            cgraph->nodes[i + n - 1], ms.x_scale, true)) {
                    params->add_alloc_dep(params->user_data, ms.mm->src[1], cgraph->nodes[i + n - 1]);
                    params->add_alloc_dep(params->user_data, ms.mm->src[2], cgraph->nodes[i + n - 1]);
                    // with the down projection in the same call (moe_ffn), they are read until its output is written
                    ggml_cuda_ink2_moe_down_match md;
                    const int nd = ggml_cuda_ink2_match_moe_down(cgraph, i + n - 1, ms.mm->src[2], md);
                    if (nd > 0) {
                        params->add_alloc_dep(params->user_data, ms.mm->src[1], cgraph->nodes[i + n - 1 + nd]);
                        params->add_alloc_dep(params->user_data, ms.mm->src[2], cgraph->nodes[i + n - 1 + nd]);
                    }
                    i += n - 1;
                    continue;
                }
                // the product with its per-expert scale (RESHAPE, REPEAT, GET_ROWS, MUL), fused at compute time by the generic
                // matcher (down projections): its kernel reads the activations while it writes the scaled output, so they stay
                // alive until that output is allocated (otherwise the fusion is refused whenever the allocator reuses them)
                ggml_tensor ** nd = cgraph->nodes + i;
                if (i + 4 < cgraph->n_nodes && nd[1]->op == GGML_OP_RESHAPE && nd[2]->op == GGML_OP_REPEAT &&
                        nd[3]->op == GGML_OP_GET_ROWS && nd[4]->op == GGML_OP_MUL &&
                        ggml_cuda_get_mul_mat_id_scale(nd[1], nd[2], nd[3], nd[4], nd[0])) {
                    params->add_alloc_dep(params->user_data, nd[0]->src[1], nd[4]);
                    params->add_alloc_dep(params->user_data, nd[0]->src[2], nd[4]);
                    i += 4;
                }
                continue;
            }
            const int n = ggml_cuda_ink2_match_any(cgraph, i);
            if (n > 0) {
                n_match[cgraph->nodes[i]->op == GGML_OP_SQR ? 0 : (n == 12 && cgraph->nodes[i + 2]->op == GGML_OP_RESHAPE ? 1 : 2)]++;
                add_alloc_deps(i, i + n - 1);
                i += n - 1;
            } else if (debug && cgraph->nodes[i]->op == GGML_OP_CONCAT && n_dump < 6) {
                // show what follows an unmatched CONCAT
                std::string s;
                for (int k = i; k < std::min(i + 14, cgraph->n_nodes); ++k) {
                    s += ggml_op_name(cgraph->nodes[k]->op);
                    s += ' ';
                }
                GGML_LOG_INFO("%s: unmatched CONCAT at %d (dim %d): %s\n", __func__, i, ggml_get_op_params_i32(cgraph->nodes[i], 0), s.c_str());
                n_dump++;
            }
        }
        if (debug) {
            GGML_LOG_INFO("%s: Inkling 2 fusions in %d nodes: clip %d, gate %d, mix %d\n", __func__, cgraph->n_nodes, n_match[0], n_match[1], n_match[2]);
            // GGML_CUDA_INK2_RAW=start,count: every node of that range, views included
            if (const char * e = getenv("GGML_CUDA_INK2_RAW")) {
                int start = 0;
                int count = 0;
                if (sscanf(e, "%d,%d", &start, &count) == 2) {
                    std::string s;
                    for (int i = std::max(0, start); i < std::min(start + count, cgraph->n_nodes); ++i) {
                        const ggml_tensor * node = cgraph->nodes[i];
                        s += node->op == GGML_OP_UNARY ? ggml_unary_op_name(ggml_get_unary_op(node)) : ggml_op_name(node->op);
                        if (node->name[0] && strncmp(node->name, "node_", 5) != 0 && strncmp(node->name, " (", 2) != 0) {
                            s += std::string("(") + std::string(node->name).substr(0, 16) + ")";
                        }
                        s += ' ';
                    }
                    GGML_LOG_INFO("%s: raw %d,%d: %s\n", __func__, start, count, s.c_str());
                }
            }
            // GGML_CUDA_INK2_DUMP=start,count: the real ops left in that node range, fused patterns shown as [KIND]
            if (const char * e = getenv("GGML_CUDA_INK2_DUMP")) {
                int start = 0;
                int count = 0;
                if (sscanf(e, "%d,%d", &start, &count) == 2) {
                    std::string s;
                    for (int i = std::max(0, start); i < std::min(start + count, cgraph->n_nodes); ++i) {
                        const int n = ggml_cuda_ink2_match_any(cgraph, i);
                        if (n > 0) {
                            s += cgraph->nodes[i]->op == GGML_OP_SQR ? "[CLIP] " : (n == 12 && cgraph->nodes[i + 2]->op == GGML_OP_RESHAPE ? "[GATE] " : "[MIX] ");
                            i += n - 1;
                            continue;
                        }
                        const ggml_tensor * node = cgraph->nodes[i];
                        if (ggml_cuda_is_view_or_noop(node)) {
                            continue;
                        }
                        s += node->op == GGML_OP_UNARY ? ggml_unary_op_name(ggml_get_unary_op(node)) : ggml_op_name(node->op);
                        if (node->name[0] && strncmp(node->name, "node_", 5) != 0 && strncmp(node->name, " (", 2) != 0) {
                            s += std::string("(") + std::string(node->name).substr(0, 14) + ")";
                        }
                        s += ' ';
                    }
                    GGML_LOG_INFO("%s: dump %d,%d: %s\n", __func__, start, count, s.c_str());
                }
            }
        }
    }

    if (!disable_fusion) {
        // add alloc deps for performance positive fusions. This may increase the overall compute buffer size.
        // TODO: consolidate fusion paths in graph_optimize and graph_compute
        ggml_cuda_set_device(cuda_ctx->device);
        for (int i = 0; i + 5 < cgraph->n_nodes; ++i) {
            if (cgraph->nodes[i]->op != GGML_OP_MUL_MAT_ID) {
                continue;
            }
            for (int j = i + 3; j + 2 < cgraph->n_nodes; ++j) {
                if (cgraph->nodes[j]->op == GGML_OP_MUL_MAT_ID && cgraph->nodes[j + 1]->op == GGML_OP_MUL_MAT_ID) {
                    break;
                }
                if (cgraph->nodes[j]->op != GGML_OP_MUL_MAT || !ggml_cuda_match_shared_expert(cgraph, i, j)) {
                    continue;
                }
                // Group both outputs before allocation so the shared result cannot alias intervening nodes.
                std::rotate(cgraph->nodes + i + 3, cgraph->nodes + j, cgraph->nodes + j + 3);
                ggml_tensor * up = cgraph->nodes[i + 2]->src[1];
                params->add_alloc_dep(params->user_data, up->src[1], cgraph->nodes[i + 5]);
                params->add_alloc_dep(params->user_data, up->src[2], cgraph->nodes[i + 5]);
                i += 5;
                break;
            }
        }
        for (int i = 0; i < cgraph->n_nodes; ++i) {
            // Keep the inputs of the fused gate/up/GLU MMQ kernel alive until the GLU so that they cannot alias its output (under op offload the weights are buffer copies).
            if (i + 2 < cgraph->n_nodes && cgraph->nodes[i + 2]->op == GGML_OP_GLU) {
                ggml_tensor * glu  = cgraph->nodes[i + 2];
                ggml_tensor * gate = glu->src[0];
                ggml_tensor * up   = glu->src[1];
                const bool ok = (gate == cgraph->nodes[i] && up == cgraph->nodes[i + 1]) || (gate == cgraph->nodes[i + 1] && up == cgraph->nodes[i]);
                if (ok && ggml_cuda_mul_mat_q_fusion_matches(up, gate, glu, cuda_ctx->device)) {
                    add_alloc_deps(i, i + 2);
                    i += 2;
                    continue;
                }
            }
            // Same with the NVFP4 scales: MUL_MAT(gate), MUL(gate scale), MUL_MAT(up), MUL(up scale), GLU.
            if (i + 4 < cgraph->n_nodes && cgraph->nodes[i + 4]->op == GGML_OP_GLU) {
                ggml_tensor * glu        = cgraph->nodes[i + 4];
                ggml_tensor * gate_scale = cgraph->nodes[i + 1];
                ggml_tensor * up_scale   = cgraph->nodes[i + 3];
                if (glu->src[0] == gate_scale && glu->src[1] == up_scale &&
                        ggml_cuda_mul_mat_q_fusion_matches(cgraph->nodes[i + 2], cgraph->nodes[i], glu, cuda_ctx->device, up_scale, gate_scale) &&
                        ggml_cuda_get_mul_mat_scale(gate_scale, cgraph->nodes[i]) && ggml_cuda_get_mul_mat_scale(up_scale, cgraph->nodes[i + 2])) {
                    add_alloc_deps(i, i + 4);
                    i += 4;
                    continue;
                }
            }
            // Same with per-expert NVFP4 scales: MUL_MAT_ID, RESHAPE, REPEAT, GET_ROWS, MUL for gate and for up, GLU.
            if (i + 10 < cgraph->n_nodes && cgraph->nodes[i + 10]->op == GGML_OP_GLU) {
                ggml_tensor * glu        = cgraph->nodes[i + 10];
                ggml_tensor * gate_scale = cgraph->nodes[i + 4];
                ggml_tensor * up_scale   = cgraph->nodes[i + 9];
                if (glu->src[0] == gate_scale && glu->src[1] == up_scale &&
                        ggml_cuda_mul_mat_q_fusion_matches(cgraph->nodes[i + 5], cgraph->nodes[i], glu, cuda_ctx->device, up_scale, gate_scale) &&
                        ggml_cuda_get_mul_mat_id_scale(cgraph->nodes[i + 1], cgraph->nodes[i + 2], cgraph->nodes[i + 3], gate_scale, cgraph->nodes[i]) &&
                        ggml_cuda_get_mul_mat_id_scale(cgraph->nodes[i + 6], cgraph->nodes[i + 7], cgraph->nodes[i + 8], up_scale, cgraph->nodes[i + 5])) {
                    add_alloc_deps(i, i + 10);
                    i += 10;
                    continue;
                }
            }

            ggml_cuda_moe_weighted_reduction_match match;
            if (ggml_cuda_match_moe_weighted_reduction(cgraph, i, match)) {
                params->add_alloc_dep(params->user_data, const_cast<ggml_tensor *>(match.experts), match.dst);
                params->add_alloc_dep(params->user_data, const_cast<ggml_tensor *>(match.weights), match.dst);
                if (match.expert_scale != nullptr) {
                    params->add_alloc_dep(
                        params->user_data, const_cast<ggml_tensor *>(match.expert_scale), match.dst);
                }
                i += match.node_count - 1;
            }

            if (cgraph->nodes[i]->op == GGML_OP_UNARY || cgraph->nodes[i]->op == GGML_OP_SOFT_MAX ||
                    cgraph->nodes[i]->op == GGML_OP_ARGSORT) {
                ggml_cuda_topk_moe_args args;
                const bool              can_fuse = ggml_cuda_topk_moe_fusion(cgraph, i, args);
                std::vector<ggml_op>    ops;
                ops.reserve(13);  // max ops; avoids gcc -Wstringop-overflow false positive

                const ggml_tensor * node = cgraph->nodes[i];

                if (can_fuse) {
                    const ggml_tensor * logits  = node->src[0];
                    ggml_tensor *       weights = nullptr;
                    ggml_tensor *       ids     = nullptr;

                    if (!args.delayed_softmax) {
                        int out_nodes[2];  // nodes which can't be elided

                        if (args.sigmoid) {
                            ops.insert(ops.end(), { GGML_OP_UNARY });
                        } else if (args.sqrt_softplus) {
                            ops.insert(ops.end(), { GGML_OP_UNARY, GGML_OP_SQRT });
                        } else {
                            ops.insert(ops.end(), { GGML_OP_SOFT_MAX });
                        }
                        const int i_probs = i + (int) ops.size() - 1;  // last node of the gating activation

                        if (args.prob_bias) {
                            ops.insert(ops.end(), { GGML_OP_RESHAPE, GGML_OP_ADD, GGML_OP_ARGSORT, GGML_OP_VIEW,
                                                    GGML_OP_GET_ROWS });
                            out_nodes[0] = i_probs + 4;
                        } else {
                            ops.insert(ops.end(), { GGML_OP_RESHAPE, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS });
                            out_nodes[0] = i_probs + 3;
                        }
                        ids = cgraph->nodes[out_nodes[0]];

                        if (args.norm) {
                            ops.insert(ops.end(),
                                       { GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_CLAMP, GGML_OP_DIV, GGML_OP_RESHAPE });
                        }
                        if (args.scale) {
                            ops.insert(ops.end(), { GGML_OP_SCALE });
                        }

                        weights      = cgraph->nodes[i + ops.size() - 1];
                        out_nodes[1] = i + ops.size() - 1;

                        if (ggml_can_fuse_subgraph(cgraph, i, ops.size(), ops.data(), out_nodes, 2) &&
                                ggml_cuda_should_use_topk_moe(node, logits, weights, ids)) {

                            add_alloc_deps(i, i + ops.size());
                            // local (Inkling 2): with the weight normalisation that follows (one kernel), the buffers of the
                            // whole run stay alive until its output is allocated
                            ggml_cuda_ink2_wnorm_match wm;
                            const int n_norm = !args.norm && ggml_cuda_ink2_arch_ok(cuda_ctx->device) ?
                                ggml_cuda_ink2_match_router_norm(cgraph, i + (int) ops.size(), weights, wm) : 0;
                            if (n_norm > 0) {
                                add_alloc_deps(i, i + ops.size() + n_norm - 1);
                                i += n_norm;
                            }
                            i += ops.size() - 1;
                        }
                    }
                }
            }
        }
    }

#ifdef USE_CUDA_GRAPH
    const uint64_t graph_key = ggml_cuda_graph_get_key(cgraph);
    const bool use_cuda_graph = ggml_cuda_graph_set_enabled(cuda_ctx, graph_key);
#else
    const bool use_cuda_graph = false;
    GGML_UNUSED(cuda_ctx);
    GGML_UNUSED(cgraph);
#endif

    static bool enable_graph_optimization = [] {
        const char * env     = getenv("GGML_CUDA_GRAPH_OPT");
        return env != nullptr && atoi(env) == 1;
    }();

    if (!enable_graph_optimization) {
        return;
    }

    ggml_cuda_stream_context & stream_context = cuda_ctx->stream_context();
    stream_context.reset();

    if (!use_cuda_graph) {
        return;
    }

    ggml_cuda_set_device(cuda_ctx->device);

    // number of out-degrees for a particular node
    std::unordered_map<const ggml_tensor *, int> fan_out;
    // reverse mapping of node to index in the cgraph
    std::unordered_map<const ggml_tensor *, int> node_indices;

    const auto & is_noop = [](const ggml_tensor * node) -> bool {
        return ggml_is_empty(node) || node->op == GGML_OP_NONE || node->op == GGML_OP_RESHAPE ||
               node->op == GGML_OP_TRANSPOSE || node->op == GGML_OP_VIEW || node->op == GGML_OP_PERMUTE;
    };

    const auto & depends_on = [](const ggml_tensor * dst, const ggml_tensor * src) -> bool {
        for (uint32_t s = 0; s < GGML_MAX_SRC; ++s) {
            if (dst->src[s] == src) {
                return true;
            }
        }
        // implicit dependency if they view the same tensor
        const ggml_tensor * dst2 = dst->view_src ? dst->view_src : dst;
        const ggml_tensor * src2 = src->view_src ? src->view_src : src;
        if (dst2 == src2) {
            return true;
        }
        return false;
    };

    for (int node_idx = 0; node_idx < cgraph->n_nodes; node_idx++) {
        const ggml_tensor * node = cgraph->nodes[node_idx];
        node_indices[node]       = node_idx;

        if (is_noop(node)) {
            continue;
        }
        for (int src_idx = 0; src_idx < GGML_MAX_SRC; ++src_idx) {
            const ggml_tensor * src = cgraph->nodes[node_idx]->src[src_idx];
            //TODO: check why nrows > 1 fails
            if (node && !is_noop(node) && ggml_nrows(node) <= 1) {
                fan_out[src] += 1;
            }
        }
    }

    // Target Q, K, V for concurrency
    // this is a more general way to find nodes which can be candidates for concurrency (although it has not been tested for anything else):
    // 1. find fan-out (fork) nodes where the same input is used at least N times (in QKV, it would be "attn-norm")
    // 2. find the join node, where 2 or more of the outputs are required (in QKV, this would "KQ" or "flash-attn")
    // 3. account for all branches from the fork to the join
    // 4. To extend lifetimes of the tensors, we interleave the branches (see below for more details)
    // 5. save the original cgraph and restore it in graph_compute, to enable fusion within streams
    // See discussion: https://github.com/ggml-org/llama.cpp/pull/16991#issuecomment-3522620030

    const int min_fan_out = 3;
    const int max_fan_out = 3;

    // store {fork_idx, join_idx}
    std::vector<std::pair<int, int>> concurrent_node_ranges;

    for (const auto & [root_node, count] : fan_out) {
        if (count >= min_fan_out && count <= max_fan_out) {
            const int root_node_idx = node_indices[root_node];

            // only optimize for attn_norm
            // TODO: make this more generic
            if (!strstr(root_node->name, "attn_norm")) {
                continue;
            }

            bool is_part_of_event = false;
            for (const auto & [start, end] : concurrent_node_ranges) {
                if (root_node_idx >= start && root_node_idx <= end) {
                    is_part_of_event = true;
                }
            }

            if (is_part_of_event) {
                continue;
            }

            std::vector<std::vector<const ggml_tensor *>> nodes_per_branch;
            for (int i = root_node_idx + 1; i < cgraph->n_nodes; ++i) {
                const ggml_tensor * node = cgraph->nodes[i];
                if (!is_noop(node) && depends_on(node, root_node)) {
                    nodes_per_branch.push_back({ node });
                }
            }

            GGML_ASSERT(nodes_per_branch.size() == (size_t) count);

            //find the join point
            const ggml_tensor * join_node = nullptr;

            const auto & belongs_to_branch = [&](const ggml_tensor *                      node,
                                                 const std::vector<const ggml_tensor *> & branch) -> bool {
                for (const ggml_tensor * n : branch) {
                    if (depends_on(node, n)) {
                        return true;
                    }
                }
                return false;
            };

            for (int i = root_node_idx + 1; i < cgraph->n_nodes; ++i) {
                const ggml_tensor * curr_node = cgraph->nodes[i];

                int num_joins = 0;
                for (size_t branch_idx = 0; branch_idx < nodes_per_branch.size(); branch_idx++) {
                    if (belongs_to_branch(curr_node, nodes_per_branch[branch_idx])) {
                        num_joins++;
                    }
                }

                if (num_joins >= 2) {
                    join_node = curr_node;
                    break;
                }

                bool found_branch = false;
                for (size_t branch_idx = 0; branch_idx < nodes_per_branch.size(); branch_idx++) {
                    std::vector<const ggml_tensor *> & branch_vec = nodes_per_branch[branch_idx];
                    if (belongs_to_branch(curr_node, branch_vec)) {
                        //continue accumulating
                        if (std::find(branch_vec.begin(), branch_vec.end(), curr_node) == branch_vec.end()) {
                            branch_vec.push_back(curr_node);
                        }
                        found_branch = true;
                    }
                }

                if (!found_branch && is_noop(curr_node)) {
                    // we can put it in any branch because it will be ignored
                    nodes_per_branch[0].push_back({ curr_node });
                }
            }

            if (join_node) {
                //Create ggml_cuda_concurrent_event
                ggml_cuda_concurrent_event concurrent_event(nodes_per_branch.size());
                concurrent_event.join_node = join_node;

                for (size_t branch_idx = 0; branch_idx < nodes_per_branch.size(); branch_idx++) {
                    for (const ggml_tensor * n : nodes_per_branch[branch_idx]) {
                        concurrent_event.stream_mapping[n] = branch_idx + 1;
                    }
                }

                int fork_node_idx = node_indices[root_node];
                int join_node_idx = node_indices[join_node];

                int       current_branch_idx = 0;
                int       current_node_idx   = fork_node_idx + 1;
                const int n_branches         = nodes_per_branch.size();

                int total_branch_nodes = 0;
                for (std::vector<const ggml_tensor *> branch_nodes : nodes_per_branch) {
                    total_branch_nodes += branch_nodes.size();
                }

                // there are other nodes in the middle which are unaccounted for
                // usually (cpy) nodes, then ignore this fork
                if (join_node_idx - fork_node_idx - 1 != total_branch_nodes) {
                    GGML_LOG_DEBUG(
                        "Skipping %s because the number of nodes in the middle is not equal to the total number of "
                        "branch nodes %d != %d\n",
                        root_node->name, join_node_idx - fork_node_idx - 1, total_branch_nodes);
                    continue;
                }

                // Save the original order of nodes in this region before interleaving
                // This is used later to restore grouping for fusion within streams
                concurrent_event.original_order.reserve(total_branch_nodes);
                for (int i = fork_node_idx + 1; i < join_node_idx; ++i) {
                    concurrent_event.original_order.push_back(cgraph->nodes[i]);
                }

                std::unordered_map<const ggml_tensor *, ggml_cuda_concurrent_event> & concurrent_events = cuda_ctx->stream_context().concurrent_events;
                GGML_ASSERT(concurrent_events.find(root_node) == concurrent_events.end());
                concurrent_events.emplace(root_node, std::move(concurrent_event));
                GGML_LOG_DEBUG("Adding stream at node %s %p\n", root_node->name, root_node);
                concurrent_node_ranges.emplace_back(fork_node_idx, join_node_idx);

                // interleave tensors to extend lifetimes so that ggml graph doesn't recycle them
                // example transformation:
                // [attn-norm, QMul, QNorm, QRope, KMul, KNorm, KRope, VMul, attn] ->
                // [attn-norm, QMul, KMul, VMul, QNorm, VNorm, QRope, KRope, attn]
                while (current_node_idx < join_node_idx) {
                    std::vector<const ggml_tensor *> & branch_nodes = nodes_per_branch[current_branch_idx];

                    bool has_node = false;
                    for (std::vector<const ggml_tensor *> branch_node : nodes_per_branch) {
                        has_node |= branch_node.size() > 0;
                    }

                    GGML_ASSERT(has_node);

                    if (branch_nodes.empty()) {
                        current_branch_idx = (current_branch_idx + 1) % n_branches;
                        continue;
                    }

                    cgraph->nodes[current_node_idx] = const_cast<ggml_tensor *>(branch_nodes.front());
                    current_node_idx++;
                    branch_nodes.erase(branch_nodes.begin());

                    // append all empty nodes
                    while (!branch_nodes.empty() && is_noop(branch_nodes.front())) {
                        cgraph->nodes[current_node_idx] = const_cast<ggml_tensor *>(branch_nodes.front());
                        current_node_idx++;
                        branch_nodes.erase(branch_nodes.begin());
                    }

                    current_branch_idx = (current_branch_idx + 1) % n_branches;
                }
            }
        }
    }
}

static const ggml_backend_i ggml_backend_cuda_interface = {
    /* .get_name                = */ ggml_backend_cuda_get_name,
    /* .free                    = */ ggml_backend_cuda_free,
    /* .set_tensor_async        = */ ggml_backend_cuda_set_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_cuda_get_tensor_async,
    /* .set_tensor_2d_async     = */ ggml_backend_cuda_set_tensor_2d_async,
    /* .get_tensor_2d_async     = */ ggml_backend_cuda_get_tensor_2d_async,
    /* .cpy_tensor_async        = */ ggml_backend_cuda_cpy_tensor_async,
    /* .synchronize             = */ ggml_backend_cuda_synchronize,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_cuda_graph_compute,
    /* .event_record            = */ ggml_backend_cuda_event_record,
    /* .event_wait              = */ ggml_backend_cuda_event_wait,
    /* .graph_optimize          = */ ggml_backend_cuda_graph_optimize,
};

static ggml_guid_t ggml_backend_cuda_guid() {
    static ggml_guid guid = { 0x2c, 0xdd, 0xe8, 0x1c, 0x65, 0xb3, 0x65, 0x73, 0x6a, 0x12, 0x88, 0x61, 0x1c, 0xc9, 0xdc, 0x25 };
    return &guid;
}

bool ggml_backend_is_cuda(ggml_backend_t backend) {
    return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_cuda_guid());
}

int ggml_backend_cuda_get_device_count() {
    return ggml_cuda_info().device_count;
}

static std::string ggml_cuda_device_description(int device) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, ggml_cuda_get_physical_device(device)));

    const ggml_cuda_device_info & info = ggml_cuda_info();
    std::string description = prop.name;
    if (info.device_count > info.physical_device_count) {
        description += " (dev p" + std::to_string(info.devices[device].physical_device) +
                       "/v" + std::to_string(info.devices[device].virtual_index) + ")";
    }
    return description;
}

void ggml_backend_cuda_get_device_description(int device, char * description, size_t description_size) {
    snprintf(description, description_size, "%s", ggml_cuda_device_description(device).c_str());
}

static int ggml_cuda_physical_device_share_count(int device) {
    const ggml_cuda_device_info & info = ggml_cuda_info();
    GGML_ASSERT(device >= 0 && device < info.device_count);
    return info.devices[device].physical_share_count;
}

void ggml_backend_cuda_get_device_memory(int device, size_t * free, size_t * total) {
    ggml_cuda_set_device(device);

    CUDA_CHECK(cudaMemGetInfo(free, total));

    // virtual devices sharing one physical GPU share its memory pool; split it between them
    const int share_count = ggml_cuda_physical_device_share_count(device);
    *free  /= share_count;
    *total /= share_count;
}

bool ggml_backend_cuda_register_host_buffer(void * buffer, size_t size) {
    if (!ggml_cuda_register_host_enabled()) {
        return false;
    }

#if CUDART_VERSION >= 11010 || defined(GGML_USE_MUSA) || defined(GGML_USE_HIP)
    cudaError_t err = cudaHostRegister(buffer, size, cudaHostRegisterPortable | cudaHostRegisterReadOnly);
    if (err != cudaSuccess) {
        // clear the error
        (void)cudaGetLastError();

        GGML_LOG_DEBUG("%s: failed to register %.2f MiB of pinned memory: %s\n", __func__,
                           size / 1024.0 / 1024.0, cudaGetErrorString(err));
        return false;
    }
    return true;
#else
    GGML_UNUSED(buffer);
    GGML_UNUSED(size);
    return false;
#endif // CUDART_VERSION >= 11010 || defined(GGML_USE_MUSA)
}

void ggml_backend_cuda_unregister_host_buffer(void * buffer) {
    if (!ggml_cuda_register_host_enabled()) {
        return;
    }

    cudaError_t err = cudaHostUnregister(buffer);
    if (err != cudaSuccess) {
        // clear the error
        (void)cudaGetLastError();
    }
}


// backend device

struct ggml_backend_cuda_device_context {
    int device;
    std::string name;
    std::string description;
    std::string pci_bus_id;
    int op_offload_min_batch_size;
};

static const char * ggml_backend_cuda_device_get_name(ggml_backend_dev_t dev) {
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *)dev->context;
    return ctx->name.c_str();
}

static const char * ggml_backend_cuda_device_get_description(ggml_backend_dev_t dev) {
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *)dev->context;
    return ctx->description.c_str();
}

#if defined(__linux__)
// Helper function to get available memory from /proc/meminfo for UMA systems
static bool ggml_backend_cuda_get_available_uma_memory(long * available_memory_kb, long * free_swap_kb) {
    FILE * meminfo_file = nullptr;
    // 2KB buffer for reading /proc/meminfo since it does not report size info, should be enough
    const size_t BUFFER_SIZE = 2048;
    auto file_buffer = std::make_unique<char[]>(BUFFER_SIZE);
    size_t bytes_read = 0;
    long huge_tlb_total_pages = -1;
    long huge_tlb_free_pages = -1;
    long huge_tlb_page_size = -1;

    if (available_memory_kb == nullptr || free_swap_kb == nullptr) {
        return false;
    }

    meminfo_file = fopen("/proc/meminfo", "r");
    if (meminfo_file == nullptr) {
        GGML_LOG_ERROR("%s: failed to open /proc/meminfo\n", __func__);
        return false;
    }

    // Read file into buffer
    bytes_read = fread(file_buffer.get(), 1, BUFFER_SIZE - 1, meminfo_file);
    fclose(meminfo_file);

    if (bytes_read == 0) {
        GGML_LOG_ERROR("%s: failed to read from /proc/meminfo\n", __func__);
        return false;
    }
    file_buffer[bytes_read] = '\0';

    *available_memory_kb = -1;
    *free_swap_kb = -1;

    // Parse the file buffer line by line
    char * line = file_buffer.get();
    char * line_next;
    while (line < file_buffer.get() + bytes_read) {
        // Find the end of the current line
        line_next = strchr(line, '\n');
        if (line_next != nullptr) {
            *line_next = '\0';
            line_next++;
        } else {
            line_next = file_buffer.get() + bytes_read;
        }

        long value;
        if (sscanf(line, "MemAvailable: %ld kB", &value) == 1) {
            *available_memory_kb = value;
        } else if (sscanf(line, "SwapFree: %ld kB", &value) == 1) {
            *free_swap_kb = value;
        } else if (sscanf(line, "HugePages_Total: %ld", &value) == 1) {
            huge_tlb_total_pages = value;
        } else if (sscanf(line, "HugePages_Free: %ld", &value) == 1) {
            huge_tlb_free_pages = value;
        } else if (sscanf(line, "Hugepagesize: %ld kB", &value) == 1) {
            huge_tlb_page_size = value;
        }

        line = line_next;
    }

    if (huge_tlb_total_pages != 0 && huge_tlb_total_pages != -1) {
        *available_memory_kb = huge_tlb_free_pages * huge_tlb_page_size;

        // Hugetlbfs pages are not swappable.
        *free_swap_kb = 0;
    }

    GGML_LOG_DEBUG("%s: final available_memory_kb: %ld\n", __func__, *available_memory_kb);
    return true;
}
#endif // defined(__linux__)

static void ggml_backend_cuda_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *)dev->context;
    ggml_cuda_set_device(ctx->device);
    cudaError_t err = cudaMemGetInfo(free, total);
    if (err != cudaSuccess) {
        (void)cudaGetLastError();
        GGML_LOG_WARN("%s: cudaMemGetInfo failed (%s), returning 0/0\n", __func__, cudaGetErrorString(err));
        *free = 0;
        *total = 0;
        return;
    }

// ref: https://github.com/ggml-org/llama.cpp/pull/17368
#if defined(__linux__) && !defined(GGML_USE_HIP)
    // Check if this is a UMA (Unified Memory Architecture) system
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, ggml_cuda_get_physical_device(ctx->device)));

    // Check if UMA is explicitly enabled via environment variable
    bool is_uma = prop.integrated > 0 || ggml_cuda_unified_memory_enabled();

    if (is_uma) {
        // For UMA systems (like DGX Spark), use system memory info
        long available_memory_kb = 0;
        long free_swap_kb = 0;

        if (ggml_backend_cuda_get_available_uma_memory(&available_memory_kb, &free_swap_kb) && available_memory_kb > 0) {
            *free = (size_t)available_memory_kb * 1024;
        } else {
            GGML_LOG_ERROR("%s: /proc/meminfo reading failed, using cudaMemGetInfo\n", __func__);
        }
    }
#endif // defined(__linux__) && !defined(GGML_USE_HIP)

    // virtual devices sharing one physical GPU share its memory pool; split it between them
    const int share_count = ggml_cuda_physical_device_share_count(ctx->device);
    *free  /= share_count;
    *total /= share_count;
}

static enum ggml_backend_dev_type ggml_backend_cuda_device_get_type(ggml_backend_dev_t dev) {
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *) dev->context;

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, ggml_cuda_get_physical_device(ctx->device)));

    return prop.integrated
        ? GGML_BACKEND_DEVICE_TYPE_IGPU
        : GGML_BACKEND_DEVICE_TYPE_GPU;
}

static bool ggml_backend_cuda_host_buffer_supported() {
    return !ggml_cuda_pinned_memory_disabled();
}

static bool ggml_backend_cuda_device_supports_cuda_host_buft(int device) {
#if defined(GGML_USE_HIP)
    if (ggml_cuda_info().devices[device].integrated) {
        return false;
    }
#else
    GGML_UNUSED(device);
#endif

    return ggml_backend_cuda_host_buffer_supported();
}

static void ggml_backend_cuda_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *)dev->context;

    props->name        = ggml_backend_cuda_device_get_name(dev);
    props->description = ggml_backend_cuda_device_get_description(dev);
    props->type        = ggml_backend_cuda_device_get_type(dev);
    props->device_id   = ctx->pci_bus_id.empty() ? nullptr : ctx->pci_bus_id.c_str();
    ggml_backend_cuda_device_get_memory(dev, &props->memory_free, &props->memory_total);

    const bool host_buffer = !ggml_cuda_pinned_memory_disabled();
#ifdef GGML_CUDA_NO_PEER_COPY
    bool events = false;
#else
    bool events = true;
#endif

    props->caps = {
        /* .async                 = */ true,
        /* .host_buffer           = */ host_buffer,
        /* .buffer_from_host_ptr  = */ false,
        /* .events                = */ events,
        /* .mmap_support          = */ props->type != GGML_BACKEND_DEVICE_TYPE_IGPU,
    };
}

static ggml_backend_t ggml_backend_cuda_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    GGML_UNUSED(params);
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *)dev->context;
    return ggml_backend_cuda_init(ctx->device);
}

static ggml_backend_buffer_type_t ggml_backend_cuda_device_get_buffer_type(ggml_backend_dev_t dev) {
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *)dev->context;
    return ggml_backend_cuda_buffer_type(ctx->device);
}

static ggml_backend_buffer_type_t ggml_backend_cuda_device_get_host_buffer_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    if (!ggml_backend_cuda_host_buffer_supported()) {
        return nullptr;
    }

    return ggml_backend_cuda_host_buffer_type();
}

// TODO: move these functions here
static bool ggml_backend_cuda_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    ggml_backend_cuda_device_context * dev_ctx = (ggml_backend_cuda_device_context *) dev->context;

    // check if all the sources are allocated on this device
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (op->src[i] && op->src[i]->buffer && ggml_backend_buft_is_cuda(op->src[i]->buffer->buft)) {
            ggml_backend_cuda_buffer_type_context * buft_ctx = (ggml_backend_cuda_buffer_type_context *)op->src[i]->buffer->buft->context;
            if (buft_ctx->device != dev_ctx->device) {
                return false;
            }
        }
    }

    switch (op->op) {
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(op)) {
                case GGML_UNARY_OP_ABS:
                case GGML_UNARY_OP_SGN:
                case GGML_UNARY_OP_NEG:
                case GGML_UNARY_OP_STEP:
                case GGML_UNARY_OP_GELU:
                case GGML_UNARY_OP_SILU:
                case GGML_UNARY_OP_RELU:
                case GGML_UNARY_OP_SIGMOID:
                case GGML_UNARY_OP_HARDSIGMOID:
                case GGML_UNARY_OP_HARDSWISH:
                case GGML_UNARY_OP_GELU_ERF:
                case GGML_UNARY_OP_GELU_QUICK:
                case GGML_UNARY_OP_TANH:
                case GGML_UNARY_OP_EXP:
                case GGML_UNARY_OP_EXPM1:
                case GGML_UNARY_OP_SOFTPLUS:
                case GGML_UNARY_OP_ELU:
                case GGML_UNARY_OP_XIELU:
                case GGML_UNARY_OP_FLOOR:
                case GGML_UNARY_OP_CEIL:
                case GGML_UNARY_OP_ROUND:
                case GGML_UNARY_OP_TRUNC:
                    if (op->src[0]->type == GGML_TYPE_BF16 && ggml_get_unary_op(op) == GGML_UNARY_OP_XIELU) {
                        return false;
                    }
                    return op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_BF16;
                default:
                    return false;
            }
        case GGML_OP_GLU:
            switch (ggml_get_glu_op(op)) {
                case GGML_GLU_OP_REGLU:
                case GGML_GLU_OP_GEGLU:
                case GGML_GLU_OP_SWIGLU:
                case GGML_GLU_OP_SWIGLU_OAI:
                case GGML_GLU_OP_GEGLU_ERF:
                case GGML_GLU_OP_GEGLU_QUICK:
                case GGML_GLU_OP_SWIGLU_CLAMP:
                    if (op->src[0]->type == GGML_TYPE_BF16 &&
                            (ggml_get_glu_op(op) == GGML_GLU_OP_SWIGLU_OAI || ggml_get_glu_op(op) == GGML_GLU_OP_SWIGLU_CLAMP)) {
                        return false;
                    }
                    if (op->src[0]->type == GGML_TYPE_F16 && ggml_get_glu_op(op) == GGML_GLU_OP_SWIGLU_OAI) {
                        return false;
                    }
                    return ggml_is_contiguous_1(op->src[0]);
                default:
                    return false;
            }
        case GGML_OP_MUL_MAT:
        case GGML_OP_MUL_MAT_ID:
            {
                struct ggml_tensor * a = op->src[0];
                struct ggml_tensor * b = op->src[1];
                if (a->nb[0] != ggml_element_size(a) || b->nb[0] != ggml_element_size(b)) {
                    return false; // TODO this could in principle be implemented though currently there is no use case.
                }
                if (b->type == GGML_TYPE_F16 && a->type != GGML_TYPE_F16 && !ggml_cuda_op_mul_mat_use_fwht(op)) {
                    return false;
                }
                const int cc_dev = ggml_cuda_info().devices[dev_ctx->device].cc;
                const bool f32_src1_ok = GGML_CUDA_CC_IS_NVIDIA(cc_dev) && cc_dev < GGML_CUDA_CC_AMPERE;
                const enum ggml_prec prec_src1 = ggml_prec_get_src(op, 1);
                const bool fp8_id_mmvf = op->op == GGML_OP_MUL_MAT_ID &&
                    a->type == GGML_TYPE_F8_E4M3 && b->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 &&
                    ggml_cuda_should_use_mmvf(a, cc_dev, ggml_cuda_info().devices[dev_ctx->device].warp_size, b->ne[2]);
                if (prec_src1 == GGML_PREC_F32 && !fp8_id_mmvf && (op->op == GGML_OP_MUL_MAT_ID || !f32_src1_ok || a->type != GGML_TYPE_F32)) {
                    return false;
                }
                if (prec_src1 == GGML_PREC_BF16 && (op->op == GGML_OP_MUL_MAT_ID || a->type != GGML_TYPE_F32)) {
                    return false;
                }
#ifdef GGML_USE_MUSA
                const int cc = ggml_cuda_info().devices[dev_ctx->device].cc;
                if (b->ne[2]*b->ne[3] > 1 && !ggml_is_transposed(a) && !ggml_is_transposed(b)) {
                    if (GGML_CUDA_CC_IS_QY1(cc) && op->op == GGML_OP_MUL_MAT &&
                            a->type == GGML_TYPE_F16 && b->type == GGML_TYPE_F16) {
                        return false;
                    }
                    if (GGML_CUDA_CC_IS_QY2(cc) && op->op == GGML_OP_MUL_MAT_ID &&
                            a->type == GGML_TYPE_Q2_K && b->type == GGML_TYPE_F32) {
                        return false;
                    }
                }
#endif // GGML_USE_MUSA
                switch (a->type) {
                    case GGML_TYPE_F32:
                    case GGML_TYPE_F16:
                    case GGML_TYPE_Q1_0:
                    case GGML_TYPE_Q2_0:
                    case GGML_TYPE_Q4_0:
                    case GGML_TYPE_Q4_1:
                    case GGML_TYPE_Q5_0:
                    case GGML_TYPE_Q5_1:
                    case GGML_TYPE_Q8_0:
                    case GGML_TYPE_MXFP4:
                    case GGML_TYPE_NVFP4:
                    case GGML_TYPE_F8_E4M3:
                    case GGML_TYPE_Q2_K:
                    case GGML_TYPE_Q3_K:
                    case GGML_TYPE_Q4_K:
                    case GGML_TYPE_Q5_K:
                    case GGML_TYPE_Q6_K:
                    case GGML_TYPE_Q8_K:
                    case GGML_TYPE_IQ1_M:
                    case GGML_TYPE_IQ1_S:
                    case GGML_TYPE_IQ1_XS:
                    case GGML_TYPE_IQ1_XXS:
                    case GGML_TYPE_IQ1_XXXS:
                    case GGML_TYPE_IQ2_S:
                    case GGML_TYPE_IQ2_XS:
                    case GGML_TYPE_IQ2_XXS:
                    case GGML_TYPE_IQ3_S:
                    case GGML_TYPE_IQ3_XXS:
                    case GGML_TYPE_IQ4_NL:
                    case GGML_TYPE_IQ4_XS:
                    case GGML_TYPE_BF16:
                        return true;
                    default:
                        return false;
                }
            }
        case GGML_OP_OUT_PROD:
            return op->type == GGML_TYPE_F32 && op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32;
        case GGML_OP_GET_ROWS:
            {
                switch (op->src[0]->type) {
                    case GGML_TYPE_F16:
                    case GGML_TYPE_F32:
                    case GGML_TYPE_BF16:
                    case GGML_TYPE_I32:
                    case GGML_TYPE_Q1_0:
                    case GGML_TYPE_Q2_0:
                    case GGML_TYPE_Q4_0:
                    case GGML_TYPE_Q4_1:
                    case GGML_TYPE_Q5_0:
                    case GGML_TYPE_Q5_1:
                    case GGML_TYPE_Q8_0:
                    case GGML_TYPE_F8_E4M3:
                    case GGML_TYPE_Q2_K:
                    case GGML_TYPE_Q3_K:
                    case GGML_TYPE_Q4_K:
                    case GGML_TYPE_Q5_K:
                    case GGML_TYPE_Q6_K:
                    case GGML_TYPE_IQ2_XXS:
                    case GGML_TYPE_IQ2_XS:
                    case GGML_TYPE_IQ2_S:
                    case GGML_TYPE_IQ3_XXS:
                    case GGML_TYPE_IQ3_S:
                    case GGML_TYPE_IQ1_S:
                    case GGML_TYPE_IQ1_M:
                    case GGML_TYPE_IQ4_XS:
                    case GGML_TYPE_MXFP8:
                        return true;
                    case GGML_TYPE_IQ4_NL:
                    case GGML_TYPE_MXFP4:
                        // 32-value sub-blocks, the row size does not guarantee
                        // the QK_K super-blocks the get_rows kernel iterates on
                        return op->src[0]->ne[0] % QK_K == 0;
                    default:
                        return false;
                }
            }
        case GGML_OP_GET_ROWS_BACK:
            return op->type == GGML_TYPE_F32 && op->src[0]->type == GGML_TYPE_F32 && op->ne[2] == 1 && op->ne[3] == 1;
        case GGML_OP_SET_ROWS:
            {
                return (
                           (
                               (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16 || op->type == GGML_TYPE_BF16 || op->type == GGML_TYPE_F8_E4M3 ||
                               op->type == GGML_TYPE_Q4_0 || op->type == GGML_TYPE_Q4_1 || op->type == GGML_TYPE_Q5_0 ||
                               op->type == GGML_TYPE_Q5_1 || op->type == GGML_TYPE_Q8_0 || op->type == GGML_TYPE_IQ4_NL) &&
                               (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_BF16)
                           ) || (
                               op->type == GGML_TYPE_F16 && op->src[0]->type == GGML_TYPE_F16
                           )
                       ) &&
                       (op->src[1]->type == GGML_TYPE_I64 || op->src[1]->type == GGML_TYPE_I32);
            }
        case GGML_OP_SET:
            {
                const ggml_type t = op->type;
                return (t == GGML_TYPE_F32 || t == GGML_TYPE_I32) &&
                    t == op->src[0]->type &&
                    t == op->src[1]->type;
            }
        case GGML_OP_CPY:
            {
                ggml_type src0_type = op->src[0]->type;
                ggml_type src1_type = op->src[1]->type;
                if (src0_type == GGML_TYPE_F8_E4M3 && src1_type == GGML_TYPE_F8_E4M3) {
                    return true;
                }
                if ((src0_type == GGML_TYPE_F32 || src0_type == GGML_TYPE_F16 || src0_type == GGML_TYPE_BF16) &&
                    src1_type == GGML_TYPE_F8_E4M3) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F8_E4M3 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if ((src0_type == GGML_TYPE_F32 || src0_type == GGML_TYPE_BF16 || src0_type == GGML_TYPE_F16) &&
                    (src1_type == GGML_TYPE_F32 || src1_type == GGML_TYPE_BF16 || src1_type == GGML_TYPE_F16)
                ) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q8_0) {
                    return true;
                }
                if (src0_type == GGML_TYPE_Q8_0 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q4_0) {
                    return true;
                }
                if (src0_type == GGML_TYPE_Q4_0 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q4_1) {
                    return true;
                }
                if (src0_type == GGML_TYPE_Q4_1 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q5_0) {
                    return true;
                }
                if (src0_type == GGML_TYPE_Q5_0 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q5_1) {
                    return true;
                }
                if (src0_type == GGML_TYPE_Q5_1 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_IQ4_NL) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_I32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_I32 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_I32 && src1_type == GGML_TYPE_I32) {
                    return true;
                }
                if (src0_type == src1_type && ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1])) {
                    return true;
                }
                return false;
            }
        case GGML_OP_DUP:
                return true;
        case GGML_OP_ARGMAX:
        case GGML_OP_COUNT_EQUAL:
            return true;
        case GGML_OP_REPEAT:
            {
                // the CUDA REPEAT path only implements F32/F16; other types assert at runtime
                ggml_type src0_type = op->src[0]->type;
                return src0_type == GGML_TYPE_F32 || src0_type == GGML_TYPE_F16;
            }
        case GGML_OP_REPEAT_BACK:
                return op->type == GGML_TYPE_F32 && (op->src[0]->ne[2]*op->src[0]->ne[3]) <= (1 << 15);
        case GGML_OP_CONCAT:
            {
                ggml_type src0_type = op->src[0]->type;
                ggml_type src1_type = op->src[1]->type;
                const int32_t dim = op->op_params[0];
                return src0_type == src1_type &&
                       src0_type == op->type &&
                       (
                           (
                               ggml_is_quantized(src0_type) &&
                               (
                                   (
                                       dim == 3 &&
                                       ggml_is_contiguous(op->src[0]) &&
                                       ggml_is_contiguous(op->src[1])
                                   ) || (
                                       dim != 3 &&
                                       ggml_is_contiguous_to_3(op->src[0]) &&
                                       ggml_is_contiguous_to_3(op->src[1])
                                   )
                               ) &&
                               op->src[0]->ne[0] % ggml_blck_size(src0_type) == 0 &&
                               op->src[1]->ne[0] % ggml_blck_size(src0_type) == 0
                           ) || (
                               !ggml_is_quantized(src0_type) &&
                               ggml_blck_size(src0_type) == 1 &&
                               (
                                   ggml_type_size(src0_type) == 1 ||
                                   ggml_type_size(src0_type) == 2 ||
                                   ggml_type_size(src0_type) == 4 ||
                                   ggml_type_size(src0_type) == 8
                               )
                           )
                       );
            }
        case GGML_OP_CONV_TRANSPOSE_1D:
            {
                ggml_type src0_type = op->src[0]->type;
                ggml_type src1_type = op->src[1]->type;
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                return false;
            }
        case GGML_OP_COL2IM_1D:
            {
                ggml_type src0_type = op->src[0]->type;
                return (src0_type == GGML_TYPE_F32 || src0_type == GGML_TYPE_F16 || src0_type == GGML_TYPE_BF16) &&
                    op->type == src0_type &&
                    ggml_is_contiguous(op->src[0]) &&
                    ggml_is_contiguous(op);
            }
        case GGML_OP_SILU_BACK:
            return ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_F32;
        case GGML_OP_NORM:
        case GGML_OP_L2_NORM:
            return ggml_is_contiguous_rows(op->src[0]);
        case GGML_OP_RMS_NORM:
            return (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_BF16) && op->type == op->src[0]->type &&
                   ggml_is_contiguous_rows(op->src[0]);
        case GGML_OP_RMS_NORM_BACK:
            return ggml_is_contiguous(op->src[0]);
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_ADD_ID:
        case GGML_OP_SQR:
        case GGML_OP_SQRT:
        case GGML_OP_SIN:
        case GGML_OP_COS:
        case GGML_OP_LOG:
            return true;
        case GGML_OP_SCALE:
            return (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_BF16) && op->type == op->src[0]->type &&
                   ggml_is_contiguous_rows(op->src[0]);
        case GGML_OP_CLAMP:
            return ggml_is_contiguous_rows(op->src[0]) && (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16);
        case GGML_OP_ADD:
        case GGML_OP_SUB:
        case GGML_OP_MUL:
        case GGML_OP_DIV:
            if (op->src[0]->type == GGML_TYPE_BF16 || op->src[1]->type == GGML_TYPE_BF16 || op->type == GGML_TYPE_BF16) {
                return op->src[0]->type == GGML_TYPE_BF16 && op->type == GGML_TYPE_BF16 &&
                    (op->src[1]->type == GGML_TYPE_BF16 || op->src[1]->type == GGML_TYPE_F32);
            }
            return (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
                   (op->src[1]->type == GGML_TYPE_F32 || op->src[1]->type == GGML_TYPE_F16) &&
                   (op->type         == GGML_TYPE_F32 || op->type         == GGML_TYPE_F16);
        case GGML_OP_SSM_SCAN: {
            const int32_t K = ggml_get_op_params_i32(op, 0);

            if (op->src[3]->ne[0] == 1) {
                // Mamba2
                // (kernel only supports (d_state == 64 || d_state == 96 || d_state == 128 || d_state == 256) && d_head % 16 == 0)
                const int64_t d_state = op->src[0]->ne[0];
                return (d_state == 64 || d_state == 96 || d_state == 128 || d_state == 256) && op->src[0]->ne[1] % 16 == 0;
            } else {
                if (K > 1) {
                    return false;
                }

                // Mamba
                // (kernel only supports d_state == 16, d_head == 1, n_head % 128 == 0, n_group == 1)
                return op->src[0]->ne[0] == 16 && op->src[0]->ne[1] == 1 && op->src[0]->ne[2] % 128 == 0 && op->src[4]->ne[1] == 1;
            }
        }
        case GGML_OP_SSM_CONV: {
            // assumes d_inner % threads == 0
            return op->src[0]->ne[1] % 128 == 0;
        }
        case GGML_OP_CONT:
            return true;
        case GGML_OP_SOFT_MAX:
            return true;
        case GGML_OP_SOFT_MAX_BACK: {
            float max_bias = 0.0f;
            memcpy(&max_bias, (const float *) op->op_params + 1, sizeof(float));
            return max_bias == 0.0f;
        }
        case GGML_OP_ROLL:
            if(op->src[0]->type == GGML_TYPE_F32) {
                return true;
            }
            return false;
        case GGML_OP_ROPE:
        case GGML_OP_ROPE_BACK: {
            return (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_BF16) &&
                   op->type == op->src[0]->type &&
                   op->src[0]->nb[0] == ggml_type_size(op->src[0]->type) && ggml_is_contiguous_2(op->src[0]);
        }
        case GGML_OP_IM2COL:
        case GGML_OP_IM2COL_3D:
        case GGML_OP_CONV_2D:
            return (ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1]));
        case GGML_OP_CONV_3D:
            return (op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_F32) &&
                   op->src[1]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 &&
                   ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1]) && ggml_is_contiguous(op);
        case GGML_OP_CONV_2D_DW:
            return (op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_F32) &&
                   op->src[1]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32;
        case GGML_OP_CONV_TRANSPOSE_2D:
        case GGML_OP_POOL_1D:
        case GGML_OP_POOL_2D:
            return true;
        case GGML_OP_ACC:
            // TODO: extend support like so:
            //return ggml_is_contiguous_rows(op->src[0]) && ggml_is_contiguous_rows(op->src[1]);
            return ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1]);
        case GGML_OP_SUM:
            return ggml_is_contiguously_allocated(op->src[0]);
        case GGML_OP_TOP_K:
            return op->src[0]->ne[0] <= INT_MAX;
        case GGML_OP_ARGSORT:
#ifndef GGML_CUDA_USE_CUB
            {
                // bitonic path: the padded row must fit in shared memory
                int64_t ncols_pad = 1;
                while (ncols_pad < op->src[0]->ne[0]) {
                    ncols_pad *= 2;
                }
                return ncols_pad * sizeof(int) <= ggml_cuda_info().devices[dev_ctx->device].smpb;
            }
#else
            return op->src[0]->ne[0] <= INT_MAX;
#endif
        case GGML_OP_SUM_ROWS:
            return op->src[0]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 && ggml_is_contiguous_rows(op->src[0]);
        case GGML_OP_MEAN:
            return op->src[0]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 && ggml_is_contiguous_rows(op->src[0]);
        case GGML_OP_GROUP_NORM:
            return ggml_is_contiguous(op->src[0]);
        case GGML_OP_PAD:
            return true;
        case GGML_OP_UPSCALE:
        case GGML_OP_PAD_REFLECT_1D:
        case GGML_OP_ARANGE:
        case GGML_OP_TIMESTEP_EMBEDDING:
            return true;
        case GGML_OP_LEAKY_RELU:
            return op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16;
        case GGML_OP_RWKV_WKV6:
        case GGML_OP_GATED_LINEAR_ATTN:
        case GGML_OP_RWKV_WKV7:
            return true;
        case GGML_OP_GATED_DELTA_NET:
            return ggml_are_same_stride(op->src[0], op->src[1]);
        case GGML_OP_DSV4_HC_COMB:
            return op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 &&
                op->src[2]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32;
        case GGML_OP_DSV4_HC_PRE:
            return op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 &&
                op->type == GGML_TYPE_F32;
        case GGML_OP_DSV4_HC_POST:
            return op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 &&
                op->src[2]->type == GGML_TYPE_F32 && (op->src[3] == nullptr || op->src[3]->type == GGML_TYPE_F32) &&
                op->type == GGML_TYPE_F32;
        case GGML_OP_FLASH_ATTN_EXT:
            return ggml_cuda_flash_attn_ext_supported(dev_ctx->device, op);
        case GGML_OP_FLASH_ATTN_EXT_BANDED:
            return ggml_cuda_flash_attn_ext_banded_supported(dev_ctx->device, op);
        case GGML_OP_CROSS_ENTROPY_LOSS:
        case GGML_OP_CROSS_ENTROPY_LOSS_BACK:
        case GGML_OP_OPT_STEP_ADAMW:
        case GGML_OP_OPT_STEP_SGD:
        case GGML_OP_FILL:
        case GGML_OP_CUMSUM:
        case GGML_OP_TRI:
            return true;
        case GGML_OP_DIAG:
            return op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16;
        case GGML_OP_SOLVE_TRI:
            return true;
        case GGML_OP_LIGHTNING_INDEXER:
            return ggml_cuda_lightning_indexer_supported(dev_ctx->device, op);
        case GGML_OP_TOP_K_MASK:
            return op->type == GGML_TYPE_F16 || op->type == GGML_TYPE_F32;
        case GGML_OP_SLEEP:
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
            return false;
#else
            return true;
#endif // defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)

        default:
            return false;
    }
}

static bool ggml_backend_cuda_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    ggml_backend_cuda_device_context * dev_ctx = (ggml_backend_cuda_device_context *) dev->context;
    const bool integrated = ggml_cuda_info().devices[dev_ctx->device].integrated;
    return (ggml_backend_buft_is_cuda(buft) && buft->device == dev) ||
        (integrated &&
         ggml_backend_buft_is_cuda_host(buft) &&
         ggml_backend_cuda_device_supports_cuda_host_buft(dev_ctx->device));
}

static int64_t get_op_batch_size(const ggml_tensor * op) {
    switch (op->op) {
        case GGML_OP_GET_ROWS:
            return 0;
        case GGML_OP_MUL_MAT:
            return op->ne[1];
        case GGML_OP_MUL_MAT_ID:
        case GGML_OP_ROPE:
        case GGML_OP_ROPE_BACK:
            return op->ne[2];
        default:
            return ggml_nrows(op);
    }
}

static bool ggml_backend_cuda_device_offload_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    ggml_backend_cuda_device_context * dev_ctx = (ggml_backend_cuda_device_context *) dev->context;

    return get_op_batch_size(op) >= dev_ctx->op_offload_min_batch_size;
}

static ggml_backend_event_t ggml_backend_cuda_device_event_new(ggml_backend_dev_t dev) {
#ifdef GGML_CUDA_NO_PEER_COPY
    GGML_UNUSED(dev);
    return nullptr;
#else
    ggml_backend_cuda_device_context * dev_ctx = (ggml_backend_cuda_device_context *)dev->context;

    ggml_cuda_set_device(dev_ctx->device);

    cudaEvent_t event;
    CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));

    return new ggml_backend_event {
        /* .device  = */ dev,
        /* .context = */ event,
    };
#endif
}

static void ggml_backend_cuda_device_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    GGML_UNUSED(dev);

    CUDA_CHECK(cudaEventDestroy((cudaEvent_t)event->context));
    delete event;
}

static void ggml_backend_cuda_device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    GGML_UNUSED(dev);
    ggml_cuda_ink2_tl(event, "ev_sync");
    CUDA_CHECK(cudaEventSynchronize((cudaEvent_t)event->context));
    ggml_cuda_ink2_tl(event, "ev_sync_end");
}

static const ggml_backend_device_i ggml_backend_cuda_device_interface = {
    /* .get_name                = */ ggml_backend_cuda_device_get_name,
    /* .get_description         = */ ggml_backend_cuda_device_get_description,
    /* .get_memory              = */ ggml_backend_cuda_device_get_memory,
    /* .get_type                = */ ggml_backend_cuda_device_get_type,
    /* .get_props               = */ ggml_backend_cuda_device_get_props,
    /* .init_backend            = */ ggml_backend_cuda_device_init_backend,
    /* .get_buffer_type         = */ ggml_backend_cuda_device_get_buffer_type,
    /* .get_host_buffer_type    = */ ggml_backend_cuda_device_get_host_buffer_type,
    /* .buffer_from_host_ptr    = */ NULL,
    /* .supports_op             = */ ggml_backend_cuda_device_supports_op,
    /* .supports_buft           = */ ggml_backend_cuda_device_supports_buft,
    /* .offload_op              = */ ggml_backend_cuda_device_offload_op,
    /* .event_new               = */ ggml_backend_cuda_device_event_new,
    /* .event_free              = */ ggml_backend_cuda_device_event_free,
    /* .event_synchronize       = */ ggml_backend_cuda_device_event_synchronize,
};

// backend reg

struct ggml_backend_cuda_reg_context {
    std::vector<ggml_backend_dev_t> devices;
};

static const char * ggml_backend_cuda_reg_get_name(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return GGML_CUDA_NAME;
}

static size_t ggml_backend_cuda_reg_get_device_count(ggml_backend_reg_t reg) {
    ggml_backend_cuda_reg_context * ctx = (ggml_backend_cuda_reg_context *)reg->context;
    return ctx->devices.size();
}

static ggml_backend_dev_t ggml_backend_cuda_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    ggml_backend_cuda_reg_context * ctx = (ggml_backend_cuda_reg_context *)reg->context;
    GGML_ASSERT(index < ctx->devices.size());
    return ctx->devices[index];
}

static ggml_backend_feature * ggml_backend_cuda_get_features(ggml_backend_reg_t reg) {
    static std::vector<ggml_backend_feature> features = []() {
        std::vector<ggml_backend_feature> features;
    #define _STRINGIFY(...) #__VA_ARGS__
    #define STRINGIFY(...) _STRINGIFY(__VA_ARGS__)

    #ifdef __CUDA_ARCH_LIST__
        features.push_back({ "ARCHS", STRINGIFY(__CUDA_ARCH_LIST__) });
    #endif

    #ifdef GGML_CUDA_FORCE_MMQ
        features.push_back({ "FORCE_MMQ", "1" });
    #endif

    #ifdef GGML_CUDA_FORCE_CUBLAS
        features.push_back({ "FORCE_CUBLAS", "1" });
    #endif

    #ifndef GGML_USE_VMM
        features.push_back({ "NO_VMM", "1" });
    #endif

    #ifdef GGML_CUDA_NO_PEER_COPY
        features.push_back({ "NO_PEER_COPY", "1" });
    #endif

    #ifdef GGML_CUDA_USE_GRAPHS
        features.push_back({ "USE_GRAPHS", "1" });
    #endif

    #ifdef GGML_CUDA_FA_QUANTS
        features.push_back({ "FA_QUANTS", GGML_CUDA_FA_QUANTS });
    #endif

    {
        const auto & info = ggml_cuda_info();
        bool has_fp8 = false;
        bool has_fp4 = false;
        for (int id = 0; id < info.device_count; ++id) {
            has_fp8 |= fp8_mma_hardware_available(info.devices[id].cc);
            has_fp4 |= blackwell_mma_available(info.devices[id].cc);
            if (has_fp8 && has_fp4) {
                break;
            }
        }
        if (has_fp8) {
            features.push_back({ "NATIVE_FP8", "1"});
        }
        if (has_fp4) {
            features.push_back({ "BLACKWELL_NATIVE_FP4", "1"});
        }
    }

    #undef _STRINGIFY
    #undef STRINGIFY

        features.push_back({ nullptr, nullptr });

        return features;
    }();

    return features.data();

    GGML_UNUSED(reg);
}

// local: batched upload of the inputs that the scheduler stages in pinned host memory (ggml_backend_sched_copy_input_async):
// one kernel on the backend's stream copies n byte ranges from the staging memory, read directly over the bus, to their device
// destinations, instead of one asynchronous copy each (on HIP every copy is a separate command with its own latency on the
// device timeline). The descriptors are in the same pinned buffer; the layout below is shared with ggml-backend.cpp.
// RDNA3.0 only, like the other local paths (the scheduler falls back to one copy per input when this returns false).
struct ggml_cuda_ink2_upload_desc {
    uint64_t src_off; // offset in the staging buffer
    uint64_t dst;     // device address
    uint64_t size;    // bytes
};

static __global__ void ggml_cuda_ink2_upload_kernel(const unsigned char * __restrict__ base, const ggml_cuda_ink2_upload_desc * __restrict__ descs) {
    const ggml_cuda_ink2_upload_desc d = descs[blockIdx.x];
    const unsigned char * src = base + d.src_off;
    unsigned char       * dst = (unsigned char *) d.dst;
    const size_t first  = (size_t) blockIdx.y*blockDim.x + threadIdx.x;
    const size_t stride = (size_t) gridDim.y*blockDim.x;
    size_t tail = 0;
    if ((((uintptr_t) src | (uintptr_t) dst) & 15) == 0) {
        for (size_t i = first; i < d.size/16; i += stride) {
            ((int4 *) dst)[i] = ((const int4 *) src)[i];
        }
        tail = d.size/16*16;
    } else if ((((uintptr_t) src | (uintptr_t) dst) & 3) == 0) {
        for (size_t i = first; i < d.size/4; i += stride) {
            ((int *) dst)[i] = ((const int *) src)[i];
        }
        tail = d.size/4*4;
    }
    for (size_t i = tail + first; i < d.size; i += stride) {
        dst[i] = src[i];
    }
}

static bool ggml_backend_cuda_ink2_upload_n(ggml_backend_t backend, const void * stage_base, const void * descs, int n) {
    if (!ggml_backend_is_cuda(backend) || n <= 0 || n > 65535) {
        return false;
    }
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;
    if (!ggml_cuda_ink2_arch_ok(cuda_ctx->device)) {
        return false;
    }
    ggml_cuda_set_device(cuda_ctx->device);
    void * dbase = nullptr;
    if (cudaHostGetDevicePointer(&dbase, const_cast<void *>(stage_base), 0) != cudaSuccess || dbase == nullptr) {
        (void) cudaGetLastError();
        return false;
    }
    const ggml_cuda_ink2_upload_desc * hd = (const ggml_cuda_ink2_upload_desc *) descs;
    uint64_t max_size = 0;
    for (int i = 0; i < n; ++i) {
        max_size = std::max(max_size, hd[i].size);
    }
    const int threads = 256;
    const int by      = (int) std::min<uint64_t>(64, std::max<uint64_t>(1, (max_size/16 + threads - 1)/threads));
    const ggml_cuda_ink2_upload_desc * dd = (const ggml_cuda_ink2_upload_desc *) ((const char *) dbase + ((const char *) descs - (const char *) stage_base));
    ggml_cuda_ink2_tl(cuda_ctx, "upload_n", n, (long long) max_size);
    ggml_cuda_ink2_upload_kernel<<<dim3((unsigned) n, (unsigned) by), threads, 0, cuda_ctx->stream()>>>((const unsigned char *) dbase, dd);
    return cudaGetLastError() == cudaSuccess;
}

static void * ggml_backend_cuda_reg_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    GGML_UNUSED(reg);
    if (strcmp(name, "ggml_backend_ink2_upload_n") == 0) {
        return (void *)ggml_backend_cuda_ink2_upload_n;
    }
    if (strcmp(name, "ggml_backend_comm_init") == 0) {
        return (void *)ggml_backend_cuda_comm_init;
    }
    if (strcmp(name, "ggml_backend_comm_free") == 0) {
        return (void *)ggml_backend_cuda_comm_free;
    }
    if (strcmp(name, "ggml_backend_comm_allreduce_tensor") == 0) {
        return (void *)ggml_backend_cuda_comm_allreduce_tensor;
    }
#ifdef GGML_USE_NCCL
    if (strcmp(name, "ggml_backend_comm_init_rank") == 0) {
        return (void *)ggml_backend_cuda_comm_init_rank;
    }
#endif // GGML_USE_NCCL
    if (strcmp(name, "ggml_backend_register_host_buffer") == 0) {
        return (void *)ggml_backend_cuda_register_host_buffer;
    }
    if (strcmp(name, "ggml_backend_unregister_host_buffer") == 0) {
        return (void *)ggml_backend_cuda_unregister_host_buffer;
    }
    if (strcmp(name, "ggml_backend_get_features") == 0) {
        return (void *)ggml_backend_cuda_get_features;
    }
    if (strcmp(name, "ggml_backend_cuda_diffusion_sample") == 0) {
        return (void *)ggml_cuda_diffusion_sample;
    }
    return nullptr;
}

static const ggml_backend_reg_i ggml_backend_cuda_reg_interface = {
    /* .get_name          = */ ggml_backend_cuda_reg_get_name,
    /* .get_device_count  = */ ggml_backend_cuda_reg_get_device_count,
    /* .get_device        = */ ggml_backend_cuda_reg_get_device,
    /* .get_proc_address  = */ ggml_backend_cuda_reg_get_proc_address,
};

// backend registry
ggml_backend_reg_t ggml_backend_cuda_reg() {
    static ggml_backend_reg reg;
    static bool initialized = false;

    {
        static std::mutex mutex;
        std::lock_guard<std::mutex> lock(mutex);
        if (!initialized) {
            ggml_backend_cuda_reg_context * ctx = new ggml_backend_cuda_reg_context;
            const int min_batch_size = ggml_cuda_op_offload_min_batch();

            const ggml_cuda_device_info & info = ggml_cuda_info();
            const bool virtual_devices = info.device_count > info.physical_device_count;

            for (int i = 0; i < info.device_count; i++) {
                const int physical_id = info.devices[i].physical_device;

                ggml_backend_cuda_device_context * dev_ctx = new ggml_backend_cuda_device_context;
                dev_ctx->device = i;
                dev_ctx->name = GGML_CUDA_NAME + std::to_string(i);
                dev_ctx->description = ggml_cuda_device_description(i);

                char pci_bus_id[32] = {};
                CUDA_CHECK(cudaDeviceGetPCIBusId(pci_bus_id, sizeof(pci_bus_id), physical_id));
                dev_ctx->pci_bus_id = pci_bus_id;
                if (virtual_devices) {
                    // make the pci bus id unique for virtual devices
                    dev_ctx->pci_bus_id += "-v" + std::to_string(i);
                }
                for (char & c : dev_ctx->pci_bus_id) {
                    c = std::tolower(c);
                }
                dev_ctx->op_offload_min_batch_size = min_batch_size;

                ggml_backend_dev_t dev = new ggml_backend_device {
                    /* .iface   = */ ggml_backend_cuda_device_interface,
                    /* .reg     = */ &reg,
                    /* .context = */ dev_ctx
                };
                ctx->devices.push_back(dev);
            }

            reg = ggml_backend_reg {
                /* .api_version = */ GGML_BACKEND_API_VERSION,
                /* .iface       = */ ggml_backend_cuda_reg_interface,
                /* .context     = */ ctx
            };
        }

        initialized = true;
    }

    return &reg;
}

ggml_backend_t ggml_backend_cuda_init(int device) {
    if (device < 0 || device >= ggml_backend_cuda_get_device_count()) {
        GGML_LOG_ERROR("%s: invalid device %d\n", __func__, device);
        return nullptr;
    }

    ggml_backend_cuda_context * ctx = new ggml_backend_cuda_context(device);
    if (ctx == nullptr) {
        GGML_LOG_ERROR("%s: failed to allocate context\n", __func__);
        return nullptr;
    }

    ggml_backend_t cuda_backend = new ggml_backend {
        /* .guid    = */ ggml_backend_cuda_guid(),
        /* .iface   = */ ggml_backend_cuda_interface,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_cuda_reg(), device),
        /* .context = */ ctx,
    };

    return cuda_backend;
}

GGML_BACKEND_DL_IMPL(ggml_backend_cuda_reg)
