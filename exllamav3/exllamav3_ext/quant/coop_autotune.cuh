#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <vector>

struct CoopAutotuneCandidate
{
    void* kernel;
    int block_dim;
    int max_num_sms;
    int max_concurrency;
    int total_sms;
    int tag;
};

struct CoopAutotuneLaunch
{
    void* kernel;
    int block_dim;
    int num_sms;
    int concurrency;
    int tag;
};

class CoopKernelAutotuner
{
public:
    // The launch this process holds for `hash`, without launching it; false when it holds none.
    // No disk cache lookup and no tuning
    static bool find(uint64_t hash, CoopAutotuneLaunch* launch_config);

    static bool launch_locked
    (
        uint64_t hash,
        void** kernel_args,
        size_t smem,
        cudaStream_t stream,
        CoopAutotuneLaunch* launch_config = nullptr
    );

    static CoopAutotuneLaunch launch
    (
        uint64_t hash,
        const std::vector<CoopAutotuneCandidate>& candidates,
        void** kernel_args,
        size_t smem,
        cudaStream_t stream,
        size_t numel_B = 1e9
    );
};
