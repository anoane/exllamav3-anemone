#include "exact_rows.h"

#include <atomic>
#include <ATen/cuda/CUDAContext.h>
#include "quant/exl3_devctx.cuh"

static std::atomic<uint64_t> one_launches{0};
static std::atomic<uint64_t> moe_grouped{0};
static std::atomic<uint64_t> router_one_launches{0};

int exact_rows_caps()
{
    return EXACT_ROWS_CAP_LINEAR | EXACT_ROWS_CAP_MGEMM | EXACT_ROWS_CAP_ROUTER | EXACT_ROWS_CAP_MOE |
           EXACT_ROWS_CAP_HC | EXACT_ROWS_CAP_ONE_LAUNCH | EXACT_ROWS_CAP_HGEMM | EXACT_ROWS_CAP_MOE_GROUPED |
           EXACT_ROWS_CAP_ROUTER_ONE_LAUNCH;
}

bool exact_rows_device_ok(int device)
{
    if (device < 0 || device >= MAX_DEVICES) return false;
    const auto* props = at::cuda::getDeviceProperties((c10::DeviceIndex) device);
    const int sm = props->major * 10 + props->minor;
    return sm == 80 || sm == 89 || sm == 120;
}

uint64_t exact_rows_one_launches()
{
    return one_launches.load(std::memory_order_relaxed);
}

void exact_rows_count_one_launch()
{
    one_launches.fetch_add(1, std::memory_order_relaxed);
}

static std::atomic<uint64_t>* served_counter(int cap)
{
    switch (cap)
    {
        case EXACT_ROWS_CAP_MOE_GROUPED: return &moe_grouped;
        case EXACT_ROWS_CAP_ROUTER_ONE_LAUNCH: return &router_one_launches;
        default: return nullptr;
    }
}

uint64_t exact_rows_served(int cap)
{
    const auto* counter = served_counter(cap);
    return counter ? counter->load(std::memory_order_relaxed) : 0;
}

void exact_rows_count_served(int cap)
{
    auto* counter = served_counter(cap);
    if (counter) counter->fetch_add(1, std::memory_order_relaxed);
}
