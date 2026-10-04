#include "exact_rows.h"

#include <atomic>

static std::atomic<uint64_t> one_launches{0};

int exact_rows_caps()
{
    return EXACT_ROWS_CAP_LINEAR | EXACT_ROWS_CAP_MGEMM | EXACT_ROWS_CAP_ROUTER | EXACT_ROWS_CAP_MOE |
           EXACT_ROWS_CAP_HC | EXACT_ROWS_CAP_ONE_LAUNCH | EXACT_ROWS_CAP_HGEMM;
}

uint64_t exact_rows_one_launches()
{
    return one_launches.load(std::memory_order_relaxed);
}

void exact_rows_count_one_launch()
{
    one_launches.fetch_add(1, std::memory_order_relaxed);
}
