#include "stable_arithmetic.h"
#include <cstdlib>
#include <cstring>
#include <c10/util/Exception.h>

bool stable_arithmetic()
{
    static const bool stable = []()
    {
        const char* value = std::getenv("EXL3_STABLE_ARITHMETIC");
        if (!value || !std::strcmp(value, "0")) return false;
        TORCH_CHECK(!std::strcmp(value, "1"), "EXL3_STABLE_ARITHMETIC must be 0 or 1");
        return true;
    }();
    return stable;
}
