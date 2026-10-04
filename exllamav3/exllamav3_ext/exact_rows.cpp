#include "exact_rows.h"

int exact_rows_caps()
{
    return EXACT_ROWS_CAP_LINEAR | EXACT_ROWS_CAP_MGEMM | EXACT_ROWS_CAP_ROUTER | EXACT_ROWS_CAP_MOE;
}
