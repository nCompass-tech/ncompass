#include "flash_fwd_launch_template.h"

namespace hstu {
#ifndef FLASHATTENTION_DISABLE_HDIM128
template void run_mha_fwd_<90, cutlass::bfloat16_t, 128, true>(
    Flash_fwd_params& params,
    cudaStream_t stream);
#endif
} // namespace hstu
