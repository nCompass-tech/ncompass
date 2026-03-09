#pragma once
#include "flash.h"
#include "static_switch.h"
#include <cuda_bf16.h>
#include <cutlass/numeric_types.h>

namespace hstu {

// Stub kernel: zeros the output tensor.
// TODO: Replace with actual HSTU flash attention implementation.
static __global__ void stub_zero_output(__nv_bfloat16* o_ptr, int64_t num_elements) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        o_ptr[idx] = __float2bfloat16(0.0f);
    }
}

template <int Arch, typename T, int kHeadDim, bool Softmax>
void run_mha_fwd_(Flash_fwd_params& params, cudaStream_t stream) {
    int64_t total_elements = (int64_t)params.total_seq_len_q * params.h * params.v_d;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    stub_zero_output<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<__nv_bfloat16*>(params.o_ptr), total_elements);
}

} // namespace hstu
