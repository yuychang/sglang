/* Copyright 2026 SGLang Team. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>

#include "utils.h"

namespace {

template <typename scalar_t>
__global__ void add_shared_kernel(
    const scalar_t* __restrict__ routed,
    const scalar_t* __restrict__ shared,
    scalar_t* __restrict__ output,
    int64_t numel) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (; idx < numel; idx += stride) {
    float value = castToFloat(routed[idx]) + castToFloat(shared[idx]);
    output[idx] = castFromFloat<scalar_t>(value);
  }
}

int64_t get_num_blocks(int64_t numel) {
  constexpr int64_t block_size = 256;
  constexpr int64_t max_blocks = 65535;
  return std::min<int64_t>((numel + block_size - 1) / block_size, max_blocks);
}

}  // namespace

void rocm_mxfp4_moe_add_shared(
    torch::Tensor& output,
    const torch::Tensor& routed_final,
    const torch::Tensor& shared_output) {
  TORCH_CHECK(output.is_cuda(), "output must be a CUDA/HIP tensor");
  TORCH_CHECK(routed_final.is_cuda(), "routed_final must be a CUDA/HIP tensor");
  TORCH_CHECK(shared_output.is_cuda(), "shared_output must be a CUDA/HIP tensor");
  TORCH_CHECK(output.is_contiguous(), "output must be contiguous");
  TORCH_CHECK(routed_final.is_contiguous(), "routed_final must be contiguous");
  TORCH_CHECK(shared_output.is_contiguous(), "shared_output must be contiguous");
  TORCH_CHECK(output.sizes() == routed_final.sizes(), "output and routed_final shapes must match");
  TORCH_CHECK(output.sizes() == shared_output.sizes(), "output and shared_output shapes must match");
  TORCH_CHECK(output.scalar_type() == routed_final.scalar_type(), "output and routed_final dtypes must match");
  TORCH_CHECK(output.scalar_type() == shared_output.scalar_type(), "output and shared_output dtypes must match");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(output));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(output.get_device());
  const int64_t numel = output.numel();
  if (numel == 0) {
    return;
  }

  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FLOAT_FP16(output.scalar_type(), c_type, [&] {
    constexpr int threads = 256;
    add_shared_kernel<c_type><<<get_num_blocks(numel), threads, 0, stream>>>(
        static_cast<const c_type*>(routed_final.data_ptr()),
        static_cast<const c_type*>(shared_output.data_ptr()),
        static_cast<c_type*>(output.data_ptr()),
        numel);
    return true;
  });
}
