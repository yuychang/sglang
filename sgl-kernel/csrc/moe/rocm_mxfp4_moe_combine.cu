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

// ROCm/HIP fused combine kernels for the Kimi-K2.5 / DeepSeek-style MXFP4 MoE
// multi-stream overlap.
//
//  * rocm_mxfp4_moe_add_shared             (P0): out = routed_final + shared
//  * rocm_mxfp4_moe_finalize_fuse_shared   (P1): deferred routed finalize +
//                                                shared add in one launch.
//
// Both accumulate in FP32 and emit BF16/FP16 (matching the surrounding SGLang
// graph). They read the current HIP stream from the PyTorch dispatcher, do no
// host synchronization, and are safe to launch inside a HIP graph capture.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>

#include "utils.h"

namespace {

// out[i] = routed_final[i] + shared_output[i]  (element-wise, FP32 accumulate)
template <typename T>
__global__ void rocm_mxfp4_moe_add_shared_kernel(
    T* __restrict__ out,
    const T* __restrict__ routed_final,
    const T* __restrict__ shared_output,
    int64_t n) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (; idx < n; idx += stride) {
    float acc = static_cast<float>(routed_final[idx]) + static_cast<float>(shared_output[idx]);
    out[idx] = static_cast<T>(acc);
  }
}

// out[t, :] = shared[t, :] + rsf * sum_k w[t,k] * routed_partial[row_map[t,k], :]
// One block per token; threads stride over the hidden dimension (coalesced).
template <typename T, bool HAS_SHARED>
__global__ void rocm_mxfp4_moe_finalize_fuse_shared_kernel(
    T* __restrict__ out,
    const T* __restrict__ routed_partial,
    const int64_t* __restrict__ row_map,
    const float* __restrict__ topk_weights,
    const T* __restrict__ shared_output,
    float routed_scaling_factor,
    int top_k,
    int64_t num_tokens,
    int64_t hidden,
    int64_t routed_row_stride) {
  int64_t token = blockIdx.x;
  if (token >= num_tokens) {
    return;
  }
  const int64_t out_base = token * hidden;
  const int64_t map_base = token * static_cast<int64_t>(top_k);

  for (int64_t col = threadIdx.x; col < hidden; col += blockDim.x) {
    float acc = 0.0f;
    for (int k = 0; k < top_k; ++k) {
      int64_t row = row_map[map_base + k];
      float w = topk_weights[map_base + k];
      float v = static_cast<float>(routed_partial[row * routed_row_stride + col]);
      acc += w * v;
    }
    acc *= routed_scaling_factor;
    if (HAS_SHARED) {
      acc += static_cast<float>(shared_output[out_base + col]);
    }
    out[out_base + col] = static_cast<T>(acc);
  }
}

}  // namespace

void rocm_mxfp4_moe_add_shared(
    const at::Tensor& routed_final, const at::Tensor& shared_output, at::Tensor& out) {
  TORCH_CHECK(routed_final.is_cuda(), "routed_final must be a CUDA/HIP tensor");
  TORCH_CHECK(shared_output.is_cuda(), "shared_output must be a CUDA/HIP tensor");
  TORCH_CHECK(out.is_cuda(), "out must be a CUDA/HIP tensor");
  TORCH_CHECK(routed_final.is_contiguous(), "routed_final must be contiguous");
  TORCH_CHECK(shared_output.is_contiguous(), "shared_output must be contiguous");
  TORCH_CHECK(out.is_contiguous(), "out must be contiguous");
  TORCH_CHECK(
      routed_final.sizes() == shared_output.sizes() && routed_final.sizes() == out.sizes(),
      "rocm_mxfp4_moe_add_shared: routed_final, shared_output and out must share the same shape");
  TORCH_CHECK(
      routed_final.scalar_type() == shared_output.scalar_type() &&
          routed_final.scalar_type() == out.scalar_type(),
      "rocm_mxfp4_moe_add_shared: dtype mismatch");

  const int64_t n = out.numel();
  if (n == 0) {
    return;
  }
  const int threads = 256;
  int64_t blocks64 = (n + threads - 1) / threads;
  const int blocks = static_cast<int>(blocks64 > 65535 ? 65535 : blocks64);

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const at::cuda::OptionalCUDAGuard device_guard(device_of(out));

  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FLOAT_FP16(out.scalar_type(), c_type, [&] {
    rocm_mxfp4_moe_add_shared_kernel<c_type><<<blocks, threads, 0, stream>>>(
        static_cast<c_type*>(out.data_ptr()),
        static_cast<const c_type*>(routed_final.data_ptr()),
        static_cast<const c_type*>(shared_output.data_ptr()),
        n);
    return true;
  });
  cudaError_t status = cudaGetLastError();
  TORCH_CHECK(status == cudaSuccess, "rocm_mxfp4_moe_add_shared launch failed: ", cudaGetErrorString(status));
}

void rocm_mxfp4_moe_finalize_fuse_shared(
    const at::Tensor& routed_partial,
    const at::Tensor& row_map,
    const at::Tensor& topk_weights,
    const c10::optional<at::Tensor>& shared_output,
    double routed_scaling_factor,
    int64_t top_k,
    at::Tensor& out) {
  TORCH_CHECK(routed_partial.is_cuda(), "routed_partial must be a CUDA/HIP tensor");
  TORCH_CHECK(row_map.is_cuda(), "row_map must be a CUDA/HIP tensor");
  TORCH_CHECK(topk_weights.is_cuda(), "topk_weights must be a CUDA/HIP tensor");
  TORCH_CHECK(out.is_cuda(), "out must be a CUDA/HIP tensor");
  TORCH_CHECK(routed_partial.dim() == 2, "routed_partial must be 2D (num_rows, hidden)");
  TORCH_CHECK(out.dim() == 2, "out must be 2D (num_tokens, hidden)");
  TORCH_CHECK(out.is_contiguous(), "out must be contiguous");
  TORCH_CHECK(topk_weights.is_contiguous(), "topk_weights must be contiguous");
  TORCH_CHECK(row_map.is_contiguous(), "row_map must be contiguous");
  TORCH_CHECK(row_map.scalar_type() == at::ScalarType::Long, "row_map must be int64");
  TORCH_CHECK(topk_weights.scalar_type() == at::ScalarType::Float, "topk_weights must be float32");
  // routed_partial must be contiguous in the hidden dimension for coalesced loads.
  TORCH_CHECK(
      routed_partial.stride(1) == 1,
      "routed_partial must be contiguous along the hidden dimension");

  const int64_t num_tokens = out.size(0);
  const int64_t hidden = out.size(1);
  TORCH_CHECK(routed_partial.size(1) == hidden, "routed_partial hidden dim must match out");
  TORCH_CHECK(row_map.size(0) == num_tokens && row_map.size(1) == top_k, "row_map must be (num_tokens, top_k)");
  TORCH_CHECK(
      topk_weights.size(0) == num_tokens && topk_weights.size(1) == top_k,
      "topk_weights must be (num_tokens, top_k)");

  const bool has_shared = shared_output.has_value();
  if (has_shared) {
    const auto& s = shared_output.value();
    TORCH_CHECK(s.is_cuda(), "shared_output must be a CUDA/HIP tensor");
    TORCH_CHECK(s.is_contiguous(), "shared_output must be contiguous");
    TORCH_CHECK(s.size(0) == num_tokens && s.size(1) == hidden, "shared_output must be (num_tokens, hidden)");
    TORCH_CHECK(s.scalar_type() == out.scalar_type(), "shared_output dtype must match out");
  }
  TORCH_CHECK(routed_partial.scalar_type() == out.scalar_type(), "routed_partial dtype must match out");

  if (num_tokens == 0 || hidden == 0) {
    return;
  }

  const int64_t routed_row_stride = routed_partial.stride(0);
  const int threads = hidden >= 1024 ? 1024 : static_cast<int>(((hidden + 63) / 64) * 64);
  const int blocks = static_cast<int>(num_tokens > 2147483647 ? 2147483647 : num_tokens);

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const at::cuda::OptionalCUDAGuard device_guard(device_of(out));

  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FLOAT_FP16(out.scalar_type(), c_type, [&] {
    const c_type* shared_ptr =
        has_shared ? static_cast<const c_type*>(shared_output.value().data_ptr()) : nullptr;
    if (has_shared) {
      rocm_mxfp4_moe_finalize_fuse_shared_kernel<c_type, true><<<blocks, threads, 0, stream>>>(
          static_cast<c_type*>(out.data_ptr()),
          static_cast<const c_type*>(routed_partial.data_ptr()),
          static_cast<const int64_t*>(row_map.data_ptr()),
          static_cast<const float*>(topk_weights.data_ptr()),
          shared_ptr,
          static_cast<float>(routed_scaling_factor),
          static_cast<int>(top_k),
          num_tokens,
          hidden,
          routed_row_stride);
    } else {
      rocm_mxfp4_moe_finalize_fuse_shared_kernel<c_type, false><<<blocks, threads, 0, stream>>>(
          static_cast<c_type*>(out.data_ptr()),
          static_cast<const c_type*>(routed_partial.data_ptr()),
          static_cast<const int64_t*>(row_map.data_ptr()),
          static_cast<const float*>(topk_weights.data_ptr()),
          shared_ptr,
          static_cast<float>(routed_scaling_factor),
          static_cast<int>(top_k),
          num_tokens,
          hidden,
          routed_row_stride);
    }
    return true;
  });
  cudaError_t status = cudaGetLastError();
  TORCH_CHECK(
      status == cudaSuccess, "rocm_mxfp4_moe_finalize_fuse_shared launch failed: ", cudaGetErrorString(status));
}
