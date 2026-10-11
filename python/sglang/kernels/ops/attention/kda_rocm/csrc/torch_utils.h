// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Adapted from vLLM csrc/libtorch_stable/torch_utils.h for the HIP-only JIT
// build of the Kimi-K3 KDA kernels: the CUDA runtime calls are spelled as
// their HIP equivalents (the sources are not hipified) and the unused cuBLAS
// helper is dropped.
#pragma once

#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/util/shim_utils.h>

#include <hip/hip_runtime.h>

#include <deque>
#include <mutex>
#include <string>
#include <vector>

// Stable ABI equivalent of TORCH_CHECK_NOT_IMPLEMENTED.
#define STD_TORCH_CHECK_NOT_IMPLEMENTED(cond, ...) \
  STD_TORCH_CHECK(cond, "NotImplementedError: ", __VA_ARGS__)

inline std::deque<std::once_flag> device_flags;
inline std::vector<hipDeviceProp_t> device_properties;
inline std::once_flag vectors_init_flag;

inline void do_init_device_vectors() {
  int device_count;
  hipError_t err = hipGetDeviceCount(&device_count);
  if (err != hipSuccess) {
    STD_TORCH_CHECK(false, "hipGetDeviceCount failed: " +
                               std::string(hipGetErrorString(err)));
  }
  device_flags.resize(device_count);
  device_properties.resize(device_count);
}

inline void initDeviceVectors() {
  std::call_once(vectors_init_flag, do_init_device_vectors);
}

inline void initDeviceProperty(int device_index) {
  hipDeviceProp_t device_prop{};
  hipError_t err = hipGetDeviceProperties(&device_prop, device_index);
  if (err != hipSuccess) {
    STD_TORCH_CHECK(false, "hipGetDeviceProperties failed: " +
                               std::string(hipGetErrorString(err)));
  }
  device_properties[device_index] = device_prop;
}

inline hipDeviceProp_t* get_device_prop() {
  initDeviceVectors();
  int device_index;
  hipError_t err = hipGetDevice(&device_index);
  if (err != hipSuccess) {
    STD_TORCH_CHECK(
        false, "hipGetDevice failed: " + std::string(hipGetErrorString(err)));
  }
  STD_TORCH_CHECK(device_index >= 0 && static_cast<size_t>(device_index) <
                                           device_properties.size(),
                  "HIP device index " + std::to_string(device_index) +
                      " out of range [0, " +
                      std::to_string(device_properties.size()) + ")");

  std::call_once(device_flags[device_index], initDeviceProperty, device_index);
  return &device_properties[device_index];
}

inline hipStream_t get_current_cuda_stream(int32_t device_index = -1) {
  void* stream_ptr = nullptr;
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_get_current_cuda_stream(device_index, &stream_ptr));
  return reinterpret_cast<hipStream_t>(stream_ptr);
}
