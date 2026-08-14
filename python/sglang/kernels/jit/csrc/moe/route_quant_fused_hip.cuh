/// K3 ROCm MoE-front preparation in one launch.
///
/// CTAs [0, M) run the production CDNA Radix-4 router. CTAs [M, 2M)
/// quantize one 3584-wide BF16 activation row to MXFP8 with 112 group-32
/// E8M0 scales. The two independent halves can overlap on the device while
/// appearing as one graph node.

#pragma once

#include "route_radix4_hip.cuh"

#include <sgl_kernel/deepseek_v4/fp8_utils.cuh>
#include <hip/hip_fp8.h>

namespace sglang {

inline constexpr uint32_t kRouteQuantHipHidden = 3584;
inline constexpr uint32_t kRouteQuantHipGroups = 112;
inline constexpr uint32_t kRouteQuantHipGroup = 32;
inline constexpr uint32_t kRouteQuantHipBlock = 256;

struct RouteQuantHipParams {
  RouteRadix4Params route;
  const bf16_t* __restrict__ x;
  uint8_t* __restrict__ out_q;
  uint8_t* __restrict__ out_s;
  uint32_t stride_x;
  uint32_t stride_q;
  uint32_t stride_s;
  uint32_t M;
};

SGL_DEVICE void quant_mxfp8_row(const RouteQuantHipParams& params, uint32_t token) {
  const uint32_t tid = threadIdx.x;
  if (tid >= kRouteQuantHipGroups * 2) return;

  const uint32_t group = tid >> 1;
  const uint32_t half = tid & 1;
  const uint32_t begin = group * kRouteQuantHipGroup + half * 16;
  const bf16_t* in = params.x + static_cast<size_t>(token) * params.stride_x;
  uint8_t* out = params.out_q + static_cast<size_t>(token) * params.stride_q;

  float amax = 0.0f;
#pragma unroll
  for (uint32_t i = 0; i < 16; ++i) {
    amax = fmaxf(amax, fabsf(__bfloat162float(in[begin + i])));
  }
  amax = fmaxf(amax, __shfl_xor(amax, 1, 64));
  amax = fmaxf(amax, 1.0e-10f);

  using deepseek_v4::fp8::cast_to_ue8m0;
  using deepseek_v4::fp8::inv_scale_ue8m0;
  const int32_t exp = cast_to_ue8m0(amax * (1.0f / 448.0f));
  const bf16_t quant_scale = __float2bfloat16(inv_scale_ue8m0(exp));

  if (half == 0) {
    params.out_s[static_cast<size_t>(token) * params.stride_s + group] =
        static_cast<uint8_t>(exp);
  }
#pragma unroll
  for (uint32_t i = 0; i < 16; ++i) {
    const bf16_t scaled = __hmul(in[begin + i], quant_scale);
    out[begin + i] = __hip_fp8_e4m3(scaled).__x;
  }
}

__global__ __launch_bounds__(kRouteQuantHipBlock) void route_quant_fused_hip_kernel(
    __grid_constant__ const RouteQuantHipParams params) {
  if (blockIdx.x < params.M) {
    route_radix4_block<
        bf16_t,
        bf16_t,
        kRadix4NumExperts,
        kRadix4TopK,
        kRadix4Block>(params.route, static_cast<int>(blockIdx.x));
  } else {
    quant_mxfp8_row(params, static_cast<uint32_t>(blockIdx.x) - params.M);
  }
}

struct RouteQuantFusedHipKernel {
  static void run(
      const tvm::ffi::TensorView scores,
      const tvm::ffi::TensorView bias,
      const tvm::ffi::TensorView out_w,
      const tvm::ffi::TensorView out_i,
      const tvm::ffi::TensorView out_packed,
      const tvm::ffi::TensorView x,
      const tvm::ffi::TensorView out_q,
      const tvm::ffi::TensorView out_s,
      int64_t topk,
      double routed_scaling_factor,
      bool renormalize,
      bool apply_scale) {
    using namespace host;

    auto M_ = SymbolicSize{"num_tokens"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLROCM>();
    TensorMatcher({M_, kRadix4NumExperts})
        .with_dtype<bf16_t>()
        .with_device(device_)
        .with_strides({-1, 1})
        .verify(scores);
    TensorMatcher({kRadix4NumExperts})
        .with_dtype<bf16_t>()
        .with_device(device_)
        .verify(bias);
    TensorMatcher({M_, kRadix4TopK})
        .with_dtype<fp32_t>()
        .with_device(device_)
        .verify(out_w);
    TensorMatcher({M_, kRadix4TopK})
        .with_dtype<int32_t>()
        .with_device(device_)
        .verify(out_i);
    TensorMatcher({M_, kRadix4TopK})
        .with_dtype<int32_t>()
        .with_device(device_)
        .verify(out_packed);
    TensorMatcher({M_, kRouteQuantHipHidden})
        .with_dtype<bf16_t>()
        .with_device(device_)
        .with_strides({-1, 1})
        .verify(x);
    TensorMatcher({M_, kRouteQuantHipHidden})
        .with_device(device_)
        .with_strides({-1, 1})
        .verify(out_q);
    TensorMatcher({M_, kRouteQuantHipGroups / 4})
        .with_dtype<int32_t>()
        .with_device(device_)
        .with_strides({-1, 1})
        .verify(out_s);
    RuntimeCheck(
        topk == kRadix4TopK && !apply_scale,
        "route_quant_fused_hip expects topk=16 and apply_scale=false");

    const auto M = static_cast<uint32_t>(M_.unwrap());
    if (M == 0) return;
    const auto params = RouteQuantHipParams{
        .route =
            {
                scores.data_ptr(),
                bias.data_ptr(),
                static_cast<fp32_t*>(out_w.data_ptr()),
                static_cast<int32_t*>(out_i.data_ptr()),
                static_cast<uint32_t>(scores.stride(0)),
                static_cast<uint32_t>(out_w.stride(0)),
                static_cast<fp32_t>(routed_scaling_factor),
                renormalize,
            },
        .x = static_cast<const bf16_t*>(x.data_ptr()),
        .out_q = static_cast<uint8_t*>(out_q.data_ptr()),
        .out_s = static_cast<uint8_t*>(out_s.data_ptr()),
        .stride_x = static_cast<uint32_t>(x.stride(0)),
        .stride_q = static_cast<uint32_t>(out_q.stride(0)),
        .stride_s = static_cast<uint32_t>(out_s.stride(0) * sizeof(int32_t)),
        .M = M,
    };
    LaunchKernel(2 * M, kRouteQuantHipBlock, device_.unwrap())(
        route_quant_fused_hip_kernel, params);
  }
};

}  // namespace sglang
