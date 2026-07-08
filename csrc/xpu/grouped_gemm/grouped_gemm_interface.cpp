#include "csrc/utils.h"
#include "grouped_gemm_interface.h"
#include <stdio.h>

#ifdef VLLM_XPU_ENABLE_XE2
  #include "xe_2/grouped_gemm_xe2.h"
#endif
#ifdef VLLM_XPU_ENABLE_XE_DEFAULT
  #include "xe_default/grouped_gemm_xe_default.h"
#endif

torch::Tensor cutlass_grouped_gemm_interface(
    torch::Tensor ptr_A,
    torch::Tensor ptr_B,
    const c10::optional<at::Tensor>& ptr_scales,
    const c10::optional<at::Tensor>& ptr_bias,
    torch::Tensor ptr_D,
    torch::Tensor rows_per_expert,
    int64_t N,
    int64_t K,
    int64_t num_experts,
    bool is_B_int4,
    bool is_B_mxfp4) {
  if (vllm::xpu::force_xe_default_kernel()) {
#ifdef VLLM_XPU_ENABLE_XE_DEFAULT
    int64_t groups = num_experts;
    return cutlass_grouped_gemm_xe_default(
        ptr_A, ptr_B, ptr_bias, ptr_D, rows_per_expert, N, K, groups);
#else
    TORCH_CHECK(
        false,
        "XE default cutlass kernel is not enabled in this build, force use XE "
        "default kernel failed.");
#endif
  } else if (vllm::xpu::is_xe2_arch()) {
#ifdef VLLM_XPU_ENABLE_XE2
    // Use XE2 cutlass kernel
    return cutlass_grouped_gemm_xe2(
        ptr_A,
        ptr_B,
        ptr_scales,
        ptr_bias,
        ptr_D,
        rows_per_expert,
        N,
        K,
        num_experts,
        is_B_int4,
        is_B_mxfp4);
#else
    TORCH_CHECK(false, "XE2 cutlass kernel is not enabled in this build.");
#endif
  } else {
#ifdef VLLM_XPU_ENABLE_XE_DEFAULT
    int64_t groups = num_experts;
    return cutlass_grouped_gemm_xe_default(
        ptr_A, ptr_B, ptr_bias, ptr_D, rows_per_expert, N, K, groups);
#else
    TORCH_CHECK(
        false, "XE default cutlass kernel is not enabled in this build.");
#endif
  }
}

torch::Tensor cutlass_batched_gemm_interface(
    torch::Tensor ptr_A,
    torch::Tensor ptr_B,
    const c10::optional<at::Tensor>& ptr_scales,
    const c10::optional<at::Tensor>& ptr_bias,
    torch::Tensor ptr_D,
    torch::Tensor expert_num_tokens,
    int64_t max_tokens_per_rank,
    int64_t N,
    int64_t K,
    int64_t num_experts,
    bool is_B_int4,
    bool is_B_mxfp4) {
  // The batched MoEGEMM kernel expects per-expert token COUNTS as int32
  // (it computes the running row/tile prefix sums internally). Passing an
  // int64 tensor or a prefix-sum offset here mis-sizes gemm_m per expert and
  // reads/writes out of bounds (UR_RESULT_ERROR_DEVICE_LOST). Pass raw counts.
  auto rows_per_expert = expert_num_tokens.to(torch::kInt32).contiguous();

  if (vllm::xpu::force_xe_default_kernel()) {
    TORCH_CHECK(
        false,
        "Batched gemm is currently only supported on XE2 arch, but default xe "
        "was requested.");
  } else if (vllm::xpu::is_xe2_arch()) {
#ifdef VLLM_XPU_ENABLE_XE2
    return cutlass_grouped_gemm_xe2(
        ptr_A,
        ptr_B,
        ptr_scales,
        ptr_bias,
        ptr_D,
        rows_per_expert,
        N,
        K,
        num_experts,
        is_B_int4,
        is_B_mxfp4,
        true,
        max_tokens_per_rank);
#else
    TORCH_CHECK(false, "XE2 cutlass kernel is not enabled in this build.");
#endif
  } else {
    TORCH_CHECK(
        false, "Batched gemm is currently only supported on XE2 arch.");
  }
  return ptr_D;
}
