#include "rmsnorm_kernel.h"

namespace kernel {
void rmsnorm_kernel_cpu(const tensor::Tensor& input, const tensor::Tensor& weight,
                        const tensor::Tensor& output, void* stream) {
  UNUSED(stream);
  CHECK(!input.is_empty());
  CHECK(!weight.is_empty());
  CHECK(!output.is_empty());

  CHECK(input.device_type() == base::DeviceType::kDeviceCPU &&
        weight.device_type() == base::DeviceType::kDeviceCPU &&
        output.device_type() == base::DeviceType::kDeviceCPU);

  const float* in_ptr = input.ptr<float>();    ///< 输入指针
  const float* wei_ptr = weight.ptr<float>();  ///< 权重指针
  const float* out_ptr = output.ptr<float>();  ///< 输出指针
  const int32_t dim = static_cast<int32_t>(input.size());

  /**
   * arma::fvec  vec(ptr_aux_mem, number_of_elements, copy_aux_mem = true, strict = false)
   * ptr_aux_mem 可写辅助内存
   * number_of_elements 张量元素大小
   * copy_aux_mem  false is 直接使用辅助内存，而不进行复制
   * strict   false is 使用辅助内存，知道大小发生变化
   *          true  is 向量的生命周期绑定到辅助存储器，向量中的元素不能改变
   */
  arma::fvec in_tensor(const_cast<float*>(in_ptr), dim, false, true);
  arma::fvec out_tensor(const_cast<float*>(out_ptr), dim, false, true);
  arma::fvec wei_tensor(const_cast<float*>(wei_ptr), dim, false, true);

#ifdef QWEN2_SUPPORT
  const float eps = 1e-6f;
#else
  const float eps = 1e-5f;
#endif

  // as_scalar 转换为标量
  const float mean = arma::as_scalar(arma::mean(arma::pow(in_tensor, 2))) + eps;
  const float rsqrt = 1.f / std::sqrt(mean);
  out_tensor = wei_tensor % (rsqrt * in_tensor);
}
}  // namespace kernel