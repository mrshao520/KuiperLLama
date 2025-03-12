#include <cub/block/block_reduce.cuh>
#include "rmsnorm_kernel.cuh"
namespace kernel {

/// BLOCK_DIM 定义线程块的大小
template <int32_t BLOCK_DIM>
static __global__ void row_rmsnorm_f32(float* in, float* wei, float* out, int size, float eps) {
  const int tid = threadIdx.x;  /// 当前线程在线程块中的索引

  /// 定义每个数据包包含的浮点数数量
  constexpr int pack_size = 4;
  /// 计算可以完整打包的数据包数量
  const int pack_num = size / pack_size;
  /// 计算打包后剩余的元素偏移量
  const int pack_off = pack_size * pack_num;

  float sum = 0.0f;
  /// 将输入指针转换为float4类型，以便进行数据打包处理
  float4* in_pack = reinterpret_cast<float4*>(in);
  /// 遍历每个数据包，计算包内每个元素的平方和
  for (int i = tid; i < pack_num; i += blockDim.x) {
    float4 in_float4 = *(in_pack + i);
    sum += in_float4.x * in_float4.x;
    sum += in_float4.y * in_float4.y;
    sum += in_float4.z * in_float4.z;
    sum += in_float4.w * in_float4.w;
  }

  /// 处理剩余的未打包元素，计算它们的平方和
  for (int i = pack_off + tid; i < size; i += blockDim.x) {
    sum += in[i] * in[i];
  }
  /// 使用CUB库的块归约操作，用于在线程块内累加所有线程的sum值
  using BlockReduce = cub::BlockReduce<float, BLOCK_DIM>;
  /// 定义共享内存，用于存储临时数据和最终归约结果
  __shared__ typename BlockReduce::TempStorage temp;
  __shared__ float shared_val;
  /// 执行规约操作
  sum = BlockReduce(temp).Sum(sum);
  /// 只有线程 0 将归约结果存储在 shared_val 中
  if (threadIdx.x == 0) {
    shared_val = sum;
  }
  /// 同步所有线程，确保线程0的写入操作完成。
  __syncthreads();
  /// 将归约结果广播给所有线程。
  sum = shared_val;
  /// 计算平方根的倒数
  const float scale = rsqrtf(sum / static_cast<float>(size) + eps);
  /// 将权重和输出指针转换为float4类型。
  float4* wei_pack = reinterpret_cast<float4*>(wei);
  float4* out_pack = reinterpret_cast<float4*>(out);
  /// 遍历每个数据包，将输入、权重和缩放因子相乘，结果存储在输出数组中
  for (int i = tid; i < pack_num; i += blockDim.x) {
    float4 in_float4 = *(in_pack + i);
    float4 wei_float4 = *(wei_pack + i);
    *(out_pack + i) =
        make_float4(scale * in_float4.x * wei_float4.x, scale * in_float4.y * wei_float4.y,
                    scale * in_float4.z * wei_float4.z, scale * in_float4.w * wei_float4.w);
  }
  /// 处理剩余的未打包元素，同样将输入、权重和缩放因子相乘，结果存储在输出数组中
  for (int i = pack_off + tid; i < size; i += blockDim.x) {
    out[i] = wei[i] * in[i] * scale;
  }
}

void rmsnorm_kernel_cu(const tensor::Tensor& input, const tensor::Tensor& weight,
                       const tensor::Tensor& output, void* stream) {
  CHECK(!input.is_empty());
  CHECK(!weight.is_empty());
  CHECK(!output.is_empty());

  CHECK(input.device_type() == base::DeviceType::kDeviceCUDA &&
        weight.device_type() == base::DeviceType::kDeviceCUDA &&
        output.device_type() == base::DeviceType::kDeviceCUDA);

#ifdef QWEN2_SUPPORT
  const float eps = 1e-6f;
#else
  const float eps = 1e-5f;
#endif
  const int32_t size = static_cast<int32_t>(input.size());
  float* in_ptr = const_cast<float*>(input.ptr<float>());
  float* wei_ptr = const_cast<float*>(weight.ptr<float>());
  float* out_ptr = const_cast<float*>(output.ptr<float>());
  constexpr int threads_num = 128;
  if (stream) {
    cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
    row_rmsnorm_f32<128><<<1, threads_num, 0, stream_>>>(in_ptr, wei_ptr, out_ptr, size, eps);
  } else {
    row_rmsnorm_f32<128><<<1, threads_num>>>(in_ptr, wei_ptr, out_ptr, size, eps);
  }
}
}  // namespace kernel