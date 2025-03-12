#include "base/alloc.h"
#include <cuda_runtime_api.h>
namespace base {
void DeviceAllocator::memcpy(const void* src_ptr, void* dest_ptr, size_t byte_size,
                             MemcpyKind memcpy_kind, void* stream, bool need_sync) const {
  /// 检查源指针和目标指针是否为空
  CHECK_NE(src_ptr, nullptr);
  CHECK_NE(dest_ptr, nullptr);

  if (!byte_size) {
    return;
  }

  /// 初始化CUDA流为空
  cudaStream_t stream_ = nullptr;
  if (stream) {
    /// 如果提供了流，则将流转换为 CUDA 流类型
    stream_ = static_cast<CUstream_st*>(stream);
  }
  if (memcpy_kind == MemcpyKind::kMemcpyCPU2CPU) {
    std::memcpy(dest_ptr, src_ptr, byte_size);
  } else if (memcpy_kind == MemcpyKind::kMemcpyCPU2CUDA) {
    if (!stream_) {
      /// 同步复制
      cudaMemcpy(dest_ptr, src_ptr, byte_size, cudaMemcpyHostToDevice);
    } else {
      /// 异步复制
      cudaMemcpyAsync(dest_ptr, src_ptr, byte_size, cudaMemcpyHostToDevice, stream_);
    }
  } else if (memcpy_kind == MemcpyKind::kMemcpyCUDA2CPU) {
    if (!stream_) {
      cudaMemcpy(dest_ptr, src_ptr, byte_size, cudaMemcpyDeviceToHost);
    } else {
      cudaMemcpyAsync(dest_ptr, src_ptr, byte_size, cudaMemcpyDeviceToHost, stream_);
    }
  } else if (memcpy_kind == MemcpyKind::kMemcpyCUDA2CUDA) {
    if (!stream_) {
      cudaMemcpy(dest_ptr, src_ptr, byte_size, cudaMemcpyDeviceToDevice);
    } else {
      cudaMemcpyAsync(dest_ptr, src_ptr, byte_size, cudaMemcpyDeviceToDevice, stream_);
    }
  } else {
    LOG(FATAL) << "Unknown memcpy kind: " << int(memcpy_kind);
  }
  if (need_sync) {
    /// 同步CUDA设备
    cudaDeviceSynchronize();
  }
}

void DeviceAllocator::memset_zero(void* ptr, size_t byte_size, void* stream, bool need_sync) {
  CHECK(device_type_ != base::DeviceType::kDeviceUnknown);
  if (device_type_ == base::DeviceType::kDeviceCPU) {
    std::memset(ptr, 0, byte_size);
  } else {
    if (stream) {
      cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
      cudaMemsetAsync(ptr, 0, byte_size, stream_);
    } else {
      cudaMemset(ptr, 0, byte_size);
    }
    if (need_sync) {
      cudaDeviceSynchronize();
    }
  }
}

}  // namespace base