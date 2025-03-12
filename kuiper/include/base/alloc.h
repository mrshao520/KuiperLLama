#ifndef KUIPER_INCLUDE_BASE_ALLOC_H_
#define KUIPER_INCLUDE_BASE_ALLOC_H_
#include <map>
#include <memory>
#include "base.h"
namespace base {
enum class MemcpyKind {
  kMemcpyCPU2CPU = 0,
  kMemcpyCPU2CUDA = 1,
  kMemcpyCUDA2CPU = 2,
  kMemcpyCUDA2CUDA = 3,
};

class DeviceAllocator {
 public:
  explicit DeviceAllocator(DeviceType device_type) : device_type_(device_type) {}

  /**
   * @brief 设备类型
   */
  virtual DeviceType device_type() const { return device_type_; }

  /**
   * @brief 释放资源
   */
  virtual void release(void* ptr) const = 0;

  /**
   * @brief 申请资源
   * @param byte_size 内存大小
   */
  virtual void* allocate(size_t byte_size) const = 0;

  /**
   * @brief 内存拷贝，用于不同存储之间复制数据
   */
  virtual void memcpy(const void* src_ptr, void* dest_ptr, size_t byte_size,
                      MemcpyKind memcpy_kind = MemcpyKind::kMemcpyCPU2CPU, void* stream = nullptr,
                      bool need_sync = false) const;

  /**
   * @brief 内存清零
   */
  virtual void memset_zero(void* ptr, size_t byte_size, void* stream, bool need_sync = false);

 private:
  DeviceType device_type_ = DeviceType::kDeviceUnknown;  ///< 设备类型
};

class CPUDeviceAllocator : public DeviceAllocator {
 public:
  explicit CPUDeviceAllocator();

  void* allocate(size_t byte_size) const override;

  void release(void* ptr) const override;
};

struct CudaMemoryBuffer {
  void* data;        ///< 存储的数据
  size_t byte_size;  ///< 数据长度
  bool busy;         ///< 是否被使用

  CudaMemoryBuffer() = default;

  CudaMemoryBuffer(void* data, size_t byte_size, bool busy)
      : data(data), byte_size(byte_size), busy(busy) {}
};

class CUDADeviceAllocator : public DeviceAllocator {
 public:
  explicit CUDADeviceAllocator();

  void* allocate(size_t byte_size) const override;

  void release(void* ptr) const override;

 private:
  mutable std::map<int, size_t> no_busy_cnt_;
  mutable std::map<int, std::vector<CudaMemoryBuffer>> big_buffers_map_;
  mutable std::map<int, std::vector<CudaMemoryBuffer>> cuda_buffers_map_;
};

class CPUDeviceAllocatorFactory {
 public:
  static std::shared_ptr<CPUDeviceAllocator> get_instance() {
    if (instance == nullptr) {
      instance = std::make_shared<CPUDeviceAllocator>();
    }
    return instance;
  }

 private:
  /// @brief 静态成员变量，存储单例实例
  static std::shared_ptr<CPUDeviceAllocator> instance;
};

class CUDADeviceAllocatorFactory {
 public:
  static std::shared_ptr<CUDADeviceAllocator> get_instance() {
    if (instance == nullptr) {
      instance = std::make_shared<CUDADeviceAllocator>();
    }
    return instance;
  }

 private:
  static std::shared_ptr<CUDADeviceAllocator> instance;
};

class DeviceAlloctorFactory {
 public:
  static std::shared_ptr<DeviceAllocator> get_instance(base::DeviceType device_type) {
    if (device_type == base::DeviceType::kDeviceCPU) {
      return CPUDeviceAllocatorFactory::get_instance();
    } else if (device_type == base::DeviceType::kDeviceCUDA) {
      return CPUDeviceAllocatorFactory::get_instance();
    } else {
      LOG(FATAL) << "This device type of allocator is not supported!";
      return nullptr;
    }
  }
};

}  // namespace base
#endif  // KUIPER_INCLUDE_BASE_ALLOC_H_