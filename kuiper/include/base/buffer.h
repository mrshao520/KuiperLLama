#ifndef KUIPER_INCLUDE_BASE_BUFFER_H_
#define KUIPER_INCLUDE_BASE_BUFFER_H_
#include <memory>
#include "base/alloc.h"
namespace base {
/**
 * @brief 管理内存缓冲区
 * NoCopyable 表示不能被复制
 * enable_shared_from_this 表示可以从 this 指针创建 shared_ptr
 */
class Buffer : public NoCopyable, std::enable_shared_from_this<Buffer> {
 private:
  size_t byte_size_ = 0;                                 ///< 缓冲区的字节大小
  void* ptr_ = nullptr;                                  ///< 指向分配的内存的指针
  bool use_external_ = false;                            ///< 是否使用外部提供的内存
  DeviceType device_type_ = DeviceType::kDeviceUnknown;  ///< 设备类型
  std::shared_ptr<DeviceAllocator> allocator_;  ///< 指向设备分配器的共享指针，用于分配和释放内存

 public:
  explicit Buffer() = default;

  explicit Buffer(size_t byte_size, std::shared_ptr<DeviceAllocator> allocator = nullptr,
                  void* ptr = nullptr, bool use_external = false);

  virtual ~Buffer();

  bool allocate();

  /// @brief 从另一个缓冲区复制数据的函数
  /// @param buffer
  void copy_from(const Buffer& buffer) const;

  /// @brief 从另一个缓冲区复制数据的函数
  /// @param buffer
  void copy_from(const Buffer* buffer) const;

  void* ptr();

  const void* ptr() const;

  size_t byte_size() const;

  /// @brief 返回分配器的共享指针
  std::shared_ptr<DeviceAllocator> allocator() const;

  DeviceType device_type() const;

  void set_device_type(DeviceType device_type);

  /// @brief 返回当前对象的 shared_ptr
  std::shared_ptr<Buffer> get_shared_from_this();

  bool is_external() const;
};
}  // namespace base

#endif