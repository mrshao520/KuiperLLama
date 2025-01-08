#include <glog/logging.h>
#include <cstdlib>
#include "base/alloc.h"

#if (defined(_POSIX_ADVISORY_INFO) && (_POSIX_ADVISORY_INFO >= 200112L))
#define KUIPER_HAVE_POSIX_MEMALIGN
#endif

namespace base
{
  CPUDeviceAllocator::CPUDeviceAllocator() : DeviceAllocator(DeviceType::kDeviceCPU) {}

  void *CPUDeviceAllocator::allocate(size_t byte_size) const
  {
    if (!byte_size)
    {
      return nullptr;
    }
#ifdef KUIPER_HAVE_POSIX_MEMALIGN
    /// 声明一个名为 data 的指针，并将其初始化为 nullptr
    void *data = nullptr;
    /// 根据 byte_size 的值来确定内存对齐的字节数
    const size_t alignment = (byte_size >= size_t(1024)) ? size_t(32) : size_t(16);
    /// posix_memalign 要求对齐参数至少和指针大小一样大
    int status = posix_memalign(
        (void **)&data, ((alignment >= sizeof(void *)) ? alignment : sizeof(void *)), byte_size);
    if (status != 0)
    {
      return nullptr;
    }
    return data;
#else
    void *data = malloc(byte_size);
    return data;
#endif
  }

  void CPUDeviceAllocator::release(void *ptr) const
  {
    if (ptr)
    {
      free(ptr);
    }
  }

  /// @brief 静态成员变量必须在类定义之外进行初始化
  std::shared_ptr<CPUDeviceAllocator> CPUDeviceAllocatorFactory::instance = nullptr;
} // namespace base