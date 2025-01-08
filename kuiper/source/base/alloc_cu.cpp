#include <cuda_runtime_api.h>
#include "base/alloc.h"
namespace base
{

  CUDADeviceAllocator::CUDADeviceAllocator() : DeviceAllocator(DeviceType::kDeviceCUDA) {}

  void *CUDADeviceAllocator::allocate(size_t byte_size) const
  {
    int id = -1;
    cudaError_t state = cudaGetDevice(&id);
    LOG(INFO) << "GPU id is " << id;
    CHECK(state == cudaSuccess);

    /// 管理超过 1024*1024 的大型内存缓冲区
    if (byte_size > 1024 * 1024)
    {
      /// 通过 id 获取对应的缓冲区列表
      auto &big_buffers = big_buffers_map_[id];

      /// 用于记录找到的最适合的缓冲区的索引，-1表示未找到
      int sel_id = -1;
      /// 遍历与当前id相关联的所有大型缓冲区
      for (int i = 0; i < big_buffers.size(); i++)
      {
        if (big_buffers[i].byte_size >= byte_size && !big_buffers[i].busy &&
            big_buffers[i].byte_size - byte_size < 1 * 1024 * 1024)
        {
          /// 第一个适合的缓冲区或者比之前的更接近所需大小
          if (sel_id == -1 || big_buffers[sel_id].byte_size > big_buffers[i].byte_size)
          {
            sel_id = i;
          }
        }
      }
      if (sel_id != -1)
      {
        big_buffers[sel_id].busy = true;
        return big_buffers[sel_id].data;
      }

      /// 未找到合适内存，使用 cudaMalloc 申请内存
      void *ptr = nullptr;
      state = cudaMalloc(&ptr, byte_size);
      if (cudaSuccess != state)
      {
        char buf[256];
        snprintf(buf, 256,
                 "Error: CUDA error when allocating %lu MB memory! maybe there's no enough memory "
                 "left on  device.",
                 byte_size >> 20);
        LOG(ERROR) << buf;
        return nullptr;
      }
      /// 如果内存分配成功，则将新的缓冲区添加到列表中，并返回新分配的内存指针
      big_buffers.emplace_back(ptr, byte_size, true);
      return ptr;
    }

    auto &cuda_buffers = cuda_buffers_map_[id];
    for (int i = 0; i < cuda_buffers.size(); i++)
    {
      if (cuda_buffers[i].byte_size >= byte_size && !cuda_buffers[i].busy)
      {
        cuda_buffers[i].busy = true;
        /// 减少 no_busy_cnt_ 映射中对应 id 的空闲内存计数
        no_busy_cnt_[id] -= cuda_buffers[i].byte_size;
        return cuda_buffers[i].data;
      }
    }

    void *ptr = nullptr;
    state = cudaMalloc(&ptr, byte_size);
    if (cudaSuccess != state)
    {
      char buf[256];
      snprintf(buf, 256,
               "Error: CUDA error when allocating %lu MB memory! maybe there's no enough memory "
               "left on  device.",
               byte_size >> 20);
      LOG(ERROR) << buf;
      return nullptr;
    }
    cuda_buffers.emplace_back(ptr, byte_size, true);

    return ptr;
  }

  void CUDADeviceAllocator::release(void *ptr) const
  {
    if (!ptr)
    {
      return;
    }
    if (cuda_buffers_map_.empty())
    {
      return;
    }

    /// 清理每个设备上超过 1GB 的空闲内存
    /// 通过释放未使用的缓冲区并将仍在使用的缓冲区保存在列表中来优化内存使用
    cudaError_t state = cudaSuccess;
    /// 遍历映射中的每个元素
    for (auto &it : cuda_buffers_map_)
    {
      /// 检查对应设备ID的空闲内存计数是否超过 1 GB
      if (no_busy_cnt_[it.first] > 1024 * 1024 * 1024)
      {
        /// 获取当前设备 ID 对应的 CUDA 缓冲区列表
        auto &cuda_buffers = it.second;
        /// 临时缓冲区，存储被使用的内存
        std::vector<CudaMemoryBuffer> temp;
        for (int i = 0; i < cuda_buffers.size(); i++)
        {
          if (!cuda_buffers[i].busy)
          {
            /// 清除未被使用的内存
            state = cudaSetDevice(it.first);
            state = cudaFree(cuda_buffers[i].data);
            CHECK(state == cudaSuccess)
                << "Error: CUDA error when release memory on device " << it.first;
          }
          else
          {
            /// 保存被使用的内存
            temp.push_back(cuda_buffers[i]);
          }
        }
        cuda_buffers.clear();
        it.second = temp;
        /// 重置对应设备 ID 的空闲内存计数为 0
        no_busy_cnt_[it.first] = 0;
      }
    }

    /// 寻找给定指针 ptr 相匹配的缓冲区，并更新缓冲区的状态
    for (auto &it : cuda_buffers_map_)
    {
      auto &cuda_buffers = it.second;
      for (int i = 0; i < cuda_buffers.size(); i++)
      {
        if (cuda_buffers[i].data == ptr)
        {
          no_busy_cnt_[it.first] += cuda_buffers[i].byte_size;
          cuda_buffers[i].busy = false;
          return;
        }
      }
      auto &big_buffers = big_buffers_map_[it.first];
      for (int i = 0; i < big_buffers.size(); i++)
      {
        if (big_buffers[i].data == ptr)
        {
          big_buffers[i].busy = false;
          return;
        }
      }
    }

    /// 对于大内存不进行释放，对于小内存进行释放
    state = cudaFree(ptr);
    CHECK(state == cudaSuccess) << "Error: CUDA error when release memory on device";
  }

  std::shared_ptr<CUDADeviceAllocator> CUDADeviceAllocatorFactory::instance = nullptr;
} // namespace base