#ifndef KUIPER_INCLUDE_OP_LAYER_H_
#define KUIPER_INCLUDE_OP_LAYER_H_
#include <base/cuda_config.h>
#include <string>
#include <vector>
#include "base/base.h"
#include "tensor/tensor.h"

namespace op {
class Layer;
enum class LayerType : uint8_t {
  kLayerUnknown = 0,
  kLayerLinear = 1,
  kLayerEncode = 2,
  kLayerEmbedding = 3,
  kLayerRMSNorm = 4,
  kLayerMatmul = 5,
  kLayerRoPe = 6,
  kLayerMHA = 7,
  kLayerSoftmax = 8,
  kLayerAdd = 9,
  kLayerSwiGLU = 10,
  kLayerGELU = 11,
};

class BaseLayer {
 public:
  explicit BaseLayer(base::DeviceType device_type, LayerType layer_type, base::DataType data_type,
                     std::string layer_name = "");

  base::DataType data_type() const;

  LayerType layer_type() const;

  virtual base::Status init() = 0;

  virtual base::Status forward() = 0;

  virtual base::Status forward(const tensor::Tensor& input1, const tensor::Tensor& output1) = 0;

  virtual base::Status forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                               const tensor::Tensor& output1) = 0;

  virtual base::Status forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                               const tensor::Tensor& input3, const tensor::Tensor& output1) = 0;

  virtual base::Status forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                               const tensor::Tensor& input3, const tensor::Tensor& input4,
                               const tensor::Tensor& output1) = 0;

  virtual base::Status forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                               const tensor::Tensor& input3, const tensor::Tensor& input4,
                               const tensor::Tensor& input5, const tensor::Tensor& output1) = 0;

  virtual void set_input(int32_t idx, const tensor::Tensor& input) = 0;

  virtual void set_output(int32_t idx, const tensor::Tensor& output) = 0;

  virtual size_t input_size() const = 0;

  virtual size_t output_size() const = 0;

  virtual base::Status check() const = 0;

  virtual tensor::Tensor& get_input(int32_t idx) = 0;

  virtual tensor::Tensor& get_output(int32_t idx) = 0;

  virtual const tensor::Tensor& get_input(int32_t idx) const = 0;

  virtual const tensor::Tensor& get_output(int32_t idx) const = 0;

  virtual base::Status set_weight(int32_t idx, const tensor::Tensor& weight);

  virtual base::Status set_weight(int32_t idx, const std::vector<int32_t>& dims,
                                  const void* weight_ptr,
                                  base::DeviceType device_type = base::DeviceType::kDeviceUnknown);

  const std::string& get_layer_name() const;

  void set_layer_name(const std::string& layer_name);

  base::DeviceType device_type() const;

  void set_device_type(base::DeviceType device_type);

 protected:
  std::string layer_name_;                                           ///< 存储层的名称
  LayerType layer_type_ = LayerType::kLayerUnknown;                  ///< 层的类型
  base::DataType data_type_ = base::DataType::kDataTypeUnknown;      ///< 数据类型，FP32
  base::DeviceType device_type_ = base::DeviceType::kDeviceUnknown;  ///< 设备类型
};

class Layer : public BaseLayer {
 public:
  explicit Layer(base::DeviceType device_type, LayerType layer_type, std::string layer_name = "");

  base::Status init() override;

  base::Status check_tensor(const tensor::Tensor& tensor, base::DeviceType device_type,
                            base::DataType data_type) const;

  base::Status check_tensor_with_dim(const tensor::Tensor& tensor, base::DeviceType device_type,
                                     base::DataType data_type, ...) const;

  base::Status check() const override;

  base::Status forward() override;

  base::Status forward(const tensor::Tensor& input1, const tensor::Tensor& output1) override;

  base::Status forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                       const tensor::Tensor& output1) override;

  base::Status forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                       const tensor::Tensor& input3, const tensor::Tensor& output1) override;

  base::Status forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                       const tensor::Tensor& input3, const tensor::Tensor& input4,
                       const tensor::Tensor& output1) override;

  base::Status forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                       const tensor::Tensor& input3, const tensor::Tensor& input4,
                       const tensor::Tensor& input5, const tensor::Tensor& output1) override;

  void set_input(int32_t idx, const tensor::Tensor& input) override;

  void set_output(int32_t idx, const tensor::Tensor& output) override;

  const tensor::Tensor& get_input(int32_t idx) const override;

  const tensor::Tensor& get_output(int32_t idx) const override;

  tensor::Tensor& get_input(int32_t idx) override;

  tensor::Tensor& get_output(int32_t idx) override;

  size_t input_size() const override;

  size_t output_size() const override;

  void reset_input_size(size_t size);

  void reset_output_size(size_t size);

  virtual void to_cuda();

  void set_cuda_config(std::shared_ptr<kernel::CudaConfig> config);

  std::shared_ptr<kernel::CudaConfig> cuda_config() const;

 protected:
  std::vector<tensor::Tensor> inputs_;               ///< 输入张量
  std::vector<tensor::Tensor> outputs_;              ///< 输出张量
  std::shared_ptr<kernel::CudaConfig> cuda_config_;  ///< CUDA配置信息
};

class LayerParam : public Layer {
 public:
  explicit LayerParam(base::DeviceType device_type, LayerType layer_type,
                      bool is_quant_layer = false, std::string layer_name = "");

  size_t weight_size() const;

  void reset_weight_size(size_t size);

  tensor::Tensor& get_weight(int32_t idx);

  const tensor::Tensor& get_weight(int32_t idx) const;

  void to_cuda() override;

  base::Status set_weight(int32_t idx, const tensor::Tensor& weight) override;

  base::Status set_weight(int32_t idx, const std::vector<int32_t>& dims, const void* weight_ptr,
                          base::DeviceType device_type = base::DeviceType::kDeviceUnknown) override;

  void set_scales(const tensor::Tensor& scales);

  void set_group_size(int32_t group_size);

  int32_t get_scale_num() const;

 protected:
  int32_t group_size_ = 0;
  bool is_quant_layer_ = false;  ///< 是否是量化层
  tensor::Tensor scales_;        ///< 用于存储量化层中的缩放因子，用于将量化后的数据转换为原始浮点数
  std::vector<tensor::Tensor> weights_;  ///< 权重张量
};
}  // namespace op
#endif  // KUIPER_INCLUDE_OP_LAYER_H_
