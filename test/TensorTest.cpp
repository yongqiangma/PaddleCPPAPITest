#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <ATen/ops/ones.h>
#include <gtest/gtest.h>
#include <torch/all.h>

#include <vector>

namespace at {
namespace test {

class TensorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    std::vector<int64_t> shape = {2, 3, 4};

    tensor = at::ones(shape, at::kFloat);
    // std::cout << "tensor dim: " << tensor.dim() << std::endl;
  }

  at::Tensor tensor;
};

TEST_F(TensorTest, ConstructFromPaddleTensor) {
  EXPECT_EQ(tensor.dim(), 3);
  EXPECT_EQ(tensor.numel(), 24);  // 2*3*4
}

// 测试 data_ptr
TEST_F(TensorTest, DataPtr) {
  // Tensor tensor(paddle_tensor_);

  void* ptr = tensor.data_ptr();
  EXPECT_NE(ptr, nullptr);

  float* float_ptr = tensor.data_ptr<float>();
  EXPECT_NE(float_ptr, nullptr);
}

// 测试 strides
TEST_F(TensorTest, Strides) {
  // Tensor tensor(paddle_tensor_);

  c10::IntArrayRef strides = tensor.strides();
  EXPECT_GT(strides.size(), 0U);  // 使用无符号字面量
}

// 测试 sizes
TEST_F(TensorTest, Sizes) {
  // Tensor tensor(paddle_tensor_);

  c10::IntArrayRef sizes = tensor.sizes();
  EXPECT_EQ(sizes.size(), 3U);
  EXPECT_EQ(sizes[0], 2U);
  EXPECT_EQ(sizes[1], 3U);
  EXPECT_EQ(sizes[2], 4U);
}

// 测试 toType
TEST_F(TensorTest, ToType) {
  // Tensor tensor(paddle_tensor_);

  Tensor double_tensor = tensor.toType(c10::ScalarType::Double);
  EXPECT_EQ(double_tensor.dtype(), c10::ScalarType::Double);
}

// 测试 numel
TEST_F(TensorTest, Numel) {
  // Tensor tensor(paddle_tensor_);

  EXPECT_EQ(tensor.numel(), 24U);  // 2*3*4
}

// 测试 device
TEST_F(TensorTest, Device) {
  // Tensor tensor(paddle_tensor_);

  c10::Device device = tensor.device();
  EXPECT_EQ(device.type(), c10::DeviceType::CPU);
}

// 测试 get_device
TEST_F(TensorTest, GetDevice) {
  // Tensor tensor(paddle_tensor_);

  c10::DeviceIndex device_idx = tensor.get_device();
  EXPECT_GE(device_idx, -1);
}

// 测试 dim 和 ndimension
TEST_F(TensorTest, DimAndNdimension) {
  // Tensor tensor(paddle_tensor_);

  EXPECT_EQ(tensor.dim(), 3);
  EXPECT_EQ(tensor.ndimension(), 3);
  EXPECT_EQ(tensor.dim(), tensor.ndimension());
}

// 测试 contiguous
TEST_F(TensorTest, Contiguous) {
  // Tensor tensor(paddle_tensor_);

  at::Tensor cont_tensor = tensor.contiguous();
  EXPECT_TRUE(cont_tensor.is_contiguous());
}

// 测试 is_contiguous
TEST_F(TensorTest, IsContiguous) {
  // Tensor tensor(paddle_tensor_);

  EXPECT_TRUE(tensor.is_contiguous());
}

// 测试 scalar_type
TEST_F(TensorTest, ScalarType) {
  // Tensor tensor(paddle_tensor_);

  c10::ScalarType stype = tensor.scalar_type();
  EXPECT_EQ(stype, c10::ScalarType::Float);
}

// 测试 fill_
TEST_F(TensorTest, Fill) {
  // Tensor tensor(paddle_tensor_);

  tensor.fill_(5.0);
  float* data = tensor.data_ptr<float>();
  EXPECT_FLOAT_EQ(data[0], 5.0f);
}

// 测试 zero_
TEST_F(TensorTest, Zero) {
  // Tensor tensor(paddle_tensor_);

  tensor.zero_();
  float* data = tensor.data_ptr<float>();
  EXPECT_FLOAT_EQ(data[0], 0.0f);
}

// 测试 is_cpu
TEST_F(TensorTest, IsCpu) {
  // Tensor tensor(paddle_tensor_);

  EXPECT_TRUE(tensor.is_cpu());
}

// 测试 is_cuda (在 CPU tensor 上应该返回 false)
TEST_F(TensorTest, IsCuda) {
  // Tensor tensor(paddle_tensor_);

  EXPECT_FALSE(tensor.is_cuda());
}

// 测试 reshape
TEST_F(TensorTest, Reshape) {
  // Tensor tensor(paddle_tensor_);

  at::Tensor reshaped = tensor.reshape({6, 4});
  EXPECT_EQ(reshaped.sizes()[0], 6);
  EXPECT_EQ(reshaped.sizes()[1], 4);
  EXPECT_EQ(reshaped.numel(), 24);
}

// 测试 transpose
TEST_F(TensorTest, Transpose) {
  // Tensor tensor(paddle_tensor_);

  at::Tensor transposed = tensor.transpose(0, 2);
  EXPECT_EQ(transposed.sizes()[0], 4);
  EXPECT_EQ(transposed.sizes()[2], 2);
}

// 测试 squeeze
TEST_F(TensorTest, Squeeze) {
  // 创建一个包含大小为1的维度的tensor: shape = {2, 1, 3, 1, 4}
  at::Tensor tensor_with_ones = at::ones({2, 1, 3, 1, 4}, at::kFloat);

  // 移除所有大小为1的维度
  at::Tensor squeezed = tensor_with_ones.squeeze();
  EXPECT_EQ(squeezed.dim(), 3);
  EXPECT_EQ(squeezed.sizes()[0], 2);
  EXPECT_EQ(squeezed.sizes()[1], 3);
  EXPECT_EQ(squeezed.sizes()[2], 4);
  EXPECT_EQ(squeezed.numel(), 24);

  // 移除指定维度（维度1，大小为1）
  at::Tensor squeezed_dim1 = tensor_with_ones.squeeze(1);
  EXPECT_EQ(squeezed_dim1.dim(), 4);
  EXPECT_EQ(squeezed_dim1.sizes()[0], 2);
  EXPECT_EQ(squeezed_dim1.sizes()[1], 3);
  EXPECT_EQ(squeezed_dim1.sizes()[2], 1);
  EXPECT_EQ(squeezed_dim1.sizes()[3], 4);
}

// 测试 squeeze_ (原位操作)
TEST_F(TensorTest, SqueezeInplace) {
  // 创建一个包含大小为1的维度的tensor: shape = {2, 1, 3, 1, 4}
  at::Tensor tensor_with_ones = at::ones({2, 1, 3, 1, 4}, at::kFloat);

  // 记录原始数据指针
  void* original_ptr = tensor_with_ones.data_ptr();

  // 原位移除所有大小为1的维度
  tensor_with_ones.squeeze_();
  EXPECT_EQ(tensor_with_ones.dim(), 3);
  EXPECT_EQ(tensor_with_ones.sizes()[0], 2);
  EXPECT_EQ(tensor_with_ones.sizes()[1], 3);
  EXPECT_EQ(tensor_with_ones.sizes()[2], 4);
  EXPECT_EQ(tensor_with_ones.numel(), 24);

  // 验证是原位操作（数据指针未改变）
  EXPECT_EQ(tensor_with_ones.data_ptr(), original_ptr);

  // 测试原位移除指定维度
  at::Tensor tensor_with_ones2 = at::ones({2, 1, 3, 1, 4}, at::kFloat);
  tensor_with_ones2.squeeze_(1);
  EXPECT_EQ(tensor_with_ones2.dim(), 4);
  EXPECT_EQ(tensor_with_ones2.sizes()[1], 3);
}

// 测试 unsqueeze
TEST_F(TensorTest, Unsqueeze) {
  // 在维度0之前添加一个大小为1的维度
  at::Tensor unsqueezed0 = tensor.unsqueeze(0);
  EXPECT_EQ(unsqueezed0.dim(), 4);
  EXPECT_EQ(unsqueezed0.sizes()[0], 1);
  EXPECT_EQ(unsqueezed0.sizes()[1], 2);
  EXPECT_EQ(unsqueezed0.sizes()[2], 3);
  EXPECT_EQ(unsqueezed0.sizes()[3], 4);
  EXPECT_EQ(unsqueezed0.numel(), 24);

  // 在维度2之前添加一个大小为1的维度
  at::Tensor unsqueezed2 = tensor.unsqueeze(2);
  EXPECT_EQ(unsqueezed2.dim(), 4);
  EXPECT_EQ(unsqueezed2.sizes()[0], 2);
  EXPECT_EQ(unsqueezed2.sizes()[1], 3);
  EXPECT_EQ(unsqueezed2.sizes()[2], 1);
  EXPECT_EQ(unsqueezed2.sizes()[3], 4);

  // 在最后添加一个大小为1的维度（使用负索引-1）
  at::Tensor unsqueezed_last = tensor.unsqueeze(-1);
  EXPECT_EQ(unsqueezed_last.dim(), 4);
  EXPECT_EQ(unsqueezed_last.sizes()[0], 2);
  EXPECT_EQ(unsqueezed_last.sizes()[1], 3);
  EXPECT_EQ(unsqueezed_last.sizes()[2], 4);
  EXPECT_EQ(unsqueezed_last.sizes()[3], 1);
}

// 测试 unsqueeze_ (原位操作)
TEST_F(TensorTest, UnsqueezeInplace) {
  // 创建一个新的tensor用于原位操作
  at::Tensor test_tensor = at::ones({2, 3, 4}, at::kFloat);

  // 记录原始数据指针
  void* original_ptr = test_tensor.data_ptr();

  // 原位在维度0之前添加一个大小为1的维度
  test_tensor.unsqueeze_(0);
  EXPECT_EQ(test_tensor.dim(), 4);
  EXPECT_EQ(test_tensor.sizes()[0], 1);
  EXPECT_EQ(test_tensor.sizes()[1], 2);
  EXPECT_EQ(test_tensor.sizes()[2], 3);
  EXPECT_EQ(test_tensor.sizes()[3], 4);
  EXPECT_EQ(test_tensor.numel(), 24);

  // 验证是原位操作（数据指针未改变）
  EXPECT_EQ(test_tensor.data_ptr(), original_ptr);

  // 测试使用负索引的原位操作
  at::Tensor test_tensor2 = at::ones({2, 3, 4}, at::kFloat);
  test_tensor2.unsqueeze_(-1);
  EXPECT_EQ(test_tensor2.dim(), 4);
  EXPECT_EQ(test_tensor2.sizes()[3], 1);
}

}  // namespace test
}  // namespace at
