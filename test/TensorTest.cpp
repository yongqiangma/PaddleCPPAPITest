#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <ATen/ops/ones.h>
#include <gtest/gtest.h>
#include <torch/all.h>

#include <sstream>
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

// 测试 layout
TEST_F(TensorTest, Layout) {
  // 默认创建的张量应该是 strided 布局
  c10::Layout layout = tensor.layout();
  EXPECT_EQ(layout, c10::Layout::Strided);
}

// 测试 layout 常量别名
TEST_F(TensorTest, LayoutConstants) {
  // 测试 c10 命名空间下的常量别名
  EXPECT_EQ(c10::kStrided, c10::Layout::Strided);
  EXPECT_EQ(c10::kSparse, c10::Layout::Sparse);
  EXPECT_EQ(c10::kSparseCsr, c10::Layout::SparseCsr);
  EXPECT_EQ(c10::kSparseCsc, c10::Layout::SparseCsc);
  EXPECT_EQ(c10::kSparseBsr, c10::Layout::SparseBsr);
  EXPECT_EQ(c10::kSparseBsc, c10::Layout::SparseBsc);
  EXPECT_EQ(c10::kMkldnn, c10::Layout::Mkldnn);
  EXPECT_EQ(c10::kJagged, c10::Layout::Jagged);
}

// 测试 at 命名空间下的 layout 常量
TEST_F(TensorTest, LayoutConstantsInAtNamespace) {
  EXPECT_EQ(at::kStrided, c10::Layout::Strided);
  EXPECT_EQ(at::kSparse, c10::Layout::Sparse);
  EXPECT_EQ(at::kSparseCsr, c10::Layout::SparseCsr);
  EXPECT_EQ(at::kSparseCsc, c10::Layout::SparseCsc);
  EXPECT_EQ(at::kSparseBsr, c10::Layout::SparseBsr);
  EXPECT_EQ(at::kSparseBsc, c10::Layout::SparseBsc);
  EXPECT_EQ(at::kMkldnn, c10::Layout::Mkldnn);
  EXPECT_EQ(at::kJagged, c10::Layout::Jagged);
}

// 测试 torch 命名空间下的 layout 常量
TEST_F(TensorTest, LayoutConstantsInTorchNamespace) {
  EXPECT_EQ(torch::kStrided, c10::Layout::Strided);
  EXPECT_EQ(torch::kSparse, c10::Layout::Sparse);
  EXPECT_EQ(torch::kSparseCsr, c10::Layout::SparseCsr);
  EXPECT_EQ(torch::kSparseCsc, c10::Layout::SparseCsc);
  EXPECT_EQ(torch::kSparseBsr, c10::Layout::SparseBsr);
  EXPECT_EQ(torch::kSparseBsc, c10::Layout::SparseBsc);
  EXPECT_EQ(torch::kMkldnn, c10::Layout::Mkldnn);
  EXPECT_EQ(torch::kJagged, c10::Layout::Jagged);
}

// 测试 layout 枚举值
TEST_F(TensorTest, LayoutEnumValues) {
  // 测试 Layout 枚举的底层值
  EXPECT_EQ(static_cast<int8_t>(c10::Layout::Strided), 0);
  EXPECT_EQ(static_cast<int8_t>(c10::Layout::Sparse), 1);
  EXPECT_EQ(static_cast<int8_t>(c10::Layout::SparseCsr), 2);
  EXPECT_EQ(static_cast<int8_t>(c10::Layout::Mkldnn), 3);
  EXPECT_EQ(static_cast<int8_t>(c10::Layout::SparseCsc), 4);
  EXPECT_EQ(static_cast<int8_t>(c10::Layout::SparseBsr), 5);
  EXPECT_EQ(static_cast<int8_t>(c10::Layout::SparseBsc), 6);
  EXPECT_EQ(static_cast<int8_t>(c10::Layout::Jagged), 7);
  EXPECT_EQ(static_cast<int8_t>(c10::Layout::NumOptions), 8);
}

// 测试 layout 输出流操作符
TEST_F(TensorTest, LayoutOutputStream) {
  std::ostringstream oss;

  oss.str("");
  oss << c10::Layout::Strided;
  EXPECT_EQ(oss.str(), "Strided");

  oss.str("");
  oss << c10::Layout::Sparse;
  EXPECT_EQ(oss.str(), "Sparse");

  oss.str("");
  oss << c10::Layout::SparseCsr;
  EXPECT_EQ(oss.str(), "SparseCsr");

  oss.str("");
  oss << c10::Layout::SparseCsc;
  EXPECT_EQ(oss.str(), "SparseCsc");

  oss.str("");
  oss << c10::Layout::SparseBsr;
  EXPECT_EQ(oss.str(), "SparseBsr");

  oss.str("");
  oss << c10::Layout::SparseBsc;
  EXPECT_EQ(oss.str(), "SparseBsc");

  oss.str("");
  oss << c10::Layout::Mkldnn;
  EXPECT_EQ(oss.str(), "Mkldnn");

  oss.str("");
  oss << c10::Layout::Jagged;
  EXPECT_EQ(oss.str(), "Jagged");
}

// 测试使用 kStrided 常量与 tensor.layout() 比较
TEST_F(TensorTest, LayoutWithConstant) {
  // 使用常量别名进行比较
  EXPECT_EQ(tensor.layout(), at::kStrided);
  EXPECT_EQ(tensor.layout(), torch::kStrided);
  EXPECT_EQ(tensor.layout(), c10::kStrided);

  // 确保不是其他布局类型
  EXPECT_NE(tensor.layout(), at::kSparse);
  EXPECT_NE(tensor.layout(), at::kSparseCsr);
  EXPECT_NE(tensor.layout(), at::kMkldnn);
}

}  // namespace test
}  // namespace at
