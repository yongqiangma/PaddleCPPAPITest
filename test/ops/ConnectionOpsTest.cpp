#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <ATen/ops/cat.h>
#include <ATen/ops/zeros.h>
#include <gtest/gtest.h>

#include <vector>

namespace at {
namespace test {

class ConnectionOpsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    tensor1 = at::zeros({2, 3}, at::kFloat);
    tensor2 = at::zeros({2, 3}, at::kFloat);

    float* data1 = tensor1.data_ptr<float>();
    float* data2 = tensor2.data_ptr<float>();
    for (int64_t i = 0; i < 6; ++i) {
      data1[i] = static_cast<float>(i);
      data2[i] = static_cast<float>(i + 6);
    }
  }

  at::Tensor tensor1;
  at::Tensor tensor2;
};

TEST_F(ConnectionOpsTest, CatDim0) {
  std::vector<at::Tensor> tensors = {tensor1, tensor2};
  at::Tensor result = at::cat(tensors, 0);

  EXPECT_EQ(result.dim(), 2);
  EXPECT_EQ(result.sizes()[0], 4);  // 2+2
  EXPECT_EQ(result.sizes()[1], 3);
  EXPECT_EQ(result.numel(), 12);

  float* data = result.data_ptr<float>();
  for (int64_t i = 0; i < 12; ++i) {
    EXPECT_FLOAT_EQ(data[i], static_cast<float>(i));
  }
}

TEST_F(ConnectionOpsTest, CatDim1) {
  std::vector<at::Tensor> tensors = {tensor1, tensor2};
  at::Tensor result = at::cat(tensors, 1);

  EXPECT_EQ(result.dim(), 2);
  EXPECT_EQ(result.sizes()[0], 2);
  EXPECT_EQ(result.sizes()[1], 6);  // 3+3
  EXPECT_EQ(result.numel(), 12);

  float* data = result.data_ptr<float>();
  float expected_values[12] = {0, 1, 2, 6, 7, 8, 3, 4, 5, 9, 10, 11};
  for (int64_t i = 0; i < 12; ++i) {
    EXPECT_FLOAT_EQ(data[i], expected_values[i]);
  }
}

TEST_F(ConnectionOpsTest, CatThreeTensors) {
  at::Tensor tensor3 = at::zeros({2, 3}, at::kFloat);
  float* data3 = tensor3.data_ptr<float>();
  for (int64_t i = 0; i < 6; ++i) {
    data3[i] = static_cast<float>(i + 12);
  }

  std::vector<at::Tensor> tensors = {tensor1, tensor2, tensor3};
  at::Tensor result = at::cat(tensors, 0);

  EXPECT_EQ(result.sizes()[0], 6);  // 2+2+2
  EXPECT_EQ(result.sizes()[1], 3);
  EXPECT_EQ(result.numel(), 18);
}

TEST_F(ConnectionOpsTest, CatWithDifferentTypes) {
  at::Tensor int_tensor = at::zeros({1, 2}, at::kInt);
  at::Tensor float_tensor = at::zeros({1, 2}, at::kInt);

  std::vector<at::Tensor> tensors = {int_tensor, float_tensor};
  at::Tensor result = at::cat(tensors, 0);

  // Tensors should be promoted to common type
  EXPECT_EQ(result.dtype(), at::kInt);
  EXPECT_EQ(result.sizes()[0], 2);
  EXPECT_EQ(result.sizes()[1], 2);
}

}  // namespace test
}  // namespace at
