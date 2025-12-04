#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/reshape.h>
#include <ATen/ops/zeros.h>
#include <ATen/ops/zeros_like.h>
#include <gtest/gtest.h>

#include <vector>

namespace at {
namespace test {

class ReshapeTest : public ::testing::Test {
 protected:
  void SetUp() override {
    original_tensor = at::zeros({2, 3}, at::kFloat);
    float* data = original_tensor.data_ptr<float>();
    for (int64_t i = 0; i < 6; ++i) {
      data[i] = static_cast<float>(i);
    }
  }
  at::Tensor original_tensor;
};

TEST_F(ReshapeTest, Reshape2DTo1D) {
  at::Tensor result = at::reshape(original_tensor, {6});

  EXPECT_EQ(result.dim(), 1);
  EXPECT_EQ(result.numel(), 6);
  EXPECT_EQ(result.sizes()[0], 6);

  float* data = result.data_ptr<float>();
  for (int64_t i = 0; i < 6; ++i) {
    EXPECT_FLOAT_EQ(data[i], static_cast<float>(i));
  }
}

TEST_F(ReshapeTest, Reshape2DTo3D) {
  at::Tensor result = at::reshape(original_tensor, {1, 2, 3});

  EXPECT_EQ(result.dim(), 3);
  EXPECT_EQ(result.numel(), 6);
  EXPECT_EQ(result.sizes()[0], 1);
  EXPECT_EQ(result.sizes()[1], 2);
  EXPECT_EQ(result.sizes()[2], 3);

  float* data = result.data_ptr<float>();
  for (int64_t i = 0; i < 6; ++i) {
    EXPECT_FLOAT_EQ(data[i], static_cast<float>(i));
  }
}

TEST_F(ReshapeTest, ReshapeAutoInferDim) {
  at::Tensor result = at::reshape(original_tensor, {-1});

  EXPECT_EQ(result.dim(), 1);
  EXPECT_EQ(result.numel(), 6);
  EXPECT_EQ(result.sizes()[0], 6);
}

TEST_F(ReshapeTest, ReshapeInferOneDim) {
  at::Tensor result = at::reshape(original_tensor, {3, -1});

  EXPECT_EQ(result.dim(), 2);
  EXPECT_EQ(result.sizes()[0], 3);
  EXPECT_EQ(result.sizes()[1], 2);  // 6/3 = 2
}

TEST_F(ReshapeTest, EmptyLike) {
  at::Tensor result = at::empty_like(original_tensor);

  EXPECT_EQ(result.dim(), 2);
  EXPECT_EQ(result.sizes()[0], 2);
  EXPECT_EQ(result.sizes()[1], 3);
  EXPECT_EQ(result.dtype(), original_tensor.dtype());
  EXPECT_NE(result.data_ptr(), nullptr);
}

TEST_F(ReshapeTest, ZerosLike) {
  at::Tensor result = at::zeros_like(original_tensor);

  EXPECT_EQ(result.dim(), 2);
  EXPECT_EQ(result.sizes()[0], 2);
  EXPECT_EQ(result.sizes()[1], 3);
  EXPECT_EQ(result.dtype(), original_tensor.dtype());

  float* data = result.data_ptr<float>();
  for (int64_t i = 0; i < 6; ++i) {
    EXPECT_FLOAT_EQ(data[i], 0.0f);
  }
}

TEST_F(ReshapeTest, EmptyLikeWithOptions) {
  at::Tensor result =
      at::empty_like(original_tensor, at::TensorOptions().dtype(at::kDouble));

  EXPECT_EQ(result.dim(), 2);
  EXPECT_EQ(result.sizes()[0], 2);
  EXPECT_EQ(result.sizes()[1], 3);
  EXPECT_EQ(result.dtype(), at::kDouble);
}

TEST_F(ReshapeTest, ZerosLikeWithOptions) {
  at::Tensor result =
      at::zeros_like(original_tensor, at::TensorOptions().dtype(at::kInt));

  EXPECT_EQ(result.dim(), 2);
  EXPECT_EQ(result.sizes()[0], 2);
  EXPECT_EQ(result.sizes()[1], 3);
  EXPECT_EQ(result.dtype(), at::kInt);

  int* data = result.data_ptr<int>();
  for (int64_t i = 0; i < 6; ++i) {
    EXPECT_EQ(data[i], 0);
  }
}

}  // namespace test
}  // namespace at
