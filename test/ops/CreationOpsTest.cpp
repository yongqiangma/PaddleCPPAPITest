#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/full.h>
#include <ATen/ops/ones.h>
#include <ATen/ops/zeros.h>
#include <gtest/gtest.h>

#include <vector>

namespace at {
namespace test {

class CreationOpsTest : public ::testing::Test {
 protected:
  void SetUp() override {}
};

TEST_F(CreationOpsTest, ZerosBasic) {
  std::vector<int64_t> shape = {2, 3};
  at::Tensor result = at::zeros(shape);

  EXPECT_EQ(result.dim(), 2);
  EXPECT_EQ(result.numel(), 6);
  EXPECT_EQ(result.sizes()[0], 2);
  EXPECT_EQ(result.sizes()[1], 3);

  float* data = result.data_ptr<float>();
  for (int64_t i = 0; i < 6; ++i) {
    EXPECT_FLOAT_EQ(data[i], 0.0f);
  }
}

TEST_F(CreationOpsTest, ZerosWithOptions) {
  at::Tensor result = at::zeros({3, 4}, at::TensorOptions().dtype(at::kDouble));

  EXPECT_EQ(result.dim(), 2);
  EXPECT_EQ(result.numel(), 12);
  EXPECT_EQ(result.dtype(), at::kDouble);

  double* data = result.data_ptr<double>();
  for (int64_t i = 0; i < 12; ++i) {
    EXPECT_DOUBLE_EQ(data[i], 0.0);
  }
}

TEST_F(CreationOpsTest, OnesBasic) {
  at::Tensor result = at::ones({2, 2});

  EXPECT_EQ(result.dim(), 2);
  EXPECT_EQ(result.numel(), 4);

  float* data = result.data_ptr<float>();
  for (int64_t i = 0; i < 4; ++i) {
    EXPECT_FLOAT_EQ(data[i], 1.0f);
  }
}

TEST_F(CreationOpsTest, OnesWithOptions) {
  at::Tensor result = at::ones({3}, at::TensorOptions().dtype(at::kInt));

  EXPECT_EQ(result.dim(), 1);
  EXPECT_EQ(result.numel(), 3);
  EXPECT_EQ(result.dtype(), at::kInt);

  int* data = result.data_ptr<int>();
  for (int64_t i = 0; i < 3; ++i) {
    EXPECT_EQ(data[i], 1);
  }
}

TEST_F(CreationOpsTest, EmptyBasic) {
  at::Tensor result = at::empty({2, 3});

  EXPECT_EQ(result.dim(), 2);
  EXPECT_EQ(result.numel(), 6);
  EXPECT_EQ(result.sizes()[0], 2);
  EXPECT_EQ(result.sizes()[1], 3);
  EXPECT_NE(result.data_ptr(), nullptr);
}

TEST_F(CreationOpsTest, EmptyWithOptions) {
  at::Tensor result = at::empty({4}, at::TensorOptions().dtype(at::kFloat));

  EXPECT_EQ(result.dim(), 1);
  EXPECT_EQ(result.numel(), 4);
  EXPECT_EQ(result.dtype(), at::kFloat);
  EXPECT_NE(result.data_ptr<float>(), nullptr);
}

TEST_F(CreationOpsTest, FullBasic) {
  at::Tensor result = at::full({2, 2}, 5.0f);

  EXPECT_EQ(result.dim(), 2);
  EXPECT_EQ(result.numel(), 4);

  float* data = result.data_ptr<float>();
  for (int64_t i = 0; i < 4; ++i) {
    EXPECT_FLOAT_EQ(data[i], 5.0f);
  }
}

TEST_F(CreationOpsTest, FullWithOptions) {
  at::Tensor result = at::full({3}, 10, at::TensorOptions().dtype(at::kLong));

  EXPECT_EQ(result.dim(), 1);
  EXPECT_EQ(result.numel(), 3);
  EXPECT_EQ(result.dtype(), at::kLong);

  int64_t* data = result.data_ptr<int64_t>();
  for (int64_t i = 0; i < 3; ++i) {
    EXPECT_EQ(data[i], 10);
  }
}

}  // namespace test
}  // namespace at
