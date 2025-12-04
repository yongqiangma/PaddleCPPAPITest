#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <ATen/ops/arange.h>
#include <gtest/gtest.h>

#include <vector>

namespace at {
namespace test {

class ArangeTest : public ::testing::Test {
 protected:
  void SetUp() override {}
};

TEST_F(ArangeTest, BasicArangeWithEnd) {
  at::Tensor result = at::arange(5, at::TensorOptions().dtype(at::kLong));
  EXPECT_EQ(result.dim(), 1);
  EXPECT_EQ(result.numel(), 5);

  int64_t* data = result.data_ptr<int64_t>();
  for (int64_t i = 0; i < 5; ++i) {
    EXPECT_EQ(data[i], i);
  }
}

TEST_F(ArangeTest, ArangeWithStartEnd) {
  at::Tensor result = at::arange(2, 7, at::TensorOptions().dtype(at::kLong));
  EXPECT_EQ(result.dim(), 1);
  EXPECT_EQ(result.numel(), 5);

  int64_t* data = result.data_ptr<int64_t>();
  for (int64_t i = 0; i < 5; ++i) {
    EXPECT_EQ(data[i], i + 2);
  }
}

TEST_F(ArangeTest, ArangeWithStartEndStep) {
  at::Tensor result =
      at::arange(1, 10, 2, at::TensorOptions().dtype(at::kLong));
  EXPECT_EQ(result.dim(), 1);
  EXPECT_EQ(result.numel(), 5);

  int64_t* data = result.data_ptr<int64_t>();
  for (int64_t i = 0; i < 5; ++i) {
    EXPECT_EQ(data[i], 1 + i * 2);
  }
}

TEST_F(ArangeTest, ArangeWithOptions) {
  at::Tensor result = at::arange(4, at::TensorOptions().dtype(at::kFloat));
  EXPECT_EQ(result.dim(), 1);
  EXPECT_EQ(result.numel(), 4);
  EXPECT_EQ(result.dtype(), at::kFloat);

  float* data = result.data_ptr<float>();
  for (int64_t i = 0; i < 4; ++i) {
    EXPECT_FLOAT_EQ(data[i], static_cast<float>(i));
  }
}

TEST_F(ArangeTest, NegativeValues) {
  at::Tensor result = at::arange(-3, 3, at::TensorOptions().dtype(at::kLong));
  EXPECT_EQ(result.dim(), 1);
  EXPECT_EQ(result.numel(), 6);

  int64_t* data = result.data_ptr<int64_t>();
  for (int64_t i = 0; i < 6; ++i) {
    EXPECT_EQ(data[i], i - 3);
  }
}

}  // namespace test
}  // namespace at
