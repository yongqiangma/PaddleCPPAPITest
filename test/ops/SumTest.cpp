#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <ATen/ops/sum.h>
#include <ATen/ops/zeros.h>
#include <gtest/gtest.h>

#include <vector>

namespace at {
namespace test {

class SumTest : public ::testing::Test {
 protected:
  void SetUp() override {
    std::vector<int64_t> shape = {2, 3};
    test_tensor = at::zeros(shape, at::kFloat);
    float* data = test_tensor.data_ptr<float>();
    for (int64_t i = 0; i < 6; ++i) {
      data[i] = static_cast<float>(i + 1);
    }
  }
  at::Tensor test_tensor;
};

TEST_F(SumTest, SumAllElements) {
  at::Tensor result = at::sum(test_tensor);
  EXPECT_EQ(result.dim(), 0);
  EXPECT_EQ(result.numel(), 1);

  float result_value = *result.data_ptr<float>();
  EXPECT_FLOAT_EQ(result_value, 21.0f);  // 1+2+3+4+5+6 = 21
}

TEST_F(SumTest, SumWithDtype) {
  at::Tensor result = at::sum(test_tensor, at::kDouble);
  EXPECT_EQ(result.dim(), 0);
  EXPECT_EQ(result.dtype(), at::kDouble);

  double result_value = *result.data_ptr<double>();
  EXPECT_DOUBLE_EQ(result_value, 21.0);
}

TEST_F(SumTest, SumAlongDim0) {
  at::Tensor result = at::sum(test_tensor, {0}, false);
  EXPECT_EQ(result.dim(), 1);
  EXPECT_EQ(result.numel(), 3);

  float* data = result.data_ptr<float>();
  EXPECT_FLOAT_EQ(data[0], 5.0f);  // 1+4
  EXPECT_FLOAT_EQ(data[1], 7.0f);  // 2+5
  EXPECT_FLOAT_EQ(data[2], 9.0f);  // 3+6
}

TEST_F(SumTest, SumAlongDim1) {
  at::Tensor result = at::sum(test_tensor, {1}, false);
  EXPECT_EQ(result.dim(), 1);
  EXPECT_EQ(result.numel(), 2);

  float* data = result.data_ptr<float>();
  EXPECT_FLOAT_EQ(data[0], 6.0f);   // 1+2+3
  EXPECT_FLOAT_EQ(data[1], 15.0f);  // 4+5+6
}

TEST_F(SumTest, SumWithKeepdim) {
  at::Tensor result = at::sum(test_tensor, {0}, true);
  EXPECT_EQ(result.dim(), 2);
  EXPECT_EQ(result.numel(), 3);
  EXPECT_EQ(result.sizes()[0], 1);
  EXPECT_EQ(result.sizes()[1], 3);

  float* data = result.data_ptr<float>();
  EXPECT_FLOAT_EQ(data[0], 5.0f);
  EXPECT_FLOAT_EQ(data[1], 7.0f);
  EXPECT_FLOAT_EQ(data[2], 9.0f);
}

TEST_F(SumTest, SumOutFunction) {
  at::Tensor output = at::zeros({}, at::kFloat);
  at::Tensor& result = at::sum_out(output, test_tensor);

  EXPECT_EQ(&result, &output);
  float result_value = *output.data_ptr<float>();
  EXPECT_FLOAT_EQ(result_value, 21.0f);
}

}  // namespace test
}  // namespace at
