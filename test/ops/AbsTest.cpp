#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <ATen/ops/abs.h>
#include <ATen/ops/zeros.h>
#include <gtest/gtest.h>

#include <vector>

namespace at {
namespace test {

class AbsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    std::vector<int64_t> shape = {4};
    test_tensor = at::zeros(shape, at::kFloat);
    float* data = test_tensor.data_ptr<float>();
    data[0] = 1.0f;
    data[1] = -2.0f;
    data[2] = 0.0f;
    data[3] = -3.5f;
  }
  at::Tensor test_tensor;
};

TEST_F(AbsTest, BasicAbs) {
  at::Tensor result = at::abs(test_tensor);
  EXPECT_EQ(result.sizes(), test_tensor.sizes());
  float* result_data = result.data_ptr<float>();
  EXPECT_FLOAT_EQ(result_data[0], 1.0f);
  EXPECT_FLOAT_EQ(result_data[1], 2.0f);
  EXPECT_FLOAT_EQ(result_data[2], 0.0f);
  EXPECT_FLOAT_EQ(result_data[3], 3.5f);
}

TEST_F(AbsTest, PositiveTensor) {
  at::Tensor positive_tensor = at::zeros({3}, at::kFloat);
  float* data = positive_tensor.data_ptr<float>();
  data[0] = 1.5f;
  data[1] = 3.0f;
  data[2] = 7.2f;

  at::Tensor result = at::abs(positive_tensor);
  float* result_data = result.data_ptr<float>();
  EXPECT_FLOAT_EQ(result_data[0], 1.5f);
  EXPECT_FLOAT_EQ(result_data[1], 3.0f);
  EXPECT_FLOAT_EQ(result_data[2], 7.2f);
}

TEST_F(AbsTest, NegativeTensor) {
  at::Tensor negative_tensor = at::zeros({3}, at::kFloat);
  float* data = negative_tensor.data_ptr<float>();
  data[0] = -1.5f;
  data[1] = -3.0f;
  data[2] = -7.2f;

  at::Tensor result = at::abs(negative_tensor);
  float* result_data = result.data_ptr<float>();
  EXPECT_FLOAT_EQ(result_data[0], 1.5f);
  EXPECT_FLOAT_EQ(result_data[1], 3.0f);
  EXPECT_FLOAT_EQ(result_data[2], 7.2f);
}

}  // namespace test
}  // namespace at
