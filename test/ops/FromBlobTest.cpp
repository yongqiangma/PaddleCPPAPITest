#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <ATen/ops/from_blob.h>
#include <gtest/gtest.h>

#include <vector>

namespace at {
namespace test {

class FromBlobTest : public ::testing::Test {
 protected:
  void SetUp() override {
    data_buffer = new float[6];
    for (int i = 0; i < 6; ++i) {
      data_buffer[i] = static_cast<float>(i);
    }
  }

  void TearDown() override { delete[] data_buffer; }

  float* data_buffer;
};

TEST_F(FromBlobTest, FromBlobBasic) {
  std::vector<int64_t> sizes = {2, 3};
  at::Tensor result = at::from_blob(data_buffer, sizes);

  EXPECT_EQ(result.dim(), 2);
  EXPECT_EQ(result.sizes()[0], 2);
  EXPECT_EQ(result.sizes()[1], 3);
  EXPECT_EQ(result.data_ptr<float>(), data_buffer);

  float* data = result.data_ptr<float>();
  for (int i = 0; i < 6; ++i) {
    EXPECT_FLOAT_EQ(data[i], static_cast<float>(i));
  }
}

TEST_F(FromBlobTest, FromBlobWithOptions) {
  std::vector<int64_t> sizes = {3, 2};
  at::Tensor result =
      at::from_blob(data_buffer, sizes, at::TensorOptions().dtype(at::kDouble));

  EXPECT_EQ(result.dim(), 2);
  EXPECT_EQ(result.sizes()[0], 3);
  EXPECT_EQ(result.sizes()[1], 2);
  EXPECT_EQ(result.dtype(), at::kDouble);
}

TEST_F(FromBlobTest, FromBlobWithStrides) {
  std::vector<int64_t> sizes = {2, 3};
  std::vector<int64_t> strides = {3, 1};  // Row-major
  at::Tensor result = at::from_blob(data_buffer, sizes, strides);

  EXPECT_EQ(result.dim(), 2);
  EXPECT_EQ(result.strides()[0], 3);
  EXPECT_EQ(result.strides()[1], 1);
}

TEST_F(FromBlobTest, FromBlob1D) {
  std::vector<int64_t> sizes = {6};
  at::Tensor result = at::from_blob(data_buffer, sizes);

  EXPECT_EQ(result.dim(), 1);
  EXPECT_EQ(result.numel(), 6);
  EXPECT_EQ(result.data_ptr<float>(), data_buffer);
}

TEST_F(FromBlobTest, FromBlobDifferentDataTypes) {
  int* int_data = new int[4]{1, 2, 3, 4};

  std::vector<int64_t> sizes = {2, 2};
  at::Tensor result =
      at::from_blob(int_data, sizes, at::TensorOptions().dtype(at::kInt));

  EXPECT_EQ(result.dtype(), at::kInt);
  EXPECT_EQ(result.numel(), 4);

  int* data = result.data_ptr<int>();
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(data[i], i + 1);
  }

  delete[] int_data;
}

}  // namespace test
}  // namespace at
