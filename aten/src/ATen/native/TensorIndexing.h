#pragma once

#include <c10/util/Optional.h>
#include <ATen/core/TensorBody.h>

namespace at {
namespace indexing {

enum class SliceIndexType { None, Integer };
enum class TensorIndexType { None, Ellipsis, Integer, Boolean, Slice, Tensor };

struct NoneIndexType { NoneIndexType() {} };
struct EllipsisIndexType { EllipsisIndexType() {} };

CAFFE2_API extern const NoneIndexType None;
CAFFE2_API extern const EllipsisIndexType Ellipsis;

// yf225 TODO: can we use `c10::optional<int64_t>` and remove this class?
struct CAFFE2_API SliceIndex {
  SliceIndex(at::indexing::NoneIndexType none);
  SliceIndex(int64_t integer);

  bool is_none() const;
  bool is_integer() const;
  int64_t integer() const;

 private:
  int64_t integer_;
  SliceIndexType type_;
};

struct CAFFE2_API Slice {
 public:
  Slice();
  Slice(int64_t start, int64_t stop, int64_t step);

  const int64_t& start() const;
  const int64_t& stop() const;
  const int64_t& step() const;

 private:
  int64_t start_;
  int64_t stop_;
  int64_t step_;
};

std::ostream& operator<<(std::ostream& stream, const Slice& slice);

struct CAFFE2_API TensorIndex {
  TensorIndex(at::indexing::NoneIndexType);
  TensorIndex(at::indexing::EllipsisIndexType);
  TensorIndex(const char *str);
  TensorIndex(int integer);
  TensorIndex(bool boolean);
  TensorIndex(std::initializer_list<SliceIndex> init_list);
  TensorIndex(Tensor tensor);

  bool is_none() const;
  bool is_ellipsis() const;

  bool is_integer() const;
  int64_t integer() const;

  bool is_boolean() const;
  bool boolean() const;

  bool is_slice() const;
  const Slice& slice() const;

  bool is_tensor() const;
  const Tensor& tensor() const;

 private:
  int64_t integer_;
  bool boolean_;
  Slice slice_;
  Tensor tensor_;
  TensorIndexType type_;
};

std::ostream& operator<<(std::ostream& stream, const TensorIndex& tensor_index);
std::ostream& operator<<(std::ostream& stream, const std::vector<TensorIndex>& tensor_indices);

} // namespace indexing
} // namespace at
