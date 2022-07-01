#ifndef DISTRIBUTED_POINT_FUNCTIONS_DPF_CUCKOO_H_
#define DISTRIBUTED_POINT_FUNCTIONS_DPF_CUCKOO_H_

#include <cstdint>
#include <vector>

#include "absl/numeric/int128.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "aes_128_fixed_key_hash.h"
#include "dpf/distributed_point_function.pb.h"

namespace distributed_point_functions {

class Cuckoo {
 public:
  static absl::StatusOr<CuckooParameters> Sample(uint64_t number_inputs);
  static absl::StatusOr<std::unique_ptr<Cuckoo>> CreateFromParameters(
      CuckooParameters);

  static uint64_t ComputeNumberOfBuckets(uint64_t number_inputs);
  absl::StatusOr<std::vector<std::vector<uint64_t>>> Hash(
      absl::Span<const absl::uint128> inputs) const;
  absl::StatusOr<std::vector<absl::uint128>> HashCuckoo(
      absl::Span<const absl::uint128> inputs) const;
  absl::StatusOr<std::vector<std::vector<absl::uint128>>> HashSimple(
      absl::Span<const absl::uint128> inputs) const;

 private:
  const static uint64_t NUMBER_HASH_FUNCTIONS = 3;

  Cuckoo(const Cuckoo&) = delete;
  Cuckoo& operator=(const Cuckoo&) = delete;
  Cuckoo(Cuckoo&&) = default;
  Cuckoo& operator=(Cuckoo&&) = default;

  Cuckoo(uint64_t number_inputs, uint64_t number_buckets,
         std::vector<Aes128FixedKeyHash>&& hash_functions);
  uint64_t HashToBucket(uint64_t hash_function, absl::uint128 input) const;

  uint64_t number_inputs_;
  uint64_t number_buckets_;
  std::vector<Aes128FixedKeyHash> hash_functions_;
};

}  // namespace distributed_point_functions

#endif  // DISTRIBUTED_POINT_FUNCTIONS_DPF_CUCKOO_H_
