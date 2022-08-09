#ifndef DISTRIBUTED_POINT_FUNCTIONS_DPF_CUCKOO_H_
#define DISTRIBUTED_POINT_FUNCTIONS_DPF_CUCKOO_H_

#include <cstdint>
#include <memory>
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

  absl::StatusOr<std::tuple<std::vector<absl::uint128>, std::vector<uint64_t>,
                            std::vector<uint8_t>>>
  HashCuckoo(absl::Span<const absl::uint128> inputs) const;

  absl::StatusOr<std::vector<std::vector<absl::uint128>>> HashSimple(
      absl::Span<const absl::uint128> inputs) const;

  // Hashes the entire domain with the 3 hash functions into buckets
  absl::StatusOr<std::vector<std::vector<absl::uint128>>> HashSimpleDomain(
      uint64_t domain_size) const;

  const CuckooParameters& GetParameters() const { return parameters_; }
  uint64_t GetNumberInputs() const { return parameters_.number_inputs(); }
  uint64_t GetNumberBuckets() const { return parameters_.number_buckets(); }

  const static uint64_t NUMBER_HASH_FUNCTIONS = 3;

 private:
  Cuckoo(const Cuckoo&) = delete;
  Cuckoo& operator=(const Cuckoo&) = delete;
  Cuckoo(Cuckoo&&) = default;
  Cuckoo& operator=(Cuckoo&&) = default;

  Cuckoo(CuckooParameters parameters,
         std::vector<Aes128FixedKeyHash>&& hash_functions);
  uint64_t HashToBucket(uint64_t hash_function, absl::uint128 input) const;

  CuckooParameters parameters_;
  std::vector<Aes128FixedKeyHash> hash_functions_;
};

}  // namespace distributed_point_functions

#endif  // DISTRIBUTED_POINT_FUNCTIONS_DPF_CUCKOO_H_
