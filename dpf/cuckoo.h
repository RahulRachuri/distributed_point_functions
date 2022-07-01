#ifndef DISTRIBUTED_POINT_FUNCTIONS_DPF_CUCKOO_H_
#define DISTRIBUTED_POINT_FUNCTIONS_DPF_CUCKOO_H_

#include <cstdint>
#include <vector>

#include <absl/numeric/int128.h>
#include <absl/types/span.h>

#include "aes_128_fixed_key_hash.h"

namespace distributed_point_functions {

class Cuckoo {
  public:
    // create new cuckoo context
    static absl::StatusOr<Cuckoo> Create(uint64_t number_inputs);

    // create from given collection of hash functions
    static Cuckoo Create(uint64_t number_buckets, absl::Span<absl::uint128>);

    static uint64_t ComputeNumberOfBuckets(uint64_t number_inputs);
    std::vector<std::vector<uint64_t>> Hash(absl::Span<const absl::uint128> inputs) const;
    std::vector<std::vector<uint64_t>> HashCuckoo(absl::Span<const absl::uint128> inputs) const;
    std::vector<std::vector<uint64_t>> HashSimple(absl::Span<const absl::uint128> inputs) const;
  private:
    const static uint64_t NUMBER_HASH_FUNCTIONS = 3;

    Cuckoo(const Cuckoo&) = delete;
    Cuckoo& operator=(const Cuckoo&) = delete;
    Cuckoo(Cuckoo&&) = delete;
    Cuckoo& operator=(Cuckoo&&) = delete;

    Cuckoo(uint64_t number_buckets, std::vector<Aes128FixedKeyHash>&& hash_functions);
    uint64_t HashToBucket(uint64_t number_bucket, uint64_t hash_function, absl::uint128 input) const;

    uint64_t number_inputs_;
    uint64_t number_buckets_;
    std::vector<Aes128FixedKeyHash> hash_functions_;

};

}  // namespace distributed_point_functions

#endif  // DISTRIBUTED_POINT_FUNCTIONS_DPF_CUCKOO_H_
