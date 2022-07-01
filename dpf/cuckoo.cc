#include "cuckoo.h"

#include <openssl/rand.h>

#include "dpf/status_macros.h"

namespace distributed_point_functions {

absl::StatusOr<Cuckoo> Cuckoo::Create(uint64_t number_inputs) {
    auto number_buckets = ComputeNumberOfBuckets(number_inputs);
    std::vector<Aes128FixedKeyHash> hash_functions(NUMBER_HASH_FUNCTIONS);

    std::array<absl::uint128, NUMBER_HASH_FUNCTIONS> keys;
    RAND_bytes(reinterpret_cast<uint8_t*>(keys.data()), sizeof(decltype(keys)::value_type) * keys.size());

    for (uint64_t i = 0; i < NUMBER_HASH_FUNCTIONS; ++i) {
        DPF_ASSIGN_OR_RETURN(hash_functions[i], Aes128FixedKeyHash::Create(keys[i]));
    }

    return Cuckoo(number_buckets, std::move(hash_functions));
}

}  // namespace distributed_point_functions
