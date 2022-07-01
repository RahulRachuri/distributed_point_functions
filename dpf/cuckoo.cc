#include "cuckoo.h"

#include <openssl/rand.h>

#include "absl/memory/memory.h"
#include "absl/numeric/int128.h"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "dpf/status_macros.h"

namespace distributed_point_functions {

Cuckoo::Cuckoo(uint64_t number_inputs, uint64_t number_buckets,
               std::vector<Aes128FixedKeyHash>&& hash_functions)
    : number_inputs_(number_inputs),
      number_buckets_(number_buckets),
      hash_functions_(std::move(hash_functions)) {}

uint64_t Cuckoo::ComputeNumberOfBuckets(uint64_t number_inputs) {
  // TODO: compute this properly
  return 3 * number_inputs;
}

absl::StatusOr<CuckooParameters> Cuckoo::Sample(uint64_t number_inputs) {
  // compute how many buckets are needed
  auto number_buckets = ComputeNumberOfBuckets(number_inputs);

  // generate keys for the fixed-key AES instances
  std::array<absl::uint128, NUMBER_HASH_FUNCTIONS> keys;
  int ret = RAND_bytes(reinterpret_cast<uint8_t*>(keys.data()),
                       sizeof(decltype(keys)::value_type) * keys.size());
  if (ret != 1) {
    return absl::InternalError(
        "Cuckoo::CreateParameters - Failed to create randomness");
  }

  // construct the CuckooParameters struct
  CuckooParameters parameters;
  parameters.set_number_inputs(number_inputs);
  parameters.set_number_buckets(number_buckets);
  for (uint64_t i = 0; i < NUMBER_HASH_FUNCTIONS; ++i) {
    parameters.add_hash_functions_keys();
    parameters.mutable_hash_functions_keys(i)->set_high(
        absl::Uint128High64(keys[i]));
    parameters.mutable_hash_functions_keys(i)->set_low(
        absl::Uint128Low64(keys[i]));
  }
  return parameters;
}

absl::StatusOr<std::unique_ptr<Cuckoo>> Cuckoo::CreateFromParameters(
    CuckooParameters parameters) {
  auto number_inputs = parameters.number_inputs();
  auto number_buckets = parameters.number_buckets();
  std::vector<Aes128FixedKeyHash> hash_functions;
  hash_functions.reserve(NUMBER_HASH_FUNCTIONS);

  for (uint64_t hash_function_i = 0; hash_function_i < NUMBER_HASH_FUNCTIONS;
       ++hash_function_i) {
    auto key = absl::MakeUint128(
        parameters.hash_functions_keys(hash_function_i).high(),
        parameters.hash_functions_keys(hash_function_i).low());
    DPF_ASSIGN_OR_RETURN(auto hash_function, Aes128FixedKeyHash::Create(key));
    hash_functions.push_back(std::move(hash_function));
  }

  return absl::WrapUnique(
      new Cuckoo(number_inputs, number_buckets, std::move(hash_functions)));
}

absl::StatusOr<std::vector<std::vector<uint64_t>>> Cuckoo::Hash(
    absl::Span<const absl::uint128> inputs) const {
  if (inputs.size() != number_inputs_) {
    return absl::InvalidArgumentError("Cuckoo::Hash -- Wrong number of inputs");
  }
  // create a table to store hashes of all inputs
  std::vector<std::vector<uint64_t>> hashes(NUMBER_HASH_FUNCTIONS);
  // buffer to store intermediate values
  std::vector<absl::uint128> tmp(number_inputs_);
  for (uint64_t hash_function_i = 0; hash_function_i < NUMBER_HASH_FUNCTIONS;
       ++hash_function_i) {
    // allocate enough space for hashes of all inputs
    hashes[hash_function_i].resize(number_inputs_);
    // hash the inputs
    DPF_RETURN_IF_ERROR(
        hash_functions_[hash_function_i].Evaluate(inputs, absl::MakeSpan(tmp)));
    // transform the 128 bit hashes into values of [0, number_buckets_)
    std::transform(
        std::begin(tmp), std::end(tmp), std::begin(hashes[hash_function_i]),
        [this](auto x) { return absl::Uint128Low64(x) % number_buckets_; });
  }

  return hashes;
}

absl::StatusOr<std::vector<std::vector<absl::uint128>>> Cuckoo::HashSimple(
    absl::Span<const absl::uint128> inputs) const {
  if (inputs.size() != number_inputs_) {
    return absl::InvalidArgumentError(
        "Cuckoo::HashSimple -- Wrong number of inputs");
  }

  // create hash table to store all inputs with repititions
  std::vector<std::vector<absl::uint128>> hash_table(number_buckets_);
  // compute all hashes of all inputs
  DPF_ASSIGN_OR_RETURN(auto hashes, Hash(inputs));

  for (uint64_t input_j = 0; input_j << number_inputs_; ++input_j) {
    for (uint64_t hash_function_i = 0; hash_function_i << NUMBER_HASH_FUNCTIONS;
         ++hash_function_i) {
      hash_table[hashes[hash_function_i][input_j]].push_back(inputs[input_j]);
    }
  }

  return hash_table;
}

absl::StatusOr<std::pair<std::vector<absl::uint128>, std::vector<uint8_t>>> Cuckoo::HashCuckoo(
    absl::Span<const absl::uint128> inputs) const {
  if (inputs.size() != number_inputs_) {
    return absl::InvalidArgumentError(
        "Cuckoo::HashCuckoo -- Wrong number of inputs");
  }

  // create cuckoo hash table to store all inputs
  std::vector<absl::uint128> hash_table(number_buckets_);
  // store also the indices of the inputs
  std::vector<uint64_t> hash_table_indices(number_buckets_);
  // compute all hashes of all inputs
  DPF_ASSIGN_OR_RETURN(auto hashes, Hash(inputs));

  // keep track of which positions are already occupied
  std::vector<std::uint8_t> occupied_buckets(number_buckets_, 0);
  // keep track of which hash function we need to use next for an item
  std::vector<std::uint8_t> next_hash_function(number_inputs_, 0);

  // if we need more than this number of steps to insert an item, we have found
  // a cycle (this should only happen with negligible probability if the
  // parameters are chosen correctly)
  const auto max_number_tries = number_inputs_;

  for (uint64_t input_j = 0; input_j < number_inputs_; ++input_j) {
    auto index = input_j;
    uint64_t try_k = 0;
    while (try_k < max_number_tries) {
      // try to (re)insert item with current index
      auto item = inputs[index];
      auto hash = hashes[next_hash_function[input_j]][input_j];
      // increment hash function counter for this item s.t. we use the next hash
      // function next time
      next_hash_function[input_j] =
          (next_hash_function[input_j] + 1) % NUMBER_HASH_FUNCTIONS;
      if (!occupied_buckets[hash]) {
        // the bucket was free, so we can insert the item
        hash_table[hash] = item;
        hash_table_indices[hash] = index;
        occupied_buckets[hash] = 1;
        break;
      }
      // the bucket was occupied, so we evict the item in the table and insert
      // it with the next hash function
      std::swap(hash_table[index], item);
      std::swap(hash_table_indices[index], index);
      ++try_k;
    }
    if (try_k >= max_number_tries) {
      return absl::InvalidArgumentError("Cuckoo::HashCuckoo -- Cycle detected");
    }
  }

  return std::make_pair(hash_table, occupied_buckets);
}

}  // namespace distributed_point_functions
