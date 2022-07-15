#include <openssl/rand.h>

#include "absl/memory/memory.h"
#include "absl/numeric/int128.h"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "cuckoo.h"
#include "dpf/status_macros.h"

namespace distributed_point_functions {

const uint64_t Cuckoo::NUMBER_HASH_FUNCTIONS;

Cuckoo::Cuckoo(CuckooParameters parameters,
               std::vector<Aes128FixedKeyHash>&& hash_functions)
    : parameters_(std::move(parameters)),
      hash_functions_(std::move(hash_functions)) {}

uint64_t Cuckoo::ComputeNumberOfBuckets(uint64_t number_inputs) {
  // return 2 * number_inputs;
  // taken from here:
  // https://github.com/ladnir/cryptoTools/blob/bd5ed567cc97022a2e30601986679c83a823eb84/cryptoTools/Common/CuckooIndex.cpp#L131-L145

  // double a = 240;
  // double b = -std::log2(number_inputs) - 256;
  //
  // const auto statSecParam = 40;
  // auto e = (statSecParam - b) / a + 0.3;
  // ^ does not work

  // taken from here:
  // https://github.com/ladnir/cryptoTools/blob/85da63e335c3ad3019af3958b48d3ff6750c3d92/cryptoTools/Common/CuckooIndex.cpp#L129-L150
  //
  double log_number_inputs = std::log2(number_inputs);
  double aMax = 123.5;
  double bMax = -130;
  double aSD = 2.3;
  double bSD = 2.18;
  double aMean = 6.3;
  double bMean = 6.45;
  const auto statSecParam = 40;
  double a =
      aMax / 2 * (1 + erf((log_number_inputs - aMean) / (aSD * std::sqrt(2))));
  double b =
      bMax / 2 * (1 + erf((log_number_inputs - bMean) / (bSD * std::sqrt(2)))) -
      log_number_inputs;
  auto e = (statSecParam - b) / a + 0.3;

  // we have the statSecParam = a e + b, where e = |cuckoo|/|set| is the
  // expenation factor therefore we have that
  //
  //   e = (statSecParam - b) / a
  //
  return std::ceil(e * number_inputs);
}

absl::StatusOr<CuckooParameters> Cuckoo::Sample(uint64_t number_inputs) {
  // compute how many buckets are needed
  const auto number_buckets = ComputeNumberOfBuckets(number_inputs);

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
      new Cuckoo(std::move(parameters), std::move(hash_functions)));
}

absl::StatusOr<std::vector<std::vector<uint64_t>>> Cuckoo::Hash(
    absl::Span<const absl::uint128> inputs) const {
  const auto number_inputs = GetNumberInputs();
  const auto number_buckets = GetNumberBuckets();
  if (inputs.size() != number_inputs) {
    return absl::InvalidArgumentError("Cuckoo::Hash -- Wrong number of inputs");
  }
  // create a table to store hashes of all inputs
  std::vector<std::vector<uint64_t>> hashes(NUMBER_HASH_FUNCTIONS);
  // buffer to store intermediate values
  std::vector<absl::uint128> tmp(number_inputs);
  for (uint64_t hash_function_i = 0; hash_function_i < NUMBER_HASH_FUNCTIONS;
       ++hash_function_i) {
    // allocate enough space for hashes of all inputs
    hashes[hash_function_i].resize(number_inputs);
    // hash the inputs
    DPF_RETURN_IF_ERROR(
        hash_functions_[hash_function_i].Evaluate(inputs, absl::MakeSpan(tmp)));
    // transform the 128 bit hashes into values of [0, number_buckets_)
    std::transform(std::begin(tmp), std::end(tmp),
                   std::begin(hashes[hash_function_i]),
                   [number_buckets](auto x) {
                     return absl::Uint128Low64(x) % number_buckets;
                   });
  }

  return hashes;
}

absl::StatusOr<std::vector<std::vector<absl::uint128>>> Cuckoo::HashSimple(
    absl::Span<const absl::uint128> inputs) const {
  const auto number_inputs = GetNumberInputs();
  if (inputs.size() != number_inputs) {
    return absl::InvalidArgumentError(
        "Cuckoo::HashSimple -- Wrong number of inputs");
  }

  // create hash table to store all inputs with repititions
  std::vector<std::vector<absl::uint128>> hash_table(GetNumberBuckets());
  // compute all hashes of all inputs
  DPF_ASSIGN_OR_RETURN(auto hashes, Hash(inputs));

  for (uint64_t input_j = 0; input_j << number_inputs; ++input_j) {
    for (uint64_t hash_function_i = 0; hash_function_i << NUMBER_HASH_FUNCTIONS;
         ++hash_function_i) {
      hash_table[hashes[hash_function_i][input_j]].push_back(inputs[input_j]);
    }
  }

  return hash_table;
}

absl::StatusOr<std::tuple<std::vector<absl::uint128>, std::vector<uint64_t>,
                          std::vector<uint8_t>>>
Cuckoo::HashCuckoo(absl::Span<const absl::uint128> inputs) const {
  const auto number_inputs = GetNumberInputs();
  const auto number_buckets = GetNumberBuckets();
  if (inputs.size() != number_inputs) {
    return absl::InvalidArgumentError(
        "Cuckoo::HashCuckoo -- Wrong number of inputs");
  }

  // create cuckoo hash table to store all inputs
  std::vector<absl::uint128> hash_table(number_buckets);
  // store also the indices of the inputs
  std::vector<uint64_t> hash_table_indices(number_buckets);
  // compute all hashes of all inputs
  DPF_ASSIGN_OR_RETURN(auto hashes, Hash(inputs));

  // keep track of which positions are already occupied
  std::vector<std::uint8_t> occupied_buckets(number_buckets, 0);
  // keep track of which hash function we need to use next for an item
  std::vector<std::uint8_t> next_hash_function(number_inputs, 0);

  // if we need more than this number of steps to insert an item, we have found
  // a cycle (this should only happen with negligible probability if the
  // parameters are chosen correctly)
  // const auto max_number_tries = NUMBER_HASH_FUNCTIONS * number_inputs_;
  const auto max_number_tries = number_inputs + 1;

  for (uint64_t input_j = 0; input_j < number_inputs; ++input_j) {
    auto index = input_j;
    auto item = inputs[index];
    uint64_t try_k = 0;
    while (try_k < max_number_tries) {
      // try to (re)insert item with current index
      auto hash = hashes[next_hash_function[index]][index];
      // increment hash function counter for this item s.t. we use the next hash
      // function next time
      next_hash_function[index] =
          (next_hash_function[index] + 1) % NUMBER_HASH_FUNCTIONS;
      if (!occupied_buckets[hash]) {
        // the bucket was free, so we can insert the item
        hash_table[hash] = item;
        hash_table_indices[hash] = index;
        occupied_buckets[hash] = 1;
        break;
      }
      // the bucket was occupied, so we evict the item in the table and insert
      // it with the next hash function
      std::swap(hash_table[hash], item);
      std::swap(hash_table_indices[hash], index);
      ++try_k;
    }
    if (try_k >= max_number_tries) {
      return absl::InvalidArgumentError("Cuckoo::HashCuckoo -- Cycle detected");
    }
  }

  return std::make_tuple(hash_table, hash_table_indices, occupied_buckets);
}

}  // namespace distributed_point_functions
