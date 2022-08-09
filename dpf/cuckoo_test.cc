// Copyright 2021 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "cuckoo.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <vector>

#include "absl/numeric/int128.h"
#include "absl/random/random.h"
#include "absl/types/span.h"
#include "internal/status_matchers.h"

namespace distributed_point_functions {

namespace {

class CuckooTest : public testing::TestWithParam<int> {
 protected:
  std::vector<absl::uint128> gen_random_numbers(uint64_t size) {
    absl::BitGen rng;
    absl::uniform_int_distribution<absl::uint128> dist;
    std::vector<absl::uint128> numbers(size);
    std::generate(std::begin(numbers), std::end(numbers),
                  [&] { return dist(rng); });
    return numbers;
  }
  std::unique_ptr<Cuckoo> create_cuckoo(uint64_t number_inputs) {
    auto cuckoo_params_status = Cuckoo::Sample(number_inputs);
    auto cuckoo_status =
        Cuckoo::CreateFromParameters(cuckoo_params_status.value());
    auto cuckoo = std::move(cuckoo_status.value());
    return cuckoo;
  }
};

TEST_P(CuckooTest, TestSample) {
  ASSERT_EQ(Cuckoo::NUMBER_HASH_FUNCTIONS, 3);
  const auto number_inputs = uint64_t(1) << GetParam();
  DPF_ASSERT_OK_AND_ASSIGN(auto cuckoo_params, Cuckoo::Sample(number_inputs));
  DPF_ASSERT_OK_AND_ASSIGN(auto cuckoo,
                           Cuckoo::CreateFromParameters(cuckoo_params));
}

TEST_P(CuckooTest, TestHash) {
  const auto number_inputs = uint64_t(1) << GetParam();
  const auto inputs = gen_random_numbers(number_inputs);

  auto cuckoo = create_cuckoo(number_inputs);
  DPF_ASSERT_OK_AND_ASSIGN(const auto hashes,
                           cuckoo->Hash(absl::MakeSpan(inputs)));
  ASSERT_EQ(hashes.size(), Cuckoo::NUMBER_HASH_FUNCTIONS);
  for (const auto& row : hashes) {
    ASSERT_EQ(row.size(), number_inputs);
  }
}

TEST_P(CuckooTest, TestHashCuckoo) {
  const auto number_inputs = uint64_t(1) << GetParam();
  const auto inputs = gen_random_numbers(number_inputs);
  auto cuckoo = create_cuckoo(number_inputs);
  DPF_ASSERT_OK_AND_ASSIGN(const auto cuckoo_table,
                           cuckoo->HashCuckoo(absl::MakeSpan(inputs)));
  const auto& [cuckoo_table_items, cuckoo_table_indices,
               cuckoo_table_occupied] = cuckoo_table;
  const auto number_buckets = cuckoo->GetNumberBuckets();
  // check dimensions
  ASSERT_EQ(cuckoo_table_items.size(), number_buckets);
  ASSERT_EQ(cuckoo_table_indices.size(), number_buckets);
  ASSERT_EQ(cuckoo_table_occupied.size(), number_buckets);
  // check that we have the right number of things in the table
  const auto num_entries_in_occupied_table = std::count_if(
      std::begin(cuckoo_table_occupied), std::end(cuckoo_table_occupied),
      [](auto x) { return x != 0; });
  ASSERT_EQ(num_entries_in_occupied_table, number_inputs);
  // keep track of which items we have seen in the cuckoo table
  std::vector<bool> found_inputs_in_table(number_inputs, false);
  for (uint64_t bucket_i = 0; bucket_i < number_buckets; ++bucket_i) {
    if (cuckoo_table_occupied[bucket_i]) {
      const auto index = cuckoo_table_indices[bucket_i];
      // check that the right item is here
      ASSERT_EQ(cuckoo_table_items[bucket_i], inputs[index]);
      // check that we have not yet seen this item
      ASSERT_FALSE(found_inputs_in_table[index]);
      // remember that we have seen this item
      found_inputs_in_table[index] = true;
    }
  }
  // check that we have found all inputs in the cuckoo table
  ASSERT_TRUE(std::all_of(std::begin(found_inputs_in_table),
                          std::end(found_inputs_in_table),
                          [](auto x) { return x; }));
}

TEST_P(CuckooTest, TestHashSimpleDomain) {
  const auto number_inputs = uint64_t(1) << GetParam();
  const auto log_domain_size = 10;
  auto cuckoo = create_cuckoo(number_inputs);
  const auto number_buckets = cuckoo->GetNumberBuckets();

  DPF_ASSERT_OK_AND_ASSIGN(const auto hash_table,
                           cuckoo->HashSimpleDomain(1 << log_domain_size));
  ASSERT_EQ(hash_table.size(), number_buckets);
  for (const auto& bucket : hash_table) {
    // Check that items inside each bucket are sorted
    ASSERT_TRUE(std::is_sorted(std::begin(bucket), std::end(bucket)));
  }

  // Check that hashing is deterministic
  DPF_ASSERT_OK_AND_ASSIGN(const auto hash_table2,
                           cuckoo->HashSimpleDomain(1 << log_domain_size));
  ASSERT_EQ(hash_table, hash_table2);

  std::vector<absl::uint128> domain(1 << log_domain_size);
  std::iota(std::begin(domain), std::end(domain), 0);

  DPF_ASSERT_OK_AND_ASSIGN(const auto hashes,
                           cuckoo->Hash(absl::MakeSpan(domain)));

  for (uint64_t element = 0; element < 1 << log_domain_size; ++element) {
    if (hashes[0][element] == hashes[1][element] &&
        hashes[0][element] == hashes[2][element]) {
      const auto hash = hashes[0][element];
      const auto it_start = std::lower_bound(
          std::begin(hash_table[hash]), std::end(hash_table[hash]), element);
      const auto it_end = std::upper_bound(std::begin(hash_table[hash]),
                                           std::end(hash_table[hash]), element);
      ASSERT_NE(it_start, std::end(hash_table[hash]));
      ASSERT_EQ(*it_start, element);
      ASSERT_EQ(std::distance(it_start, it_end), 3);
    } else if (hashes[0][element] == hashes[1][element]) {
      const auto hash = hashes[0][element];
      const auto it_start = std::lower_bound(
          std::begin(hash_table[hash]), std::end(hash_table[hash]), element);
      const auto it_end = std::upper_bound(std::begin(hash_table[hash]),
                                           std::end(hash_table[hash]), element);
      ASSERT_NE(it_start, std::end(hash_table[hash]));
      ASSERT_EQ(*it_start, element);
      ASSERT_EQ(std::distance(it_start, it_end), 2);

      const auto hash_other = hashes[2][element];
      ASSERT_TRUE(std::binary_search(std::begin(hash_table[hash_other]),
                                     std::end(hash_table[hash_other]),
                                     element));

    } else if (hashes[0][element] == hashes[2][element]) {
      const auto hash = hashes[0][element];
      const auto it_start = std::lower_bound(
          std::begin(hash_table[hash]), std::end(hash_table[hash]), element);
      const auto it_end = std::upper_bound(std::begin(hash_table[hash]),
                                           std::end(hash_table[hash]), element);
      ASSERT_NE(it_start, std::end(hash_table[hash]));
      ASSERT_EQ(*it_start, element);
      ASSERT_EQ(std::distance(it_start, it_end), 2);

      const auto hash_other = hashes[1][element];
      ASSERT_TRUE(std::binary_search(std::begin(hash_table[hash_other]),
                                     std::end(hash_table[hash_other]),
                                     element));
    } else if (hashes[1][element] == hashes[2][element]) {
      const auto hash = hashes[1][element];
      const auto it_start = std::lower_bound(
          std::begin(hash_table[hash]), std::end(hash_table[hash]), element);
      const auto it_end = std::upper_bound(std::begin(hash_table[hash]),
                                           std::end(hash_table[hash]), element);
      ASSERT_NE(it_start, std::end(hash_table[hash]));
      ASSERT_EQ(*it_start, element);
      ASSERT_EQ(std::distance(it_start, it_end), 2);

      const auto hash_other = hashes[0][element];
      ASSERT_TRUE(std::binary_search(std::begin(hash_table[hash_other]),
                                     std::end(hash_table[hash_other]),
                                     element));
    } else {
      for (int hash_j = 0; hash_j < 3; ++hash_j) {
        const auto hash = hashes[hash_j][element];
        ASSERT_TRUE(std::binary_search(std::begin(hash_table[hash]),
                                       std::end(hash_table[hash]), element));
      }
    }
  }

  const auto num_items_in_hash_table = std::transform_reduce(
      std::begin(hash_table), std::end(hash_table), 0, std::plus<>{},
      [](const auto& b) { return b.size(); });
  ASSERT_EQ(num_items_in_hash_table, 3 * (1 << log_domain_size));
}

INSTANTIATE_TEST_SUITE_P(CuckooTestSuite, CuckooTest, testing::Range(5, 10));

}  // namespace

}  // namespace distributed_point_functions
