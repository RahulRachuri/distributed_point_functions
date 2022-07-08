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

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <vector>

#include "absl/numeric/int128.h"
#include "absl/random/random.h"
#include "absl/types/span.h"
#include "cuckoo.h"
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
      auto cuckoo_status = Cuckoo::CreateFromParameters(cuckoo_params_status.value());
      auto cuckoo = std::move(cuckoo_status.value());
      return cuckoo;
  }
};

TEST_P(CuckooTest, TestSample) {
  ASSERT_EQ(Cuckoo::NUMBER_HASH_FUNCTIONS, 3);
  auto number_inputs = uint64_t(1) << GetParam();
  DPF_ASSERT_OK_AND_ASSIGN(auto cuckoo_params, Cuckoo::Sample(number_inputs));
  DPF_ASSERT_OK_AND_ASSIGN(auto cuckoo, Cuckoo::CreateFromParameters(cuckoo_params));
}

TEST_P(CuckooTest, TestHash) {
  auto number_inputs = uint64_t(1) << GetParam();
  auto inputs = gen_random_numbers(number_inputs);

  auto cuckoo = create_cuckoo(number_inputs);
  DPF_ASSERT_OK_AND_ASSIGN(const auto hashes, cuckoo->Hash(absl::MakeSpan(inputs)));
  ASSERT_EQ(hashes.size(), Cuckoo::NUMBER_HASH_FUNCTIONS);
  for (const auto& row : hashes) {
    ASSERT_EQ(row.size(), number_inputs);
  }
}

TEST_P(CuckooTest, TestHashCuckoo) {
  auto number_inputs = uint64_t(1) << GetParam();
  auto inputs = gen_random_numbers(number_inputs);
  auto cuckoo = create_cuckoo(number_inputs);
  DPF_ASSERT_OK_AND_ASSIGN(auto cuckoo_table, cuckoo->HashCuckoo(absl::MakeSpan(inputs)));
  const auto& [cuckoo_table_items, cuckoo_table_indices, cuckoo_table_occupied] = cuckoo_table;
  ASSERT_EQ(cuckoo_table_items.size(), number_buckets);
  for (const auto& row : hashes) {
    ASSERT_EQ(row.size(), number_inputs);
  }
}

INSTANTIATE_TEST_SUITE_P(CuckooTestSuite, CuckooTest, testing::Range(5, 10));

}  // namespace

}  // namespace distributed_point_functions
