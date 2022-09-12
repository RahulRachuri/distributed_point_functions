#ifndef DISTRIBUTED_POINT_FUNCTIONS_DPF_MULTI_POINT_DISTRIBUTED_POINT_FUNCTION_H_
#define DISTRIBUTED_POINT_FUNCTIONS_DPF_MULTI_POINT_DISTRIBUTED_POINT_FUNCTION_H_

#include <memory>

#include "absl/status/statusor.h"
#include "dpf/cuckoo.h"
#include "dpf/distributed_point_function.h"
#include "dpf/distributed_point_function.pb.h"
#include "dpf/internal/mp_proto_validator.h"
#include "dpf/status_macros.h"

namespace distributed_point_functions {

class MultiPointDistributedPointFunction {
 public:
  static absl::StatusOr<std::unique_ptr<MultiPointDistributedPointFunction>>
  Create(const MpDpfParameters& parameters);

  absl::StatusOr<std::pair<MpDpfKey, MpDpfKey>> GenerateKeys(
      absl::Span<const absl::uint128> alphas, absl::Span<const Value> betas);

  template <typename T>
  absl::StatusOr<std::vector<T>> EvaluateAt(
      const MpDpfKey& key,
      absl::Span<const absl::uint128> evaluation_points) const;

  // Returns the DpfParameters of this DPF.
  const MpDpfParameters& parameters() const { return parameters_; }

  void SetValueTypeRegistrationFunction(
      std::function<absl::Status(DistributedPointFunction&)> f) {
    value_type_registration_function_ = f;
  }

 private:
  MultiPointDistributedPointFunction(
      const MpDpfParameters& parameters,
      std::unique_ptr<Cuckoo>&& cuckoo_context,
      std::unique_ptr<dpf_internal::MpProtoValidator>&& mp_proto_validator);

  const MpDpfParameters parameters_;
  std::unique_ptr<Cuckoo> cuckoo_context_;
  std::vector<DistributedPointFunction> single_point_dpfs_;
  std::unique_ptr<dpf_internal::MpProtoValidator> mp_proto_validator_;
  std::function<absl::Status(DistributedPointFunction&)>
      value_type_registration_function_;
};

template <typename T>
absl::StatusOr<std::vector<T>> MultiPointDistributedPointFunction::EvaluateAt(
    const MpDpfKey& key,
    absl::Span<const absl::uint128> evaluation_points) const {
  const auto num_evaluation_points = evaluation_points.size();
  const int log_domain_size = parameters_.dpf_parameters().log_domain_size();
  const auto num_buckets = cuckoo_context_->GetNumberBuckets();

  absl::uint128 max_evaluation_point = absl::Uint128Max();
  if (log_domain_size < 128) {
    max_evaluation_point = (absl::uint128{1} << log_domain_size) - 1;
  }
  // Check if `evaluation_points` are inside the domain.
  for (size_t i = 0; i < num_evaluation_points; ++i) {
    if (evaluation_points[i] > max_evaluation_point) {
      return absl::InvalidArgumentError(absl::StrCat(
          "`evaluation_points[", i, "]` larger than the domain size"));
    }
  }

  DPF_RETURN_IF_ERROR(mp_proto_validator_->ValidateMpDpfKey(key));
  if (num_evaluation_points == 0) {
    return std::vector<T>{};  // Nothing to do.
  }

  // hash all the evaluation points with all three hash functions
  DPF_ASSIGN_OR_RETURN(const auto hashes,
                       cuckoo_context_->Hash(evaluation_points));
  // hash the complete domain into buckets
  DPF_ASSIGN_OR_RETURN(const auto simple_htable,
                       cuckoo_context_->HashSimpleDomain(1 << log_domain_size));

  // function that maps: (bucket, item) to position of item in bucket
  const auto pos = [&simple_htable](auto bucket_i, auto item) {
    const auto it = std::lower_bound(std::begin(simple_htable[bucket_i]),
                                     std::end(simple_htable[bucket_i]), item);
    assert(*it == item);
    assert(it != std::end(simple_htable[bucket_i]));

    return std::distance(std::begin(simple_htable[bucket_i]), it);
  };

  // parameters of all the single-point dpfs (one for each bucket)
  std::vector<DpfParameters> sp_dpf_params(num_buckets);
  // buffer for all single-point dpf objects
  std::vector<std::unique_ptr<DistributedPointFunction>> sp_dpfs;
  sp_dpfs.reserve(num_buckets);

  // value type of the dpf
  const auto value_type = parameters_.dpf_parameters().value_type();

  // set up all the sp-dpfs for each bucket
  for (uint64_t bucket_i = 0; bucket_i < num_buckets; ++bucket_i) {
    // if the bucket is empty, we do not need to do anything
    if (key.bucket_sizes(bucket_i) != 0) {
      // setup the sp-dpf parameter corresponding to the bucket size
      *sp_dpf_params[bucket_i].mutable_value_type() = value_type;
      sp_dpf_params[bucket_i].set_log_domain_size(static_cast<int32_t>(
        std::ceil(std::log2(simple_htable[bucket_i].size()))));

      // create the sp-dpf object
      DPF_ASSIGN_OR_RETURN(
        auto sp_dpf, DistributedPointFunction::Create(sp_dpf_params[bucket_i]));
      sp_dpfs.push_back(std::move(sp_dpf));
    } else {
      // if the bucket is empty, we don't need an sp-dpf, so just fill this position with a nullptr
      sp_dpfs.emplace_back();
    }
  }

  // buffer holding all the outputs
  std::vector<T> output(num_evaluation_points);

  // figure out if there are collision
  // std::vector<uint8_t> hash_bit_map(num_evaluation_points, 0);

  // for (size_t i = 0; i < num_evaluation_points; ++i) {
  //   if(hashes[0][i] != hashes[1][i]) {
  //     hash_bit_map[i] |= 1;
  //   }
  //   if(hashes[0][i] != hashes[2][i] && hashes[1][i] != hashes[2][i]) {
  //     hash_bit_map[i] |= 2;
  //   }
  // }
  std::array<std::vector<uint8_t>, 2> hash_bit_maps;
  hash_bit_maps[0] = std::vector<uint8_t>(num_evaluation_points, 0);
  hash_bit_maps[1] = std::vector<uint8_t>(num_evaluation_points, 0);

  for (size_t i = 0; i < num_evaluation_points; ++i) {
    if(hashes[0][i] != hashes[1][i]) {
      // hash_bit_map[i] |= 1;
      hash_bit_maps[0][i] = 1;
    }
    if(hashes[0][i] != hashes[2][i] && hashes[1][i] != hashes[2][i]) {
      // hash_bit_map[i] |= 2;
      hash_bit_maps[1][i] = 1;
    }
    // if (hash_bit_maps[0][i] == 0 && hash_bit_maps[1][i] == 0) {
    //   std::cerr << "3-way COLLISION at index " << i << "\n";
    // } else  if (hash_bit_maps[0][i] == 0 || hash_bit_maps[1][i] == 0) {
    //   std::cerr << "2-way COLLISION at index " << i << "\n";
    // }
  }

  // compute the output for each evaluation point
  for (size_t i = 0; i < num_evaluation_points; ++i) {
    std::cerr << "i = " << i << " -> (" << hashes[0][i] << ", " << hashes[1][i] << "' " << hashes[2][i] << ")\n";

    {
      const auto hash = hashes[0][i];
      assert(sp_dpfs[hash] != nullptr);
      const auto& sp_key = key.dpf_keys(hash);
      const absl::uint128 pos_point = pos(hash, evaluation_points[i]);
      assert(simple_htable[hash][static_cast<size_t>(pos_point)] == evaluation_points[i]);

      DPF_ASSIGN_OR_RETURN(auto tmp,
                           sp_dpfs[hash]->EvaluateAt<T>(
                               sp_key, 0, absl::MakeSpan(&pos_point, 1)));
      output[i] = tmp[0];
    }

    for (size_t j = 1; j < Cuckoo::NUMBER_HASH_FUNCTIONS; ++j) {
      // if ((hash_bit_map[i] & (1 << (j - 1))) == 0) {
      if (hash_bit_maps[j-1][i] == 0) {
        continue;
      }
      const auto hash = hashes[j][i];
      assert(sp_dpfs[hash] != nullptr);
      const auto& sp_key = key.dpf_keys(hash);
      const absl::uint128 pos_point = pos(hash, evaluation_points[i]);
      assert(simple_htable[hash][static_cast<size_t>(pos_point)] == evaluation_points[i]);

      DPF_ASSIGN_OR_RETURN(auto tmp,
                           sp_dpfs[hash]->EvaluateAt<T>(
                               sp_key, 0, absl::MakeSpan(&pos_point, 1)));

      output[i] = output[i] + tmp[0];
      // output[i] += output_j;
    }
  }

  return output;
}

}  // namespace distributed_point_functions

#endif  // DISTRIBUTED_POINT_FUNCTIONS_DPF_MULTI_POINT_DISTRIBUTED_POINT_FUNCTION_H_
