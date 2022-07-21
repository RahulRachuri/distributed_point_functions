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
      absl::Span<absl::uint128> alphas, absl::Span<const Value> betas);

  template <typename T>
  absl::StatusOr<std::vector<T>> EvaluateAt(
      const MpDpfKey& key,
      absl::Span<const absl::uint128> evaluation_points) const;

  // Returns the DpfParameters of this DPF.
  inline const MpDpfParameters& parameters() const { return parameters_; }

 private:
  MultiPointDistributedPointFunction(
      const MpDpfParameters& parameters,
      std::unique_ptr<Cuckoo>&& cuckoo_context,
      std::unique_ptr<dpf_internal::MpProtoValidator>&& mp_proto_validator);

  const MpDpfParameters parameters_;
  std::unique_ptr<Cuckoo> cuckoo_context_;
  std::vector<DistributedPointFunction> single_point_dpfs_;
  std::unique_ptr<dpf_internal::MpProtoValidator> mp_proto_validator_;
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

  DPF_ASSIGN_OR_RETURN(const auto hashes,
                       cuckoo_context_->Hash(evaluation_points));
  DPF_ASSIGN_OR_RETURN(const auto simple_htable,
                       cuckoo_context_->HashSimple(evaluation_points));

  const auto pos = [&simple_htable](auto bucket_i, auto item) {
    const auto iter = std::find(std::begin(simple_htable[bucket_i]),
                                std::end(simple_htable[bucket_i]), item);

    return std::distance(std::begin(simple_htable[bucket_i]), iter);
  };

  std::vector<DpfParameters> sp_dpf_params(num_buckets);
  std::vector<std::unique_ptr<DistributedPointFunction>> sp_dpfs;
  sp_dpfs.reserve(num_buckets);

  const auto value_type = parameters_.dpf_parameters().value_type();
  for (uint64_t bucket_i = 0; bucket_i < num_buckets; ++bucket_i) {
    *sp_dpf_params[bucket_i].mutable_value_type() = value_type;

    sp_dpf_params[bucket_i].set_log_domain_size(static_cast<int32_t>(
        std::ceil(std::log2(simple_htable[bucket_i].size()))));

    DPF_ASSIGN_OR_RETURN(
        auto sp_dpf, DistributedPointFunction::Create(sp_dpf_params[bucket_i]));

    sp_dpfs.push_back(std::move(sp_dpf));
  }

  std::vector<T> output(num_evaluation_points);

  for (size_t i = 0; i < num_evaluation_points; ++i) {
    {
      const auto hash = hashes[0][i];
      const auto& sp_key = key.dpf_keys(hash);
      const auto pos_point = pos(hash, evaluation_points[i]);

      DPF_ASSIGN_OR_RETURN(
          output[i],
          sp_dpfs[hash]->EvaluateAt(sp_key, 0, absl::MakeSpan(&pos_point, 1)));
    }

    for (size_t j = 1; j < Cuckoo::NUMBER_HASH_FUNCTIONS; ++j) {
      const auto hash = hashes[j][i];
      const auto& sp_key = key.dpf_keys(hash);
      const auto pos_point = pos(hash, evaluation_points[i]);

      DPF_ASSIGN_OR_RETURN(
          auto output_j,
          sp_dpfs[hash]->EvaluateAt(sp_key, 0, absl::MakeSpan(&pos_point, 1)));

      output[i] += output_j;
    }
  }

  return output;
}

}  // namespace distributed_point_functions

#endif  // DISTRIBUTED_POINT_FUNCTIONS_DPF_MULTI_POINT_DISTRIBUTED_POINT_FUNCTION_H_
