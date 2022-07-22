#include "multi_point_distributed_point_function.h"

#include <algorithm>
#include <cmath>

#include "absl/memory/memory.h"
#include "status_macros.h"

namespace distributed_point_functions {

MultiPointDistributedPointFunction::MultiPointDistributedPointFunction(
    const MpDpfParameters& parameters, std::unique_ptr<Cuckoo>&& cuckoo_context,
    std::unique_ptr<dpf_internal::MpProtoValidator>&& mp_proto_validator)
    : parameters_(parameters),
      cuckoo_context_(std::move(cuckoo_context)),
      mp_proto_validator_(std::move(mp_proto_validator)),
      value_type_registration_function_(
          [](auto& _) { return absl::OkStatus(); }) {}

absl::StatusOr<std::unique_ptr<MultiPointDistributedPointFunction>>
MultiPointDistributedPointFunction::Create(const MpDpfParameters& parameters) {
  // Initialise cuckoo context. single_point_dpfs is inititalised in
  // GenerateKeys().
  uint64_t number_inputs = parameters.number_points();

  // Create mp_proto_validator and validate parameters.
  DPF_ASSIGN_OR_RETURN(auto mp_proto_validator,
                       dpf_internal::MpProtoValidator::Create(parameters));

  DPF_ASSIGN_OR_RETURN(auto cuckoo_parameters, Cuckoo::Sample(number_inputs));
  DPF_ASSIGN_OR_RETURN(auto cuckoo_context,
                       Cuckoo::CreateFromParameters(cuckoo_parameters));
  return absl::WrapUnique(new MultiPointDistributedPointFunction(
      parameters, std::move(cuckoo_context), std::move(mp_proto_validator)));
}

absl::StatusOr<std::pair<MpDpfKey, MpDpfKey>>
MultiPointDistributedPointFunction::GenerateKeys(
    absl::Span<const absl::uint128> alphas, absl::Span<const Value> betas) {
  // Check validity of alphas
  if (alphas.size() != parameters_.number_points()) {
    return absl::InvalidArgumentError(
        "`alphas` has to have the same size as `number_points` passed at "
        "construction");
  }
  const auto log_domain_size = parameters_.dpf_parameters().log_domain_size();
  if (log_domain_size < 128) {
    for (const auto alpha_i : alphas) {
      if (alpha_i >= (absl::uint128{1} << log_domain_size)) {
        return absl::InvalidArgumentError(
            "each `alpha` must be smaller than the output domain size");
      }
    }
  }
  // Check validity of betas
  if (betas.size() != parameters_.number_points()) {
    return absl::InvalidArgumentError(
        "`betas` has to have the same size as `number_points` passed at "
        "construction");
  }

  const auto value_type = parameters_.dpf_parameters().value_type();
  for (const auto& beta_i : betas) {
    DPF_RETURN_IF_ERROR(mp_proto_validator_->GetProtoValidator().ValidateValue(
        beta_i, value_type));
  }

  DPF_ASSIGN_OR_RETURN(auto cuckoo_table, cuckoo_context_->HashCuckoo(alphas));
  const auto& [cuckoo_table_items, cuckoo_table_indices,
               cuckoo_table_occupied] = cuckoo_table;
  DPF_ASSIGN_OR_RETURN(auto simple_htable,
                       cuckoo_context_->HashSimpleDomain(1 << log_domain_size));

  auto pos = [&simple_htable](auto bucket_i, auto item) {
    auto it = std::lower_bound(std::begin(simple_htable[bucket_i]),
                               std::end(simple_htable[bucket_i]), item);
    assert(*it == item);  // the item should be in this bucket
    return std::distance(std::begin(simple_htable[bucket_i]), it);
  };

  const auto num_buckets = cuckoo_context_->GetNumberBuckets();

  // std::vector<std::unique_ptr<DistributedPointFunction>> sp_dpfs;
  // sp_dpfs.reserve(num_buckets);

  std::vector<DpfKey> keys_0;
  std::vector<DpfKey> keys_1;

  keys_0.reserve(num_buckets);
  keys_1.reserve(num_buckets);

  std::vector<uint64_t> bucket_sizes(num_buckets, 0);

  for (uint64_t bucket_i = 0; bucket_i < num_buckets; ++bucket_i) {
    const auto bucket_size = simple_htable[bucket_i].size();

    // if bucket is empty, add invalid dummy keys to the arrays to make the
    // indices work
    if (bucket_size == 0) {
      keys_0.emplace_back();
      keys_1.emplace_back();
      continue;
    }

    // remember the bucket size
    bucket_sizes[bucket_i] = bucket_size;

    // create a single point dpf instance
    DpfParameters sp_dpf_parameters;
    *sp_dpf_parameters.mutable_value_type() = value_type;
    sp_dpf_parameters.set_log_domain_size(
        static_cast<int32_t>(std::ceil(std::log2(bucket_size))));

    DPF_ASSIGN_OR_RETURN(auto sp_dpf,
                         DistributedPointFunction::Create(sp_dpf_parameters));
    DPF_RETURN_IF_ERROR(value_type_registration_function_(*sp_dpf));

    absl::uint128 a;
    Value b;

    if (cuckoo_table_occupied[bucket_i]) {
      a = pos(bucket_i, cuckoo_table_items[bucket_i]);
      b = betas[cuckoo_table_indices[bucket_i]];
    } else {
      a = 0;
      DPF_ASSIGN_OR_RETURN(b, dpf_internal::MakeZero(value_type));
    }

    DPF_ASSIGN_OR_RETURN(auto keys, sp_dpf->GenerateKeys(a, b));
    keys_0.push_back(std::move(keys.first));
    keys_1.push_back(std::move(keys.second));
  }

  MpDpfKey key_0;
  MpDpfKey key_1;
  key_0.set_party(0);
  key_1.set_party(1);
  *key_0.mutable_dpf_keys() = {std::begin(keys_0), std::end(keys_0)};
  *key_1.mutable_dpf_keys() = {std::begin(keys_1), std::end(keys_1)};
  *key_0.mutable_cuckoo_parameters() = cuckoo_context_->GetParameters();
  *key_1.mutable_cuckoo_parameters() = cuckoo_context_->GetParameters();
  *key_0.mutable_bucket_sizes() = {std::begin(bucket_sizes),
                                   std::end(bucket_sizes)};
  *key_1.mutable_bucket_sizes() = {std::begin(bucket_sizes),
                                   std::end(bucket_sizes)};
  return std::make_pair(key_0, key_1);
}

}  // namespace distributed_point_functions
