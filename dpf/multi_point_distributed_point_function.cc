#include "absl/memory/memory.h"

#include <cmath>

#include "multi_point_distributed_point_function.h"
#include "status_macros.h"

namespace distributed_point_functions {

MultiPointDistributedPointFunction::MultiPointDistributedPointFunction(
    const MpDpfParameters& parameters,
    std::unique_ptr<Cuckoo>&& cuckoo_context_)
    : parameters_(parameters), cuckoo_context_(std::move(cuckoo_context_)) {}

absl::StatusOr<std::unique_ptr<MultiPointDistributedPointFunction>>
MultiPointDistributedPointFunction::Create(const MpDpfParameters& parameters) {
  uint64_t number_inputs = parameters.number_points();
  DPF_ASSIGN_OR_RETURN(auto cuckoo_parameters, Cuckoo::Sample(number_inputs));
  DPF_ASSIGN_OR_RETURN(auto cuckoo_context,
                       Cuckoo::CreateFromParameters(cuckoo_parameters));
  return absl::WrapUnique(new MultiPointDistributedPointFunction(
      parameters, std::move(cuckoo_context)));
}

absl::StatusOr<std::pair<DpfKey, DpfKey>>
MultiPointDistributedPointFunction::GenerateKeys(
    absl::Span<absl::uint128> alphas, absl::Span<const Value> betas) {
  // Check validity of alphas
  if (alphas.size() != parameters_.size()) {
    return absl::InvalidArgumentError(
        "`alphas` has to have the same size as `parameters` passed at "
        "construction");
  }
  int log_domain_size = parameters_.log_domain_size();
  if (log_domain_size < 128)
      for (auto alpha_i : alphas) {
          if (alpha_i >= (absl::uint128{1} << log_domain_size)) {
            return absl::InvalidArgumentError(
                "each `alpha` must be smaller than the output domain size");
          }
      }
  }
  // Check validity of betas
  if (betas.size() != parameters_.size()) {
    return absl::InvalidArgumentError(
        "`betas` has to have the same size as `parameters` passed at "
        "construction");
  }
  // for (int i = 0; i < static_cast<int>(parameters_.size()); ++i) {
  //   absl::Status status = proto_validator_->ValidateValue(beta[i], i);
  //   if (!status.ok()) {
  //     return status;
  //   }
  // }
  // TODO

  DPF_ASSIGN_OR_RETURN(auto cuckoo_table, cuckoo_context_->HashCuckoo(alphas));
  const auto& [cuckoo_table_items, cuckoo_table_indices, cuckoo_table_occupied] = cuckoo_table;
  DPF_ASSIGN_OR_RETURN(auto simple_htable, cuckoo_context_->HashSimple(alphas));

  auto pos = [&simple_htable](auto bucket_i, auto item){
    auto iter = std::find(std::begin(simple_htable[bucket_i]), std::end(simple_htable[bucket_i]), item);

    return std::distance(std::begin(simple_htable[bucket_i]), iter);
  };

  const auto num_buckets = cuckoo_context_->GetNumBuckets();

  // std::vector<std::unique_ptr<DistributedPointFunction>> sp_dpfs;
  // sp_dpfs.reserve(num_buckets);

  const auto value_type = parameters_.dpf_parameters().value_type();  

  std::vector<DpfParameters> sp_dpf_params(num_buckets);

  std::vector<DpfKey> keys_1;
  std::vector<DpfKey> keys_2;

  keys_1.reserve(num_buckets);
  keys_2.reserve(num_buckets);

  for (uint64_t bucket_i = 0; bucket_i < num_buckets; ++bucket_i) {
    *sp_dpf_params[bucket_i].mutable_value_type() = value_type;

    sp_dpf_params[bucket_i].set_log_domain_size(static_cast<int32_t>(std::ceil(std::log2(simple_htable[bucket_i].size()))));

    DPF_ASSIGN_OR_RETURN(auto sp_dpf, DistributedPointFunction::Create(sp_dpf_params[bucket_i]) );

    // sp_dpfs.push_back(std::move(sp_dpf));

    absl::uint128 a;
    Value b;

    if(cuckoo_table_occupied[bucket_i]) {
      a = pos(bucket_i, cuckoo_table_items[bucket_i]);
      b = betas[cuckoo_table_indices[bucket_i]];
    }
    else {

    }

    DPF_ASSIGN_OR_RETURN(auto keys, sp_dpf.GenerateKeys() )
  }

}

}  // namespace distributed_point_functions
