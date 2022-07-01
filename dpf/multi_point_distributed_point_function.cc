#include <absl/memory/memory.h>

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
  // TODO
}

}  // namespace distributed_point_functions
