#include "dpf/internal/mp_proto_validator.h"

#include "dpf/internal/proto_validator.h"
#include "dpf/status_macros.h"

namespace distributed_point_functions {
namespace dpf_internal {

absl::StatusOr<std::unique_ptr<MpProtoValidator>> MpProtoValidator::Create(
    const MpDpfParameters& parameters) {
  DPF_RETURN_IF_ERROR(ValidateMpParameters(parameters));

  DPF_ASSIGN_OR_RETURN(
      auto proto_validator,
      ProtoValidator::Create(absl::MakeSpan(&parameters.dpf_parameters(), 1)));

  return absl::WrapUnique(
      new MpProtoValidator(parameters, std::move(proto_validator)));
}

// Checks the validity of `parameters`.
// Returns OK on success, and INVALID_ARGUMENT otherwise.
absl::Status MpProtoValidator::ValidateMpParameters(
    const MpDpfParameters& parameters) {
  DPF_RETURN_IF_ERROR(ProtoValidator::ValidateParameters(
      absl::MakeSpan(&parameters.dpf_parameters(), 1)));

  if (parameters.number_points() == 0 ||
      parameters.number_points() >
          (1 << parameters.dpf_parameters().log_domain_size())) {
    return absl::InvalidArgumentError(
        "number of points must be in [1, log_domain_size]");
  }

  return absl::OkStatus();
}

// Checks that `key` is valid for the `parameters` passed at construction.
// Returns OK on success, and INVALID_ARGUMENT otherwise.
absl::Status MpProtoValidator::ValidateMpDpfKey(const MpDpfKey& key) const {
  // Check that numbe_points matches number of dpf_keys
  if (key.cuckoo_parameters().number_buckets() != key.dpf_keys_size()) {
    return absl::InvalidArgumentError(
        "number of dpf keys must be equal to number of buckets");
  }

  for (const auto& dpf_key : key.dpf_keys()) {
    DPF_RETURN_IF_ERROR(proto_validator_->ValidateDpfKey(dpf_key));
  }

  // Check cuckoo parameters
  if (parameters_.number_points() != key.cuckoo_parameters().number_inputs()) {
    return absl::InvalidArgumentError(
        "number of inputs in cuckoo params must be equal to number of points");
  }

  if (key.cuckoo_parameters().number_buckets() < parameters_.number_points()) {
    return absl::InvalidArgumentError(
        "number of buckets in cuckoo params cannot be less than number of "
        "points");
  }

  if (key.cuckoo_parameters().hash_functions_keys_size() != 3) {
    return absl::InvalidArgumentError(
        "number of hash functions must be equal to 3");
  }

  return absl::OkStatus();
}

MpProtoValidator::MpProtoValidator(
    MpDpfParameters parameters, std::unique_ptr<ProtoValidator> proto_validator)
    : parameters_(std::move(parameters)),
      proto_validator_(std::move(proto_validator)) {}

}  // namespace dpf_internal
}  // namespace distributed_point_functions
