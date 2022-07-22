#include "dpf/internal/mp_proto_validator.h"

#include "dpf/internal/proto_validator.h"
#include "dpf/status_macros.h"

namespace distributed_point_functions {
namespace dpf_internal {

absl::StatusOr<std::unique_ptr<MpProtoValidator>> MpProtoValidator::Create(
    const MpDpfParameters& parameters) {
  DPF_RETURN_IF_ERROR(ValidateMpParameters(parameters));

  return absl::WrapUnique(new MpProtoValidator(parameters, nullptr));
}

// Checks the validity of `parameters`.
// Returns OK on success, and INVALID_ARGUMENT otherwise.
absl::Status MpProtoValidator::ValidateMpParameters(
    const MpDpfParameters& parameters) {
  if (!parameters.has_dpf_parameters()) {
    return absl::InvalidArgumentError(
        "ValidateMpParameters: dpf parameters are required");
  }
  DPF_RETURN_IF_ERROR(ProtoValidator::ValidateParameters(
      absl::MakeSpan(&parameters.dpf_parameters(), 1)));

  if (parameters.number_points() == uint64_t(0) ||
      parameters.number_points() >
          (uint64_t(1) << parameters.dpf_parameters().log_domain_size())) {
    return absl::InvalidArgumentError(
        "ValidateMpParameters: number of points must be in [1, "
        "log_domain_size]");
  }

  return absl::OkStatus();
}

// Checks that `key` is valid for the `parameters` passed at construction.
// Returns OK on success, and INVALID_ARGUMENT otherwise.
absl::Status MpProtoValidator::ValidateMpDpfKey(const MpDpfKey& key) const {
  // Check that numbe_points matches number of dpf_keys
  if (key.cuckoo_parameters().number_buckets() !=
      static_cast<uint64_t>(key.dpf_keys_size())) {
    return absl::InvalidArgumentError(
        "ValidateMpDpfKey: number of dpf keys must be equal to number of "
        "buckets");
  }

  if (key.cuckoo_parameters().number_buckets() !=
      static_cast<uint64_t>(key.bucket_sizes_size())) {
    return absl::InvalidArgumentError(
        "ValidateMpDpfKey: need to have the size of each bucket when hashing "
        "the full domain");
  }

  const auto value_type = parameters_.dpf_parameters().value_type();
  for (size_t i = 1; i < key.cuckoo_parameters().number_buckets(); ++i) {
    if (key.bucket_sizes(i) != 0) {
      DpfParameters sp_dpf_parameters;
      *sp_dpf_parameters.mutable_value_type() = value_type;
      sp_dpf_parameters.set_log_domain_size(
          static_cast<int32_t>(std::ceil(std::log2(key.bucket_sizes(i)))));
      DPF_ASSIGN_OR_RETURN(auto validator,
                           dpf_internal::ProtoValidator::Create(
                               absl::MakeSpan(&sp_dpf_parameters, 1)));
      DPF_RETURN_IF_ERROR(validator->ValidateDpfKey(key.dpf_keys(i)));
    }
  }

  // Check cuckoo parameters
  if (parameters_.number_points() != key.cuckoo_parameters().number_inputs()) {
    return absl::InvalidArgumentError(
        "ValidateMpDpfKey: number of inputs in cuckoo params must be equal to "
        "number of points");
  }

  if (key.cuckoo_parameters().number_buckets() < parameters_.number_points()) {
    return absl::InvalidArgumentError(
        "ValidateMpDpfKey: number of buckets in cuckoo params cannot be less "
        "than number of "
        "points");
  }

  if (key.cuckoo_parameters().hash_functions_keys_size() != 3) {
    return absl::InvalidArgumentError(
        "ValidateMpDpfKey: number of hash functions must be equal to 3");
  }

  return absl::OkStatus();
}

MpProtoValidator::MpProtoValidator(
    MpDpfParameters parameters, std::unique_ptr<ProtoValidator> proto_validator)
    : parameters_(std::move(parameters)),
      proto_validator_(std::move(proto_validator)) {}

}  // namespace dpf_internal
}  // namespace distributed_point_functions
