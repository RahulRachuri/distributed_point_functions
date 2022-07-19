#include "dpf/status_macros.h"
#include "dpf/internal/mp_proto_validator.h"
#include "dpf/internal/proto_validator.h"

namespace distributed_point_functions {
namespace dpf_internal {

absl::StatusOr<std::unique_ptr<MpProtoValidator>> MpProtoValidator::Create(
      const MpDpfParameters& parameters) {

    DPF_RETURN_IF_ERROR(ValidateMpParameters(parameters));

    DPF_ASSIGN_OR_RETURN(auto proto_validator, ProtoValidator::Create(parameters.dpf_parameters()) );

    return absl::WrapUnique(new MpProtoValidator(
        std::move(parameters)), std::move(proto_validator));

    }

  // Checks the validity of `parameters`.
  // Returns OK on success, and INVALID_ARGUMENT otherwise.
  absl::Status MpProtoValidator::ValidateMpParameters(
      const MpDpfParameters& parameters) {

    DPF_RETURN_IF_ERROR(ProtoValidator::ValidateParameters(absl::MakeSpan({parameters.dpf_parameters()})) );

    if(parameters.number_points() == 0 || parameters.number_points() > ( 1 << parameters.dpf_parameters().log_domain_size() ) ) {
        return absl::InvalidArgumentError("number of points must be in [1, log_domain_size]");
    }

    return absl::OkStatus();

    }

  // Checks that `key` is valid for the `parameters` passed at construction.
  // Returns OK on success, and INVALID_ARGUMENT otherwise.
  absl::Status MpProtoValidator::ValidateMpDpfKey(const MpDpfKey& key) {
    // Check that key has the seed
    if (!key.has_seed()) {
        return absl::InvalidArgumentError("key seed must be present");
    }

    // Check cuckoo parameters
    
    
    return absl::OkStatus();

  }

  MpProtoValidator::MpProtoValidator(MpDpfParameters parameters, std::unique_ptr<ProtoValidator> proto_validator) : parameters_(std::move(parameters)), proto_validator_(std::move(proto_validator)) {}


}  // namespace dpf_internal
}  // namespace distributed_point_functions
