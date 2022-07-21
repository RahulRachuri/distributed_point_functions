#ifndef DISTRIBUTED_POINT_FUNCTIONS_DPF_INTERNAL_MP_PROTO_VALIDATOR_H_
#define DISTRIBUTED_POINT_FUNCTIONS_DPF_INTERNAL_MP_PROTO_VALIDATOR_H_

#include "absl/status/statusor.h"
#include "dpf/distributed_point_function.pb.h"
#include "dpf/internal/proto_validator.h"

namespace distributed_point_functions {
namespace dpf_internal {
// ProtoValidator is used to validate protos for DPF parameters, keys, and
// evaluation contexts. Also holds information computed from the DPF parameters,
// such as the mappings between hierarchy and tree levels.
class MpProtoValidator {
 public:
  // The negative logarithm of the total variation distance from uniform that a
  // *full* evaluation of a hierarchy level is allowed to have. Used as the
  // default value for DpfParameters that don't have an explicit per-element
  // security parameter set.
  static constexpr double kDefaultSecurityParameter = 40;

  // Security parameters that differ by less than this are considered equal.
  static constexpr double kSecurityParameterEpsilon = 0.0001;

  // Checks the validity of `parameters` and returns a ProtoValidator, which
  // will be used to validate DPF keys and evaluation contexts afterwards.
  //
  // Returns INVALID_ARGUMENT if `parameters` are invalid.
  static absl::StatusOr<std::unique_ptr<MpProtoValidator>> Create(
      const MpDpfParameters& parameters);

  // Checks the validity of `parameters`.
  // Returns OK on success, and INVALID_ARGUMENT otherwise.
  static absl::Status ValidateMpParameters(const MpDpfParameters& parameters);

  // Checks that `key` is valid for the `parameters` passed at construction.
  // Returns OK on success, and INVALID_ARGUMENT otherwise.
  absl::Status ValidateMpDpfKey(const MpDpfKey& key) const;

  // MpProtoValidator is not copyable.
  MpProtoValidator(const MpProtoValidator&) = delete;
  MpProtoValidator& operator=(const MpProtoValidator&) = delete;

  // MpProtoValidator is movable.
  MpProtoValidator(MpProtoValidator&&) = default;
  MpProtoValidator& operator=(MpProtoValidator&&) = default;

  // Getters.
  const MpDpfParameters& parameters() const { return parameters_; }

  const ProtoValidator& proto_valid() const { return *proto_validator_; }

 private:
  MpProtoValidator(MpDpfParameters parameters,
                   std::unique_ptr<ProtoValidator> proto_validator);

  // The MpDpfParameters passed at construction.
  MpDpfParameters parameters_;

  std::unique_ptr<ProtoValidator> proto_validator_;
};

}  // namespace dpf_internal
}  // namespace distributed_point_functions

#endif  // DISTRIBUTED_POINT_FUNCTIONS_DPF_INTERNAL_MP_PROTO_VALIDATOR_H_
