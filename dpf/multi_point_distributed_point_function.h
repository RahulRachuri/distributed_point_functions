#ifndef DISTRIBUTED_POINT_FUNCTIONS_DPF_MULTI_POINT_DISTRIBUTED_POINT_FUNCTION_H_
#define DISTRIBUTED_POINT_FUNCTIONS_DPF_MULTI_POINT_DISTRIBUTED_POINT_FUNCTION_H_

#include <absl/status/statusor.h>
#include "dpf/distributed_point_function.h"
#include "dpf/distributed_point_function.pb.h"

namespace distributed_point_functions {

class MultiPointDistributedPointFunction {
 public:

  static absl::StatusOr<std::unique_ptr<MultiPointDistributedPointFunction>> Create(
      const MpDpfParameters& parameters);

  absl::StatusOr<std::pair<DpfKey, DpfKey>> GenerateKeys(
      absl::Span<absl::uint128> alphas, absl::Span<const Value> betas);

  template <typename T>
  absl::StatusOr<std::vector<T>> EvaluateAt(
      const DpfKey& key, int hierarchy_level,
      absl::Span<const absl::uint128> evaluation_points) const;

  // Returns the DpfParameters of this DPF.
  inline const MpDpfParameters& parameters() const {
    return parameters_;
  }

 private:
  const MpDpfParameters parameters_;
  std::vector<DistributedPointFunction> single_point_dpfs_;
};

}  // namespace distributed_point_functions

#endif  // DISTRIBUTED_POINT_FUNCTIONS_DPF_MULTI_POINT_DISTRIBUTED_POINT_FUNCTION_H_
