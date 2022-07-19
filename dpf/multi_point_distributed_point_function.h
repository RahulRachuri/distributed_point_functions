#ifndef DISTRIBUTED_POINT_FUNCTIONS_DPF_MULTI_POINT_DISTRIBUTED_POINT_FUNCTION_H_
#define DISTRIBUTED_POINT_FUNCTIONS_DPF_MULTI_POINT_DISTRIBUTED_POINT_FUNCTION_H_

#include "absl/status/statusor.h"
#include "dpf/cuckoo.h"
#include "dpf/distributed_point_function.h"
#include "dpf/distributed_point_function.pb.h"
#include <memory>

namespace distributed_point_functions {

class MultiPointDistributedPointFunction {
 public:

  static absl::StatusOr<std::unique_ptr<MultiPointDistributedPointFunction>> Create(
      const MpDpfParameters& parameters);

  absl::StatusOr<std::pair<MpDpfKey, MpDpfKey>> GenerateKeys(
      absl::Span<absl::uint128> alphas, absl::Span<const Value> betas);

  template <typename T>
  absl::StatusOr<std::vector<T>> EvaluateAt(
      const MpDpfKey& key, int hierarchy_level,
      absl::Span<const absl::uint128> evaluation_points) const;

  // Returns the DpfParameters of this DPF.
  inline const MpDpfParameters& parameters() const {
    return parameters_;
  }

 private:
  MultiPointDistributedPointFunction(const MpDpfParameters& parameters, std::unique_ptr<Cuckoo>&& cuckoo_context_);

  const MpDpfParameters parameters_;
  std::unique_ptr<Cuckoo> cuckoo_context_;
  std::vector<DistributedPointFunction> single_point_dpfs_;
};

}  // namespace distributed_point_functions

#endif  // DISTRIBUTED_POINT_FUNCTIONS_DPF_MULTI_POINT_DISTRIBUTED_POINT_FUNCTION_H_
