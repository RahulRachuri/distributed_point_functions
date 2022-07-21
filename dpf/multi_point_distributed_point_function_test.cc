#include "dpf/multi_point_distributed_point_function.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "absl/numeric/int128.h"
#include "absl/random/random.h"
#include "absl/strings/str_format.h"
#include "absl/utility/utility.h"
#include "dpf/internal/status_matchers.h"

namespace distributed_point_functions {
namespace {

using dpf_internal::IsOk;
using dpf_internal::IsOkAndHolds;
using dpf_internal::StatusIs;
using ::testing::Ne;
using ::testing::StartsWith;

TEST(MultiPointDistributedPointFunction, TestCreate) {
  for (int log_domain_size = 0; log_domain_size <= 128; ++log_domain_size) {
    for (int element_bitsize = 1; element_bitsize <= 128;
         element_bitsize *= 2) {
            for(int num_points = 1; num_points < 5 && num_points < 1 << log_domain_size; ++num_points) {
                MpDpfParameters parameters;
                parameters.set_number_points(num_points);
                 parameters.mutable_dpf_parameters()->set_log_domain_size(log_domain_size);
                parameters.mutable_dpf_parameters()->mutable_value_type()->mutable_integer()->set_bitsize(
                    element_bitsize);

                EXPECT_THAT(MultiPointDistributedPointFunction::Create(parameters),
                            IsOkAndHolds(Ne(nullptr)))
                    << "log_domain_size=" << log_domain_size
                    << " element_bitsize=" << element_bitsize << "num_points=" << num_points;
            }
      
    }
  }
}

TEST(MultiPointDistributedPointFunction, CreateFailsForMissingValueType) {

    MpDpfParameters parameters;
    parameters.set_number_points(5);
        parameters.mutable_dpf_parameters()->set_log_domain_size(10);

  EXPECT_THAT(
      MultiPointDistributedPointFunction::Create(parameters),
      StatusIs(absl::StatusCode::kInvalidArgument, "`value_type` is required"));
}

TEST(MultiPointDistributedPointFunction, CreateFailsForInvalidValueType) {
    MpDpfParameters parameters;
    parameters.set_number_points(5);
        parameters.mutable_dpf_parameters()->set_log_domain_size(10);

    *(parameters.mutable_dpf_parameters()->mutable_value_type()) = ValueType{};
  

  EXPECT_THAT(MultiPointDistributedPointFunction::Create(parameters),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       StartsWith("ValidateValueType: Unsupported ValueType")));
}

TEST(MultiPointDistributedPointFunction, CreateFailsForMissingDPFParameters) {
  MpDpfParameters parameters;
  parameters.set_number_points(5);

  EXPECT_THAT(MultiPointDistributedPointFunction::Create(parameters),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       StartsWith("ValidateMpParameters: dpf parameters are required")));

}

TEST(MultiPointDistributedPointFunction, CreateFailsForTooManyPoints) {
  MpDpfParameters parameters;
  parameters.set_number_points(20);
  parameters.mutable_dpf_parameters()->set_log_domain_size(2);
  parameters.mutable_dpf_parameters()->mutable_value_type()->mutable_integer()->set_bitsize(
                    64);

  EXPECT_THAT(MultiPointDistributedPointFunction::Create(parameters),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       StartsWith("ValidateMpParameters: number of points must be")));
}                       

TEST(MultiPointDistributedPointFunction, TestGenerateKeysTemplate) {
  MpDpfParameters parameters;

  parameters.set_number_points(2);
  parameters.mutable_dpf_parameters()->set_log_domain_size(10);
  *(parameters.mutable_dpf_parameters()->mutable_value_type()) =
      ToValueType<Tuple<uint32_t, uint64_t>>();    
  DPF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<MultiPointDistributedPointFunction> mp_dpf,
      MultiPointDistributedPointFunction::Create(parameters));

  std::vector<absl::uint128> indices = {47, 69};
  std::vector<Value> values = {ToValue(Tuple<uint32_t, uint64_t>{123, 456}), ToValue(Tuple<uint32_t, uint64_t>{234, 1337}) };
  absl::StatusOr<std::pair<MpDpfKey, MpDpfKey>> keys = mp_dpf->GenerateKeys(absl::MakeSpan(indices), absl::MakeSpan(values));
  EXPECT_THAT(keys, IsOk());
}

class RegularDpfKeyGenerationTest
    : public testing::TestWithParam<std::tuple<int, int, int>> {
 public:
  void SetUp() {
    std::tie(number_points_, log_domain_size_, element_bitsize_) = GetParam();
    if (number_points_ > (1 << log_domain_size_)) {
      return;
    }
    MpDpfParameters parameters;
    parameters.set_number_points(number_points_);
    parameters.mutable_dpf_parameters()->set_log_domain_size(log_domain_size_);
    parameters.mutable_dpf_parameters()->mutable_value_type()->mutable_integer()->set_bitsize(
        element_bitsize_);
    DPF_ASSERT_OK_AND_ASSIGN(mp_dpf_,
                             MultiPointDistributedPointFunction::Create(parameters));
    DPF_ASSERT_OK_AND_ASSIGN(
        mp_proto_validator_, dpf_internal::MpProtoValidator::Create({parameters}));
  }

 protected:
  int number_points_;
  int log_domain_size_;
  int element_bitsize_;
  std::unique_ptr<MultiPointDistributedPointFunction> mp_dpf_;
  std::unique_ptr<dpf_internal::MpProtoValidator> mp_proto_validator_;
};

TEST_P(RegularDpfKeyGenerationTest, KeyHasCorrectFormat) {
  if (number_points_ > (1 << log_domain_size_)) {
      return;
    }
  MpDpfKey key_a, key_b;
  std::vector<absl::uint128> indices(number_points_);
  std::iota(std::begin(indices), std::end(indices), 0);
  std::vector<Value> values(number_points_, ToValue(absl::uint128{3339090} & ((1 << element_bitsize_) - 1)));

  DPF_ASSERT_OK_AND_ASSIGN(std::tie(key_a, key_b), mp_dpf_->GenerateKeys(absl::MakeSpan(indices), absl::MakeSpan(values)));

  // Check that party is set correctly.
  EXPECT_EQ(key_a.party(), 0);
  EXPECT_EQ(key_b.party(), 1);
  // Check that keys are accepted by proto_validator_.
  DPF_EXPECT_OK(mp_proto_validator_->ValidateMpDpfKey(key_a));
  DPF_EXPECT_OK(mp_proto_validator_->ValidateMpDpfKey(key_b));
}

TEST_P(RegularDpfKeyGenerationTest, FailsIfBetaHasTheWrongSize) {
  EXPECT_THAT(
      dpf_->GenerateKeysIncremental(0, {1, 2}),
      StatusIs(absl::StatusCode::kInvalidArgument,
               "`beta` has to have the same size as `parameters` passed at "
               "construction"));
}
#if 0
TEST_P(RegularDpfKeyGenerationTest, FailsIfAlphaIsTooLarge) {
  if (log_domain_size_ >= 128) {
    // Alpha is an absl::uint128, so never too large in this case.
    return;
  }

  EXPECT_THAT(dpf_->GenerateKeys((absl::uint128{1} << log_domain_size_), 1),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "`alpha` must be smaller than the output domain size"));
}

TEST_P(RegularDpfKeyGenerationTest, FailsIfBetaIsTooLarge) {
  if (element_bitsize_ >= 128) {
    // Beta is an absl::uint128, so never too large in this case.
    return;
  }

  const auto beta = absl::uint128{1} << element_bitsize_;

  // Not testing error message, as it's an implementation detail of
  // ProtoValidator.
  EXPECT_THAT(dpf_->GenerateKeys(0, beta),
              StatusIs(absl::StatusCode::kInvalidArgument));
}
#endif

INSTANTIATE_TEST_SUITE_P(VaryNumberPointsAndDomainAndElementSizes, RegularDpfKeyGenerationTest,
                         testing::Combine(testing::Values(1, 2, 4, 8, 16), testing::Values(0, 1, 2, 3, 4, 5, 6,
                                                          7, 8, 9, 10, 32, 62),
                                          testing::Values(8, 16, 32, 64, 128)));
#if 0
struct DpfTestParameters {
  int log_domain_size;
  int element_bitsize;

  friend std::ostream& operator<<(std::ostream& os,
                                  const DpfTestParameters& parameters) {
    return os << "{ log_domain_size: " << parameters.log_domain_size
              << ", element_bitsize: " << parameters.element_bitsize << " }";
  }
};

class IncrementalDpfTest : public testing::TestWithParam<
                               std::tuple</*parameters*/
                                          std::vector<DpfTestParameters>,
                                          /*alpha*/ absl::uint128,
                                          /*beta*/ std::vector<absl::uint128>,
                                          /*level_step*/ int>> {
 protected:
  void SetUp() {
    const std::vector<DpfTestParameters>& parameters = std::get<0>(GetParam());
    parameters_.resize(parameters.size());
    for (int i = 0; i < static_cast<int>(parameters.size()); ++i) {
      parameters_[i].set_log_domain_size(parameters[i].log_domain_size);
      parameters_[i].mutable_value_type()->mutable_integer()->set_bitsize(
          parameters[i].element_bitsize);
    }
    DPF_ASSERT_OK_AND_ASSIGN(
        dpf_, MultiPointDistributedPointFunction::CreateIncremental(parameters_));
    alpha_ = std::get<1>(GetParam());
    beta_ = std::get<2>(GetParam());
    DPF_ASSERT_OK_AND_ASSIGN(keys_,
                             dpf_->GenerateKeysIncremental(alpha_, beta_));
    level_step_ = std::get<3>(
        GetParam());  // Number of hierarchy level to evaluate at once.
    DPF_ASSERT_OK_AND_ASSIGN(proto_validator_,
                             dpf_internal::ProtoValidator::Create(parameters_));
  }

  // Returns the prefix of `index` for the domain of `hierarchy_level`.
  absl::uint128 GetPrefixForLevel(int hierarchy_level, absl::uint128 index) {
    absl::uint128 result = 0;
    int shift_amount = parameters_.back().log_domain_size() -
                       parameters_[hierarchy_level].log_domain_size();
    if (shift_amount < 128) {
      result = index >> shift_amount;
    }
    return result;
  }

  // Evaluates both contexts `ctx0` and `ctx1` at `hierarchy level`, using the
  // appropriate prefixes of `evaluation_points`. Checks that the expansion of
  // both keys form correct DPF shares, i.e., they add up to
  // `beta_[ctx.hierarchy_level()]` under prefixes of `alpha_`, and to 0
  // otherwise.
  template <typename T>
  void EvaluateAndCheckLevel(int hierarchy_level,
                             absl::Span<const absl::uint128> evaluation_points,
                             EvaluationContext& ctx0, EvaluationContext& ctx1) {
    int previous_hierarchy_level = ctx0.previous_hierarchy_level();
    int current_log_domain_size =
        parameters_[hierarchy_level].log_domain_size();
    int previous_log_domain_size = 0;
    int num_expansions = 1;
    bool is_first_evaluation = previous_hierarchy_level < 0;
    // Generate prefixes if we're not on the first level.
    std::vector<absl::uint128> prefixes;
    if (!is_first_evaluation) {
      num_expansions = static_cast<int>(evaluation_points.size());
      prefixes.resize(evaluation_points.size());
      previous_log_domain_size =
          parameters_[previous_hierarchy_level].log_domain_size();
      for (int i = 0; i < static_cast<int>(evaluation_points.size()); ++i) {
        prefixes[i] =
            GetPrefixForLevel(previous_hierarchy_level, evaluation_points[i]);
      }
    }

    absl::StatusOr<std::vector<T>> result_0 =
        dpf_->EvaluateUntil<T>(hierarchy_level, prefixes, ctx0);
    absl::StatusOr<std::vector<T>> result_1 =
        dpf_->EvaluateUntil<T>(hierarchy_level, prefixes, ctx1);

    // Check results are ok.
    DPF_EXPECT_OK(result_0);
    DPF_EXPECT_OK(result_1);
    if (result_0.ok() && result_1.ok()) {
      // Check output sizes match.
      ASSERT_EQ(result_0->size(), result_1->size());
      int64_t outputs_per_prefix =
          int64_t{1} << (current_log_domain_size - previous_log_domain_size);
      int64_t expected_output_size = num_expansions * outputs_per_prefix;
      ASSERT_EQ(result_0->size(), expected_output_size);

      // Iterate over the outputs and check that they sum up to 0 or to
      // beta_[current_hierarchy_level].
      absl::uint128 previous_alpha_prefix = 0;
      if (!is_first_evaluation) {
        previous_alpha_prefix =
            GetPrefixForLevel(previous_hierarchy_level, alpha_);
      }
      absl::uint128 current_alpha_prefix =
          GetPrefixForLevel(hierarchy_level, alpha_);
      for (int64_t i = 0; i < expected_output_size; ++i) {
        int prefix_index = i / outputs_per_prefix;
        int prefix_expansion_index = i % outputs_per_prefix;
        // The output is on the path to alpha, if we're at the first level or
        // under a prefix of alpha, and the current block in the expansion of
        // the prefix is also on the path to alpha.
        if ((is_first_evaluation ||
             prefixes[prefix_index] == previous_alpha_prefix) &&
            prefix_expansion_index ==
                (current_alpha_prefix % outputs_per_prefix)) {
          // We need to static_cast here since otherwise operator+ returns an
          // unsigned int without doing a modular reduction, which causes the
          // test to fail on types with sizeof(T) < sizeof(unsigned).
          EXPECT_EQ(static_cast<T>((*result_0)[i] + (*result_1)[i]),
                    beta_[hierarchy_level])
              << "i=" << i << ", hierarchy_level=" << hierarchy_level
              << "\nparameters={\n"
              << parameters_[hierarchy_level].DebugString() << "}\n"
              << "previous_alpha_prefix=" << previous_alpha_prefix << "\n"
              << "current_alpha_prefix=" << current_alpha_prefix << "\n";
        } else {
          EXPECT_EQ(static_cast<T>((*result_0)[i] + (*result_1)[i]), 0)
              << "i=" << i << ", hierarchy_level=" << hierarchy_level
              << "\nparameters={\n"
              << parameters_[hierarchy_level].DebugString() << "}\n";
        }
      }
    }
  }

  std::vector<DpfParameters> parameters_;
  std::unique_ptr<MultiPointDistributedPointFunction> dpf_;
  absl::uint128 alpha_;
  std::vector<absl::uint128> beta_;
  std::pair<DpfKey, DpfKey> keys_;
  int level_step_;
  std::unique_ptr<dpf_internal::ProtoValidator> proto_validator_;
};

TEST_P(IncrementalDpfTest, CreateEvaluationContextCreatesValidContext) {
  DPF_ASSERT_OK_AND_ASSIGN(EvaluationContext ctx,
                           dpf_->CreateEvaluationContext(keys_.first));
  DPF_EXPECT_OK(proto_validator_->ValidateEvaluationContext(ctx));
}

TEST_P(IncrementalDpfTest, FailsIfPrefixNotPresentInCtx) {
  if (parameters_.size() < 3 || parameters_[0].log_domain_size() < 1 ||
      parameters_[0].value_type().integer().bitsize() != 128 ||
      parameters_[1].value_type().integer().bitsize() != 128 ||
      parameters_[2].value_type().integer().bitsize() != 128) {
    return;
  }
  DPF_ASSERT_OK_AND_ASSIGN(EvaluationContext ctx,
                           dpf_->CreateEvaluationContext(keys_.first));

  // Evaluate on prefixes 0 and 1, then delete partial evaluations for prefix 0.
  DPF_ASSERT_OK(dpf_->EvaluateNext<absl::uint128>({}, ctx));
  DPF_ASSERT_OK(dpf_->EvaluateNext<absl::uint128>({0, 1}, ctx));
  ctx.mutable_partial_evaluations()->erase(ctx.partial_evaluations().begin());

  // The missing prefix corresponds to hierarchy level 1, even though we
  // evaluate at level 2.
  EXPECT_THAT(dpf_->EvaluateNext<absl::uint128>({0}, ctx),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "Prefix not present in ctx.partial_evaluations at "
                       "hierarchy level 1"));
}

TEST_P(IncrementalDpfTest, FailsIfDuplicatePrefixInCtx) {
  if (parameters_.size() < 3 || parameters_[0].log_domain_size() < 1 ||
      parameters_[0].value_type().integer().bitsize() != 128 ||
      parameters_[1].value_type().integer().bitsize() != 128 ||
      parameters_[2].value_type().integer().bitsize() != 128) {
    return;
  }
  DPF_ASSERT_OK_AND_ASSIGN(EvaluationContext ctx,
                           dpf_->CreateEvaluationContext(keys_.first));

  // Evaluate on prefixes 0 and 1, then delete partial evaluations for prefix 0.
  DPF_ASSERT_OK(dpf_->EvaluateNext<absl::uint128>({}, ctx));
  DPF_ASSERT_OK(dpf_->EvaluateNext<absl::uint128>({0, 1}, ctx));
  *(ctx.add_partial_evaluations()) = ctx.partial_evaluations(0);

  // The missing prefix corresponds to hierarchy level 1, even though we
  // evaluate at level 2.
  EXPECT_THAT(dpf_->EvaluateNext<absl::uint128>({0}, ctx),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "Duplicate prefix in `ctx.partial_evaluations()`"));
}

TEST_P(IncrementalDpfTest, EvaluationFailsOnEmptyContext) {
  if (parameters_[0].value_type().integer().bitsize() != 128) {
    return;
  }
  EvaluationContext ctx;

  // We don't check the error message, since it depends on the ProtoValidator
  // implementation which is tested in the corresponding unit test.
  EXPECT_THAT(dpf_->EvaluateNext<absl::uint128>({}, ctx),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_P(IncrementalDpfTest, EvaluationFailsIfHierarchyLevelNegative) {
  if (parameters_[0].value_type().integer().bitsize() != 128) {
    return;
  }
  DPF_ASSERT_OK_AND_ASSIGN(EvaluationContext ctx,
                           dpf_->CreateEvaluationContext(keys_.first));

  EXPECT_THAT(dpf_->EvaluateUntil<absl::uint128>(-1, {}, ctx),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "`hierarchy_level` must be non-negative and less than "
                       "parameters_.size()"));
}

TEST_P(IncrementalDpfTest, EvaluationFailsIfHierarchyLevelTooLarge) {
  if (parameters_[0].value_type().integer().bitsize() != 128) {
    return;
  }
  DPF_ASSERT_OK_AND_ASSIGN(EvaluationContext ctx,
                           dpf_->CreateEvaluationContext(keys_.first));

  EXPECT_THAT(dpf_->EvaluateUntil<absl::uint128>(parameters_.size(), {}, ctx),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "`hierarchy_level` must be non-negative and less than "
                       "parameters_.size()"));
}

TEST_P(IncrementalDpfTest, EvaluationFailsIfValueTypeDoesntMatch) {
  using SomeStrangeType = Tuple<uint8_t, uint32_t, uint8_t, uint16_t, uint8_t>;
  DPF_ASSERT_OK_AND_ASSIGN(EvaluationContext ctx,
                           dpf_->CreateEvaluationContext(keys_.first));

  EXPECT_THAT(
      dpf_->EvaluateUntil<SomeStrangeType>(0, {}, ctx),
      StatusIs(absl::StatusCode::kInvalidArgument,
               "Value type T doesn't match parameters at `hierarchy_level`"));
}

TEST_P(IncrementalDpfTest, EvaluationFailsIfLevelAlreadyEvaluated) {
  if (parameters_.size() < 2 ||
      parameters_[0].value_type().integer().bitsize() != 128) {
    return;
  }
  DPF_ASSERT_OK_AND_ASSIGN(EvaluationContext ctx,
                           dpf_->CreateEvaluationContext(keys_.first));

  DPF_ASSERT_OK(dpf_->EvaluateUntil<absl::uint128>(0, {}, ctx));

  EXPECT_THAT(dpf_->EvaluateUntil<absl::uint128>(0, {}, ctx),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "`hierarchy_level` must be greater than "
                       "`ctx.previous_hierarchy_level`"));
}

TEST_P(IncrementalDpfTest, EvaluationFailsIfPrefixesNotEmptyOnFirstCall) {
  if (parameters_[0].value_type().integer().bitsize() != 128) {
    return;
  }
  DPF_ASSERT_OK_AND_ASSIGN(EvaluationContext ctx,
                           dpf_->CreateEvaluationContext(keys_.first));

  EXPECT_THAT(
      dpf_->EvaluateUntil<absl::uint128>(0, {0}, ctx),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          "`prefixes` must be empty if and only if this is the first call with "
          "`ctx`."));
}

TEST_P(IncrementalDpfTest, EvaluationFailsIfPrefixOutOfRange) {
  if (parameters_.size() < 2 ||
      parameters_[0].value_type().integer().bitsize() != 128 ||
      parameters_[1].value_type().integer().bitsize() != 128) {
    return;
  }
  DPF_ASSERT_OK_AND_ASSIGN(EvaluationContext ctx,
                           dpf_->CreateEvaluationContext(keys_.first));

  DPF_ASSERT_OK(dpf_->EvaluateUntil<absl::uint128>(0, {}, ctx));
  auto invalid_prefix = absl::uint128{1} << parameters_[0].log_domain_size();

  EXPECT_THAT(dpf_->EvaluateUntil<absl::uint128>(1, {invalid_prefix}, ctx),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       StrFormat("Index %d out of range for hierarchy level 0",
                                 invalid_prefix)));
}

TEST_P(IncrementalDpfTest, TestCorrectness) {
  DPF_ASSERT_OK_AND_ASSIGN(EvaluationContext ctx0,
                           dpf_->CreateEvaluationContext(keys_.first));
  DPF_ASSERT_OK_AND_ASSIGN(EvaluationContext ctx1,
                           dpf_->CreateEvaluationContext(keys_.second));

  // Generate a random set of evaluation points. The library should be able to
  // handle duplicates, so fixing the size to 1000 works even for smaller
  // domains.
  absl::BitGen rng;
  absl::uniform_int_distribution<uint64_t> dist;
  const int kNumEvaluationPoints = 1000;
  std::vector<absl::uint128> evaluation_points(kNumEvaluationPoints);
  for (int i = 0; i < kNumEvaluationPoints - 1; ++i) {
    evaluation_points[i] = absl::MakeUint128(dist(rng), dist(rng));
    if (parameters_.back().log_domain_size() < 128) {
      evaluation_points[i] %= absl::uint128{1}
                              << parameters_.back().log_domain_size();
    }
  }
  evaluation_points.back() = alpha_;  // Always evaluate on alpha_.

  int num_levels = static_cast<int>(parameters_.size());
  for (int i = level_step_ - 1; i < num_levels; i += level_step_) {
    switch (parameters_[i].value_type().integer().bitsize()) {
      case 8:
        EvaluateAndCheckLevel<uint8_t>(i, evaluation_points, ctx0, ctx1);
        break;
      case 16:
        EvaluateAndCheckLevel<uint16_t>(i, evaluation_points, ctx0, ctx1);
        break;
      case 32:
        EvaluateAndCheckLevel<uint32_t>(i, evaluation_points, ctx0, ctx1);
        break;
      case 64:
        EvaluateAndCheckLevel<uint64_t>(i, evaluation_points, ctx0, ctx1);
        break;
      case 128:
        EvaluateAndCheckLevel<absl::uint128>(i, evaluation_points, ctx0, ctx1);
        break;
      default:
        ASSERT_TRUE(0) << "Unsupported element_bitsize";
    }
  }
}

INSTANTIATE_TEST_SUITE_P(
    OneHierarchyLevelVaryElementSizes, IncrementalDpfTest,
    testing::Combine(
        // DPF parameters.
        testing::Values(
            // Vary element sizes, small domain size.
            std::vector<DpfTestParameters>{
                {.log_domain_size = 4, .element_bitsize = 8}},
            std::vector<DpfTestParameters>{
                {.log_domain_size = 4, .element_bitsize = 16}},
            std::vector<DpfTestParameters>{
                {.log_domain_size = 4, .element_bitsize = 32}},
            std::vector<DpfTestParameters>{
                {.log_domain_size = 4, .element_bitsize = 64}},
            std::vector<DpfTestParameters>{
                {.log_domain_size = 4, .element_bitsize = 128}},
            // Vary element sizes, medium domain size.
            std::vector<DpfTestParameters>{
                {.log_domain_size = 10, .element_bitsize = 8}},
            std::vector<DpfTestParameters>{
                {.log_domain_size = 10, .element_bitsize = 16}},
            std::vector<DpfTestParameters>{
                {.log_domain_size = 10, .element_bitsize = 32}},
            std::vector<DpfTestParameters>{
                {.log_domain_size = 10, .element_bitsize = 64}},
            std::vector<DpfTestParameters>{
                {.log_domain_size = 10, .element_bitsize = 128}}),
        testing::Values(0, 1, 15),  // alpha
        testing::Values(std::vector<absl::uint128>(1, 1),
                        std::vector<absl::uint128>(1, 100),
                        std::vector<absl::uint128>(1, 255)),  // beta
        testing::Values(1)));                                 // level_step

INSTANTIATE_TEST_SUITE_P(
    OneHierarchyLevelVaryDomainSizes, IncrementalDpfTest,
    testing::Combine(
        // DPF parameters.
        testing::Values(
            // Vary domain sizes, small element size.
            std::vector<DpfTestParameters>{
                {.log_domain_size = 0, .element_bitsize = 8}},
            std::vector<DpfTestParameters>{
                {.log_domain_size = 1, .element_bitsize = 8}},
            std::vector<DpfTestParameters>{
                {.log_domain_size = 2, .element_bitsize = 8}},
            std::vector<DpfTestParameters>{
                {.log_domain_size = 3, .element_bitsize = 8}},
            std::vector<DpfTestParameters>{
                {.log_domain_size = 4, .element_bitsize = 8}},
            std::vector<DpfTestParameters>{
                {.log_domain_size = 5, .element_bitsize = 8}},
            std::vector<DpfTestParameters>{
                {.log_domain_size = 6, .element_bitsize = 8}},
            std::vector<DpfTestParameters>{
                {.log_domain_size = 7, .element_bitsize = 8}},
            std::vector<DpfTestParameters>{
                {.log_domain_size = 8, .element_bitsize = 8}},
            std::vector<DpfTestParameters>{
                {.log_domain_size = 9, .element_bitsize = 8}},
            // Vary domain sizes, medium element size.
            std::vector<DpfTestParameters>{
                {.log_domain_size = 0, .element_bitsize = 64}},
            std::vector<DpfTestParameters>{
                {.log_domain_size = 1, .element_bitsize = 64}},
            std::vector<DpfTestParameters>{
                {.log_domain_size = 2, .element_bitsize = 64}},
            std::vector<DpfTestParameters>{
                {.log_domain_size = 3, .element_bitsize = 64}},
            std::vector<DpfTestParameters>{
                {.log_domain_size = 4, .element_bitsize = 64}},
            std::vector<DpfTestParameters>{
                {.log_domain_size = 5, .element_bitsize = 64}},
            std::vector<DpfTestParameters>{
                {.log_domain_size = 6, .element_bitsize = 64}},
            std::vector<DpfTestParameters>{
                {.log_domain_size = 7, .element_bitsize = 64}},
            std::vector<DpfTestParameters>{
                {.log_domain_size = 8, .element_bitsize = 64}},
            std::vector<DpfTestParameters>{
                {.log_domain_size = 9, .element_bitsize = 64}},
            // Vary domain sizes, large element size.
            std::vector<DpfTestParameters>{
                {.log_domain_size = 0, .element_bitsize = 128}},
            std::vector<DpfTestParameters>{
                {.log_domain_size = 1, .element_bitsize = 128}},
            std::vector<DpfTestParameters>{
                {.log_domain_size = 2, .element_bitsize = 128}},
            std::vector<DpfTestParameters>{
                {.log_domain_size = 3, .element_bitsize = 128}},
            std::vector<DpfTestParameters>{
                {.log_domain_size = 4, .element_bitsize = 128}},
            std::vector<DpfTestParameters>{
                {.log_domain_size = 5, .element_bitsize = 128}},
            std::vector<DpfTestParameters>{
                {.log_domain_size = 6, .element_bitsize = 128}},
            std::vector<DpfTestParameters>{
                {.log_domain_size = 7, .element_bitsize = 128}},
            std::vector<DpfTestParameters>{
                {.log_domain_size = 8, .element_bitsize = 128}},
            std::vector<DpfTestParameters>{
                {.log_domain_size = 9, .element_bitsize = 128}}),
        testing::Values(0),  // alpha
        testing::Values(std::vector<absl::uint128>(1, 1),
                        std::vector<absl::uint128>(1, 100),
                        std::vector<absl::uint128>(1, 255)),  // beta
        testing::Values(1)));                                 // level_step

INSTANTIATE_TEST_SUITE_P(
    TwoHierarchyLevels, IncrementalDpfTest,
    testing::Combine(
        // DPF parameters.
        testing::Values(
            // Equal element sizes.
            std::vector<DpfTestParameters>{
                {.log_domain_size = 5, .element_bitsize = 8},
                {.log_domain_size = 10, .element_bitsize = 8}},
            std::vector<DpfTestParameters>{
                {.log_domain_size = 5, .element_bitsize = 16},
                {.log_domain_size = 10, .element_bitsize = 16}},
            std::vector<DpfTestParameters>{
                {.log_domain_size = 5, .element_bitsize = 32},
                {.log_domain_size = 10, .element_bitsize = 32}},
            std::vector<DpfTestParameters>{
                {.log_domain_size = 5, .element_bitsize = 64},
                {.log_domain_size = 10, .element_bitsize = 64}},
            std::vector<DpfTestParameters>{
                {.log_domain_size = 5, .element_bitsize = 128},
                {.log_domain_size = 10, .element_bitsize = 128}},
            // First correction is in seed word.
            std::vector<DpfTestParameters>{
                {.log_domain_size = 0, .element_bitsize = 8},
                {.log_domain_size = 10, .element_bitsize = 128}},
            std::vector<DpfTestParameters>{
                {.log_domain_size = 0, .element_bitsize = 16},
                {.log_domain_size = 10, .element_bitsize = 128}},
            std::vector<DpfTestParameters>{
                {.log_domain_size = 0, .element_bitsize = 32},
                {.log_domain_size = 10, .element_bitsize = 128}},
            std::vector<DpfTestParameters>{
                {.log_domain_size = 0, .element_bitsize = 64},
                {.log_domain_size = 10, .element_bitsize = 128}},
            std::vector<DpfTestParameters>{
                {.log_domain_size = 0, .element_bitsize = 128},
                {.log_domain_size = 10, .element_bitsize = 128}}),
        testing::Values(0, 1, 2, 100, 1023),  // alpha
        testing::Values(std::vector<absl::uint128>({1, 2}),
                        std::vector<absl::uint128>({80, 90}),
                        std::vector<absl::uint128>({255, 255})),  // beta
        testing::Values(1, 2)));                                  // level_step

INSTANTIATE_TEST_SUITE_P(
    ThreeHierarchyLevels, IncrementalDpfTest,
    testing::Combine(
        // DPF parameters.
        testing::Values<std::vector<DpfTestParameters>>(
            // Equal element sizes.
            std::vector<DpfTestParameters>{
                {.log_domain_size = 5, .element_bitsize = 8},
                {.log_domain_size = 10, .element_bitsize = 8},
                {.log_domain_size = 15, .element_bitsize = 8}},
            std::vector<DpfTestParameters>{
                {.log_domain_size = 5, .element_bitsize = 16},
                {.log_domain_size = 10, .element_bitsize = 16},
                {.log_domain_size = 15, .element_bitsize = 16}},
            std::vector<DpfTestParameters>{
                {.log_domain_size = 5, .element_bitsize = 32},
                {.log_domain_size = 10, .element_bitsize = 32},
                {.log_domain_size = 15, .element_bitsize = 32}},
            std::vector<DpfTestParameters>{
                {.log_domain_size = 5, .element_bitsize = 64},
                {.log_domain_size = 10, .element_bitsize = 64},
                {.log_domain_size = 15, .element_bitsize = 64}},
            std::vector<DpfTestParameters>{
                {.log_domain_size = 5, .element_bitsize = 128},
                {.log_domain_size = 10, .element_bitsize = 128},
                {.log_domain_size = 15, .element_bitsize = 128}},
            // Varying element sizes
            std::vector<DpfTestParameters>{
                {.log_domain_size = 5, .element_bitsize = 8},
                {.log_domain_size = 10, .element_bitsize = 16},
                {.log_domain_size = 15, .element_bitsize = 32}},
            // Small level distances.
            std::vector<DpfTestParameters>{
                {.log_domain_size = 4, .element_bitsize = 8},
                {.log_domain_size = 5, .element_bitsize = 8},
                {.log_domain_size = 6, .element_bitsize = 8}},
            std::vector<DpfTestParameters>{
                {.log_domain_size = 3, .element_bitsize = 16},
                {.log_domain_size = 4, .element_bitsize = 16},
                {.log_domain_size = 5, .element_bitsize = 16}},
            std::vector<DpfTestParameters>{
                {.log_domain_size = 2, .element_bitsize = 32},
                {.log_domain_size = 3, .element_bitsize = 32},
                {.log_domain_size = 4, .element_bitsize = 32}},
            std::vector<DpfTestParameters>{
                {.log_domain_size = 1, .element_bitsize = 64},
                {.log_domain_size = 2, .element_bitsize = 64},
                {.log_domain_size = 3, .element_bitsize = 64}},
            std::vector<DpfTestParameters>{
                {.log_domain_size = 0, .element_bitsize = 128},
                {.log_domain_size = 1, .element_bitsize = 128},
                {.log_domain_size = 2, .element_bitsize = 128}}),
        testing::Values(0, 1),                                   // alpha
        testing::Values(std::vector<absl::uint128>({1, 2, 3})),  // beta
        testing::Values(1, 2)));                                 // level_step

INSTANTIATE_TEST_SUITE_P(
    MaximumOutputDomainSize, IncrementalDpfTest,
    testing::Combine(
        // DPF parameters. We want to be able to evaluate at every bit, so this
        // lambda returns a vector with 129 parameters with log domain sizes
        // 0...128.
        testing::Values([]() -> std::vector<DpfTestParameters> {
          std::vector<DpfTestParameters> parameters(129);
          for (int i = 0; i < static_cast<int>(parameters.size()); ++i) {
            parameters[i].log_domain_size = i;
            parameters[i].element_bitsize = 64;
          }
          return parameters;
        }()),
        testing::Values(absl::MakeUint128(23, 42)),                 // alpha
        testing::Values(std::vector<absl::uint128>(129, 1234567)),  // beta
        testing::Values(1, 2, 3, 5, 7)));  // level_step

template <typename T>
class DpfEvaluationTest : public ::testing::Test {
 protected:
  void SetUp() { SetUp(10, 23); }
  void SetUp(int log_domain_size, absl::uint128 alpha) {
    log_domain_size_ = log_domain_size;
    alpha_ = alpha;
    SetTo42(beta_);
    parameters_.set_log_domain_size(log_domain_size_);
    parameters_.set_security_parameter(48);
    *(parameters_.mutable_value_type()) = ToValueType<T>();
    DPF_ASSERT_OK_AND_ASSIGN(dpf_,
                             MultiPointDistributedPointFunction::Create(parameters_));
    DPF_ASSERT_OK(this->dpf_->template RegisterValueType<T>());
    DPF_ASSERT_OK_AND_ASSIGN(
        keys_, this->dpf_->GenerateKeys(this->alpha_, this->beta_));
  }

  // Helper function that recursively sets all elements of a tuple to 42.
  template <typename T0>
  static void SetTo42(T0& x) {
    x = T0(42);
  }
  template <typename T0, typename... Tn>
  static void SetTo42(T0& x0, Tn&... xn) {
    SetTo42(x0);
    SetTo42(xn...);
  }
  template <typename... Tn>
  static void SetTo42(Tuple<Tn...>& x) {
    absl::apply([](auto&... in) { SetTo42(in...); }, x.value());
  }

  int log_domain_size_;
  absl::uint128 alpha_;
  T beta_;
  DpfParameters parameters_;
  std::unique_ptr<MultiPointDistributedPointFunction> dpf_;
  std::pair<DpfKey, DpfKey> keys_;
};

using MyIntModN = IntModN<uint32_t, 4294967291u>;                // 2**32 - 5.
using MyIntModN64 = IntModN<uint64_t, 18446744073709551557ull>;  // 2**64 - 59.
#ifdef ABSL_HAVE_INTRINSIC_INT128
using MyIntModN128 =
    IntModN<absl::uint128, (unsigned __int128)(absl::MakeUint128(
                               65535u, 18446744073709551551ull))>;  // 2**80-65
#endif
using DpfEvaluationTypes = ::testing::Types<
    // Tuple
    Tuple<uint8_t>, Tuple<uint32_t>, Tuple<absl::uint128>,
    Tuple<uint32_t, uint32_t>, Tuple<uint32_t, uint64_t>,
    Tuple<uint64_t, uint64_t>, Tuple<uint8_t, uint16_t, uint32_t, uint64_t>,
    Tuple<uint32_t, uint32_t, uint32_t, uint32_t>,
    Tuple<uint32_t, Tuple<uint32_t, uint32_t>, uint32_t>,
    Tuple<uint32_t, absl::uint128>,
    // IntModN
    MyIntModN, Tuple<MyIntModN>, Tuple<uint32_t, MyIntModN>,
    Tuple<absl::uint128, MyIntModN>, Tuple<MyIntModN, Tuple<MyIntModN>>,
    Tuple<MyIntModN, MyIntModN, MyIntModN, MyIntModN, MyIntModN>,
    Tuple<MyIntModN64, MyIntModN64>
#ifdef ABSL_HAVE_INTRINSIC_INT128
    ,
    Tuple<MyIntModN128, MyIntModN128>,
#endif
    // XorWrapper
    XorWrapper<uint8_t>, XorWrapper<absl::uint128>,
    Tuple<XorWrapper<uint32_t>, absl::uint128>>;
TYPED_TEST_SUITE(DpfEvaluationTest, DpfEvaluationTypes);

TYPED_TEST(DpfEvaluationTest, TestRegularDpf) {
  DPF_ASSERT_OK_AND_ASSIGN(
      EvaluationContext ctx_1,
      this->dpf_->CreateEvaluationContext(this->keys_.first));
  DPF_ASSERT_OK_AND_ASSIGN(
      EvaluationContext ctx_2,
      this->dpf_->CreateEvaluationContext(this->keys_.second));
  DPF_ASSERT_OK_AND_ASSIGN(
      std::vector<TypeParam> output_1,
      this->dpf_->template EvaluateNext<TypeParam>({}, ctx_1));
  DPF_ASSERT_OK_AND_ASSIGN(
      std::vector<TypeParam> output_2,
      this->dpf_->template EvaluateNext<TypeParam>({}, ctx_2));

  EXPECT_EQ(output_1.size(), 1 << this->log_domain_size_);
  EXPECT_EQ(output_2.size(), 1 << this->log_domain_size_);
  for (int i = 0; i < (1 << this->log_domain_size_); ++i) {
    TypeParam sum = output_1[i] + output_2[i];
    if (i == this->alpha_) {
      EXPECT_EQ(sum, this->beta_);
    } else {
      EXPECT_EQ(sum, TypeParam{});
    }
  }
}

TYPED_TEST(DpfEvaluationTest, TestBatchSinglePointEvaluation) {
  // Set Up with a large output domain, to make sure this works.
  for (int log_domain_size : {0, 1, 2, 32, 128}) {
    absl::uint128 max_evaluation_point = absl::Uint128Max();
    if (log_domain_size < 128) {
      max_evaluation_point = (absl::uint128{1} << log_domain_size) - 1;
    }
    const absl::uint128 alpha = 23 & max_evaluation_point;
    this->SetUp(log_domain_size, alpha);
    for (int num_evaluation_points : {0, 1, 2, 100, 1000}) {
      std::vector<absl::uint128> evaluation_points(num_evaluation_points);
      for (int i = 0; i < num_evaluation_points; ++i) {
        evaluation_points[i] = i & max_evaluation_point;
      }
      DPF_ASSERT_OK_AND_ASSIGN(std::vector<TypeParam> output_1,
                               this->dpf_->template EvaluateAt<TypeParam>(
                                   this->keys_.first, 0, evaluation_points));
      DPF_ASSERT_OK_AND_ASSIGN(std::vector<TypeParam> output_2,
                               this->dpf_->template EvaluateAt<TypeParam>(
                                   this->keys_.second, 0, evaluation_points));
      ASSERT_EQ(output_1.size(), output_2.size());
      ASSERT_EQ(output_1.size(), num_evaluation_points);

      for (int i = 0; i < num_evaluation_points; ++i) {
        TypeParam sum = output_1[i] + output_2[i];
        if (evaluation_points[i] == this->alpha_) {
          EXPECT_EQ(sum, this->beta_)
              << "i=" << i << ", log_domain_size=" << log_domain_size;
        } else {
          EXPECT_EQ(sum, TypeParam{})
              << "i=" << i << ", log_domain_size=" << log_domain_size;
        }
      }
    }
  }
}
#endif
}  // namespace
}  // namespace distributed_point_functions
