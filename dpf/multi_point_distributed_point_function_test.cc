#include "dpf/multi_point_distributed_point_function.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <iterator>

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
      for (int num_points = 1;
           num_points < 5 && num_points < 1 << log_domain_size; ++num_points) {
        MpDpfParameters parameters;
        parameters.set_number_points(num_points);
        parameters.mutable_dpf_parameters()->set_log_domain_size(
            log_domain_size);
        parameters.mutable_dpf_parameters()
            ->mutable_value_type()
            ->mutable_integer()
            ->set_bitsize(element_bitsize);

        EXPECT_THAT(MultiPointDistributedPointFunction::Create(parameters),
                    IsOkAndHolds(Ne(nullptr)))
            << "log_domain_size=" << log_domain_size
            << " element_bitsize=" << element_bitsize
            << "num_points=" << num_points;
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

  EXPECT_THAT(
      MultiPointDistributedPointFunction::Create(parameters),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          StartsWith("ValidateMpParameters: dpf parameters are required")));
}

TEST(MultiPointDistributedPointFunction, CreateFailsForTooManyPoints) {
  MpDpfParameters parameters;
  parameters.set_number_points(20);
  parameters.mutable_dpf_parameters()->set_log_domain_size(2);
  parameters.mutable_dpf_parameters()
      ->mutable_value_type()
      ->mutable_integer()
      ->set_bitsize(64);

  EXPECT_THAT(
      MultiPointDistributedPointFunction::Create(parameters),
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

  mp_dpf->SetValueTypeRegistrationFunction([](auto& sp_dpf) {
    return sp_dpf.template RegisterValueType<Tuple<uint32_t, uint64_t>>();
  });

  std::vector<absl::uint128> indices = {47, 69};
  std::vector<Value> values = {ToValue(Tuple<uint32_t, uint64_t>{123, 456}),
                               ToValue(Tuple<uint32_t, uint64_t>{234, 1337})};
  absl::StatusOr<std::pair<MpDpfKey, MpDpfKey>> keys =
      mp_dpf->GenerateKeys(absl::MakeSpan(indices), absl::MakeSpan(values));
  EXPECT_THAT(keys, IsOk());
}

class RegularMpDpfKeyGenerationTest
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
    parameters.mutable_dpf_parameters()
        ->mutable_value_type()
        ->mutable_integer()
        ->set_bitsize(element_bitsize_);
    DPF_ASSERT_OK_AND_ASSIGN(
        mp_dpf_, MultiPointDistributedPointFunction::Create(parameters));
    DPF_ASSERT_OK_AND_ASSIGN(
        mp_proto_validator_,
        dpf_internal::MpProtoValidator::Create({parameters}));
  }

  std::pair<std::vector<absl::uint128>, std::vector<Value>> GeneratePoints() {
    std::vector<absl::uint128> indices(number_points_);
    std::iota(std::begin(indices), std::end(indices), 0);
    std::vector<Value> values(number_points_,
                              ToValue(absl::uint128{0x1337424247471337} &
                                      ((1 << element_bitsize_) - 1)));
    return std::make_pair(indices, values);
  }

 protected:
  int number_points_;
  int log_domain_size_;
  int element_bitsize_;
  std::unique_ptr<MultiPointDistributedPointFunction> mp_dpf_;
  std::unique_ptr<dpf_internal::MpProtoValidator> mp_proto_validator_;
};

TEST_P(RegularMpDpfKeyGenerationTest, KeyHasCorrectFormat) {
  if (number_points_ > (1 << log_domain_size_)) {
    return;
  }
  MpDpfKey key_a, key_b;
  const auto [indices, values] = GeneratePoints();
  DPF_ASSERT_OK_AND_ASSIGN(
      std::tie(key_a, key_b),
      mp_dpf_->GenerateKeys(absl::MakeSpan(indices), absl::MakeSpan(values)));

  // Check that party is set correctly.
  EXPECT_EQ(key_a.party(), 0);
  EXPECT_EQ(key_b.party(), 1);
  // Check that keys are accepted by proto_validator_.
  DPF_EXPECT_OK(mp_proto_validator_->ValidateMpDpfKey(key_a));
  DPF_EXPECT_OK(mp_proto_validator_->ValidateMpDpfKey(key_b));
}

TEST_P(RegularMpDpfKeyGenerationTest, FailsIfAlphaHasTheWrongSize) {
  if (number_points_ > (1 << log_domain_size_)) {
    return;
  }
  auto [indices, values] = GeneratePoints();
  indices.push_back(absl::uint128(1));
  EXPECT_THAT(
      mp_dpf_->GenerateKeys(indices, values),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          "`alphas` has to have the same size as `number_points` passed at "
          "construction"));
}

TEST_P(RegularMpDpfKeyGenerationTest, FailsIfBetaHasTheWrongSize) {
  if (number_points_ > (1 << log_domain_size_)) {
    return;
  }
  auto [indices, values] = GeneratePoints();
  values.push_back(ToValue(absl::uint128(1)));
  EXPECT_THAT(
      mp_dpf_->GenerateKeys(indices, values),
      StatusIs(absl::StatusCode::kInvalidArgument,
               "`betas` has to have the same size as `number_points` passed at "
               "construction"));
}

TEST_P(RegularMpDpfKeyGenerationTest, FailsIfAlphaIsTooLarge) {
  if (number_points_ > (1 << log_domain_size_)) {
    return;
  }
  if (log_domain_size_ >= 128) {
    // Alpha is an absl::uint128, so never too large in this case.
    return;
  }

  auto [indices, values] = GeneratePoints();
  indices[0] = absl::uint128{1} << log_domain_size_;

  EXPECT_THAT(
      mp_dpf_->GenerateKeys(indices, values),
      StatusIs(absl::StatusCode::kInvalidArgument,
               "each `alpha` must be smaller than the output domain size"));
}

TEST_P(RegularMpDpfKeyGenerationTest, FailsIfBetaIsTooLarge) {
  if (number_points_ > (1 << log_domain_size_)) {
    return;
  }
  if (element_bitsize_ >= 128) {
    // Beta is an absl::uint128, so never too large in this case.
    return;
  }

  auto [indices, values] = GeneratePoints();
  values[0] = ToValue(absl::uint128{1} << element_bitsize_);

  // Not testing error message, as it's an implementation detail of
  // ProtoValidator.
  EXPECT_THAT(mp_dpf_->GenerateKeys(indices, values),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

INSTANTIATE_TEST_SUITE_P(VaryNumberPointsAndDomainAndElementSizes,
                         RegularMpDpfKeyGenerationTest,
                         testing::Combine(testing::Values(1, 2, 4, 8, 16),
                                          testing::Values(0, 1, 2, 8, 10, 12),
                                          testing::Values(8, 32, 128)));

template <typename T>
class MpDpfEvaluationTest : public ::testing::Test {
 protected:
  void SetUp() {
    std::vector<absl::uint128> alphas = {42,  1,  13}; //, 37,  42,
                                         //47, 69, 96, 116, 127};
    //                                   ^ Lenny's numbers     ^ Rahy's numbers
    SetUp(3, 7, absl::MakeSpan(alphas));
  }
  void SetUp(int number_points, int log_domain_size,
             absl::Span<const absl::uint128> alphas) {
    log_domain_size_ = log_domain_size;
    alphas_ = {std::begin(alphas), std::end(alphas)};
    std::generate_n(std::back_inserter(betas_), number_points, [] {
      T x;
      MpDpfEvaluationTest::SetTo42(x);
      return x;
    });
    std::transform(std::begin(this->betas_), std::end(this->betas_),
                   std::back_inserter(this->betas_values_), ToValue<T>);
    std::cerr << "betas_ = [ ";
    for (const auto beta_i : betas_) {
      std::cerr << beta_i << ", ";
    }
    std::cerr << " ]\n";
    // SetTo42(betas_);
    parameters_.set_number_points(number_points);
    parameters_.mutable_dpf_parameters()->set_log_domain_size(log_domain_size_);
    parameters_.mutable_dpf_parameters()->set_security_parameter(48);
    *(parameters_.mutable_dpf_parameters()->mutable_value_type()) =
        ToValueType<T>();
    DPF_ASSERT_OK_AND_ASSIGN(
        mp_dpf_, MultiPointDistributedPointFunction::Create(parameters_));
    this->mp_dpf_->SetValueTypeRegistrationFunction(
        [](auto& sp_dpf) { return sp_dpf.template RegisterValueType<T>(); });
    // DPF_ASSERT_OK(this->mp_dpf_->template RegisterValueType<T>());
    DPF_ASSERT_OK_AND_ASSIGN(
        keys_, this->mp_dpf_->GenerateKeys(
                   this->alphas_, absl::MakeSpan(this->betas_values_)));
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
  std::vector<absl::uint128> alphas_;
  std::vector<T> betas_;
  std::vector<Value> betas_values_;
  MpDpfParameters parameters_;
  std::unique_ptr<MultiPointDistributedPointFunction> mp_dpf_;
  std::pair<MpDpfKey, MpDpfKey> keys_;
};

using MyIntModN = IntModN<uint32_t, 4294967291u>;                // 2**32 - 5.
using MyIntModN64 = IntModN<uint64_t, 18446744073709551557ull>;  // 2**64 - 59.
#ifdef ABSL_HAVE_INTRINSIC_INT128
using MyIntModN128 =
    IntModN<absl::uint128, (unsigned __int128)(absl::MakeUint128(
                               65535u, 18446744073709551551ull))>;  // 2**80-65
#endif
using MpDpfEvaluationTypes = ::testing::Types<
    uint8_t, uint16_t
    //    // Tuple
    //    Tuple<uint8_t>, Tuple<uint32_t>, Tuple<absl::uint128>,
    //    Tuple<uint32_t, uint32_t>, Tuple<uint32_t, uint64_t>,
    //    Tuple<uint64_t, uint64_t>, Tuple<uint8_t, uint16_t, uint32_t,
    //    uint64_t>, Tuple<uint32_t, uint32_t, uint32_t, uint32_t>,
    //    Tuple<uint32_t, Tuple<uint32_t, uint32_t>, uint32_t>,
    //    Tuple<uint32_t, absl::uint128>,
    //    // IntModN
    //    MyIntModN, Tuple<MyIntModN>, Tuple<uint32_t, MyIntModN>,
    //    Tuple<absl::uint128, MyIntModN>, Tuple<MyIntModN, Tuple<MyIntModN>>,
    //    Tuple<MyIntModN, MyIntModN, MyIntModN, MyIntModN, MyIntModN>,
    //    Tuple<MyIntModN64, MyIntModN64>
    //#ifdef ABSL_HAVE_INTRINSIC_INT128
    //    ,
    //    Tuple<MyIntModN128, MyIntModN128>,
    //#endif
    //    // XorWrapper
    //    XorWrapper<uint8_t>, XorWrapper<absl::uint128>,
    //    Tuple<XorWrapper<uint32_t>, absl::uint128>
    >;
TYPED_TEST_SUITE(MpDpfEvaluationTest, MpDpfEvaluationTypes);

TYPED_TEST(MpDpfEvaluationTest, TestRegularDpf) {
  std::vector<absl::uint128> evaluation_points(1 << this->log_domain_size_);
  std::iota(std::begin(evaluation_points), std::end(evaluation_points), 0);

  DPF_ASSERT_OK_AND_ASSIGN(
      std::vector<TypeParam> output_1,
      this->mp_dpf_->template EvaluateAt<TypeParam>(
          this->keys_.first, absl::MakeSpan(evaluation_points)));
  DPF_ASSERT_OK_AND_ASSIGN(
      std::vector<TypeParam> output_2,
      this->mp_dpf_->template EvaluateAt<TypeParam>(
          this->keys_.second, absl::MakeSpan(evaluation_points)));

  EXPECT_EQ(output_1.size(), 1 << this->log_domain_size_);
  EXPECT_EQ(output_2.size(), 1 << this->log_domain_size_);
  for (int i = 0; i < (1 << this->log_domain_size_); ++i) {
    TypeParam sum = output_1[i] + output_2[i];
    // check if i is one of the alpha_j's
    if (auto it =
            std::find(std::begin(this->alphas_), std::end(this->alphas_), i);
        it != std::end(this->alphas_)) {
      // compute the index j of alpha_j in the array of alphas
      auto point_idx = std::distance(std::begin(this->alphas_), it);
      // const auto alpha_j = *it;
      // find the corresponding beta_j
      const auto beta_j = this->betas_[point_idx];
      // check that alpha_j maps to beta_j
      EXPECT_EQ(sum, beta_j) << "wrong value '" << sum << "' != '" << beta_j << "' at index " << i;
    } else {
      // i is not one of the alpha
      // -> check that i maps to 0
      EXPECT_EQ(sum, TypeParam{}) << "wrong value '" << sum << "' != 0 at index " << i;
    }
  }
}

#if 0

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
