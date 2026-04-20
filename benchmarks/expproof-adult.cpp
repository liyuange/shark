#include <shark/protocols/init.hpp>
#include <shark/protocols/finalize.hpp>
#include <shark/protocols/input.hpp>
#include <shark/protocols/output.hpp>
#include <shark/protocols/relutruncate.hpp>
#include <shark/protocols/matmul.hpp>
#include <shark/protocols/add.hpp>
#include <shark/protocols/ars.hpp>

#include <shark/protocols/common.hpp>
#include <shark/utils/timer.hpp>

using u64 = shark::u64;
using namespace shark::protocols;

namespace {

constexpr u64 kFeatureCount = 14;
constexpr u64 kHidden = 16;
constexpr u64 kOutputCount = 1;
constexpr int kFractionalBits = 16;

void FillTensor(shark::span<u64>& tensor, u64 value) {
  for (u64 i = 0; i < tensor.size(); ++i) {
    tensor[i] = value;
  }
}

shark::span<u64> RunExpProofAdultPredictionModel(
    shark::span<u64>& input_tensor,
    u64 batch_size,
    shark::span<u64>& W1,
    shark::span<u64>& B1,
    shark::span<u64>& W2,
    shark::span<u64>& B2,
    shark::span<u64>& W3,
    shark::span<u64>& B3) {
  auto a1 = matmul::call(batch_size, kFeatureCount, kHidden, input_tensor, W1);
  auto a2 = add::call(a1, B1);
  auto a3 = relutruncate::call(a2, kFractionalBits);
  auto a4 = matmul::call(batch_size, kHidden, kHidden, a3, W2);
  auto a5 = add::call(a4, B2);
  auto a6 = relutruncate::call(a5, kFractionalBits);
  auto a7 = matmul::call(batch_size, kHidden, kOutputCount, a6, W3);
  auto a8 = add::call(a7, B3);
  return ars::call(a8, kFractionalBits);
}

void PrintPlainBenchmarkSummary(u64 parameter_count) {
  if (party == DEALER) {
    return;
  }

  const shark::utils::TimerStat input = shark::utils::timers["input"];
  const shark::utils::TimerStat inference =
      shark::utils::timers["plain-shark-inference"];
  const u64 local_peer_comm_bytes =
      shark::protocols::peer == nullptr
          ? 0
          : shark::protocols::peer->bytesReceived() +
                shark::protocols::peer->bytesSent();
  const u64 local_dealer_comm_bytes =
      shark::protocols::dealer == nullptr
          ? 0
          : shark::protocols::dealer->bytesReceived();
  const u64 local_protocol_comm_bytes =
      local_peer_comm_bytes + local_dealer_comm_bytes;

  std::cout << "benchmark-model-parameter-count: " << parameter_count << std::endl;
  std::cout << "benchmark-plain-perturbations: 0" << std::endl;
  std::cout << "benchmark-plain-input-time: " << input.accumulated_time << " ms"
            << std::endl;
  std::cout << "benchmark-plain-shark-inference: "
            << inference.accumulated_time << " ms" << std::endl;
  std::cout << "benchmark-plain-shark-total: "
            << inference.accumulated_time << " ms" << std::endl;
  std::cout << "benchmark-local-dealer-comm-bytes: "
            << local_dealer_comm_bytes << " bytes" << std::endl;
  std::cout << "benchmark-local-peer-comm-bytes: "
            << local_peer_comm_bytes << " bytes" << std::endl;
  std::cout << "benchmark-local-protocol-comm-bytes: "
            << local_protocol_comm_bytes << " bytes" << std::endl;
}

}  // namespace

int main(int argc, char** argv) {
  init::from_args(argc, argv);

  const u64 parameter_count =
      kFeatureCount * kHidden + kHidden + kHidden * kHidden + kHidden +
      kHidden * kOutputCount + kOutputCount;

  shark::span<u64> base_input(kFeatureCount);
  shark::span<u64> W1(kFeatureCount * kHidden);
  shark::span<u64> B1(kHidden);
  shark::span<u64> W2(kHidden * kHidden);
  shark::span<u64> B2(kHidden);
  shark::span<u64> W3(kHidden * kOutputCount);
  shark::span<u64> B3(kOutputCount);

  if (party == CLIENT) {
    FillTensor(base_input, 1ull << kFractionalBits);
  }

  if (party == SERVER) {
    FillTensor(W1, 1ull << kFractionalBits);
    FillTensor(B1, 1ull << (2 * kFractionalBits));
    FillTensor(W2, 1ull << kFractionalBits);
    FillTensor(B2, 1ull << (2 * kFractionalBits));
    FillTensor(W3, 1ull << kFractionalBits);
    FillTensor(B3, 1ull << (2 * kFractionalBits));
  }

  shark::utils::start_timer("input");
  input::call(base_input, CLIENT);
  input::call(W1, SERVER);
  input::call(B1, SERVER);
  input::call(W2, SERVER);
  input::call(B2, SERVER);
  input::call(W3, SERVER);
  input::call(B3, SERVER);
  shark::utils::stop_timer("input");

  if (party != DEALER) {
    peer->sync();
  }

  shark::utils::start_timer("plain-shark-inference");
  auto base_output = RunExpProofAdultPredictionModel(
      base_input, 1, W1, B1, W2, B2, W3, B3);
  shark::utils::stop_timer("plain-shark-inference");

  (void)base_output;

  PrintPlainBenchmarkSummary(parameter_count);
  output::call(base_output);
  finalize::call();
  shark::utils::print_all_timers();
}
