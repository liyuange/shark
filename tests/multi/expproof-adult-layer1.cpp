#include <shark/protocols/init.hpp>
#include <shark/protocols/finalize.hpp>
#include <shark/protocols/input.hpp>
#include <shark/protocols/output.hpp>
#include <shark/protocols/matmul.hpp>
#include <shark/protocols/common.hpp>
#include <shark/utils/assert.hpp>
#include <iostream>

using u64 = shark::u64;
using namespace shark::protocols;

namespace {

constexpr u64 kBatch = 1;
constexpr u64 kFeatures = 14;
constexpr u64 kHidden = 16;
constexpr u64 kOutput = 1;
constexpr u64 kFractionalBits = 16;

void fill(shark::span<u64>& X, u64 value) {
    for (u64 i = 0; i < X.size(); ++i) {
        X[i] = value;
    }
}

void log_stage(const char* stage) {
    if (party != DEALER) {
        std::cerr << "stage: " << stage << std::endl;
    }
}

}  // namespace

int main(int argc, char** argv)
{
    init::from_args(argc, argv);
    const int preload_level = (argc > 4) ? atoi(argv[4]) : 5;
    const int preload_mask = (argc > 5) ? atoi(argv[5]) : ((1 << preload_level) - 1);

    shark::span<u64> base_input(kBatch * kFeatures);
    shark::span<u64> W1(kFeatures * kHidden);
    shark::span<u64> B1(kHidden);
    shark::span<u64> W2(kHidden * kHidden);
    shark::span<u64> B2(kHidden);
    shark::span<u64> W3(kHidden * kOutput);
    shark::span<u64> B3(kOutput);
    shark::span<u64> tail_scalar(1);

    if (party == CLIENT) {
        fill(base_input, 1ull << kFractionalBits);
    }

    if (party == SERVER) {
        fill(W1, 1ull << kFractionalBits);
        fill(B1, 1ull << (2 * kFractionalBits));
        fill(W2, 1ull << kFractionalBits);
        fill(B2, 1ull << (2 * kFractionalBits));
        fill(W3, 1ull << kFractionalBits);
        fill(B3, 1ull << (2 * kFractionalBits));
        fill(tail_scalar, 1ull << (2 * kFractionalBits));
    }

    log_stage("input-base");
    input::call(base_input, CLIENT);
    log_stage("input-W1");
    input::call(W1, SERVER);
    if ((preload_mask & 1) != 0) {
        log_stage("input-B1");
        input::call(B1, SERVER);
    }
    if ((preload_mask & 2) != 0) {
        log_stage("input-W2");
        input::call(W2, SERVER);
    }
    if ((preload_mask & 4) != 0) {
        log_stage("input-B2");
        input::call(B2, SERVER);
    }
    if ((preload_mask & 8) != 0) {
        log_stage("input-W3");
        input::call(W3, SERVER);
    }
    if ((preload_mask & 16) != 0) {
        log_stage("input-B3");
        input::call(B3, SERVER);
    }
    if ((preload_mask & 32) != 0) {
        log_stage("input-tail-scalar");
        input::call(tail_scalar, SERVER);
    }

    if (party != DEALER) {
        log_stage("sync");
        peer->sync();
    }

    log_stage("matmul");
    auto Z = matmul::call(kBatch, kFeatures, kHidden, base_input, W1);
    log_stage("output");
    output::call(Z);
    log_stage("finalize");
    finalize::call();

    if (party != DEALER) {
        const u64 expected = kFeatures * (1ull << (2 * kFractionalBits));
        for (u64 i = 0; i < Z.size(); ++i) {
            always_assert(Z[i] == expected);
        }
    }
}
