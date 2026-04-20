#include <shark/protocols/init.hpp>
#include <shark/protocols/finalize.hpp>
#include <shark/protocols/input.hpp>
#include <shark/protocols/output.hpp>
#include <shark/protocols/matmul.hpp>
#include <shark/protocols/common.hpp>
#include <shark/utils/assert.hpp>

using u64 = shark::u64;
using namespace shark::protocols;

namespace {

constexpr u64 kBatch = 1;
constexpr u64 kFeatures = 14;
constexpr u64 kHidden = 16;
constexpr u64 kFractionalBits = 16;

void fill(shark::span<u64>& X, u64 value) {
    for (u64 i = 0; i < X.size(); ++i) {
        X[i] = value;
    }
}

}  // namespace

int main(int argc, char** argv)
{
    init::from_args(argc, argv);

    shark::span<u64> X(kBatch * kFeatures);
    shark::span<u64> Y(kFeatures * kHidden);

    if (party == CLIENT) {
        fill(X, 1ull << kFractionalBits);
    }

    if (party == SERVER) {
        fill(Y, 1ull << kFractionalBits);
    }

    input::call(X, CLIENT);
    input::call(Y, SERVER);

    if (party != DEALER) {
        peer->sync();
    }

    auto Z = matmul::call(kBatch, kFeatures, kHidden, X, Y);
    output::call(Z);
    finalize::call();

    if (party != DEALER) {
        const u64 expected = kFeatures * (1ull << (2 * kFractionalBits));
        for (u64 i = 0; i < Z.size(); ++i) {
            always_assert(Z[i] == expected);
        }
    }
}
