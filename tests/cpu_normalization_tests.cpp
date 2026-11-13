#include "fusion/backend.hpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

bool near(float actual, float expected) {
    return std::abs(actual - expected) <= 1e-6f;
}

template <typename Function>
bool rejects(const char* name, Function function) {
    try {
        function();
        std::cerr << name << " was accepted\n";
    } catch (const std::invalid_argument&) {
        return true;
    }
    return false;
}

}  // namespace

int main() {
    auto backend = fusion::make_cpu_backend();
    bool ok = true;
    const std::vector<fusion::WeightMap> weights{
        {2, 1, {1.0f, 4.0f}},
        {2, 1, {2.0f, 0.0f}},
        {2, 1, {1.0f, 4.0f}},
    };
    const auto normalized = backend->normalize_weights(weights);
    if (normalized.size() != 3) {
        std::cerr << "Unexpected number of weight maps\n";
        return 1;
    }
    for (const auto& map : normalized) {
        if (map.width != 2 || map.height != 1 || map.pixels.size() != 2) {
            std::cerr << "Unexpected normalized map dimensions\n";
            return 1;
        }
    }
    const float expected[3][2] = {{0.25f, 0.5f}, {0.5f, 0.0f}, {0.25f, 0.5f}};
    for (std::size_t p = 0; p < 2; ++p) {
        float sum = 0.0f;
        for (std::size_t i = 0; i < 3; ++i) {
            if (!near(normalized[i].pixels[p], expected[i][p])) {
                std::cerr << "Incorrect normalized weight at map " << i << ", pixel " << p << '\n';
                ok = false;
            }
            sum += normalized[i].pixels[p];
        }
        if (!near(sum, 1.0f)) {
            std::cerr << "Normalized weights do not sum to one at pixel " << p << '\n';
            ok = false;
        }
    }
    if (weights[0].pixels[0] != 1.0f || weights[1].pixels[0] != 2.0f) {
        std::cerr << "Input weights were modified\n";
        ok = false;
    }
    const auto single = backend->normalize_weights({{1, 1, {1e-12f}}});
    if (!near(single[0].pixels[0], 1.0f)) {
        std::cerr << "Single exposure weight is not one\n";
        ok = false;
    }

    const fusion::RgbImage constant{1, 1, {0.5f, 0.5f, 0.5f}};
    const auto raw = backend->compute_weights({constant, constant, constant});
    const auto equal = backend->normalize_weights(raw);
    for (const auto& map : equal) {
        if (!near(map.pixels[0], 1.0f / 3.0f)) {
            std::cerr << "Equal epsilon weights are not evenly normalized\n";
            ok = false;
        }
    }

    ok &= rejects("empty input", [&] { backend->normalize_weights({}); });
    ok &= rejects("zero dimensions", [&] { backend->normalize_weights({{0, 1, {}}}); });
    ok &= rejects("overflowing dimensions", [&] {
        backend->normalize_weights({{std::numeric_limits<std::size_t>::max(), 2, {}}});
    });
    ok &= rejects("mismatched dimensions", [&] {
        backend->normalize_weights({{1, 1, {1.0f}}, {2, 1, {1.0f, 1.0f}}});
    });
    ok &= rejects("wrong pixel count", [&] { backend->normalize_weights({{2, 1, {1.0f}}}); });
    ok &= rejects("negative value", [&] { backend->normalize_weights({{1, 1, {-1.0f}}}); });
    ok &= rejects("NaN value", [&] {
        backend->normalize_weights({{1, 1, {std::numeric_limits<float>::quiet_NaN()}}});
    });
    ok &= rejects("infinite value", [&] {
        backend->normalize_weights({{1, 1, {std::numeric_limits<float>::infinity()}}});
    });
    ok &= rejects("zero weight sum", [&] {
        backend->normalize_weights({{1, 1, {0.0f}}, {1, 1, {0.0f}}});
    });
    ok &= rejects("overflowing weight sum", [&] {
        const float max = std::numeric_limits<float>::max();
        backend->normalize_weights({{1, 1, {max}}, {1, 1, {max}}});
    });
    return ok ? 0 : 1;
}
