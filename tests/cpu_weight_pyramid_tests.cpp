#include "fusion/backend.hpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

int reflect(int index, int length) {
    if (length == 1) return 0;
    const int period = 2 * (length - 1);
    index = (index % period + period) % period;
    return index < length ? index : period - index;
}

double reference_pixel(const fusion::WeightMap& input, int x, int y) {
    constexpr double kernel[] = {1.0 / 16, 4.0 / 16, 6.0 / 16, 4.0 / 16, 1.0 / 16};
    double result = 0.0;
    for (int dy = -2; dy <= 2; ++dy) {
        for (int dx = -2; dx <= 2; ++dx) {
            const auto sx = reflect(2 * x + dx, static_cast<int>(input.width));
            const auto sy = reflect(2 * y + dy, static_cast<int>(input.height));
            result += kernel[dx + 2] * kernel[dy + 2] *
                      input.pixels[static_cast<std::size_t>(sy) * input.width + sx];
        }
    }
    return result;
}

bool check_pyramid(fusion::Backend& backend, const fusion::WeightMap& input,
                   std::size_t levels) {
    const auto pyramid = backend.build_weight_pyramid(input, levels);
    if (pyramid.size() != levels || pyramid[0].width != input.width ||
        pyramid[0].height != input.height || pyramid[0].pixels != input.pixels) {
        std::cerr << "Unexpected pyramid level zero\n";
        return false;
    }
    for (std::size_t level = 1; level < levels; ++level) {
        const auto& previous = pyramid[level - 1];
        const auto& current = pyramid[level];
        const auto width = previous.width / 2 + previous.width % 2;
        const auto height = previous.height / 2 + previous.height % 2;
        if (current.width != width || current.height != height ||
            current.pixels.size() != width * height) {
            std::cerr << "Incorrect dimensions at level " << level << '\n';
            return false;
        }
        for (std::size_t y = 0; y < height; ++y) {
            for (std::size_t x = 0; x < width; ++x) {
                const double expected = reference_pixel(previous, static_cast<int>(x), static_cast<int>(y));
                if (std::abs(current.pixels[y * width + x] - expected) > 1e-6) {
                    std::cerr << "Incorrect value at level " << level << ", " << x << "," << y << '\n';
                    return false;
                }
            }
        }
    }
    return true;
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
    const fusion::WeightMap odd{5, 3, {
        1.0f, 0.0f, 0.2f, 0.4f, 0.6f,
        0.0f, 0.3f, 0.7f, 0.2f, 0.8f,
        0.5f, 0.1f, 0.9f, 0.4f, 0.0f,
    }};
    ok &= check_pyramid(*backend, odd, 4);
    ok &= check_pyramid(*backend, {2, 2, {1.0f, 0.0f, 0.0f, 0.0f}}, 2);
    ok &= check_pyramid(*backend, {1, 3, {0.0f, 1.0f, 0.25f}}, 3);
    ok &= check_pyramid(*backend, {3, 1, {0.0f, 1.0f, 0.25f}}, 3);
    ok &= check_pyramid(*backend, {1, 1, {0.5f}}, 1);
    ok &= check_pyramid(*backend, {17, 1, std::vector<float>(17, 0.25f)}, 5);
    const auto constant = backend->build_weight_pyramid({5, 3, std::vector<float>(15, 0.25f)}, 4);
    for (const auto& level : constant) {
        for (float value : level.pixels) {
            if (std::abs(value - 0.25f) > 1e-6f) {
                std::cerr << "Constant weights changed after reduction\n";
                ok = false;
            }
        }
    }
    fusion::WeightMap complement = odd;
    for (float& value : complement.pixels) value = 1.0f - value;
    const auto first = backend->build_weight_pyramid(odd, 4);
    const auto second = backend->build_weight_pyramid(complement, 4);
    for (std::size_t level = 0; level < first.size(); ++level) {
        for (std::size_t p = 0; p < first[level].pixels.size(); ++p) {
            if (std::abs(first[level].pixels[p] + second[level].pixels[p] - 1.0f) > 1e-6f) {
                std::cerr << "Weights do not sum to one at level " << level << '\n';
                ok = false;
            }
        }
    }
    ok &= rejects("zero levels", [&] { backend->build_weight_pyramid(odd, 0); });
    ok &= rejects("levels past 1x1", [&] { backend->build_weight_pyramid(odd, 5); });
    ok &= rejects("levels past five", [&] {
        backend->build_weight_pyramid({33, 1, std::vector<float>(33, 0.5f)}, 6);
    });
    ok &= rejects("zero dimensions", [&] { backend->build_weight_pyramid({0, 1, {}}, 1); });
    ok &= rejects("overflowing dimensions", [&] {
        backend->build_weight_pyramid({std::numeric_limits<std::size_t>::max(), 2, {}}, 1);
    });
    ok &= rejects("incorrect pixel count", [&] { backend->build_weight_pyramid({2, 1, {0.5f}}, 1); });
    ok &= rejects("negative weight", [&] { backend->build_weight_pyramid({1, 1, {-0.1f}}, 1); });
    ok &= rejects("weight above one", [&] { backend->build_weight_pyramid({1, 1, {1.1f}}, 1); });
    ok &= rejects("NaN weight", [&] {
        backend->build_weight_pyramid({1, 1, {std::numeric_limits<float>::quiet_NaN()}}, 1);
    });
    ok &= rejects("infinite weight", [&] {
        backend->build_weight_pyramid({1, 1, {std::numeric_limits<float>::infinity()}}, 1);
    });
    return ok ? 0 : 1;
}
