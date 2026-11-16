#include "fusion/backend.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

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

bool compare(const char* name, const std::vector<fusion::WeightMap>& actual,
             const std::vector<fusion::WeightMap>& expected, float absolute_tolerance,
             float relative_tolerance) {
    if (actual.size() != expected.size() || actual.empty()) {
        std::cerr << name << ": incorrect map count\n";
        return false;
    }
    double maximum = 0.0;
    double squared_error = 0.0;
    std::size_t count = 0;
    bool ok = true;
    for (std::size_t i = 0; i < expected.size(); ++i) {
        if (actual[i].width != expected[i].width || actual[i].height != expected[i].height ||
            actual[i].pixels.size() != expected[i].pixels.size()) {
            std::cerr << name << ": incorrect dimensions for map " << i << '\n';
            return false;
        }
        for (std::size_t p = 0; p < expected[i].pixels.size(); ++p) {
            const double error = std::abs(static_cast<double>(actual[i].pixels[p]) -
                                          expected[i].pixels[p]);
            if (!std::isfinite(actual[i].pixels[p]) ||
                error > absolute_tolerance + relative_tolerance * std::abs(expected[i].pixels[p])) {
                if (ok) {
                    std::cerr << name << ": mismatch at map " << i << ", pixel " << p
                              << ": " << actual[i].pixels[p] << " vs "
                              << expected[i].pixels[p] << '\n';
                }
                ok = false;
            }
            maximum = std::max(maximum, error);
            squared_error += error * error;
            ++count;
        }
    }
    std::cout << name << ": max absolute error " << maximum
              << ", RMSE " << std::sqrt(squared_error / count) << '\n';
    return ok;
}

fusion::RgbImage patterned(std::size_t width, std::size_t height, std::size_t exposure) {
    fusion::RgbImage image{width, height, std::vector<float>(width * height * 3)};
    for (std::size_t p = 0; p < width * height; ++p) {
        for (std::size_t c = 0; c < 3; ++c) {
            image.pixels[3 * p + c] =
                static_cast<float>((p * 37 + c * 71 + exposure * 53) % 256) / 255.0f;
        }
    }
    return image;
}

bool check_images(fusion::Backend& cpu, fusion::Backend& cuda, const char* name,
                  const std::vector<fusion::RgbImage>& images) {
    const auto cpu_raw = cpu.compute_weights(images);
    const auto cuda_raw = cuda.compute_weights(images);
    bool ok = compare(name, cuda_raw, cpu_raw, 1e-10f, 3e-5f);
    const auto cpu_normalized = cpu.normalize_weights(cpu_raw);
    const auto cuda_normalized = cuda.normalize_weights(cuda_raw);
    ok &= compare("normalized weights", cuda_normalized, cpu_normalized, 2e-5f, 0.0f);
    if (cuda_normalized.size() != images.size()) return false;
    for (const auto& map : cuda_normalized) {
        if (map.width != images.front().width || map.height != images.front().height ||
            map.pixels.size() != images.front().width * images.front().height) {
            return false;
        }
    }
    for (std::size_t p = 0; p < images.front().width * images.front().height; ++p) {
        float sum = 0.0f;
        for (const auto& map : cuda_normalized) sum += map.pixels[p];
        if (std::abs(sum - 1.0f) > 1e-6f) {
            std::cerr << name << ": normalized weights do not sum to one at " << p << '\n';
            return false;
        }
    }
    return ok;
}

}  // namespace

int main() {
    auto cpu = fusion::make_cpu_backend();
    auto cuda = fusion::make_cuda_backend();
    bool ok = true;

    for (const auto dimensions : {std::pair<std::size_t, std::size_t>{37, 19},
                                  {1, 7}, {7, 1}, {1, 1}, {2, 2}}) {
        std::vector<fusion::RgbImage> images;
        for (std::size_t i = 0; i < 3; ++i) {
            images.push_back(patterned(dimensions.first, dimensions.second, i));
        }
        ok &= check_images(*cpu, *cuda, "patterned raw weights", images);
    }
    ok &= check_images(*cpu, *cuda, "single exposure raw weights", {patterned(9, 5, 0)});

    const fusion::RgbImage constant{3, 2, std::vector<float>(18, 0.5f)};
    auto impulse = constant;
    impulse.pixels[0] = 1.0f;
    impulse.pixels[1] = 0.1f;
    impulse.pixels[2] = 0.3f;
    ok &= check_images(*cpu, *cuda, "constant and corner impulse raw weights",
                       {constant, impulse, constant});
    const auto constant_weights = cuda->compute_weights({constant});
    for (float value : constant_weights.front().pixels) {
        if (std::abs(value - 1e-12f) > 1e-18f) {
            std::cerr << "Constant image did not produce epsilon weight\n";
            ok = false;
        }
    }

    const std::vector<fusion::WeightMap> supplied{
        {4, 1, {1.0f, 0.0f, 1e-12f, 1000.0f}},
        {4, 1, {2.0f, 2.0f, 1e-12f, 0.5f}},
        {4, 1, {1.0f, 0.0f, 1e-12f, 7.0f}},
    };
    ok &= compare("supplied normalized weights", cuda->normalize_weights(supplied),
                  cpu->normalize_weights(supplied), 1e-6f, 0.0f);

    ok &= rejects("empty images", [&] { cuda->compute_weights({}); });
    ok &= rejects("zero image dimensions", [&] { cuda->compute_weights({{0, 1, {}}}); });
    ok &= rejects("overflowing image dimensions", [&] {
        cuda->compute_weights({{std::numeric_limits<std::size_t>::max(), 2, {}}});
    });
    ok &= rejects("mismatched image dimensions", [&] {
        cuda->compute_weights({patterned(2, 1, 0), patterned(1, 1, 0)});
    });
    ok &= rejects("incorrect image size", [&] { cuda->compute_weights({{1, 1, {0.0f}}}); });
    ok &= rejects("out-of-range pixel", [&] {
        cuda->compute_weights({{1, 1, {0.0f, 0.0f, 1.1f}}});
    });
    ok &= rejects("NaN pixel", [&] {
        cuda->compute_weights({{1, 1, {0.0f, std::numeric_limits<float>::quiet_NaN(), 0.0f}}});
    });
    ok &= rejects("infinite pixel", [&] {
        cuda->compute_weights({{1, 1, {0.0f, std::numeric_limits<float>::infinity(), 0.0f}}});
    });

    ok &= rejects("empty weights", [&] { cuda->normalize_weights({}); });
    ok &= rejects("zero weight dimensions", [&] { cuda->normalize_weights({{0, 1, {}}}); });
    ok &= rejects("overflowing weight dimensions", [&] {
        cuda->normalize_weights({{std::numeric_limits<std::size_t>::max(), 2, {}}});
    });
    ok &= rejects("mismatched weight dimensions", [&] {
        cuda->normalize_weights({{1, 1, {1.0f}}, {2, 1, {1.0f, 1.0f}}});
    });
    ok &= rejects("incorrect weight size", [&] { cuda->normalize_weights({{2, 1, {1.0f}}}); });
    ok &= rejects("negative weight", [&] { cuda->normalize_weights({{1, 1, {-1.0f}}}); });
    ok &= rejects("NaN weight", [&] {
        cuda->normalize_weights({{1, 1, {std::numeric_limits<float>::quiet_NaN()}}});
    });
    ok &= rejects("infinite weight", [&] {
        cuda->normalize_weights({{1, 1, {std::numeric_limits<float>::infinity()}}});
    });
    ok &= rejects("zero sum", [&] {
        cuda->normalize_weights({{2, 1, {1.0f, 0.0f}}, {2, 1, {1.0f, 0.0f}}});
    });
    ok &= rejects("overflowing sum", [&] {
        const float max = std::numeric_limits<float>::max();
        cuda->normalize_weights({{1, 1, {max}}, {1, 1, {max}}});
    });
    return ok ? 0 : 1;
}
