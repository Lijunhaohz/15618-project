#include "fusion/backend.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

struct Errors {
    double maximum = 0.0;
    double squared_sum = 0.0;
    std::size_t count = 0;

    template <typename Level>
    bool compare(const char* name, const Level& actual, const Level& expected,
                 std::size_t level) {
        if (actual.width != expected.width || actual.height != expected.height ||
            actual.pixels.size() != expected.pixels.size()) {
            std::cerr << name << ": incorrect dimensions at level " << level << '\n';
            return false;
        }
        bool ok = true;
        for (std::size_t p = 0; p < expected.pixels.size(); ++p) {
            const double error = std::abs(static_cast<double>(actual.pixels[p]) -
                                          expected.pixels[p]);
            if (!std::isfinite(actual.pixels[p]) || error > 1e-6) {
                if (ok) {
                    std::cerr << name << ": mismatch at level " << level << ", element " << p
                              << ": " << actual.pixels[p] << " vs " << expected.pixels[p] << '\n';
                }
                ok = false;
            }
            maximum = std::max(maximum, error);
            squared_sum += error * error;
            ++count;
        }
        return ok;
    }

    void report(const char* name) const {
        std::cout << name << ": " << count << " values, max absolute error " << maximum
                  << ", RMSE " << std::sqrt(squared_sum / count) << '\n';
    }
};

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

std::size_t level_count(std::size_t width, std::size_t height) {
    std::size_t levels = 1;
    while (levels < 5 && (width > 1 || height > 1)) {
        width = width / 2 + width % 2;
        height = height / 2 + height % 2;
        ++levels;
    }
    return levels;
}

bool check_case(fusion::Backend& cpu, fusion::Backend& cuda, const fusion::WeightMap& weight,
                const fusion::RgbImage& image, Errors& weight_errors,
                Errors& image_errors, Errors& gaussian_errors) {
    bool ok = true;
    const auto levels = level_count(image.width, image.height);
    const auto expected_weight = cpu.build_weight_pyramid(weight, levels);
    const auto actual_weight = cuda.build_weight_pyramid(weight, levels);
    const auto expected_image = cpu.build_laplacian_pyramid(image, levels);
    const auto actual_image = cuda.build_laplacian_pyramid(image, levels);
    if (expected_weight.size() != levels || actual_weight.size() != levels ||
        expected_image.size() != levels || actual_image.size() != levels) {
        std::cerr << "Incorrect pyramid depth\n";
        return false;
    }
    for (std::size_t level = 0; level < levels; ++level) {
        ok &= weight_errors.compare("weight pyramid", actual_weight[level],
                                    expected_weight[level], level);
        ok &= image_errors.compare("Laplacian pyramid", actual_image[level],
                                   expected_image[level], level);
    }

    // At each requested depth the coarsest Laplacian level is the Gaussian level.
    for (std::size_t depth = 1; depth <= levels; ++depth) {
        const auto expected = cpu.build_laplacian_pyramid(image, depth);
        const auto actual = cuda.build_laplacian_pyramid(image, depth);
        if (expected.size() != depth || actual.size() != depth) {
            std::cerr << "Incorrect explicitly requested pyramid depth\n";
            return false;
        }
        ok &= gaussian_errors.compare("image Gaussian pyramid", actual.back(), expected.back(),
                                      depth - 1);
    }
    return ok;
}

}  // namespace

int main() {
    auto cpu = fusion::make_cpu_backend();
    auto cuda = fusion::make_cuda_backend();
    Errors weights;
    Errors images;
    Errors gaussian;
    bool ok = true;
    for (const auto dimensions : {std::pair<std::size_t, std::size_t>{37, 19},
                                  {8, 6}, {5, 3}, {1, 17}, {17, 1},
                                  {1, 2}, {2, 1}, {2, 2}, {1, 1}}) {
        const auto width = dimensions.first;
        const auto height = dimensions.second;
        fusion::WeightMap weight{width, height, std::vector<float>(width * height)};
        fusion::RgbImage image{width, height, std::vector<float>(width * height * 3)};
        for (std::size_t p = 0; p < width * height; ++p) {
            weight.pixels[p] = static_cast<float>((p * 37 + 13) % 256) / 255.0f;
            for (std::size_t c = 0; c < 3; ++c) {
                image.pixels[3 * p + c] =
                    static_cast<float>((p * 43 + c * 71 + 9) % 256) / 255.0f;
            }
        }
        ok &= check_case(*cpu, *cuda, weight, image, weights, images, gaussian);
    }

    const fusion::WeightMap constant_weight{5, 3, std::vector<float>(15, 0.25f)};
    const fusion::RgbImage constant_image{5, 3, std::vector<float>(45, 0.25f)};
    ok &= check_case(*cpu, *cuda, constant_weight, constant_image, weights, images, gaussian);
    for (const std::size_t position : {std::size_t{0}, std::size_t{4}, std::size_t{7}}) {
        fusion::WeightMap impulse_weight{5, 3, std::vector<float>(15, 0.0f)};
        fusion::RgbImage impulse_image{5, 3, std::vector<float>(45, 0.0f)};
        impulse_weight.pixels[position] = 1.0f;
        impulse_image.pixels[3 * position] = 1.0f;
        impulse_image.pixels[3 * position + 1] = 0.5f;
        ok &= check_case(*cpu, *cuda, impulse_weight, impulse_image, weights, images, gaussian);
    }

    fusion::WeightMap varied{5, 3, std::vector<float>(15)};
    for (std::size_t p = 0; p < varied.pixels.size(); ++p) {
        varied.pixels[p] = static_cast<float>((p * 37 + 13) % 256) / 255.0f;
    }
    fusion::WeightMap complement = varied;
    for (float& value : complement.pixels) value = 1.0f - value;
    const auto first = cuda->build_weight_pyramid(varied, 4);
    const auto second = cuda->build_weight_pyramid(complement, 4);
    for (std::size_t level = 0; level < first.size(); ++level) {
        for (std::size_t p = 0; p < first[level].pixels.size(); ++p) {
            if (std::abs(first[level].pixels[p] + second[level].pixels[p] - 1.0f) > 1e-6f) {
                std::cerr << "Weight pyramids do not sum to one\n";
                ok = false;
            }
        }
    }

    ok &= rejects("zero weight levels", [&] { cuda->build_weight_pyramid(constant_weight, 0); });
    ok &= rejects("weight levels past 1x1", [&] { cuda->build_weight_pyramid(constant_weight, 5); });
    ok &= rejects("weight levels past five", [&] {
        cuda->build_weight_pyramid({33, 1, std::vector<float>(33, 0.5f)}, 6);
    });
    ok &= rejects("zero weight dimensions", [&] { cuda->build_weight_pyramid({0, 1, {}}, 1); });
    ok &= rejects("overflowing weight dimensions", [&] {
        cuda->build_weight_pyramid({std::numeric_limits<std::size_t>::max(), 2, {}}, 1);
    });
    ok &= rejects("incorrect weight count", [&] { cuda->build_weight_pyramid({2, 1, {0.5f}}, 1); });
    ok &= rejects("negative weight", [&] { cuda->build_weight_pyramid({1, 1, {-0.1f}}, 1); });
    ok &= rejects("weight above one", [&] { cuda->build_weight_pyramid({1, 1, {1.1f}}, 1); });
    ok &= rejects("NaN weight", [&] {
        cuda->build_weight_pyramid({1, 1, {std::numeric_limits<float>::quiet_NaN()}}, 1);
    });
    ok &= rejects("infinite weight", [&] {
        cuda->build_weight_pyramid({1, 1, {std::numeric_limits<float>::infinity()}}, 1);
    });

    ok &= rejects("zero image levels", [&] { cuda->build_laplacian_pyramid(constant_image, 0); });
    ok &= rejects("image levels past 1x1", [&] { cuda->build_laplacian_pyramid(constant_image, 5); });
    ok &= rejects("image levels past five", [&] {
        cuda->build_laplacian_pyramid({33, 1, std::vector<float>(99, 0.5f)}, 6);
    });
    ok &= rejects("zero image dimensions", [&] { cuda->build_laplacian_pyramid({0, 1, {}}, 1); });
    ok &= rejects("overflowing image dimensions", [&] {
        cuda->build_laplacian_pyramid({std::numeric_limits<std::size_t>::max(), 2, {}}, 1);
    });
    ok &= rejects("incorrect pixel count", [&] {
        cuda->build_laplacian_pyramid({1, 1, {0.5f}}, 1);
    });
    ok &= rejects("out-of-range pixel", [&] {
        cuda->build_laplacian_pyramid({1, 1, {0.5f, -0.1f, 1.0f}}, 1);
    });
    ok &= rejects("NaN pixel", [&] {
        cuda->build_laplacian_pyramid({1, 1, {0.5f, std::numeric_limits<float>::quiet_NaN(), 1.0f}}, 1);
    });
    ok &= rejects("infinite pixel", [&] {
        cuda->build_laplacian_pyramid({1, 1, {0.5f, std::numeric_limits<float>::infinity(), 1.0f}}, 1);
    });

    weights.report("Gaussian weight levels");
    images.report("Laplacian image levels");
    gaussian.report("Gaussian image levels");
    return ok ? 0 : 1;
}
