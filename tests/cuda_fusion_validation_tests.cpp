#include "fusion/backend.hpp"
#include "fusion/image_io.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr double max_error_tolerance = 2e-5;
constexpr double rmse_tolerance = 2e-6;

std::size_t level_count(std::size_t width, std::size_t height) {
    std::size_t levels = 1;
    while (levels < 5 && (width > 1 || height > 1)) {
        width = width / 2 + width % 2;
        height = height / 2 + height % 2;
        ++levels;
    }
    return levels;
}

bool compare(const char* label, const fusion::RgbImage& actual,
             const fusion::RgbImage& expected, double max_tolerance, double rmse_limit) {
    if (actual.width != expected.width || actual.height != expected.height ||
        actual.pixels.size() != expected.pixels.size() || actual.pixels.empty()) {
        std::cerr << label << ": invalid output shape\n";
        return false;
    }
    double maximum = 0.0;
    double squared_sum = 0.0;
    std::size_t max_index = 0;
    for (std::size_t i = 0; i < actual.pixels.size(); ++i) {
        if (!std::isfinite(actual.pixels[i]) || !std::isfinite(expected.pixels[i])) {
            std::cerr << label << ": non-finite value at element " << i << '\n';
            return false;
        }
        const double error = std::abs(static_cast<double>(actual.pixels[i]) - expected.pixels[i]);
        if (error > maximum) {
            maximum = error;
            max_index = i;
        }
        squared_sum += error * error;
    }
    const double rmse = std::sqrt(squared_sum / actual.pixels.size());
    std::cout << std::scientific << std::setprecision(9) << label
              << ": max_abs=" << maximum << " rmse=" << rmse
              << " max_element=" << max_index << '\n';
    if (maximum > max_tolerance || rmse > rmse_limit) {
        std::cerr << label << ": error exceeds max_abs=" << max_tolerance
                  << " or rmse=" << rmse_limit << '\n';
        return false;
    }
    return true;
}

fusion::RgbImage constant(std::size_t width, std::size_t height, float value) {
    return {width, height, std::vector<float>(width * height * 3, value)};
}

fusion::RgbImage pattern(std::size_t width, std::size_t height, std::size_t exposure) {
    fusion::RgbImage image{width, height, std::vector<float>(width * height * 3)};
    for (std::size_t p = 0; p < width * height; ++p) {
        for (std::size_t channel = 0; channel < 3; ++channel) {
            image.pixels[p * 3 + channel] =
                static_cast<float>((p * 37 + channel * 71 + exposure * 53 + 7) % 256) / 255.0f;
        }
    }
    return image;
}

bool check_case(fusion::Backend& cpu, fusion::Backend& cuda, const char* name,
                const std::vector<fusion::RgbImage>& images, bool expect_input = false) {
    const auto& first = images.front();
    std::cout << name << ": " << first.width << 'x' << first.height << ", "
              << images.size() << " exposures, " << level_count(first.width, first.height)
              << " levels\n";
    const auto expected = cpu.fuse(images, {});
    const auto actual = cuda.fuse(images, {});
    bool ok = compare(name, actual, expected, max_error_tolerance, rmse_tolerance);
    if (expect_input) {
        ok &= compare("  input round trip", actual, first, 1e-6, 1e-6);
    }
    return ok;
}

}  // namespace

int main(int argc, char* argv[]) {
    if (argc != 1 && (argc != 5 || std::string(argv[1]) != "--real")) {
        std::cerr << "Usage: cuda_fusion_validation_tests [--real IMAGE IMAGE IMAGE]\n";
        return 1;
    }
    auto cpu = fusion::make_cpu_backend();
    auto cuda = fusion::make_cuda_backend();
    if (argc == 5) {
        std::vector<fusion::RgbImage> images;
        for (int i = 2; i < argc; ++i) images.push_back(fusion::load_image(argv[i]));
        return check_case(*cpu, *cuda, "memorial bracket", images) ? 0 : 1;
    }

    bool ok = true;
    ok &= check_case(*cpu, *cuda, "constant 7x5",
                     {constant(7, 5, 64.0f / 255.0f), constant(7, 5, 128.0f / 255.0f),
                      constant(7, 5, 192.0f / 255.0f)});
    auto impulse = constant(7, 5, 0.0f);
    impulse.pixels[0] = 1.0f;
    impulse.pixels[1] = 0.4f;
    impulse.pixels[2] = 0.1f;
    ok &= check_case(*cpu, *cuda, "corner impulse 7x5", {impulse, impulse}, true);
    ok &= check_case(*cpu, *cuda, "odd pattern 7x5",
                     {pattern(7, 5, 0), pattern(7, 5, 1), pattern(7, 5, 2)});
    ok &= check_case(*cpu, *cuda, "one column 1x17",
                     {pattern(1, 17, 0), pattern(1, 17, 1), pattern(1, 17, 2)});
    ok &= check_case(*cpu, *cuda, "one row 17x1",
                     {pattern(17, 1, 0), pattern(17, 1, 1), pattern(17, 1, 2)});
    ok &= check_case(*cpu, *cuda, "one pixel 1x1",
                     {pattern(1, 1, 0), pattern(1, 1, 1), pattern(1, 1, 2)});
    ok &= check_case(*cpu, *cuda, "single exposure 19x11", {pattern(19, 11, 0)}, true);
    return ok ? 0 : 1;
}
