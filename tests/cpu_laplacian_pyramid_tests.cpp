#include "fusion/backend.hpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

struct ReferenceImage {
    int width;
    int height;
    std::vector<double> pixels;
};

int reflect(int index, int length) {
    if (length == 1) return 0;
    const int period = 2 * (length - 1);
    index = (index % period + period) % period;
    return index < length ? index : period - index;
}

constexpr double kernel[] = {1.0 / 16, 4.0 / 16, 6.0 / 16, 4.0 / 16, 1.0 / 16};

ReferenceImage reduce_reference(const ReferenceImage& input) {
    const int width = (input.width + 1) / 2;
    const int height = (input.height + 1) / 2;
    ReferenceImage output{width, height, std::vector<double>(width * height * 3)};
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            for (int channel = 0; channel < 3; ++channel) {
                double value = 0.0;
                for (int dy = -2; dy <= 2; ++dy) {
                    for (int dx = -2; dx <= 2; ++dx) {
                        const int sx = reflect(2 * x + dx, input.width);
                        const int sy = reflect(2 * y + dy, input.height);
                        value += kernel[dx + 2] * kernel[dy + 2] *
                                 input.pixels[(sy * input.width + sx) * 3 + channel];
                    }
                }
                output.pixels[(y * width + x) * 3 + channel] = value;
            }
        }
    }
    return output;
}

ReferenceImage expand_reference(const ReferenceImage& input, int width, int height) {
    ReferenceImage output{width, height, std::vector<double>(width * height * 3)};
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            for (int channel = 0; channel < 3; ++channel) {
                double value = 0.0;
                for (int dy = -2; dy <= 2; ++dy) {
                    for (int dx = -2; dx <= 2; ++dx) {
                        if ((x - dx) % 2 != 0 || (y - dy) % 2 != 0) continue;
                        const int sx = reflect((x - dx) / 2, input.width);
                        const int sy = reflect((y - dy) / 2, input.height);
                        value += 4 * kernel[dx + 2] * kernel[dy + 2] *
                                 input.pixels[(sy * input.width + sx) * 3 + channel];
                    }
                }
                output.pixels[(y * width + x) * 3 + channel] = value;
            }
        }
    }
    return output;
}

bool check_pyramid(fusion::Backend& backend, const fusion::RgbImage& image, std::size_t levels) {
    const auto actual = backend.build_laplacian_pyramid(image, levels);
    if (actual.size() != levels) {
        std::cerr << "Unexpected pyramid size\n";
        return false;
    }
    std::vector<ReferenceImage> gaussian;
    gaussian.push_back({static_cast<int>(image.width), static_cast<int>(image.height),
                        std::vector<double>(image.pixels.begin(), image.pixels.end())});
    while (gaussian.size() < levels) gaussian.push_back(reduce_reference(gaussian.back()));

    for (std::size_t level = 0; level < levels; ++level) {
        const auto& g = gaussian[level];
        const auto& result = actual[level];
        if (result.width != static_cast<std::size_t>(g.width) ||
            result.height != static_cast<std::size_t>(g.height) ||
            result.pixels.size() != g.pixels.size()) {
            std::cerr << "Unexpected dimensions at level " << level << '\n';
            return false;
        }
        std::vector<double> expected = g.pixels;
        if (level + 1 < levels) {
            const auto expanded = expand_reference(gaussian[level + 1], g.width, g.height);
            for (std::size_t p = 0; p < expected.size(); ++p) expected[p] -= expanded.pixels[p];
        }
        for (std::size_t p = 0; p < expected.size(); ++p) {
            if (std::abs(result.pixels[p] - expected[p]) > 1e-6) {
                std::cerr << "Incorrect value at level " << level << ", element " << p
                          << ": " << result.pixels[p] << " vs " << expected[p] << '\n';
                return false;
            }
        }
    }

    ReferenceImage reconstructed{static_cast<int>(actual.back().width),
                                 static_cast<int>(actual.back().height),
                                 std::vector<double>(actual.back().pixels.begin(), actual.back().pixels.end())};
    for (std::size_t level = levels - 1; level > 0; --level) {
        reconstructed = expand_reference(reconstructed, static_cast<int>(actual[level - 1].width),
                                         static_cast<int>(actual[level - 1].height));
        for (std::size_t p = 0; p < reconstructed.pixels.size(); ++p) {
            reconstructed.pixels[p] += actual[level - 1].pixels[p];
        }
    }
    for (std::size_t p = 0; p < image.pixels.size(); ++p) {
        if (std::abs(reconstructed.pixels[p] - image.pixels[p]) > 1e-6) {
            std::cerr << "Reference reconstruction mismatch at element " << p << '\n';
            return false;
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
    const fusion::RgbImage odd{3, 3, {
        1.0f, 0.0f, 0.0f,  0.2f, 0.6f, 0.8f,  0.4f, 0.1f, 0.5f,
        0.3f, 0.9f, 0.2f,  0.6f, 0.3f, 0.1f,  0.7f, 0.5f, 0.9f,
        0.8f, 0.2f, 0.3f,  0.1f, 0.7f, 0.4f,  0.0f, 0.0f, 1.0f,
    }};
    const fusion::RgbImage even{4, 2, {
        0.0f, 0.0f, 0.0f,  1.0f, 0.0f, 0.0f,  0.0f, 1.0f, 0.0f,  0.0f, 0.0f, 1.0f,
        0.2f, 0.3f, 0.4f,  0.5f, 0.6f, 0.7f,  0.9f, 0.8f, 0.1f,  1.0f, 1.0f, 1.0f,
    }};
    ok &= check_pyramid(*backend, odd, 3);
    ok &= check_pyramid(*backend, odd, 1);
    ok &= check_pyramid(*backend, even, 3);
    ok &= check_pyramid(*backend, {1, 3, {0.0f, 0.5f, 1.0f, 0.8f, 0.2f, 0.3f, 0.2f, 0.7f, 0.1f}}, 3);
    ok &= check_pyramid(*backend, {3, 1, {0.0f, 0.5f, 1.0f, 0.8f, 0.2f, 0.3f, 0.2f, 0.7f, 0.1f}}, 3);
    ok &= check_pyramid(*backend, {1, 1, {0.1f, 0.5f, 1.0f}}, 1);
    ok &= check_pyramid(*backend, {17, 1, std::vector<float>(51, 0.25f)}, 5);
    const auto constant = backend->build_laplacian_pyramid({3, 2, std::vector<float>(18, 0.25f)}, 3);
    for (std::size_t level = 0; level + 1 < constant.size(); ++level) {
        for (float value : constant[level].pixels) {
            if (std::abs(value) > 1e-6f) {
                std::cerr << "Constant image has nonzero Laplacian\n";
                ok = false;
            }
        }
    }
    ok &= rejects("zero levels", [&] { backend->build_laplacian_pyramid(odd, 0); });
    ok &= rejects("levels past 1x1", [&] { backend->build_laplacian_pyramid(odd, 4); });
    ok &= rejects("levels past five", [&] {
        backend->build_laplacian_pyramid({33, 1, std::vector<float>(99, 0.5f)}, 6);
    });
    ok &= rejects("zero dimensions", [&] { backend->build_laplacian_pyramid({0, 1, {}}, 1); });
    ok &= rejects("overflowing dimensions", [&] {
        backend->build_laplacian_pyramid({std::numeric_limits<std::size_t>::max(), 2, {}}, 1);
    });
    ok &= rejects("incorrect pixel count", [&] {
        backend->build_laplacian_pyramid({1, 1, {0.5f}}, 1);
    });
    ok &= rejects("out-of-range value", [&] {
        backend->build_laplacian_pyramid({1, 1, {0.5f, -0.1f, 1.0f}}, 1);
    });
    ok &= rejects("NaN value", [&] {
        backend->build_laplacian_pyramid({1, 1, {0.5f, std::numeric_limits<float>::quiet_NaN(), 1.0f}}, 1);
    });
    return ok ? 0 : 1;
}
