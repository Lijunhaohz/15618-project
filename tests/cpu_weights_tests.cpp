#include "fusion/backend.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

bool close(float actual, double expected) {
    return std::abs(actual - expected) <= 1e-15 + 2e-5 * std::abs(expected);
}

double reference_weight(const fusion::RgbImage& image, std::size_t x, std::size_t y) {
    const auto reflect = [](int coordinate, int size) {
        if (size == 1) return 0;
        if (coordinate < 0) return -coordinate;
        if (coordinate >= size) return 2 * size - 2 - coordinate;
        return coordinate;
    };
    const auto intensity = [&](int px, int py) {
        const auto p = static_cast<std::size_t>(reflect(py, static_cast<int>(image.height))) *
                           image.width +
                       static_cast<std::size_t>(reflect(px, static_cast<int>(image.width)));
        return 0.299 * image.pixels[3 * p] + 0.587 * image.pixels[3 * p + 1] +
               0.114 * image.pixels[3 * p + 2];
    };
    const int px = static_cast<int>(x);
    const int py = static_cast<int>(y);
    const double contrast = std::abs(intensity(px - 1, py) + intensity(px + 1, py) +
                                     intensity(px, py - 1) + intensity(px, py + 1) -
                                     4 * intensity(px, py));
    const auto offset = 3 * (y * image.width + x);
    const double r = image.pixels[offset];
    const double g = image.pixels[offset + 1];
    const double b = image.pixels[offset + 2];
    const double mean = (r + g + b) / 3;
    const double saturation = std::sqrt(((r - mean) * (r - mean) +
                                         (g - mean) * (g - mean) +
                                         (b - mean) * (b - mean)) / 3);
    const auto well_exposed = [](double value) {
        return std::exp(-(value - 0.5) * (value - 0.5) / 0.08);
    };
    return contrast * saturation * well_exposed(r) * well_exposed(g) * well_exposed(b) + 1e-12;
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

bool compare_image(fusion::Backend& backend, const fusion::RgbImage& image) {
    const auto result = backend.compute_weights({image});
    if (result.size() != 1 || result[0].width != image.width ||
        result[0].height != image.height || result[0].pixels.size() != image.width * image.height) {
        std::cerr << "Unexpected weight map dimensions\n";
        return false;
    }
    for (std::size_t y = 0; y < image.height; ++y) {
        for (std::size_t x = 0; x < image.width; ++x) {
            const auto actual = result[0].pixels[y * image.width + x];
            const auto expected = reference_weight(image, x, y);
            if (!close(actual, expected)) {
                std::cerr << "Weight mismatch at " << x << "," << y << ": "
                          << actual << " vs " << expected << '\n';
                return false;
            }
        }
    }
    return true;
}

}  // namespace

int main() {
    auto backend = fusion::make_cpu_backend();
    bool ok = true;
    const fusion::RgbImage image{3, 2, {
        0.2f, 0.5f, 0.7f,  0.8f, 0.4f, 0.3f,  0.1f, 0.3f, 0.5f,
        0.9f, 0.7f, 0.2f,  0.3f, 0.8f, 0.4f,  0.6f, 0.2f, 0.9f,
    }};
    const fusion::RgbImage one_column{1, 3, {
        0.2f, 0.5f, 0.7f, 0.8f, 0.4f, 0.3f, 0.1f, 0.3f, 0.5f,
    }};
    const fusion::RgbImage one_row{3, 1, one_column.pixels};
    const fusion::RgbImage constant{2, 1, {0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f}};
    const fusion::RgbImage one_pixel{1, 1, {0.2f, 0.4f, 0.8f}};
    ok &= compare_image(*backend, image);
    ok &= compare_image(*backend, one_column);
    ok &= compare_image(*backend, one_row);
    ok &= compare_image(*backend, constant);
    ok &= compare_image(*backend, one_pixel);
    const auto multiple = backend->compute_weights({image, image});
    ok &= multiple.size() == 2 && multiple[0].pixels == multiple[1].pixels;
    ok &= rejects("empty input", [&] { backend->compute_weights({}); });
    ok &= rejects("zero dimensions", [&] { backend->compute_weights({{0, 1, {}}}); });
    ok &= rejects("overflowing dimensions", [&] {
        backend->compute_weights({{std::numeric_limits<std::size_t>::max(), 2, {}}});
    });
    ok &= rejects("mismatched dimensions", [&] { backend->compute_weights({image, one_pixel}); });
    ok &= rejects("wrong pixel count", [&] { backend->compute_weights({{1, 1, {0.5f}}}); });
    ok &= rejects("out-of-range value", [&] { backend->compute_weights({{1, 1, {0.5f, 1.1f, 0.0f}}}); });
    ok &= rejects("NaN value", [&] {
        backend->compute_weights({{1, 1, {0.5f, std::numeric_limits<float>::quiet_NaN(), 0.0f}}});
    });
    ok &= rejects("infinite value", [&] {
        backend->compute_weights({{1, 1, {0.5f, std::numeric_limits<float>::infinity(), 0.0f}}});
    });
    return ok ? 0 : 1;
}
