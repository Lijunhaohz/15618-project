#include "fusion/backend.hpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

bool same_image(const fusion::RgbImage& actual, const fusion::RgbImage& expected,
                float tolerance = 1e-6f) {
    if (actual.width != expected.width || actual.height != expected.height ||
        actual.pixels.size() != expected.pixels.size()) {
        std::cerr << "Unexpected fused image dimensions\n";
        return false;
    }
    for (std::size_t p = 0; p < actual.pixels.size(); ++p) {
        if (!std::isfinite(actual.pixels[p]) ||
            std::abs(actual.pixels[p] - expected.pixels[p]) > tolerance) {
            std::cerr << "Fusion mismatch at element " << p << ": "
                      << actual.pixels[p] << " vs " << expected.pixels[p] << '\n';
            return false;
        }
    }
    return true;
}

fusion::RgbImage assemble_stages(fusion::Backend& backend,
                                 const std::vector<fusion::RgbImage>& images,
                                 std::size_t levels) {
    const auto weights = backend.normalize_weights(backend.compute_weights(images));
    std::vector<fusion::WeightPyramid> weight_pyramids;
    std::vector<fusion::ImagePyramid> image_pyramids;
    for (std::size_t i = 0; i < images.size(); ++i) {
        weight_pyramids.push_back(backend.build_weight_pyramid(weights[i], levels));
        image_pyramids.push_back(backend.build_laplacian_pyramid(images[i], levels));
    }
    return backend.reconstruct(backend.blend(weight_pyramids, image_pyramids));
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
    const fusion::RgbImage first{3, 3, {
        1.0f, 0.0f, 0.0f,  0.2f, 0.6f, 0.8f,  0.4f, 0.1f, 0.5f,
        0.3f, 0.9f, 0.2f,  0.6f, 0.3f, 0.1f,  0.7f, 0.5f, 0.9f,
        0.8f, 0.2f, 0.3f,  0.1f, 0.7f, 0.4f,  0.0f, 0.0f, 1.0f,
    }};
    const fusion::RgbImage second{3, 3, {
        0.3f, 0.6f, 0.2f,  0.9f, 0.2f, 0.4f,  0.1f, 0.7f, 0.8f,
        0.5f, 0.4f, 0.1f,  0.2f, 0.8f, 0.6f,  0.4f, 0.3f, 0.7f,
        0.2f, 0.5f, 0.9f,  0.8f, 0.1f, 0.6f,  0.6f, 0.4f, 0.3f,
    }};
    const std::vector<fusion::RgbImage> images{first, second};

    for (std::size_t levels = 1; levels <= 3; ++levels) {
        fusion::FusionOptions options;
        options.levels = levels;
        ok &= same_image(backend->fuse(images, options), assemble_stages(*backend, images, levels));
        ok &= same_image(backend->fuse({first}, options), first);
        ok &= same_image(backend->fuse({first, first}, options), first);
    }
    ok &= same_image(backend->fuse(images, {}), assemble_stages(*backend, images, 3));
    ok &= same_image(backend->fuse({first}, {}), first);

    fusion::FusionOptions one_level;
    one_level.levels = 1;
    const auto automatic = backend->fuse(images, {});
    const auto single_level = backend->fuse(images, one_level);
    bool different_depth_changes_result = false;
    for (std::size_t p = 0; p < automatic.pixels.size(); ++p) {
        if (std::abs(automatic.pixels[p] - single_level.pixels[p]) > 1e-5f) {
            different_depth_changes_result = true;
        }
    }
    if (!different_depth_changes_result) {
        std::cerr << "Test input does not distinguish automatic depth from one level\n";
        ok = false;
    }
    const auto normalized = backend->normalize_weights(backend->compute_weights(images));
    fusion::RgbImage direct{first.width, first.height, std::vector<float>(first.pixels.size())};
    for (std::size_t p = 0; p < first.width * first.height; ++p) {
        for (std::size_t channel = 0; channel < 3; ++channel) {
            direct.pixels[3 * p + channel] =
                normalized[0].pixels[p] * first.pixels[3 * p + channel] +
                normalized[1].pixels[p] * second.pixels[3 * p + channel];
        }
    }
    ok &= same_image(backend->fuse(images, one_level), direct);

    const fusion::RgbImage dark{2, 1, std::vector<float>(6, 0.0f)};
    const fusion::RgbImage bright{2, 1, std::vector<float>(6, 1.0f)};
    ok &= same_image(backend->fuse({dark, bright}, {}), {2, 1, std::vector<float>(6, 0.5f)});
    const fusion::RgbImage large{17, 1, std::vector<float>(51, 0.25f)};
    ok &= same_image(backend->fuse({large}, {}), large);
    fusion::FusionOptions five_levels;
    five_levels.levels = 5;
    ok &= same_image(backend->fuse({large}, {}), backend->fuse({large}, five_levels));
    const fusion::RgbImage one_pixel{1, 1, {0.1f, 0.5f, 1.0f}};
    ok &= same_image(backend->fuse({one_pixel}, {}), one_pixel);
    if (images[0].pixels != first.pixels || images[1].pixels != second.pixels) {
        std::cerr << "Input images were modified\n";
        ok = false;
    }

    ok &= rejects("empty input", [&] { backend->fuse({}, {}); });
    ok &= rejects("mismatched dimensions", [&] { backend->fuse({first, one_pixel}, {}); });
    ok &= rejects("incorrect pixel count", [&] { backend->fuse({{1, 1, {0.5f}}}, {}); });
    ok &= rejects("non-finite value", [&] {
        backend->fuse({{1, 1, {0.5f, std::numeric_limits<float>::quiet_NaN(), 0.0f}}}, {});
    });
    ok &= rejects("out-of-range value", [&] { backend->fuse({{1, 1, {0.5f, 1.1f, 0.0f}}}, {}); });
    ok &= rejects("zero levels", [&] {
        fusion::FusionOptions options;
        options.levels = 0;
        backend->fuse(images, options);
    });
    ok &= rejects("levels past 1x1", [&] {
        fusion::FusionOptions options;
        options.levels = 4;
        backend->fuse(images, options);
    });
    ok &= rejects("levels past five", [&] {
        fusion::FusionOptions options;
        options.levels = 6;
        backend->fuse({{33, 1, std::vector<float>(99, 0.5f)}}, options);
    });
    return ok ? 0 : 1;
}
