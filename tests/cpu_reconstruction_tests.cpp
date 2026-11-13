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
        std::cerr << "Unexpected reconstructed dimensions\n";
        return false;
    }
    for (std::size_t p = 0; p < actual.pixels.size(); ++p) {
        if (!std::isfinite(actual.pixels[p]) ||
            std::abs(actual.pixels[p] - expected.pixels[p]) > tolerance) {
            std::cerr << "Reconstruction mismatch at element " << p << ": "
                      << actual.pixels[p] << " vs " << expected.pixels[p] << '\n';
            return false;
        }
    }
    return true;
}

bool check_roundtrip(fusion::Backend& backend, const fusion::RgbImage& input, std::size_t levels) {
    const auto pyramid = backend.build_laplacian_pyramid(input, levels);
    return same_image(backend.reconstruct(pyramid), input);
}

template <typename Exception, typename Function>
bool rejects(const char* name, Function function) {
    try {
        function();
        std::cerr << name << " was accepted\n";
    } catch (const Exception&) {
        return true;
    } catch (const std::exception& error) {
        std::cerr << name << " reported an unexpected error: " << error.what() << '\n';
    }
    return false;
}

}  // namespace

int main() {
    auto backend = fusion::make_cpu_backend();
    bool ok = true;
    const fusion::ImagePyramid manual{
        {3, 1, {-0.5f, 0.25f, 1.0f, 0.0f, 0.0f, 0.0f, 0.5f, -0.25f, -1.0f}},
        {2, 1, {-2.0f, 1.0f, 4.0f, 2.0f, 3.0f, 0.0f}},
    };
    const fusion::RgbImage expected{3, 1, {
        -1.5f, 1.75f, 4.0f, 0.0f, 2.0f, 2.0f, 1.5f, 2.25f, 0.0f,
    }};
    ok &= same_image(backend->reconstruct(manual), expected);
    if (manual[0].pixels[0] != -0.5f || manual[1].pixels[0] != -2.0f) {
        std::cerr << "Input pyramid was modified\n";
        ok = false;
    }
    ok &= same_image(backend->reconstruct({{1, 1, {-0.5f, 0.5f, 1.5f}}}),
                     {1, 1, {-0.5f, 0.5f, 1.5f}});

    const fusion::RgbImage odd{3, 3, {
        1.0f, 0.0f, 0.0f,  0.2f, 0.6f, 0.8f,  0.4f, 0.1f, 0.5f,
        0.3f, 0.9f, 0.2f,  0.6f, 0.3f, 0.1f,  0.7f, 0.5f, 0.9f,
        0.8f, 0.2f, 0.3f,  0.1f, 0.7f, 0.4f,  0.0f, 0.0f, 1.0f,
    }};
    ok &= check_roundtrip(*backend, odd, 3);
    ok &= check_roundtrip(*backend, odd, 1);
    ok &= check_roundtrip(*backend, {4, 2, {
        0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f,
        0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.9f, 0.8f, 0.1f, 1.0f, 1.0f, 1.0f,
    }}, 3);
    ok &= check_roundtrip(*backend, {1, 3, {0.0f, 0.5f, 1.0f, 0.8f, 0.2f, 0.3f, 0.2f, 0.7f, 0.1f}}, 3);
    ok &= check_roundtrip(*backend, {3, 1, {0.0f, 0.5f, 1.0f, 0.8f, 0.2f, 0.3f, 0.2f, 0.7f, 0.1f}}, 3);
    ok &= check_roundtrip(*backend, {1, 1, {0.1f, 0.5f, 1.0f}}, 1);
    ok &= check_roundtrip(*backend, {17, 1, std::vector<float>(51, 0.25f)}, 5);

    const auto weights = backend->normalize_weights(backend->compute_weights({odd}));
    const auto weight_pyramid = backend->build_weight_pyramid(weights[0], 3);
    const auto image_pyramid = backend->build_laplacian_pyramid(odd, 3);
    ok &= same_image(backend->reconstruct(backend->blend({weight_pyramid}, {image_pyramid})), odd);

    ok &= rejects<std::invalid_argument>("empty pyramid", [&] { backend->reconstruct({}); });
    ok &= rejects<std::invalid_argument>("zero dimensions", [&] { backend->reconstruct({{0, 1, {}}}); });
    ok &= rejects<std::invalid_argument>("overflowing dimensions", [&] {
        backend->reconstruct({{std::numeric_limits<std::size_t>::max(), 2, {}}});
    });
    ok &= rejects<std::invalid_argument>("too many levels", [&] {
        backend->reconstruct({{1, 1, {0.0f, 0.0f, 0.0f}}, {1, 1, {0.0f, 0.0f, 0.0f}}});
    });
    ok &= rejects<std::invalid_argument>("incorrect level dimensions", [&] {
        backend->reconstruct({{3, 1, std::vector<float>(9)}, {1, 1, std::vector<float>(3)}});
    });
    ok &= rejects<std::invalid_argument>("incorrect pixel count", [&] {
        backend->reconstruct({{1, 1, {0.0f, 0.0f}}});
    });
    ok &= rejects<std::invalid_argument>("NaN value", [&] {
        backend->reconstruct({{1, 1, {0.0f, std::numeric_limits<float>::quiet_NaN(), 0.0f}}});
    });
    ok &= rejects<std::invalid_argument>("infinite value", [&] {
        backend->reconstruct({{1, 1, {0.0f, std::numeric_limits<float>::infinity(), 0.0f}}});
    });
    ok &= rejects<std::overflow_error>("overflow during reconstruction", [&] {
        const float max = std::numeric_limits<float>::max();
        backend->reconstruct({{2, 1, {max, 0.0f, 0.0f, max, 0.0f, 0.0f}},
                              {1, 1, {max, 0.0f, 0.0f}}});
    });
    return ok ? 0 : 1;
}
