#include "fusion/backend.hpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

bool near(float actual, double expected) {
    return std::abs(actual - expected) <= 1e-6 * (1.0 + std::abs(expected));
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

bool check_blend(fusion::Backend& backend, const std::vector<fusion::WeightPyramid>& weights,
                 const std::vector<fusion::ImagePyramid>& images) {
    const auto result = backend.blend(weights, images);
    if (result.size() != images.front().size()) {
        std::cerr << "Unexpected output pyramid size\n";
        return false;
    }
    for (std::size_t level = 0; level < result.size(); ++level) {
        const auto& output = result[level];
        if (output.width != images[0][level].width || output.height != images[0][level].height ||
            output.pixels.size() != output.width * output.height * 3) {
            std::cerr << "Incorrect dimensions at level " << level << '\n';
            return false;
        }
        for (std::size_t p = 0; p < output.width * output.height; ++p) {
            for (std::size_t channel = 0; channel < 3; ++channel) {
                double expected = 0.0;
                for (std::size_t i = 0; i < images.size(); ++i) {
                    expected += static_cast<double>(weights[i][level].pixels[p]) *
                                images[i][level].pixels[3 * p + channel];
                }
                if (!near(output.pixels[3 * p + channel], expected)) {
                    std::cerr << "Incorrect blend at level " << level << ", pixel " << p
                              << ", channel " << channel << '\n';
                    return false;
                }
            }
        }
    }
    return true;
}

}  // namespace

int main() {
    auto backend = fusion::make_cpu_backend();
    bool ok = true;
    const std::vector<fusion::WeightPyramid> weights{
        {{3, 1, {0.25f, 0.0f, 1.0f}}, {2, 1, {0.5f, 0.2f}}},
        {{3, 1, {0.75f, 1.0f, 0.0f}}, {2, 1, {0.5f, 0.8f}}},
    };
    const std::vector<fusion::ImagePyramid> images{
        {{3, 1, {-2.0f, 0.5f, 2.0f, 0.3f, -0.5f, 0.1f, 0.8f, 0.6f, -0.7f}},
         {2, 1, {0.2f, 0.3f, 0.4f, -0.5f, 0.6f, 2.0f}}},
        {{3, 1, {1.0f, -1.0f, 0.2f, 0.7f, 0.6f, -0.1f, 0.1f, 0.2f, 0.3f}},
         {2, 1, {0.6f, -0.3f, 0.1f, 1.5f, 0.1f, -1.0f}}},
    };
    ok &= check_blend(*backend, weights, images);
    const auto single = backend->blend(
        {{{1, 1, {1.0f}}}}, {{{1, 1, {-0.5f, 0.2f, 1.5f}}}});
    if (single.size() != 1 || single[0].pixels != std::vector<float>({-0.5f, 0.2f, 1.5f})) {
        std::cerr << "Single exposure was not preserved\n";
        ok = false;
    }
    if (images[0][0].pixels[0] != -2.0f || weights[0][0].pixels[0] != 0.25f) {
        std::cerr << "Input pyramids were modified\n";
        ok = false;
    }

    const std::vector<fusion::RgbImage> inputs{
        {3, 1, {0.1f, 0.3f, 0.8f, 0.2f, 0.4f, 0.6f, 0.7f, 0.3f, 0.1f}},
        {3, 1, {0.5f, 0.6f, 0.7f, 0.9f, 0.2f, 0.3f, 0.1f, 0.8f, 0.4f}},
        {3, 1, {0.3f, 0.2f, 0.1f, 0.4f, 0.5f, 0.8f, 0.6f, 0.7f, 0.9f}},
    };
    const auto normalized = backend->normalize_weights(backend->compute_weights(inputs));
    std::vector<fusion::WeightPyramid> computed_weights;
    std::vector<fusion::ImagePyramid> computed_images;
    for (std::size_t i = 0; i < inputs.size(); ++i) {
        computed_weights.push_back(backend->build_weight_pyramid(normalized[i], 3));
        computed_images.push_back(backend->build_laplacian_pyramid(inputs[i], 3));
    }
    ok &= check_blend(*backend, computed_weights, computed_images);

    ok &= rejects<std::invalid_argument>("empty inputs", [&] { backend->blend({}, {}); });
    ok &= rejects<std::invalid_argument>("different exposure counts", [&] {
        backend->blend({weights[0]}, images);
    });
    ok &= rejects<std::invalid_argument>("empty pyramid", [&] { backend->blend({{}}, {{}}); });
    ok &= rejects<std::invalid_argument>("different level counts", [&] {
        auto wrong = weights;
        wrong[0].pop_back();
        backend->blend(wrong, images);
    });
    ok &= rejects<std::invalid_argument>("incorrect image dimensions", [&] {
        auto wrong = images;
        wrong[1][1].width = 1;
        backend->blend(weights, wrong);
    });
    ok &= rejects<std::invalid_argument>("incorrect weight dimensions", [&] {
        auto wrong = weights;
        wrong[0][1].height = 2;
        backend->blend(wrong, images);
    });
    ok &= rejects<std::invalid_argument>("invalid level progression", [&] {
        auto wrong_weights = weights;
        auto wrong_images = images;
        wrong_weights[0][1].width = 3;
        wrong_images[0][1].width = 3;
        backend->blend(wrong_weights, wrong_images);
    });
    ok &= rejects<std::invalid_argument>("incorrect image buffer size", [&] {
        auto wrong = images;
        wrong[0][0].pixels.pop_back();
        backend->blend(weights, wrong);
    });
    ok &= rejects<std::invalid_argument>("incorrect weight buffer size", [&] {
        auto wrong = weights;
        wrong[0][0].pixels.pop_back();
        backend->blend(wrong, images);
    });
    ok &= rejects<std::invalid_argument>("negative weight", [&] {
        auto wrong = weights;
        wrong[0][0].pixels[0] = -1.0f;
        backend->blend(wrong, images);
    });
    ok &= rejects<std::invalid_argument>("non-finite weight", [&] {
        auto wrong = weights;
        wrong[0][0].pixels[0] = std::numeric_limits<float>::quiet_NaN();
        backend->blend(wrong, images);
    });
    ok &= rejects<std::invalid_argument>("non-finite Laplacian", [&] {
        auto wrong = images;
        wrong[1][0].pixels[0] = std::numeric_limits<float>::infinity();
        backend->blend(weights, wrong);
    });
    ok &= rejects<std::invalid_argument>("invalid level count", [&] {
        backend->blend({{{1, 1, {1.0f}}, {1, 1, {1.0f}}}},
                       {{{1, 1, {1.0f, 1.0f, 1.0f}}, {1, 1, {1.0f, 1.0f, 1.0f}}}});
    });
    ok &= rejects<std::overflow_error>("overflow in blend", [&] {
        const float max = std::numeric_limits<float>::max();
        backend->blend({{{1, 1, {2.0f}}}}, {{{1, 1, {max, 0.0f, 0.0f}}}});
    });
    return ok ? 0 : 1;
}
