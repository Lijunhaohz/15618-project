/**
 * Implements the CPU backend for image fusion.
 */
#include "fusion/backend.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace fusion {
namespace {

/**
 * Validates that the input images are non-empty, have matching dimensions,
 * and contain valid pixel values in the range [0, 1].
 */
void validate_image(const RgbImage& image) {
    const auto width = image.width;
    const auto height = image.height;
    if (width == 0 || height == 0 ||
        width > std::numeric_limits<std::size_t>::max() / height ||
        width * height > std::numeric_limits<std::size_t>::max() / 3) {
        throw std::invalid_argument("Invalid image dimensions");
    }
    if (image.pixels.size() != width * height * 3) {
        throw std::invalid_argument("Input image must contain exactly three values per pixel");
    }
    for (const float value : image.pixels) {
        if (!std::isfinite(value) || value < 0.0f || value > 1.0f) {
            throw std::invalid_argument("Input image values must be finite and in [0, 1]");
        }
    }
}

void validate_images(const std::vector<RgbImage>& images) {
    if (images.empty()) {
        throw std::invalid_argument("At least one input image is required");
    }

    const auto width = images.front().width;
    const auto height = images.front().height;
    for (const auto& image : images) {
        if (image.width != width || image.height != height) {
            throw std::invalid_argument("Input image dimensions must match");
        }
        validate_image(image);
    }
}

std::size_t previous_index(std::size_t index, std::size_t length) {
    return index == 0 ? (length == 1 ? 0 : 1) : index - 1;
}

std::size_t next_index(std::size_t index, std::size_t length) {
    return index == length - 1 ? (length == 1 ? 0 : index - 1) : index + 1;
}

std::size_t reflected_index(std::size_t index, int offset, std::size_t length) {
    if (length == 1) {
        return 0;
    }
    int direction = offset < 0 ? -1 : 1;
    for (int step = 0; step < std::abs(offset); ++step) {
        if ((direction < 0 && index == 0) || (direction > 0 && index == length - 1)) {
            direction = -direction;
        }
        index = direction < 0 ? index - 1 : index + 1;
    }
    return index;
}

std::size_t half_rounded_up(std::size_t value) {
    return value / 2 + value % 2;
}

constexpr float kernel[] = {1.0f / 16.0f, 4.0f / 16.0f, 6.0f / 16.0f,
                            4.0f / 16.0f, 1.0f / 16.0f};

std::size_t max_pyramid_levels(std::size_t width, std::size_t height) {
    std::size_t levels = 1;
    while (levels < 5 && (width > 1 || height > 1)) {
        width = half_rounded_up(width);
        height = half_rounded_up(height);
        ++levels;
    }
    return levels;
}

template <typename Image>
Image reduce(const Image& input, std::size_t channels) {
    const auto width = half_rounded_up(input.width);
    const auto height = half_rounded_up(input.height);
    std::vector<float> horizontal(width * input.height * channels);
    for (std::size_t y = 0; y < input.height; ++y) {
        for (std::size_t x = 0; x < width; ++x) {
            for (std::size_t channel = 0; channel < channels; ++channel) {
                float value = 0.0f;
                for (int offset = -2; offset <= 2; ++offset) {
                    const auto sx = reflected_index(2 * x, offset, input.width);
                    value += kernel[offset + 2] *
                             input.pixels[(y * input.width + sx) * channels + channel];
                }
                horizontal[(y * width + x) * channels + channel] = value;
            }
        }
    }

    Image output{width, height, std::vector<float>(width * height * channels)};
    for (std::size_t y = 0; y < height; ++y) {
        for (std::size_t x = 0; x < width; ++x) {
            for (std::size_t channel = 0; channel < channels; ++channel) {
                float value = 0.0f;
                for (int offset = -2; offset <= 2; ++offset) {
                    const auto sy = reflected_index(2 * y, offset, input.height);
                    value += kernel[offset + 2] * horizontal[(sy * width + x) * channels + channel];
                }
                output.pixels[(y * width + x) * channels + channel] = value;
            }
        }
    }
    return output;
}

RgbImage expand_image(const RgbImage& input, std::size_t width, std::size_t height) {
    std::vector<float> horizontal(width * input.height * 3);
    for (std::size_t y = 0; y < input.height; ++y) {
        for (std::size_t x = 0; x < width; ++x) {
            for (std::size_t channel = 0; channel < 3; ++channel) {
                float value = 0.0f;
                for (int offset = -2; offset <= 2; ++offset) {
                    // x - 2*i = offset, so i = x/2 + (x%2 - offset)/2.
                    const int delta = static_cast<int>(x % 2) - offset;
                    if (delta % 2 != 0) continue;
                    const auto sx = reflected_index(x / 2, delta / 2, input.width);
                    value += 2.0f * kernel[offset + 2] *
                             input.pixels[(y * input.width + sx) * 3 + channel];
                }
                horizontal[(y * width + x) * 3 + channel] = value;
            }
        }
    }
    RgbImage output{width, height, std::vector<float>(width * height * 3)};
    for (std::size_t y = 0; y < height; ++y) {
        for (std::size_t x = 0; x < width; ++x) {
            for (std::size_t channel = 0; channel < 3; ++channel) {
                float value = 0.0f;
                for (int offset = -2; offset <= 2; ++offset) {
                    const int delta = static_cast<int>(y % 2) - offset;
                    if (delta % 2 != 0) continue;
                    const auto sy = reflected_index(y / 2, delta / 2, input.height);
                    value += 2.0f * kernel[offset + 2] * horizontal[(sy * width + x) * 3 + channel];
                }
                output.pixels[(y * width + x) * 3 + channel] = value;
            }
        }
    }
    return output;
}

class CpuBackend final : public Backend {
public:
    std::vector<WeightMap> compute_weights(const std::vector<RgbImage>& images) override {
        validate_images(images);
        std::vector<WeightMap> weights; // each input image will have a corresponding weight map
        weights.reserve(images.size());

        for (const auto& image : images) {
            const auto width = image.width;
            const auto height = image.height;

            // Convert the RGB image to grayscale using the luminance formula.
            std::vector<float> gray(width * height);
            for (std::size_t p = 0; p < gray.size(); ++p) {
                gray[p] = 0.299f * image.pixels[3 * p] +
                          0.587f * image.pixels[3 * p + 1] +
                          0.114f * image.pixels[3 * p + 2];
            }

            // Compute the contrast, saturation, and exposedness weights for each pixel.
            WeightMap map{width, height, std::vector<float>(gray.size())};
            for (std::size_t y = 0; y < height; ++y) {
                for (std::size_t x = 0; x < width; ++x) {
                    const auto p = y * width + x;
                    const float contrast = std::abs(
                        gray[y * width + previous_index(x, width)] +
                        gray[y * width + next_index(x, width)] +
                        gray[previous_index(y, height) * width + x] +
                        gray[next_index(y, height) * width + x] - 4.0f * gray[p]);
                    const float r = image.pixels[3 * p];
                    const float g = image.pixels[3 * p + 1];
                    const float b = image.pixels[3 * p + 2];
                    const float mean = (r + g + b) / 3.0f;
                    const float saturation = std::sqrt(
                        ((r - mean) * (r - mean) + (g - mean) * (g - mean) +
                         (b - mean) * (b - mean)) / 3.0f);
                    const float exposedness =
                        std::exp(-(r - 0.5f) * (r - 0.5f) / (2.0f * 0.2f * 0.2f)) *
                        std::exp(-(g - 0.5f) * (g - 0.5f) / (2.0f * 0.2f * 0.2f)) *
                        std::exp(-(b - 0.5f) * (b - 0.5f) / (2.0f * 0.2f * 0.2f));
                    map.pixels[p] = contrast * saturation * exposedness + 1e-12f;
                }
            }
            weights.push_back(std::move(map));
        }
        return weights;
    }

    std::vector<WeightMap> normalize_weights(const std::vector<WeightMap>& weights) override {
        // Validate the input weight maps.
        if (weights.empty()) {
            throw std::invalid_argument("At least one weight map is required");
        }

        const auto width = weights.front().width;
        const auto height = weights.front().height;
        if (width == 0 || height == 0 ||
            width > std::numeric_limits<std::size_t>::max() / height) {
            throw std::invalid_argument("Invalid weight map dimensions");
        }
        const auto pixel_count = width * height;
        for (const auto& map : weights) {
            if (map.width != width || map.height != height) {
                throw std::invalid_argument("Weight map dimensions must match");
            }
            if (map.pixels.size() != pixel_count) {
                throw std::invalid_argument("Weight map must contain one value per pixel");
            }
            for (float value : map.pixels) {
                if (!std::isfinite(value) || value < 0.0f) {
                    throw std::invalid_argument("Weights must be finite and nonnegative");
                }
            }
        }

        // Initialize the normalized weight maps with the same dimensions and pixel count as the input maps.
        std::vector<WeightMap> normalized;
        normalized.reserve(weights.size());
        for (std::size_t i = 0; i < weights.size(); ++i) {
            normalized.push_back({width, height, std::vector<float>(pixel_count)});
        }

        for (std::size_t p = 0; p < pixel_count; ++p) {
            float sum = 0.0f;
            for (const auto& map : weights) {
                sum += map.pixels[p];
            }
            if (!std::isfinite(sum) || sum <= 0.0f) {
                throw std::invalid_argument("Weight sum must be finite and positive at every pixel");
            }
            for (std::size_t i = 0; i < weights.size(); ++i) {
                normalized[i].pixels[p] = weights[i].pixels[p] / sum;
            }
        }
        return normalized;
    }

    WeightPyramid build_weight_pyramid(const WeightMap& weights, std::size_t levels) override {
        if (weights.width == 0 || weights.height == 0 ||
            weights.width > std::numeric_limits<std::size_t>::max() / weights.height) {
            throw std::invalid_argument("Invalid weight map dimensions");
        }
        if (weights.pixels.size() != weights.width * weights.height) {
            throw std::invalid_argument("Weight map must contain one value per pixel");
        }
        for (float value : weights.pixels) {
            if (!std::isfinite(value) || value < 0.0f || value > 1.0f) {
                throw std::invalid_argument("Normalized weights must be finite and in [0, 1]");
            }
        }

        if (levels == 0 || levels > max_pyramid_levels(weights.width, weights.height)) {
            throw std::invalid_argument("Invalid weight pyramid level count");
        }

        WeightPyramid pyramid;
        pyramid.reserve(levels);
        pyramid.push_back(weights);
        while (pyramid.size() < levels) {
            pyramid.push_back(reduce(pyramid.back(), 1));
        }
        return pyramid;
    }

    ImagePyramid build_laplacian_pyramid(const RgbImage& image, std::size_t levels) override {
        validate_image(image);
        if (levels == 0 || levels > max_pyramid_levels(image.width, image.height)) {
            throw std::invalid_argument("Invalid image pyramid level count");
        }

        ImagePyramid gaussian;
        gaussian.reserve(levels);
        gaussian.push_back(image);
        while (gaussian.size() < levels) {
            gaussian.push_back(reduce(gaussian.back(), 3));
        }

        ImagePyramid laplacian;
        laplacian.reserve(levels);
        for (std::size_t level = 0; level + 1 < levels; ++level) {
            auto expanded = expand_image(gaussian[level + 1],
                                         gaussian[level].width, gaussian[level].height);
            for (std::size_t p = 0; p < expanded.pixels.size(); ++p) {
                expanded.pixels[p] = gaussian[level].pixels[p] - expanded.pixels[p];
            }
            laplacian.push_back(std::move(expanded));
        }
        laplacian.push_back(std::move(gaussian.back()));
        return laplacian;
    }

    ImagePyramid blend(
        const std::vector<WeightPyramid>& weights,
        const std::vector<ImagePyramid>& images) override {
        if (images.empty() || images.size() != weights.size()) {
            throw std::invalid_argument("Each exposure must have an image and weight pyramid");
        }
        const auto levels = images.front().size();
        if (levels == 0) {
            throw std::invalid_argument("Pyramids must contain at least one level");
        }
        const auto width = images.front().front().width;
        const auto height = images.front().front().height;
        if (width == 0 || height == 0 ||
            width > std::numeric_limits<std::size_t>::max() / height ||
            width * height > std::numeric_limits<std::size_t>::max() / 3) {
            throw std::invalid_argument("Invalid pyramid dimensions");
        }
        if (levels > max_pyramid_levels(width, height)) {
            throw std::invalid_argument("Invalid pyramid level count");
        }

        for (std::size_t i = 0; i < images.size(); ++i) {
            if (images[i].size() != levels || weights[i].size() != levels) {
                throw std::invalid_argument("All pyramids must have the same number of levels");
            }
            auto level_width = width;
            auto level_height = height;
            for (std::size_t level = 0; level < levels; ++level) {
                const auto& image = images[i][level];
                const auto& weight = weights[i][level];
                const auto pixel_count = level_width * level_height;
                if (image.width != level_width || image.height != level_height ||
                    weight.width != level_width || weight.height != level_height) {
                    throw std::invalid_argument("Pyramid level dimensions do not match");
                }
                if (image.pixels.size() != pixel_count * 3 || weight.pixels.size() != pixel_count) {
                    throw std::invalid_argument("Invalid pyramid level pixel count");
                }
                for (float value : image.pixels) {
                    if (!std::isfinite(value)) {
                        throw std::invalid_argument("Laplacian values must be finite");
                    }
                }
                for (float value : weight.pixels) {
                    if (!std::isfinite(value) || value < 0.0f) {
                        throw std::invalid_argument("Pyramid weights must be finite and nonnegative");
                    }
                }
                level_width = half_rounded_up(level_width);
                level_height = half_rounded_up(level_height);
            }
        }

        ImagePyramid result;
        result.reserve(levels);
        for (std::size_t level = 0; level < levels; ++level) {
            const auto level_width = images.front()[level].width;
            const auto level_height = images.front()[level].height;
            RgbImage blended{level_width, level_height,
                             std::vector<float>(level_width * level_height * 3, 0.0f)};
            for (std::size_t i = 0; i < images.size(); ++i) {
                for (std::size_t p = 0; p < level_width * level_height; ++p) {
                    for (std::size_t channel = 0; channel < 3; ++channel) {
                        const auto index = p * 3 + channel;
                        blended.pixels[index] += weights[i][level].pixels[p] *
                                                 images[i][level].pixels[index];
                        if (!std::isfinite(blended.pixels[index])) {
                            throw std::overflow_error("Non-finite value produced while blending");
                        }
                    }
                }
            }
            result.push_back(std::move(blended));
        }
        return result;
    }

    RgbImage reconstruct(const ImagePyramid& blended) override {
        if (blended.empty()) {
            throw std::invalid_argument("Image pyramid must contain at least one level");
        }
        auto width = blended.front().width;
        auto height = blended.front().height;
        if (width == 0 || height == 0 ||
            width > std::numeric_limits<std::size_t>::max() / height ||
            width * height > std::numeric_limits<std::size_t>::max() / 3) {
            throw std::invalid_argument("Invalid pyramid dimensions");
        }
        if (blended.size() > max_pyramid_levels(width, height)) {
            throw std::invalid_argument("Invalid pyramid level count");
        }
        for (const auto& level : blended) {
            if (level.width != width || level.height != height) {
                throw std::invalid_argument("Pyramid level dimensions do not match");
            }
            if (level.pixels.size() != width * height * 3) {
                throw std::invalid_argument("Invalid pyramid level pixel count");
            }
            for (float value : level.pixels) {
                if (!std::isfinite(value)) {
                    throw std::invalid_argument("Pyramid values must be finite");
                }
            }
            width = half_rounded_up(width);
            height = half_rounded_up(height);
        }

        RgbImage reconstructed = blended.back();
        for (std::size_t level = blended.size() - 1; level > 0; --level) {
            const auto& finer = blended[level - 1];
            auto expanded = expand_image(reconstructed, finer.width, finer.height);
            for (std::size_t p = 0; p < expanded.pixels.size(); ++p) {
                expanded.pixels[p] += finer.pixels[p];
                if (!std::isfinite(expanded.pixels[p])) {
                    throw std::overflow_error("Non-finite value produced during reconstruction");
                }
            }
            reconstructed = std::move(expanded);
        }
        return reconstructed;
    }

    RgbImage fuse(const std::vector<RgbImage>& images, const FusionOptions& options) override {
        validate_images(images);
        const auto max_levels = max_pyramid_levels(images.front().width, images.front().height);
        const auto levels = options.levels.value_or(max_levels);
        if (levels == 0 || levels > max_levels) {
            throw std::invalid_argument("Invalid fusion pyramid level count");
        }

        const auto weights = normalize_weights(compute_weights(images));
        std::vector<WeightPyramid> weight_pyramids;
        std::vector<ImagePyramid> image_pyramids;
        weight_pyramids.reserve(images.size());
        image_pyramids.reserve(images.size());
        for (std::size_t i = 0; i < images.size(); ++i) {
            weight_pyramids.push_back(build_weight_pyramid(weights[i], levels));
            image_pyramids.push_back(build_laplacian_pyramid(images[i], levels));
        }
        return reconstruct(blend(weight_pyramids, image_pyramids));
    }
};

}  // namespace

std::unique_ptr<Backend> make_cpu_backend() {
    return std::make_unique<CpuBackend>();
}

}  // namespace fusion
