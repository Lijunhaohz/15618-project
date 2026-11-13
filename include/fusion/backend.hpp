/**
 * Declares the Backend interface for image fusion, along with related options and factory functions.
 */

#pragma once

#include "fusion/image.hpp"

#include <cstddef>
#include <memory>
#include <optional>
#include <vector>

namespace fusion {

struct FusionOptions {
    // An unset value selects the automatic depth described in the specification.
    std::optional<std::size_t> levels;
};

class Backend {
public:
    virtual ~Backend() = default;

    virtual std::vector<WeightMap> compute_weights(
        const std::vector<RgbImage>& images) = 0;

    virtual std::vector<WeightMap> normalize_weights(
        const std::vector<WeightMap>& weights) = 0;

    virtual WeightPyramid build_weight_pyramid(
        const WeightMap& weights, std::size_t levels) = 0;

    virtual ImagePyramid build_laplacian_pyramid(
        const RgbImage& image, std::size_t levels) = 0;

    // Weight pyramids must be normalized across exposures at each pixel.
    virtual ImagePyramid blend(
        const std::vector<WeightPyramid>& weights,
        const std::vector<ImagePyramid>& images) = 0;

    virtual RgbImage reconstruct(const ImagePyramid& blended) = 0;
    
    virtual RgbImage fuse(
        const std::vector<RgbImage>& images, const FusionOptions& options) = 0;
};

std::unique_ptr<Backend> make_cpu_backend();
std::unique_ptr<Backend> make_cuda_backend();

}  // namespace fusion
