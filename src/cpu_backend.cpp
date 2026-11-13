/**
 * Implements the CPU backend for image fusion.
 */
#include "fusion/backend.hpp"

#include <stdexcept>

namespace fusion {
namespace {

class CpuBackend final : public Backend {
public:
    std::vector<WeightMap> compute_weights(const std::vector<RgbImage>&) override {
        throw std::logic_error("CPU weight computation is not implemented");
    }

    std::vector<WeightMap> normalize_weights(const std::vector<WeightMap>&) override {
        throw std::logic_error("CPU weight normalization is not implemented");
    }

    WeightPyramid build_weight_pyramid(const WeightMap&, std::size_t) override {
        throw std::logic_error("CPU weight pyramid construction is not implemented");
    }

    ImagePyramid build_laplacian_pyramid(const RgbImage&, std::size_t) override {
        throw std::logic_error("CPU Laplacian pyramid construction is not implemented");
    }

    ImagePyramid blend(
        const std::vector<WeightPyramid>&, const std::vector<ImagePyramid>&) override {
        throw std::logic_error("CPU pyramid blending is not implemented");
    }

    RgbImage reconstruct(const ImagePyramid&) override {
        throw std::logic_error("CPU reconstruction is not implemented");
    }

    RgbImage fuse(const std::vector<RgbImage>&, const FusionOptions&) override {
        throw std::logic_error("CPU fusion pipeline is not implemented");
    }
};

}  // namespace

std::unique_ptr<Backend> make_cpu_backend() {
    return std::make_unique<CpuBackend>();
}

}  // namespace fusion
