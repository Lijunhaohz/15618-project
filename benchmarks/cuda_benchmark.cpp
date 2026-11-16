#include "benchmark_utils.hpp"
#include "cuda_timing.hpp"
#include "fusion/image_io.hpp"
#include "input_paths.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

void verify(const fusion::RgbImage& actual, const fusion::RgbImage& expected,
            double tolerance, const char* label) {
    if (actual.width != expected.width || actual.height != expected.height ||
        actual.pixels.size() != expected.pixels.size()) {
        throw std::runtime_error(std::string(label) + ": image dimensions do not match");
    }
    double maximum = 0.0;
    double squared_sum = 0.0;
    for (std::size_t p = 0; p < actual.pixels.size(); ++p) {
        if (!std::isfinite(actual.pixels[p]) || !std::isfinite(expected.pixels[p])) {
            throw std::runtime_error(std::string(label) + ": non-finite pixel");
        }
        const double error = std::abs(static_cast<double>(actual.pixels[p]) - expected.pixels[p]);
        maximum = std::max(maximum, error);
        squared_sum += error * error;
    }
    const auto rmse = std::sqrt(squared_sum / actual.pixels.size());
    std::cout << label << ": max_abs=" << std::scientific << maximum << " rmse=" << rmse << '\n';
    if (maximum > tolerance || rmse > 2e-6) {
        throw std::runtime_error(std::string(label) + ": error exceeds tolerance");
    }
}

}  // namespace

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: cuda_benchmark INPUT [INPUT ...] (image files or directories)\n";
        return 1;
    }
    try {
        std::vector<fusion::RgbImage> images;
        const std::vector<std::string> paths(argv + 1, argv + argc);
        const auto inputs = fusion::detail::expand_inputs(paths);
        images.reserve(inputs.size());
        for (const auto& input : inputs) images.push_back(fusion::load_image(input));
        auto cpu = fusion::make_cpu_backend();
        auto cuda = fusion::make_cuda_backend();
        const auto expected = cpu->fuse(images, {});

        for (int i = 0; i < benchmark::warmups; ++i) {
            fusion::detail::CudaStageTimings ignored;
            fusion::detail::profile_cuda_fuse(images, {}, ignored);
            cuda->fuse(images, {});
        }

        std::vector<double> quality, normalization, gaussian, laplacian, blend, reconstruction;
        std::vector<double> pipeline, wall;
        fusion::RgbImage profiled, ordinary;
        for (int i = 0; i < benchmark::iterations; ++i) {
            fusion::detail::CudaStageTimings timings;
            profiled = fusion::detail::profile_cuda_fuse(images, {}, timings);
            quality.push_back(timings.quality_ms);
            normalization.push_back(timings.normalization_ms);
            gaussian.push_back(timings.gaussian_ms);
            laplacian.push_back(timings.laplacian_ms);
            blend.push_back(timings.blend_ms);
            reconstruction.push_back(timings.reconstruction_ms);
            pipeline.push_back(timings.pipeline_ms);

            const auto start = std::chrono::steady_clock::now();
            auto result = cuda->fuse(images, {});
            const auto end = std::chrono::steady_clock::now();
            wall.push_back(std::chrono::duration<double, std::milli>(end - start).count());
            ordinary = std::move(result);
        }

        std::cout << images.front().width << 'x' << images.front().height << ", "
                  << images.size() << " exposures, "
                  << benchmark::levels_for(images.front().width, images.front().height)
                  << " levels; " << benchmark::warmups << " warmups, " << benchmark::iterations << " trials\n";
        verify(profiled, expected, 2e-5, "Profiled CUDA vs CPU");
        verify(ordinary, expected, 2e-5, "Uninstrumented CUDA vs CPU");
        verify(profiled, ordinary, 1e-6, "Profiled vs uninstrumented CUDA");
        benchmark::report("Quality (events)", quality);
        benchmark::report("Normalization (events)", normalization);
        benchmark::report("Gaussian pyramids (events)", gaussian);
        benchmark::report("Laplacian pyramids (events)", laplacian);
        benchmark::report("Blend (events)", blend);
        benchmark::report("Reconstruction (events)", reconstruction);
        benchmark::report("Pipeline (events)", pipeline);
        benchmark::report("Fuse incl. transfers (wall)", wall);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
