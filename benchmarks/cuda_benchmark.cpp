#include "cuda_timing.hpp"
#include "fusion/image_io.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr int warmups = 2;
constexpr int iterations = 10;

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const auto middle = values.size() / 2;
    return (values[middle - 1] + values[middle]) / 2.0;
}

void report(const char* label, const std::vector<double>& values) {
    const auto bounds = std::minmax_element(values.begin(), values.end());
    std::cout << std::left << std::setw(24) << label << std::right
              << std::fixed << std::setprecision(3) << " median=" << std::setw(9)
              << median(values) << " ms  min=" << std::setw(9) << *bounds.first
              << " ms  max=" << std::setw(9) << *bounds.second << " ms\n";
}

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

std::size_t levels_for(std::size_t width, std::size_t height) {
    std::size_t levels = 1;
    while (levels < 5 && (width > 1 || height > 1)) {
        width = width / 2 + width % 2;
        height = height / 2 + height % 2;
        ++levels;
    }
    return levels;
}

}  // namespace

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: cuda_benchmark IMAGE [IMAGE ...]\n";
        return 1;
    }
    try {
        std::vector<fusion::RgbImage> images;
        for (int i = 1; i < argc; ++i) images.push_back(fusion::load_image(argv[i]));
        auto cpu = fusion::make_cpu_backend();
        auto cuda = fusion::make_cuda_backend();
        const auto expected = cpu->fuse(images, {});

        for (int i = 0; i < warmups; ++i) {
            fusion::detail::CudaStageTimings ignored;
            fusion::detail::profile_cuda_fuse(images, {}, ignored);
            cuda->fuse(images, {});
        }

        std::vector<double> quality, normalization, gaussian, laplacian, blend, reconstruction;
        std::vector<double> pipeline, wall;
        fusion::RgbImage profiled, ordinary;
        for (int i = 0; i < iterations; ++i) {
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
                  << levels_for(images.front().width, images.front().height)
                  << " levels; " << warmups << " warmups, " << iterations << " trials\n";
        verify(profiled, expected, 2e-5, "Profiled CUDA vs CPU");
        verify(ordinary, expected, 2e-5, "Uninstrumented CUDA vs CPU");
        verify(profiled, ordinary, 1e-6, "Profiled vs uninstrumented CUDA");
        report("Quality (events)", quality);
        report("Normalization (events)", normalization);
        report("Gaussian pyramids (events)", gaussian);
        report("Laplacian pyramids (events)", laplacian);
        report("Blend (events)", blend);
        report("Reconstruction (events)", reconstruction);
        report("Pipeline (events)", pipeline);
        report("Fuse incl. transfers (wall)", wall);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
