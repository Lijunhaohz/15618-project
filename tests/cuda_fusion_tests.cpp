#include "fusion/backend.hpp"
#include "cuda_timing.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

struct Errors {
    double maximum = 0.0;
    double squared_sum = 0.0;
    std::size_t count = 0;

    bool compare(const char* name, const fusion::RgbImage& actual,
                 const fusion::RgbImage& expected, double tolerance) {
        if (actual.width != expected.width || actual.height != expected.height ||
            actual.pixels.size() != expected.pixels.size()) {
            std::cerr << name << ": incorrect dimensions\n";
            return false;
        }
        bool ok = true;
        for (std::size_t p = 0; p < actual.pixels.size(); ++p) {
            const double error = std::abs(static_cast<double>(actual.pixels[p]) - expected.pixels[p]);
            if (!std::isfinite(actual.pixels[p]) || error > tolerance) {
                if (ok) {
                    std::cerr << name << ": mismatch at element " << p << ": "
                              << actual.pixels[p] << " vs " << expected.pixels[p] << '\n';
                }
                ok = false;
            }
            maximum = std::max(maximum, error);
            squared_sum += error * error;
            ++count;
        }
        return ok;
    }

    void report(const char* name) const {
        std::cout << name << ": " << count << " values, max absolute error " << maximum
                  << ", RMSE " << std::sqrt(squared_sum / count) << '\n';
    }
};

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
    auto cpu = fusion::make_cpu_backend();
    auto cuda = fusion::make_cuda_backend();
    Errors cpu_errors, stage_errors;
    bool ok = true;
    std::vector<fusion::RgbImage> images;
    for (std::size_t i = 0; i < 3; ++i) {
        fusion::RgbImage image{37, 19, std::vector<float>(37 * 19 * 3)};
        for (std::size_t p = 0; p < image.pixels.size(); ++p) {
            image.pixels[p] = static_cast<float>((p * 37 + i * 53 + 7) % 256) / 255.0f;
        }
        images.push_back(std::move(image));
    }
    for (std::size_t levels = 1; levels <= 5; ++levels) {
        fusion::FusionOptions options;
        options.levels = levels;
        const auto actual = cuda->fuse(images, options);
        ok &= cpu_errors.compare("CUDA vs CPU fusion", actual, cpu->fuse(images, options), 2e-5);
        ok &= stage_errors.compare("CUDA fusion vs separate stages", actual,
                                   assemble_stages(*cuda, images, levels), 1e-6);
    }
    ok &= cpu_errors.compare("automatic depth", cuda->fuse(images, {}), cpu->fuse(images, {}), 2e-5);
    for (const std::size_t levels : {std::size_t{1}, std::size_t{5}}) {
        fusion::FusionOptions options;
        options.levels = levels;
        fusion::detail::CudaStageTimings timings;
        const auto profiled = fusion::detail::profile_cuda_fuse(images, options, timings);
        ok &= stage_errors.compare("profiled CUDA fusion", profiled, cuda->fuse(images, options), 1e-6);
        const float stages[] = {timings.quality_ms, timings.normalization_ms, timings.gaussian_ms,
                                timings.laplacian_ms, timings.blend_ms, timings.reconstruction_ms};
        float total = 0.0f;
        for (const float stage : stages) {
            if (!std::isfinite(stage) || stage < 0.0f) {
                std::cerr << "Invalid stage timing\n";
                ok = false;
            }
            total += stage;
        }
        if (!std::isfinite(timings.pipeline_ms) || timings.pipeline_ms <= 0.0f ||
            std::abs(total - timings.pipeline_ms) > 0.01f) {
            std::cerr << "Pipeline timing does not match stage intervals\n";
            ok = false;
        }
    }
    ok &= stage_errors.compare("single exposure", cuda->fuse({images[0]}, {}), images[0], 1e-6);
    const fusion::RgbImage tiny{1, 1, {0.1f, 0.5f, 1.0f}};
    ok &= stage_errors.compare("single pixel", cuda->fuse({tiny}, {}), tiny, 1e-6);
    const fusion::RgbImage dark{2, 1, std::vector<float>(6, 0.0f)};
    const fusion::RgbImage bright{2, 1, std::vector<float>(6, 1.0f)};
    ok &= cpu_errors.compare("constant bracket", cuda->fuse({dark, bright}, {}),
                             cpu->fuse({dark, bright}, {}), 2e-5);

    ok &= rejects("empty input", [&] { cuda->fuse({}, {}); });
    ok &= rejects("mismatched dimensions", [&] { cuda->fuse({images[0], tiny}, {}); });
    ok &= rejects("incorrect pixel count", [&] { cuda->fuse({{1, 1, {0.5f}}}, {}); });
    ok &= rejects("non-finite pixel", [&] {
        cuda->fuse({{1, 1, {0.5f, std::numeric_limits<float>::quiet_NaN(), 0.0f}}}, {});
    });
    ok &= rejects("out-of-range pixel", [&] { cuda->fuse({{1, 1, {0.5f, 1.1f, 0.0f}}}, {}); });
    ok &= rejects("zero levels", [&] {
        fusion::FusionOptions options; options.levels = 0; cuda->fuse(images, options);
    });
    ok &= rejects("levels past 1x1", [&] {
        fusion::FusionOptions options; options.levels = 2; cuda->fuse({tiny}, options);
    });
    ok &= rejects("levels past five", [&] {
        fusion::FusionOptions options; options.levels = 6; cuda->fuse(images, options);
    });
    cpu_errors.report("CUDA fusion vs CPU");
    stage_errors.report("CUDA fusion vs individual stages and single exposure");
    return ok ? 0 : 1;
}
