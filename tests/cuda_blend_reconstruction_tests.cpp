#include "fusion/backend.hpp"

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
        for (std::size_t p = 0; p < expected.pixels.size(); ++p) {
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

std::size_t level_count(std::size_t width, std::size_t height) {
    std::size_t levels = 1;
    while (levels < 5 && (width > 1 || height > 1)) {
        width = width / 2 + width % 2;
        height = height / 2 + height % 2;
        ++levels;
    }
    return levels;
}

bool compare_pyramids(Errors& errors, const char* name, const fusion::ImagePyramid& actual,
                      const fusion::ImagePyramid& expected, double tolerance) {
    if (actual.size() != expected.size()) {
        std::cerr << name << ": incorrect number of levels\n";
        return false;
    }
    bool ok = true;
    for (std::size_t level = 0; level < actual.size(); ++level) {
        if (!errors.compare(name, actual[level], expected[level], tolerance)) {
            std::cerr << "at level " << level << '\n';
            ok = false;
        }
    }
    return ok;
}

bool check_case(fusion::Backend& cpu, fusion::Backend& cuda,
                const std::vector<fusion::RgbImage>& input, Errors& blended_errors,
                Errors& reconstruction_errors, Errors& staged_blended_errors,
                Errors& staged_final_errors, Errors& round_trip_errors) {
    const auto levels = level_count(input.front().width, input.front().height);
    const auto cpu_weights = cpu.normalize_weights(cpu.compute_weights(input));
    const auto cuda_weights = cuda.normalize_weights(cuda.compute_weights(input));
    std::vector<fusion::WeightPyramid> cpu_weight_pyramids, cuda_weight_pyramids;
    std::vector<fusion::ImagePyramid> cpu_image_pyramids, cuda_image_pyramids;
    for (std::size_t i = 0; i < input.size(); ++i) {
        cpu_weight_pyramids.push_back(cpu.build_weight_pyramid(cpu_weights[i], levels));
        cuda_weight_pyramids.push_back(cuda.build_weight_pyramid(cuda_weights[i], levels));
        cpu_image_pyramids.push_back(cpu.build_laplacian_pyramid(input[i], levels));
        cuda_image_pyramids.push_back(cuda.build_laplacian_pyramid(input[i], levels));
    }
    const auto reference = cpu.blend(cpu_weight_pyramids, cpu_image_pyramids);
    const auto isolated = cuda.blend(cpu_weight_pyramids, cpu_image_pyramids);
    const auto staged = cuda.blend(cuda_weight_pyramids, cuda_image_pyramids);
    bool ok = compare_pyramids(blended_errors, "blend", isolated, reference, 1e-6);
    ok &= compare_pyramids(staged_blended_errors, "staged blend", staged, reference, 2e-5);

    for (std::size_t level = 0; level < levels; ++level) {
        const fusion::ImagePyramid suffix(reference.begin() + level, reference.end());
        ok &= reconstruction_errors.compare("reconstruction", cuda.reconstruct(suffix),
                                             cpu.reconstruct(suffix), 1e-6);
    }
    const auto expected_final = cpu.reconstruct(reference);
    const auto staged_final = cuda.reconstruct(staged);
    ok &= staged_final_errors.compare("staged final", staged_final, expected_final, 2e-5);
    if (input.size() == 1) {
        ok &= round_trip_errors.compare("single-exposure round trip", staged_final,
                                        input.front(), 2e-5);
    }
    return ok;
}

}  // namespace

int main() {
    auto cpu = fusion::make_cpu_backend();
    auto cuda = fusion::make_cuda_backend();
    Errors blended_errors, reconstruction_errors, staged_blended_errors, staged_final_errors;
    Errors round_trip_errors;
    bool ok = true;
    for (const auto dimensions : {std::pair<std::size_t, std::size_t>{37, 19},
                                  {8, 6}, {5, 3}, {1, 17}, {17, 1}, {2, 2}, {1, 1}}) {
        std::vector<fusion::RgbImage> input;
        for (std::size_t i = 0; i < 3; ++i) {
            fusion::RgbImage image{dimensions.first, dimensions.second,
                                   std::vector<float>(dimensions.first * dimensions.second * 3)};
            for (std::size_t p = 0; p < image.pixels.size(); ++p) {
                image.pixels[p] = static_cast<float>((p * 37 + i * 53 + 7) % 256) / 255.0f;
            }
            input.push_back(std::move(image));
        }
        ok &= check_case(*cpu, *cuda, input, blended_errors, reconstruction_errors,
                         staged_blended_errors, staged_final_errors, round_trip_errors);
        ok &= check_case(*cpu, *cuda, {input.front()}, blended_errors, reconstruction_errors,
                         staged_blended_errors, staged_final_errors, round_trip_errors);
    }
    const fusion::RgbImage constant{5, 3, std::vector<float>(45, 0.25f)};
    auto impulse = constant;
    impulse.pixels[0] = 1.0f;
    impulse.pixels[1] = 0.0f;
    ok &= check_case(*cpu, *cuda, {constant, impulse, constant}, blended_errors,
                     reconstruction_errors, staged_blended_errors, staged_final_errors,
                     round_trip_errors);

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
    const auto manual = cuda->blend(weights, images);
    ok &= compare_pyramids(blended_errors, "manual blend", manual,
                           cpu->blend(weights, images), 1e-6);
    ok &= reconstruction_errors.compare("manual reconstruction", cuda->reconstruct(images[0]),
                                         cpu->reconstruct(images[0]), 1e-6);
    const fusion::ImagePyramid unclipped{{1, 1, {-0.5f, 0.2f, 1.5f}}};
    ok &= reconstruction_errors.compare("unclipped reconstruction", cuda->reconstruct(unclipped),
                                         unclipped.front(), 0.0);
    const auto unclipped_blend = cuda->blend({{{1, 1, {2.0f}}}}, {unclipped});
    ok &= compare_pyramids(blended_errors, "unnormalized manual blend", unclipped_blend,
                           cpu->blend({{{1, 1, {2.0f}}}}, {unclipped}), 1e-6);

    ok &= rejects<std::invalid_argument>("empty blend", [&] { cuda->blend({}, {}); });
    ok &= rejects<std::invalid_argument>("exposure count", [&] { cuda->blend({weights[0]}, images); });
    ok &= rejects<std::invalid_argument>("empty blend pyramid", [&] { cuda->blend({{}}, {{}}); });
    ok &= rejects<std::invalid_argument>("blend level count", [&] {
        auto wrong = weights; wrong[0].pop_back(); cuda->blend(wrong, images);
    });
    ok &= rejects<std::invalid_argument>("blend image dimensions", [&] {
        auto wrong = images; wrong[1][1].width = 1; cuda->blend(weights, wrong);
    });
    ok &= rejects<std::invalid_argument>("blend weight dimensions", [&] {
        auto wrong = weights; wrong[0][1].height = 2; cuda->blend(wrong, images);
    });
    ok &= rejects<std::invalid_argument>("blend image size", [&] {
        auto wrong = images; wrong[0][0].pixels.pop_back(); cuda->blend(weights, wrong);
    });
    ok &= rejects<std::invalid_argument>("blend weight size", [&] {
        auto wrong = weights; wrong[0][0].pixels.pop_back(); cuda->blend(wrong, images);
    });
    ok &= rejects<std::invalid_argument>("negative blend weight", [&] {
        auto wrong = weights; wrong[0][0].pixels[0] = -1.0f; cuda->blend(wrong, images);
    });
    ok &= rejects<std::invalid_argument>("non-finite blend weight", [&] {
        auto wrong = weights; wrong[0][0].pixels[0] = std::numeric_limits<float>::quiet_NaN();
        cuda->blend(wrong, images);
    });
    ok &= rejects<std::invalid_argument>("non-finite blend image", [&] {
        auto wrong = images; wrong[1][0].pixels[0] = std::numeric_limits<float>::infinity();
        cuda->blend(weights, wrong);
    });
    ok &= rejects<std::invalid_argument>("blend levels past 1x1", [&] {
        cuda->blend({{{1, 1, {1.0f}}, {1, 1, {1.0f}}}},
                    {{{1, 1, {1.0f, 1.0f, 1.0f}}, {1, 1, {1.0f, 1.0f, 1.0f}}}});
    });
    ok &= rejects<std::overflow_error>("blend overflow", [&] {
        const float max = std::numeric_limits<float>::max();
        cuda->blend({{{1, 1, {2.0f}}}}, {{{1, 1, {max, 0.0f, 0.0f}}}});
    });
    ok &= rejects<std::overflow_error>("intermediate blend overflow", [&] {
        const float max = std::numeric_limits<float>::max();
        cuda->blend({{{1, 1, {1.0f}}}, {{1, 1, {1.0f}}}, {{1, 1, {1.0f}}}},
                    {{{1, 1, {max, 0.0f, 0.0f}}}, {{1, 1, {max, 0.0f, 0.0f}}},
                     {{1, 1, {-max, 0.0f, 0.0f}}}});
    });
    ok &= rejects<std::invalid_argument>("empty reconstruction", [&] { cuda->reconstruct({}); });
    ok &= rejects<std::invalid_argument>("zero reconstruction dimensions", [&] {
        cuda->reconstruct({{0, 1, {}}});
    });
    ok &= rejects<std::invalid_argument>("overflowing reconstruction dimensions", [&] {
        cuda->reconstruct({{std::numeric_limits<std::size_t>::max(), 2, {}}});
    });
    ok &= rejects<std::invalid_argument>("reconstruction levels past 1x1", [&] {
        cuda->reconstruct({{1, 1, {0.0f, 0.0f, 0.0f}}, {1, 1, {0.0f, 0.0f, 0.0f}}});
    });
    ok &= rejects<std::invalid_argument>("reconstruction dimensions", [&] {
        cuda->reconstruct({{3, 1, std::vector<float>(9)}, {1, 1, std::vector<float>(3)}});
    });
    ok &= rejects<std::invalid_argument>("reconstruction size", [&] {
        cuda->reconstruct({{1, 1, {0.0f, 0.0f}}});
    });
    ok &= rejects<std::invalid_argument>("non-finite reconstruction", [&] {
        cuda->reconstruct({{1, 1, {0.0f, std::numeric_limits<float>::quiet_NaN(), 0.0f}}});
    });
    ok &= rejects<std::overflow_error>("reconstruction overflow", [&] {
        const float max = std::numeric_limits<float>::max();
        cuda->reconstruct({{2, 1, {max, 0.0f, 0.0f, max, 0.0f, 0.0f}},
                           {1, 1, {max, 0.0f, 0.0f}}});
    });

    blended_errors.report("Isolated blended levels");
    reconstruction_errors.report("Isolated reconstruction");
    staged_blended_errors.report("Staged blended levels");
    staged_final_errors.report("Staged final images");
    round_trip_errors.report("Single-exposure round trips");
    return ok ? 0 : 1;
}
