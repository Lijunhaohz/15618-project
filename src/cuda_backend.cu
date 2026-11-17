/**
 * Implements the CUDA backend for image fusion.
 */
#include "fusion/backend.hpp"
#include "cuda_timing.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace fusion {
namespace {

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

void check_cuda_device() {
    int count = 0;
    check_cuda(cudaGetDeviceCount(&count), "CUDA backend unavailable (cudaGetDeviceCount)");
    if (count == 0) throw std::runtime_error("CUDA backend unavailable: no CUDA device detected");
}

class CudaEvent {
public:
    CudaEvent() {
        check_cuda(cudaEventCreate(&event_), "cudaEventCreate");
    }

    CudaEvent(const CudaEvent&) = delete;
    CudaEvent& operator=(const CudaEvent&) = delete;

    ~CudaEvent() {
        if (event_) {
            const auto status = cudaEventDestroy(event_);
            if (status != cudaSuccess) {
                std::fprintf(stderr, "cudaEventDestroy during cleanup: %s\n",
                             cudaGetErrorString(status));
            }
        }
    }

    cudaEvent_t get() const { return event_; }

    void release() {
        const auto event = event_;
        event_ = nullptr;
        check_cuda(cudaEventDestroy(event), "cudaEventDestroy");
    }

private:
    cudaEvent_t event_ = nullptr;
};

class StageEvents {
public:
    StageEvents() {
        events_.reserve(7);
        for (int i = 0; i < 7; ++i) events_.push_back(std::make_unique<CudaEvent>());
    }

    void record(std::size_t index) {
        check_cuda(cudaEventRecord(events_[index]->get()), "cudaEventRecord");
    }

    void finish(detail::CudaStageTimings& timings) {
        check_cuda(cudaEventSynchronize(events_.back()->get()), "cudaEventSynchronize");
        timings.quality_ms = elapsed(0, 1);
        timings.normalization_ms = elapsed(1, 2);
        timings.gaussian_ms = elapsed(2, 3);
        timings.laplacian_ms = elapsed(3, 4);
        timings.blend_ms = elapsed(4, 5);
        timings.reconstruction_ms = elapsed(5, 6);
        timings.pipeline_ms = elapsed(0, 6);
        for (auto& event : events_) event->release();
    }

private:
    float elapsed(std::size_t start, std::size_t end) const {
        float milliseconds = 0.0f;
        check_cuda(cudaEventElapsedTime(&milliseconds, events_[start]->get(), events_[end]->get()),
                   "cudaEventElapsedTime");
        return milliseconds;
    }

    std::vector<std::unique_ptr<CudaEvent>> events_;
};

// Thread-local scope keeps allocation accounting out of ordinary timed calls.
thread_local detail::CudaStageTimings* allocation_stats = nullptr;

class AllocationScope {
public:
    explicit AllocationScope(detail::CudaStageTimings* stats) : previous_(allocation_stats) {
        allocation_stats = stats;
        if (stats) {
            stats->live_device_bytes = stats->peak_device_bytes = 0;
            stats->total_allocated_bytes = stats->allocation_count = 0;
        }
    }
    ~AllocationScope() { allocation_stats = previous_; }
private:
    detail::CudaStageTimings* previous_;
};

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t count) : data_(nullptr), bytes_(0), stats_(allocation_stats) {
        if (count > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
            throw std::overflow_error("CUDA allocation size overflow");
        }
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&data_), count * sizeof(T)),
                   "cudaMalloc");
        bytes_ = count * sizeof(T);
        if (stats_) {
            stats_->live_device_bytes += bytes_;
            stats_->peak_device_bytes = std::max(stats_->peak_device_bytes, stats_->live_device_bytes);
            stats_->total_allocated_bytes += bytes_;
            ++stats_->allocation_count;
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    // Explicit release reports errors on success paths; this covers exception unwinding.
    ~DeviceBuffer() {
        if (data_) {
            const auto status = cudaFree(data_);
            if (status == cudaSuccess && stats_) stats_->live_device_bytes -= bytes_;
            if (status != cudaSuccess) {
                std::fprintf(stderr, "cudaFree during cleanup: %s\n", cudaGetErrorString(status));
            }
        }
    }

    T* get() const { return data_; }

    void release() {
        if (!data_) return;
        check_cuda(cudaFree(data_), "cudaFree");
        if (stats_) stats_->live_device_bytes -= bytes_;
        data_ = nullptr;
    }

private:
    T* data_;
    std::size_t bytes_;
    detail::CudaStageTimings* stats_;
};

std::size_t checked_product(std::size_t a, std::size_t b) {
    if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a) {
        throw std::overflow_error("CUDA buffer size overflow");
    }
    return a * b;
}

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

void validate_weights(const std::vector<WeightMap>& weights) {
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
        for (const float value : map.pixels) {
            if (!std::isfinite(value) || value < 0.0f) {
                throw std::invalid_argument("Weights must be finite and nonnegative");
            }
        }
    }
}

constexpr unsigned int threads_per_block = 256;

unsigned int block_count(std::size_t work) {
    const auto blocks = work / threads_per_block + (work % threads_per_block != 0);
    return static_cast<unsigned int>(std::min<std::size_t>(blocks, 65535));
}

void check_kernel(const char* operation) {
    // Check both launch errors and errors from executing the kernel.
    check_cuda(cudaGetLastError(), operation);
    check_cuda(cudaDeviceSynchronize(), operation);
}

std::size_t half_rounded_up(std::size_t value) {
    return value / 2 + value % 2;
}

std::size_t max_pyramid_levels(std::size_t width, std::size_t height) {
    std::size_t levels = 1;
    while (levels < 5 && (width > 1 || height > 1)) {
        width = half_rounded_up(width);
        height = half_rounded_up(height);
        ++levels;
    }
    return levels;
}

__device__ std::size_t reflected_index(std::size_t index, int offset, std::size_t length) {
    // Reflect without repeating the edge, including repeated reflections for short axes.
    if (length == 1) return 0;
    int direction = offset < 0 ? -1 : 1;
    const int steps = offset < 0 ? -offset : offset;
    for (int step = 0; step < steps; ++step) {
        if ((direction < 0 && index == 0) || (direction > 0 && index == length - 1)) {
            direction = -direction;
        }
        index = direction < 0 ? index - 1 : index + 1;
    }
    return index;
}

__device__ float gaussian_coefficient(int offset) {
    constexpr float kernel[] = {1.0f / 16.0f, 4.0f / 16.0f, 6.0f / 16.0f,
                                4.0f / 16.0f, 1.0f / 16.0f};
    return kernel[offset + 2];
}

__global__ void reduce_kernel(const float* input, float* output,
                              std::size_t input_width, std::size_t input_height,
                              std::size_t output_width, std::size_t output_height,
                              std::size_t channels, bool horizontal) {
    const std::size_t count = output_width * output_height * channels;
    for (std::size_t p = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         p < count; p += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
        const std::size_t channel = p % channels;
        const std::size_t x = (p / channels) % output_width;
        const std::size_t y = p / channels / output_width;
        float value = 0.0f;
        for (int offset = -2; offset <= 2; ++offset) {
            const auto sx = horizontal ? reflected_index(2 * x, offset, input_width) : x;
            const auto sy = horizontal ? y : reflected_index(2 * y, offset, input_height);
            value += gaussian_coefficient(offset) * input[(sy * input_width + sx) * channels + channel];
        }
        output[p] = value;
    }
}

__global__ void expand_kernel(const float* input, float* output,
                              std::size_t input_width, std::size_t input_height,
                              std::size_t output_width, std::size_t output_height,
                              std::size_t channels, bool horizontal) {
    const std::size_t count = output_width * output_height * channels;
    for (std::size_t p = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         p < count; p += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
        const std::size_t channel = p % channels;
        const std::size_t x = (p / channels) % output_width;
        const std::size_t y = p / channels / output_width;
        const auto coordinate = horizontal ? x : y;
        float value = 0.0f;
        for (int offset = -2; offset <= 2; ++offset) {
            // Only coarse samples where coordinate - offset is even contribute.
            const int delta = static_cast<int>(coordinate % 2) - offset;
            if (delta % 2 != 0) continue;
            const auto sx = horizontal ? reflected_index(x / 2, delta / 2, input_width) : x;
            const auto sy = horizontal ? y : reflected_index(y / 2, delta / 2, input_height);
            value += 2.0f * gaussian_coefficient(offset) *
                     input[(sy * input_width + sx) * channels + channel];
        }
        output[p] = value;
    }
}

__global__ void subtract_kernel(const float* fine, const float* expanded,
                                float* output, std::size_t count) {
    for (std::size_t p = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         p < count; p += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
        output[p] = fine[p] - expanded[p];
    }
}

__global__ void blend_kernel(const float* weights, const float* images, float* output,
                             std::size_t pixels, std::size_t exposures, int* invalid) {
    // Each exposure occupies a contiguous plane; RGB channels are interleaved within it.
    const std::size_t count = pixels * 3;
    for (std::size_t p = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         p < count; p += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
        float value = 0.0f;
        for (std::size_t i = 0; i < exposures; ++i) {
            value += weights[i * pixels + p / 3] * images[i * count + p];
            if (!isfinite(value)) {
                atomicExch(invalid, 1);
                break;
            }
        }
        output[p] = value;
    }
}

__global__ void add_kernel(const float* fine, const float* expanded, float* output,
                           std::size_t count, int* invalid) {
    for (std::size_t p = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         p < count; p += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
        const float value = fine[p] + expanded[p];
        if (!isfinite(value)) atomicExch(invalid, 1);
        output[p] = value;
    }
}

void check_nonfinite(const DeviceBuffer<int>& flag, const char* message) {
    int invalid = 0;
    check_cuda(cudaMemcpy(&invalid, flag.get(), sizeof(int), cudaMemcpyDeviceToHost),
               "copy CUDA non-finite flag");
    if (invalid) throw std::overflow_error(message);
}

void expand_device(const float* input, std::size_t input_width, std::size_t input_height,
                   float* output, std::size_t output_width, std::size_t output_height) {
    const auto horizontal_count = checked_product(checked_product(output_width, input_height), 3);
    const auto count = checked_product(checked_product(output_width, output_height), 3);
    DeviceBuffer<float> horizontal(horizontal_count);
    expand_kernel<<<block_count(horizontal_count), threads_per_block>>>(
        input, horizontal.get(), input_width, input_height, output_width, input_height, 3, true);
    check_kernel("CUDA horizontal expansion kernel");
    expand_kernel<<<block_count(count), threads_per_block>>>(
        horizontal.get(), output, output_width, input_height, output_width, output_height, 3, false);
    check_kernel("CUDA vertical expansion kernel");
    horizontal.release();
}

void reduce_device(const float* input, std::size_t width, std::size_t height,
                   float* output, std::size_t channels) {
    const auto next_width = half_rounded_up(width);
    const auto next_height = half_rounded_up(height);
    const auto horizontal_count = checked_product(checked_product(next_width, height), channels);
    const auto count = checked_product(checked_product(next_width, next_height), channels);
    DeviceBuffer<float> horizontal(horizontal_count);
    reduce_kernel<<<block_count(horizontal_count), threads_per_block>>>(
        input, horizontal.get(), width, height, next_width, height, channels, true);
    check_kernel("CUDA horizontal reduction kernel");
    reduce_kernel<<<block_count(count), threads_per_block>>>(
        horizontal.get(), output, next_width, height, next_width, next_height, channels, false);
    check_kernel("CUDA vertical reduction kernel");
    horizontal.release();
}

struct DeviceLevel {
    std::size_t width;
    std::size_t height;
    std::unique_ptr<DeviceBuffer<float>> pixels;
};

std::vector<DeviceLevel> gaussian_levels(const float* input, std::size_t width,
                                         std::size_t height, std::size_t channels,
                                         std::size_t levels) {
    std::vector<DeviceLevel> result;
    result.reserve(levels);
    auto count = checked_product(checked_product(width, height), channels);
    result.push_back({width, height, std::make_unique<DeviceBuffer<float>>(count)});
    check_cuda(cudaMemcpy(result.back().pixels->get(), input, count * sizeof(float),
                          cudaMemcpyHostToDevice), "copy pyramid input to CUDA");

    while (result.size() < levels) {
        const auto& previous = result.back();
        const auto next_width = half_rounded_up(previous.width);
        const auto next_height = half_rounded_up(previous.height);
        count = checked_product(checked_product(next_width, next_height), channels);
        auto next = std::make_unique<DeviceBuffer<float>>(count);
        reduce_device(previous.pixels->get(), previous.width, previous.height, next->get(), channels);
        result.push_back({next_width, next_height, std::move(next)});
    }
    return result;
}

void release_levels(std::vector<DeviceLevel>& levels) {
    for (auto& level : levels) level.pixels->release();
}

struct DeviceReconstruction {
    // For one level, pixels points into the input pyramid and storage is empty.
    const float* pixels;
    std::unique_ptr<DeviceBuffer<float>> storage;
};

DeviceReconstruction reconstruct_device(const std::vector<DeviceLevel>& blended,
                                        const DeviceBuffer<int>& invalid) {
    const float* current = blended.back().pixels->get();
    std::unique_ptr<DeviceBuffer<float>> reconstructed;
    for (std::size_t level = blended.size() - 1; level > 0; --level) {
        const auto& finer = blended[level - 1];
        const auto& coarse = blended[level];
        const auto count = finer.width * finer.height * 3;
        DeviceBuffer<float> expanded(count);
        auto next = std::make_unique<DeviceBuffer<float>>(count);
        expand_device(current, coarse.width, coarse.height, expanded.get(),
                      finer.width, finer.height);
        add_kernel<<<block_count(count), threads_per_block>>>(
            finer.pixels->get(), expanded.get(), next->get(), count, invalid.get());
        check_kernel("CUDA reconstruction addition kernel");
        check_nonfinite(invalid, "Non-finite value produced during reconstruction");
        expanded.release();
        if (reconstructed) reconstructed->release();
        reconstructed = std::move(next);
        current = reconstructed->get();
    }
    return {current, std::move(reconstructed)};
}

__global__ void grayscale_kernel(const float* images, float* gray, std::size_t count) {
    for (std::size_t p = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         p < count; p += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
        gray[p] = 0.299f * images[3 * p] + 0.587f * images[3 * p + 1] +
                  0.114f * images[3 * p + 2];
    }
}

__global__ void weight_kernel(const float* images, const float* gray, float* weights,
                              std::size_t width, std::size_t height, std::size_t count) {
    const std::size_t pixels = width * height;
    for (std::size_t p = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         p < count; p += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
        const std::size_t pixel = p % pixels;
        const std::size_t base = p - pixel;
        const std::size_t x = pixel % width;
        const std::size_t y = pixel / width;
        const std::size_t left = x == 0 ? (width == 1 ? 0 : 1) : x - 1;
        const std::size_t right = x == width - 1 ? (width == 1 ? 0 : x - 1) : x + 1;
        const std::size_t up = y == 0 ? (height == 1 ? 0 : 1) : y - 1;
        const std::size_t down = y == height - 1 ? (height == 1 ? 0 : y - 1) : y + 1;
        const float contrast = fabsf(gray[base + y * width + left] +
                                     gray[base + y * width + right] +
                                     gray[base + up * width + x] +
                                     gray[base + down * width + x] - 4.0f * gray[p]);
        const float r = images[3 * p];
        const float g = images[3 * p + 1];
        const float b = images[3 * p + 2];
        const float mean = (r + g + b) / 3.0f;
        const float saturation = sqrtf(((r - mean) * (r - mean) +
                                        (g - mean) * (g - mean) +
                                        (b - mean) * (b - mean)) / 3.0f);
        const float exposedness =
            expf(-(r - 0.5f) * (r - 0.5f) / (2.0f * 0.2f * 0.2f)) *
            expf(-(g - 0.5f) * (g - 0.5f) / (2.0f * 0.2f * 0.2f)) *
            expf(-(b - 0.5f) * (b - 0.5f) / (2.0f * 0.2f * 0.2f));
        weights[p] = contrast * saturation * exposedness + 1e-12f;
    }
}

__global__ void normalize_kernel(float* weights, std::size_t pixels,
                                 std::size_t exposures, int* invalid_sum) {
    // One thread owns a pixel across all exposure planes, so normalization is in place.
    for (std::size_t p = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         p < pixels; p += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
        float sum = 0.0f;
        for (std::size_t i = 0; i < exposures; ++i) {
            sum += weights[i * pixels + p];
        }
        if (!isfinite(sum) || sum <= 0.0f) {
            atomicExch(invalid_sum, 1);
            continue;
        }
        for (std::size_t i = 0; i < exposures; ++i) {
            weights[i * pixels + p] /= sum;
        }
    }
}

class CudaBackend final : public Backend {
public:
    std::vector<WeightMap> compute_weights(const std::vector<RgbImage>& images) override {
        validate_images(images);
        const auto width = images.front().width;
        const auto height = images.front().height;
        const auto pixels = width * height;
        const auto count = checked_product(pixels, images.size());
        const auto image_count = checked_product(count, 3);
        DeviceBuffer<float> device_images(image_count);
        DeviceBuffer<float> device_gray(count);
        DeviceBuffer<float> device_weights(count);

        for (std::size_t i = 0; i < images.size(); ++i) {
            check_cuda(cudaMemcpy(device_images.get() + i * pixels * 3, images[i].pixels.data(),
                                  pixels * 3 * sizeof(float), cudaMemcpyHostToDevice),
                       "copy images to CUDA");
        }

        grayscale_kernel<<<block_count(count), threads_per_block>>>(
            device_images.get(), device_gray.get(), count);
        check_kernel("CUDA grayscale kernel");
        weight_kernel<<<block_count(count), threads_per_block>>>(
            device_images.get(), device_gray.get(), device_weights.get(), width, height, count);
        check_kernel("CUDA weight kernel");

        std::vector<WeightMap> result;
        result.reserve(images.size());
        for (std::size_t i = 0; i < images.size(); ++i) {
            result.push_back({width, height, std::vector<float>(pixels)});
            check_cuda(cudaMemcpy(result.back().pixels.data(), device_weights.get() + i * pixels,
                                  pixels * sizeof(float), cudaMemcpyDeviceToHost),
                       "copy weights from CUDA");
        }
        device_weights.release();
        device_gray.release();
        device_images.release();
        return result;
    }

    std::vector<WeightMap> normalize_weights(const std::vector<WeightMap>& weights) override {
        validate_weights(weights);
        const auto width = weights.front().width;
        const auto height = weights.front().height;
        const auto pixels = width * height;
        const auto count = checked_product(pixels, weights.size());
        DeviceBuffer<float> device_weights(count);
        DeviceBuffer<int> device_invalid_sum(1);

        for (std::size_t i = 0; i < weights.size(); ++i) {
            check_cuda(cudaMemcpy(device_weights.get() + i * pixels, weights[i].pixels.data(),
                                  pixels * sizeof(float), cudaMemcpyHostToDevice),
                       "copy weights to CUDA");
        }

        check_cuda(cudaMemset(device_invalid_sum.get(), 0, sizeof(int)), "initialize CUDA sum flag");
        normalize_kernel<<<block_count(pixels), threads_per_block>>>(
            device_weights.get(), pixels, weights.size(), device_invalid_sum.get());
        check_kernel("CUDA normalization kernel");
        int invalid_sum = 0;
        check_cuda(cudaMemcpy(&invalid_sum, device_invalid_sum.get(), sizeof(int),
                              cudaMemcpyDeviceToHost), "copy CUDA sum flag");
        if (invalid_sum) {
            throw std::invalid_argument("Weight sum must be finite and positive at every pixel");
        }

        std::vector<WeightMap> result;
        result.reserve(weights.size());
        for (std::size_t i = 0; i < weights.size(); ++i) {
            result.push_back({width, height, std::vector<float>(pixels)});
            check_cuda(cudaMemcpy(result.back().pixels.data(), device_weights.get() + i * pixels,
                                  pixels * sizeof(float), cudaMemcpyDeviceToHost),
                       "copy normalized weights from CUDA");
        }
        device_invalid_sum.release();
        device_weights.release();
        return result;
    }

    WeightPyramid build_weight_pyramid(const WeightMap& weights, std::size_t levels) override {
        if (weights.width == 0 || weights.height == 0 ||
            weights.width > std::numeric_limits<std::size_t>::max() / weights.height) {
            throw std::invalid_argument("Invalid weight map dimensions");
        }
        if (weights.pixels.size() != weights.width * weights.height) {
            throw std::invalid_argument("Weight map must contain one value per pixel");
        }
        for (const float value : weights.pixels) {
            if (!std::isfinite(value) || value < 0.0f || value > 1.0f) {
                throw std::invalid_argument("Normalized weights must be finite and in [0, 1]");
            }
        }
        if (levels == 0 || levels > max_pyramid_levels(weights.width, weights.height)) {
            throw std::invalid_argument("Invalid weight pyramid level count");
        }

        auto device_levels = gaussian_levels(weights.pixels.data(), weights.width,
                                             weights.height, 1, levels);
        WeightPyramid result;
        result.reserve(levels);
        for (const auto& level : device_levels) {
            const auto count = level.width * level.height;
            result.push_back({level.width, level.height, std::vector<float>(count)});
            check_cuda(cudaMemcpy(result.back().pixels.data(), level.pixels->get(),
                                  count * sizeof(float), cudaMemcpyDeviceToHost),
                       "copy CUDA weight pyramid level");
        }

        release_levels(device_levels);
        return result;
    }

    ImagePyramid build_laplacian_pyramid(const RgbImage& image, std::size_t levels) override {
        validate_image(image);
        if (levels == 0 || levels > max_pyramid_levels(image.width, image.height)) {
            throw std::invalid_argument("Invalid image pyramid level count");
        }
        auto gaussian = gaussian_levels(image.pixels.data(), image.width, image.height, 3, levels);
        ImagePyramid result;
        result.reserve(levels);

        for (std::size_t level = 0; level + 1 < levels; ++level) {
            const auto& fine = gaussian[level];
            const auto& coarse = gaussian[level + 1];
            const auto count = checked_product(checked_product(fine.width, fine.height), 3);
            DeviceBuffer<float> expanded(count);
            DeviceBuffer<float> laplacian(count);
            expand_device(coarse.pixels->get(), coarse.width, coarse.height,
                          expanded.get(), fine.width, fine.height);
            subtract_kernel<<<block_count(count), threads_per_block>>>(
                fine.pixels->get(), expanded.get(), laplacian.get(), count);
            check_kernel("CUDA Laplacian subtraction kernel");
            result.push_back({fine.width, fine.height, std::vector<float>(count)});
            check_cuda(cudaMemcpy(result.back().pixels.data(), laplacian.get(),
                                  count * sizeof(float), cudaMemcpyDeviceToHost),
                       "copy CUDA Laplacian pyramid level");
            laplacian.release();
            expanded.release();
        }

        const auto& coarsest = gaussian.back();
        const auto count = coarsest.width * coarsest.height * 3;
        result.push_back({coarsest.width, coarsest.height, std::vector<float>(count)});
        check_cuda(cudaMemcpy(result.back().pixels.data(), coarsest.pixels->get(),
                              count * sizeof(float), cudaMemcpyDeviceToHost),
                   "copy CUDA coarsest Gaussian level");
        release_levels(gaussian);
        return result;
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
                const auto pixels = level_width * level_height;
                if (image.width != level_width || image.height != level_height ||
                    weight.width != level_width || weight.height != level_height) {
                    throw std::invalid_argument("Pyramid level dimensions do not match");
                }
                if (image.pixels.size() != pixels * 3 || weight.pixels.size() != pixels) {
                    throw std::invalid_argument("Invalid pyramid level pixel count");
                }
                for (const float value : image.pixels) {
                    if (!std::isfinite(value)) {
                        throw std::invalid_argument("Laplacian values must be finite");
                    }
                }
                for (const float value : weight.pixels) {
                    if (!std::isfinite(value) || value < 0.0f) {
                        throw std::invalid_argument("Pyramid weights must be finite and nonnegative");
                    }
                }
                level_width = half_rounded_up(level_width);
                level_height = half_rounded_up(level_height);
            }
        }

        std::vector<DeviceLevel> device_weights;
        std::vector<DeviceLevel> device_images;
        std::vector<DeviceLevel> device_blended;
        device_weights.reserve(levels);
        device_images.reserve(levels);
        device_blended.reserve(levels);

        for (std::size_t level = 0; level < levels; ++level) {
            const auto level_width = images.front()[level].width;
            const auto level_height = images.front()[level].height;
            const auto pixels = level_width * level_height;
            const auto count = pixels * 3;
            device_weights.push_back({level_width, level_height,
                std::make_unique<DeviceBuffer<float>>(checked_product(pixels, images.size()))});
            device_images.push_back({level_width, level_height,
                std::make_unique<DeviceBuffer<float>>(checked_product(count, images.size()))});
            device_blended.push_back({level_width, level_height,
                std::make_unique<DeviceBuffer<float>>(count)});
            for (std::size_t i = 0; i < images.size(); ++i) {
                check_cuda(cudaMemcpy(device_weights.back().pixels->get() + i * pixels,
                                      weights[i][level].pixels.data(), pixels * sizeof(float),
                                      cudaMemcpyHostToDevice), "copy CUDA blend weights");
                check_cuda(cudaMemcpy(device_images.back().pixels->get() + i * count,
                                      images[i][level].pixels.data(), count * sizeof(float),
                                      cudaMemcpyHostToDevice), "copy CUDA blend images");
            }
        }

        DeviceBuffer<int> invalid(1);
        check_cuda(cudaMemset(invalid.get(), 0, sizeof(int)), "initialize CUDA non-finite flag");
        for (std::size_t level = 0; level < levels; ++level) {
            const auto pixels = device_images[level].width * device_images[level].height;
            const auto count = pixels * 3;
            blend_kernel<<<block_count(count), threads_per_block>>>(
                device_weights[level].pixels->get(), device_images[level].pixels->get(),
                device_blended[level].pixels->get(), pixels, images.size(), invalid.get());
            check_kernel("CUDA blending kernel");
            check_nonfinite(invalid, "Non-finite value produced while blending");
        }

        ImagePyramid result;
        result.reserve(levels);
        for (const auto& level : device_blended) {
            const auto count = level.width * level.height * 3;
            result.push_back({level.width, level.height, std::vector<float>(count)});
            check_cuda(cudaMemcpy(result.back().pixels.data(), level.pixels->get(),
                                  count * sizeof(float), cudaMemcpyDeviceToHost),
                       "copy CUDA blended level");
        }
        invalid.release();
        release_levels(device_blended);
        release_levels(device_images);
        release_levels(device_weights);
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
            for (const float value : level.pixels) {
                if (!std::isfinite(value)) {
                    throw std::invalid_argument("Pyramid values must be finite");
                }
            }
            width = half_rounded_up(width);
            height = half_rounded_up(height);
        }

        std::vector<DeviceLevel> device_levels;
        device_levels.reserve(blended.size());
        for (const auto& level : blended) {
            const auto count = level.width * level.height * 3;
            device_levels.push_back({level.width, level.height,
                                     std::make_unique<DeviceBuffer<float>>(count)});
            check_cuda(cudaMemcpy(device_levels.back().pixels->get(), level.pixels.data(),
                                  count * sizeof(float), cudaMemcpyHostToDevice),
                       "copy CUDA reconstruction level");
        }

        DeviceBuffer<int> invalid(1);
        check_cuda(cudaMemset(invalid.get(), 0, sizeof(int)), "initialize CUDA non-finite flag");
        auto reconstructed = reconstruct_device(device_levels, invalid);
        const auto& finest = blended.front();
        const auto count = finest.width * finest.height * 3;
        RgbImage result{finest.width, finest.height, std::vector<float>(count)};
        check_cuda(cudaMemcpy(result.pixels.data(), reconstructed.pixels, count * sizeof(float),
                              cudaMemcpyDeviceToHost), "copy CUDA reconstruction");
        if (reconstructed.storage) reconstructed.storage->release();
        invalid.release();
        release_levels(device_levels);
        return result;
    }

    RgbImage fuse(const std::vector<RgbImage>& images, const FusionOptions& options) override {
        return fuse_impl(images, options, nullptr);
    }

    RgbImage fuse_profiled(const std::vector<RgbImage>& images, const FusionOptions& options,
                           detail::CudaStageTimings& timings) {
        return fuse_impl(images, options, &timings);
    }

private:
    RgbImage fuse_impl(const std::vector<RgbImage>& images, const FusionOptions& options,
                       detail::CudaStageTimings* timings) {
        AllocationScope allocation_scope(timings);
        validate_images(images);
        const auto width = images.front().width;
        const auto height = images.front().height;
        const auto levels = options.levels.value_or(max_pyramid_levels(width, height));
        if (levels == 0 || levels > max_pyramid_levels(width, height)) {
            throw std::invalid_argument("Invalid fusion pyramid level count");
        }
        const auto exposures = images.size();
        const auto pixels = width * height;
        const auto weight_count = checked_product(pixels, exposures);
        const auto image_count = checked_product(weight_count, 3);
        std::unique_ptr<StageEvents> events;
        if (timings) events = std::make_unique<StageEvents>();

        // All pyramid levels remain device-resident until reconstruction finishes.
        // Per-level buffers store exposure planes consecutively, with interleaved RGB.
        std::vector<DeviceLevel> gaussian_images, gaussian_weights, laplacian, blended;
        gaussian_images.reserve(levels);
        gaussian_weights.reserve(levels);
        laplacian.reserve(levels);
        blended.reserve(levels);
        gaussian_images.push_back({width, height, std::make_unique<DeviceBuffer<float>>(image_count)});
        gaussian_weights.push_back({width, height, std::make_unique<DeviceBuffer<float>>(weight_count)});
        DeviceBuffer<float> gray(weight_count);
        DeviceBuffer<int> invalid(1);

        for (std::size_t i = 0; i < exposures; ++i) {
            check_cuda(cudaMemcpy(gaussian_images[0].pixels->get() + i * pixels * 3,
                                  images[i].pixels.data(), pixels * 3 * sizeof(float),
                                  cudaMemcpyHostToDevice), "copy images to CUDA");
        }

        if (events) events->record(0);
        grayscale_kernel<<<block_count(weight_count), threads_per_block>>>(
            gaussian_images[0].pixels->get(), gray.get(), weight_count);
        check_kernel("CUDA grayscale kernel");

        weight_kernel<<<block_count(weight_count), threads_per_block>>>(
            gaussian_images[0].pixels->get(), gray.get(), gaussian_weights[0].pixels->get(),
            width, height, weight_count);
        check_kernel("CUDA weight kernel");

        if (events) events->record(1);
        check_cuda(cudaMemset(invalid.get(), 0, sizeof(int)), "initialize CUDA sum flag");

        normalize_kernel<<<block_count(pixels), threads_per_block>>>(
            gaussian_weights[0].pixels->get(), pixels, exposures, invalid.get());
        check_kernel("CUDA normalization kernel");
        
        int invalid_sum = 0;
        check_cuda(cudaMemcpy(&invalid_sum, invalid.get(), sizeof(int), cudaMemcpyDeviceToHost),
                   "copy CUDA sum flag");
        if (invalid_sum) {
            throw std::invalid_argument("Weight sum must be finite and positive at every pixel");
        }

        if (events) events->record(2);
        // Reduce each exposure independently using the same dimensions at every level.
        for (std::size_t level = 1; level < levels; ++level) {
            const auto& previous_image = gaussian_images.back();
            const auto& previous_weight = gaussian_weights.back();
            const auto next_width = half_rounded_up(previous_image.width);
            const auto next_height = half_rounded_up(previous_image.height);
            const auto previous_pixels = previous_image.width * previous_image.height;
            const auto next_pixels = next_width * next_height;
            auto next_image = std::make_unique<DeviceBuffer<float>>(
                checked_product(checked_product(next_pixels, exposures), 3));
            auto next_weight = std::make_unique<DeviceBuffer<float>>(
                checked_product(next_pixels, exposures));
            for (std::size_t i = 0; i < exposures; ++i) {
                reduce_device(previous_image.pixels->get() + i * previous_pixels * 3,
                              previous_image.width, previous_image.height,
                              next_image->get() + i * next_pixels * 3, 3);
                reduce_device(previous_weight.pixels->get() + i * previous_pixels,
                              previous_weight.width, previous_weight.height,
                              next_weight->get() + i * next_pixels, 1);
            }
            gaussian_images.push_back({next_width, next_height, std::move(next_image)});
            gaussian_weights.push_back({next_width, next_height, std::move(next_weight)});
        }

        if (events) events->record(3);
        // The coarsest Laplacian level is the Gaussian level; finer levels subtract expansion.
        for (std::size_t level = 0; level < levels; ++level) {
            const auto& fine = gaussian_images[level];
            const auto level_pixels = fine.width * fine.height;
            const auto level_count = level_pixels * 3;
            const auto total_count = checked_product(level_count, exposures);
            auto current = std::make_unique<DeviceBuffer<float>>(total_count);
            if (level + 1 == levels) {
                check_cuda(cudaMemcpy(current->get(), fine.pixels->get(), total_count * sizeof(float),
                                      cudaMemcpyDeviceToDevice), "copy CUDA coarsest image level");
            } else {
                const auto& coarse = gaussian_images[level + 1];
                const auto coarse_count = coarse.width * coarse.height * 3;
                DeviceBuffer<float> expanded(level_count);
                for (std::size_t i = 0; i < exposures; ++i) {
                    expand_device(coarse.pixels->get() + i * coarse_count, coarse.width, coarse.height,
                                  expanded.get(), fine.width, fine.height);
                    subtract_kernel<<<block_count(level_count), threads_per_block>>>(
                        fine.pixels->get() + i * level_count, expanded.get(),
                        current->get() + i * level_count, level_count);
                    check_kernel("CUDA Laplacian subtraction kernel");
                }
                expanded.release();
            }
            laplacian.push_back({fine.width, fine.height, std::move(current)});
        }

        if (events) events->record(4);
        check_cuda(cudaMemset(invalid.get(), 0, sizeof(int)), "initialize CUDA non-finite flag");
        for (std::size_t level = 0; level < levels; ++level) {
            const auto& image = laplacian[level];
            const auto level_pixels = image.width * image.height;
            const auto count = level_pixels * 3;
            auto output = std::make_unique<DeviceBuffer<float>>(count);
            blend_kernel<<<block_count(count), threads_per_block>>>(
                gaussian_weights[level].pixels->get(), image.pixels->get(), output->get(),
                level_pixels, exposures, invalid.get());
            check_kernel("CUDA blending kernel");
            check_nonfinite(invalid, "Non-finite value produced while blending");
            blended.push_back({image.width, image.height, std::move(output)});
        }

        if (events) events->record(5);
        auto reconstructed = reconstruct_device(blended, invalid);
        if (events) {
            events->record(6);
            events->finish(*timings);
        }
        RgbImage result{width, height, std::vector<float>(pixels * 3)};
        check_cuda(cudaMemcpy(result.pixels.data(), reconstructed.pixels,
                              pixels * 3 * sizeof(float), cudaMemcpyDeviceToHost),
                   "copy CUDA fused image");
        if (reconstructed.storage) reconstructed.storage->release();
        release_levels(blended);
        release_levels(laplacian);
        release_levels(gaussian_weights);
        release_levels(gaussian_images);
        invalid.release();
        gray.release();
        return result;
    }
};

}  // namespace

std::unique_ptr<Backend> make_cuda_backend() {
    check_cuda_device();
    return std::make_unique<CudaBackend>();
}

RgbImage detail::profile_cuda_fuse(const std::vector<RgbImage>& images,
                                   const FusionOptions& options, CudaStageTimings& timings) {
    check_cuda_device();
    CudaBackend backend;
    return backend.fuse_profiled(images, options, timings);
}

}  // namespace fusion
