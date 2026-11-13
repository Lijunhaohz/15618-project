/**
 * Defines structures for RGB images and weight maps, as well as image pyramids.
 */

#pragma once

#include <cstddef>
#include <vector>

namespace fusion {

/**
 * Represents an RGB image with interleaved RGB pixel values.
 */
struct RgbImage {
    std::size_t width = 0;
    std::size_t height = 0;
    // Interleaved RGB values in row-major order ([R, G, B, R, G, B, ...])
    std::vector<float> pixels;
};

/** 
 * Represents a weight map for an image, typically used in image fusion.
 * Typically, each value corresponds to the weight of the corresponding pixel to the 
 * final fused image.
 */
struct WeightMap {
    std::size_t width = 0;
    std::size_t height = 0;
    // Weight values in row-major order
    std::vector<float> pixels;
};

using ImagePyramid = std::vector<RgbImage>;
using WeightPyramid = std::vector<WeightMap>;

}  // namespace fusion
