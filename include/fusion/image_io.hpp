/**
 * Declares functions for loading and saving RGB images.
 */
 
#pragma once

#include "fusion/image.hpp"

#include <string>

namespace fusion {

// Supports 8-bit RGB PNG and binary P6 PPM files.
RgbImage load_image(const std::string& path);
void save_image(const std::string& path, const RgbImage& image);

}  // namespace fusion
