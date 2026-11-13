/**
 * Declares functions for loading and saving RGB images.
 */
 
#pragma once

#include "fusion/image.hpp"

#include <string>

namespace fusion {

RgbImage load_image(const std::string& path);
void save_image(const std::string& path, const RgbImage& image);

}  // namespace fusion
