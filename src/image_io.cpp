/**
 * Implements functions for loading and saving RGB images.
 */
#include "fusion/image_io.hpp"

#include <stdexcept>

namespace fusion {

RgbImage load_image(const std::string&) {
    throw std::logic_error("Image loading is not implemented");
}

void save_image(const std::string&, const RgbImage&) {
    throw std::logic_error("Image saving is not implemented");
}

}  // namespace fusion
