#include "fusion/image_io.hpp"
#include "png_io.h"

#include <algorithm>
#include <charconv>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace fusion {
namespace {

std::string extension(const std::string& path) {
    auto ext = std::filesystem::path(path).extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    if (ext != ".png" && ext != ".ppm") {
        throw std::invalid_argument("Unsupported image format: " + path);
    }
    return ext;
}

std::size_t sample_count(std::size_t width, std::size_t height) {
    const auto max = std::numeric_limits<std::size_t>::max();
    if (width == 0 || height == 0 || width > max / height || width * height > max / 3) {
        throw std::invalid_argument("Invalid image dimensions");
    }
    return width * height * 3;
}

std::string read_ppm_token(std::istream& input) {
    while (true) {
        const int ch = input.peek();
        if (ch == std::char_traits<char>::eof()) {
            throw std::runtime_error("Unexpected end of PPM header");
        }
        if (std::isspace(static_cast<unsigned char>(ch))) {
            input.get();
        } else if (ch == '#') {
            input.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        } else {
            break;
        }
    }
    std::string token;
    while (true) {
        const int ch = input.peek();
        if (ch == std::char_traits<char>::eof() || ch == '#' ||
            std::isspace(static_cast<unsigned char>(ch))) break;
        if (token.size() == 64) throw std::runtime_error("PPM header token is too long");
        token.push_back(static_cast<char>(input.get()));
    }
    return token;
}

std::size_t parse_ppm_number(const std::string& token) {
    std::size_t value = 0;
    const auto result = std::from_chars(token.data(), token.data() + token.size(), value);
    if (token.empty() || result.ec != std::errc{} || result.ptr != token.data() + token.size()) {
        throw std::runtime_error("Invalid PPM header number");
    }
    return value;
}

RgbImage load_ppm(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("Cannot open PPM for reading: " + path);
    if (read_ppm_token(input) != "P6") throw std::runtime_error("Only binary P6 PPM is supported: " + path);
    const auto width = parse_ppm_number(read_ppm_token(input));
    const auto height = parse_ppm_number(read_ppm_token(input));
    if (parse_ppm_number(read_ppm_token(input)) != 255) {
        throw std::runtime_error("PPM maximum sample value must be 255: " + path);
    }
    const int separator = input.get();
    if (separator == std::char_traits<char>::eof() ||
        !std::isspace(static_cast<unsigned char>(separator))) {
        throw std::runtime_error("Missing PPM pixel data separator: " + path);
    }
    const auto count = sample_count(width, height);
    if (count > static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max())) {
        throw std::runtime_error("PPM image is too large: " + path);
    }
    std::vector<unsigned char> bytes(count);
    input.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(count));
    if (!input) throw std::runtime_error("Cannot read PPM pixel data: " + path);
    RgbImage image{width, height, std::vector<float>(count)};
    for (std::size_t i = 0; i < count; ++i) image.pixels[i] = bytes[i] / 255.0f;
    return image;
}

RgbImage load_png(const std::string& path) {
    unsigned char* raw = nullptr;
    std::size_t width = 0;
    std::size_t height = 0;
    char error[256] = {};
    if (!fusion_read_png(path.c_str(), &raw, &width, &height, error, sizeof(error))) {
        throw std::runtime_error("Cannot read PNG " + path + ": " + error);
    }
    std::unique_ptr<unsigned char, decltype(&std::free)> bytes(raw, &std::free);
    const auto count = sample_count(width, height);
    RgbImage image{width, height, std::vector<float>(count)};
    for (std::size_t i = 0; i < count; ++i) image.pixels[i] = bytes.get()[i] / 255.0f;
    return image;
}

std::vector<unsigned char> encode(const RgbImage& image) {
    const auto count = sample_count(image.width, image.height);
    if (image.pixels.size() != count) throw std::invalid_argument("Image must have three values per pixel");
    std::vector<unsigned char> bytes(count);
    for (std::size_t i = 0; i < count; ++i) {
        if (!std::isfinite(image.pixels[i])) {
            throw std::invalid_argument("Image output values must be finite");
        }
        bytes[i] = static_cast<unsigned char>(std::lround(std::clamp(image.pixels[i], 0.0f, 1.0f) * 255.0f));
    }
    return bytes;
}

void save_ppm(const std::string& path, const RgbImage& image,
              const std::vector<unsigned char>& bytes) {
    if (bytes.size() > static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max())) {
        throw std::runtime_error("PPM image is too large: " + path);
    }
    std::ofstream output(path, std::ios::binary);
    if (!output) throw std::runtime_error("Cannot open PPM for writing: " + path);
    output << "P6\n" << image.width << ' ' << image.height << "\n255\n";
    output.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    output.close();
    if (!output) throw std::runtime_error("Cannot write PPM: " + path);
}

}  // namespace

RgbImage load_image(const std::string& path) {
    return extension(path) == ".png" ? load_png(path) : load_ppm(path);
}

void save_image(const std::string& path, const RgbImage& image) {
    const auto ext = extension(path);
    const auto bytes = encode(image);
    if (ext == ".ppm") {
        save_ppm(path, image, bytes);
    } else {
        char error[256] = {};
        if (!fusion_write_png(path.c_str(), bytes.data(), image.width, image.height,
                              error, sizeof(error))) {
            throw std::runtime_error("Cannot write PNG " + path + ": " + error);
        }
    }
}

}  // namespace fusion
