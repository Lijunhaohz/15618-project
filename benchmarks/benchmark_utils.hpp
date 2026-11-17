#pragma once

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>
#include <charconv>
#include "fusion/backend.hpp"

namespace benchmark {

struct Arguments {
    int warmups = 2;
    int iterations = 10;
    fusion::FusionOptions options;
    std::vector<std::string> paths;
};

inline Arguments parse_arguments(int argc, char* argv[]) {
    Arguments result;
    bool positional = false;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (!positional && arg == "--") { positional = true; continue; }
        if (!positional && (arg == "--levels" || arg == "--warmups" || arg == "--trials")) {
            if (++i == argc) throw std::invalid_argument("Missing value for " + arg);
            const std::string value = argv[i];
            int number = 0;
            const auto parsed = std::from_chars(value.data(), value.data() + value.size(), number);
            if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size() ||
                number < (arg == "--warmups" ? 0 : 1))
                throw std::invalid_argument("Invalid value for " + arg);
            if (arg == "--levels") result.options.levels = static_cast<std::size_t>(number);
            else if (arg == "--warmups") result.warmups = number;
            else result.iterations = number;
        } else {
            if (!positional && !arg.empty() && arg[0] == '-')
                throw std::invalid_argument("Unknown option: " + arg);
            result.paths.push_back(arg);
        }
    }
    if (result.paths.empty()) throw std::invalid_argument("At least one input is required");
    return result;
}

inline double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const auto middle = values.size() / 2;
    return values.size() % 2 ? values[middle] : (values[middle - 1] + values[middle]) / 2.0;
}

inline void report(const char* label, const std::vector<double>& values) {
    for (std::size_t i = 0; i < values.size(); ++i)
        std::cout << "Sample: " << label << " trial=" << i << " ms="
                  << std::scientific << std::setprecision(9) << values[i] << '\n';
    const auto bounds = std::minmax_element(values.begin(), values.end());
    std::cout << std::left << std::setw(24) << label << std::right
              << std::fixed << std::setprecision(3) << " median=" << std::setw(9)
              << median(values) << " ms  min=" << std::setw(9) << *bounds.first
              << " ms  max=" << std::setw(9) << *bounds.second << " ms\n";
}

inline std::size_t levels_for(std::size_t width, std::size_t height) {
    std::size_t levels = 1;
    while (levels < 5 && (width > 1 || height > 1)) {
        width = width / 2 + width % 2;
        height = height / 2 + height % 2;
        ++levels;
    }
    return levels;
}

}  // namespace benchmark
