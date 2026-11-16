#pragma once

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <vector>

namespace benchmark {

constexpr int warmups = 2;
constexpr int iterations = 10;

inline double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const auto middle = values.size() / 2;
    return (values[middle - 1] + values[middle]) / 2.0;
}

inline void report(const char* label, const std::vector<double>& values) {
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
