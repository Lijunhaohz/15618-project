#pragma once

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

namespace fusion::detail {

inline bool is_metadata_file(const std::filesystem::path& path) {
    auto ext = path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return ext == ".md" || ext == ".txt" || ext == ".json" || ext == ".csv" ||
           path.filename() == ".DS_Store";
}

inline std::vector<std::string> expand_inputs(const std::vector<std::string>& paths) {
    std::vector<std::string> inputs;
    for (const auto& path : paths) {
        if (!std::filesystem::is_directory(path)) {
            inputs.push_back(path);
            continue;
        }
        std::vector<std::string> directory_inputs;
        for (const auto& entry : std::filesystem::directory_iterator(path)) {
            if (entry.is_regular_file() && !is_metadata_file(entry.path())) {
                directory_inputs.push_back(entry.path().string());
            }
        }
        if (directory_inputs.empty()) {
            throw std::invalid_argument("Input directory contains no image files: " + path);
        }
        std::sort(directory_inputs.begin(), directory_inputs.end());
        inputs.insert(inputs.end(), directory_inputs.begin(), directory_inputs.end());
    }
    return inputs;
}

}  // namespace fusion::detail
