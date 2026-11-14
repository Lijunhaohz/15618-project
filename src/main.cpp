#include "fusion/backend.hpp"
#include "fusion/image_io.hpp"

#include <algorithm>
#include <charconv>
#include <cctype>
#include <filesystem>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

namespace {

void print_help() {
    std::cout << "Usage: exposure_fusion --output FILE [--levels N] [--] INPUT...\n"
              << "Fuse aligned 8-bit RGB PNG or P6 PPM images using the CPU.\n"
              << "Inputs may be image files, directories, or a mixture of both.\n"
              << "  -o, --output FILE  Output PNG or PPM path (required)\n"
              << "  --levels N         Pyramid levels; defaults to at most five\n"
              << "  --                  Treat remaining arguments as input paths\n"
              << "  -h, --help         Show this help\n";
}

struct Arguments {
    std::string output;
    std::vector<std::string> inputs;
    fusion::FusionOptions options;
};

std::string_view option_value(int argc, char* argv[], int& index, std::string_view option) {
    if (index + 1 >= argc) {
        throw std::invalid_argument("Missing value for " + std::string(option));
    }
    return argv[++index];
}

Arguments parse_arguments(int argc, char* argv[]) {
    Arguments args;
    bool positional_only = false;
    bool output_set = false;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg = argv[i];
        if (!positional_only && arg == "--") {
            positional_only = true;
        } else if (!positional_only && (arg == "--output" || arg == "-o")) {
            if (output_set) throw std::invalid_argument("Output path specified more than once");
            args.output = option_value(argc, argv, i, arg);
            output_set = true;
        } else if (!positional_only && arg == "--levels") {
            if (args.options.levels) throw std::invalid_argument("Levels specified more than once");
            const auto value = option_value(argc, argv, i, arg);
            std::size_t levels = 0;
            const auto result = std::from_chars(value.data(), value.data() + value.size(), levels);
            if (value.empty() || result.ec != std::errc{} ||
                result.ptr != value.data() + value.size() || levels == 0) {
                throw std::invalid_argument("Levels must be a positive integer");
            }
            args.options.levels = levels;
        } else if (!positional_only && !arg.empty() && arg.front() == '-') {
            throw std::invalid_argument("Unknown option: " + std::string(arg));
        } else {
            if (arg.empty()) throw std::invalid_argument("Input path must not be empty");
            args.inputs.emplace_back(arg);
        }
    }
    if (!output_set || args.output.empty()) throw std::invalid_argument("Output path is required");
    if (args.inputs.empty()) throw std::invalid_argument("At least one input image is required");
    return args;
}

bool is_metadata_file(const std::filesystem::path& path) {
    auto ext = path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return ext == ".md" || ext == ".txt" || ext == ".json" || ext == ".csv" ||
           path.filename() == ".DS_Store";
}

std::vector<std::string> expand_inputs(const std::vector<std::string>& paths) {
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

void check_output_path(const Arguments& args) {
    const auto output = std::filesystem::absolute(args.output).lexically_normal();
    const bool output_exists = std::filesystem::exists(output);
    for (const auto& input_path : args.inputs) {
        const auto input = std::filesystem::absolute(input_path).lexically_normal();
        if (output == input ||
            (output_exists && std::filesystem::exists(input) && std::filesystem::equivalent(output, input))) {
            throw std::invalid_argument("Output path must not refer to an input image: " + input_path);
        }
    }
}

}  // namespace

int main(int argc, char* argv[]) {
    if (argc == 2 && (std::string_view(argv[1]) == "--help" || std::string_view(argv[1]) == "-h")) {
        print_help();
        return 0;
    }

    try {
        auto args = parse_arguments(argc, argv);
        args.inputs = expand_inputs(args.inputs);
        check_output_path(args);
        std::vector<fusion::RgbImage> images;
        images.reserve(args.inputs.size());
        for (const auto& input : args.inputs) {
            images.push_back(fusion::load_image(input));
        }
        auto backend = fusion::make_cpu_backend();
        const auto fused = backend->fuse(images, args.options);
        fusion::save_image(args.output, fused);
        std::cout << "Saved fused image to " << args.output << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
