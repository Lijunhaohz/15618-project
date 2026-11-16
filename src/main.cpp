#include "fusion/backend.hpp"
#include "fusion/image_io.hpp"
#include "input_paths.hpp"

#include <charconv>
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
    std::cout << "Usage: exposure_fusion --output FILE [--backend cpu|cuda] [--levels N] [--] INPUT...\n"
              << "Fuse aligned 8-bit RGB PNG or P6 PPM images.\n"
              << "Inputs may be image files, directories, or a mixture of both.\n"
              << "  -o, --output FILE  Output PNG or PPM path (required)\n"
              << "  --backend NAME     cpu (default) or cuda\n"
              << "  --levels N         Pyramid levels; defaults to at most five\n"
              << "  --                  Treat remaining arguments as input paths\n"
              << "  -h, --help         Show this help\n";
}

struct Arguments {
    enum class Backend { cpu, cuda };
    std::string output;
    std::vector<std::string> inputs;
    fusion::FusionOptions options;
    Backend backend = Backend::cpu;
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
    bool backend_set = false;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg = argv[i];
        if (!positional_only && arg == "--") {
            positional_only = true;
        } else if (!positional_only && (arg == "--output" || arg == "-o")) {
            if (output_set) throw std::invalid_argument("Output path specified more than once");
            args.output = option_value(argc, argv, i, arg);
            output_set = true;
        } else if (!positional_only && arg == "--backend") {
            if (backend_set) throw std::invalid_argument("Backend specified more than once");
            const auto value = option_value(argc, argv, i, arg);
            if (value == "cpu") {
                args.backend = Arguments::Backend::cpu;
            } else if (value == "cuda") {
                args.backend = Arguments::Backend::cuda;
            } else {
                throw std::invalid_argument("Backend must be cpu or cuda");
            }
            backend_set = true;
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
        args.inputs = fusion::detail::expand_inputs(args.inputs);
        check_output_path(args);
        auto backend = args.backend == Arguments::Backend::cuda
            ? fusion::make_cuda_backend() : fusion::make_cpu_backend();
        std::vector<fusion::RgbImage> images;
        images.reserve(args.inputs.size());
        for (const auto& input : args.inputs) {
            images.push_back(fusion::load_image(input));
        }
        const auto fused = backend->fuse(images, args.options);
        fusion::save_image(args.output, fused);
        std::cout << "Saved fused image to " << args.output << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
