#include "benchmark_utils.hpp"
#include "fusion/backend.hpp"
#include "fusion/image_io.hpp"
#include "input_paths.hpp"

#include <chrono>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: cpu_benchmark [--levels N] [--warmups N] [--trials N] [--] INPUT...\n";
        return 1;
    }
    try {
        const auto args = benchmark::parse_arguments(argc, argv);
        const auto inputs = fusion::detail::expand_inputs(args.paths);
        std::vector<fusion::RgbImage> images;
        images.reserve(inputs.size());
        for (const auto& input : inputs) images.push_back(fusion::load_image(input));
        auto cpu = fusion::make_cpu_backend();

        for (int i = 0; i < args.warmups; ++i) cpu->fuse(images, args.options);

        std::vector<double> wall;
        for (int i = 0; i < args.iterations; ++i) {
            const auto start = std::chrono::steady_clock::now();
            auto result = cpu->fuse(images, args.options);
            const auto end = std::chrono::steady_clock::now();
            wall.push_back(std::chrono::duration<double, std::milli>(end - start).count());
        }

        std::cout << images.front().width << 'x' << images.front().height << ", "
                  << images.size() << " exposures, "
                  << args.options.levels.value_or(benchmark::levels_for(images.front().width, images.front().height))
                  << " levels; " << args.warmups << " warmups, "
                  << args.iterations << " trials\n";
        benchmark::report("CPU fuse (wall)", wall);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
