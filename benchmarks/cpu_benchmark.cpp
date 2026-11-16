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
        std::cerr << "Usage: cpu_benchmark INPUT [INPUT ...] (image files or directories)\n";
        return 1;
    }
    try {
        const std::vector<std::string> paths(argv + 1, argv + argc);
        const auto inputs = fusion::detail::expand_inputs(paths);
        std::vector<fusion::RgbImage> images;
        images.reserve(inputs.size());
        for (const auto& input : inputs) images.push_back(fusion::load_image(input));
        auto cpu = fusion::make_cpu_backend();

        for (int i = 0; i < benchmark::warmups; ++i) cpu->fuse(images, {});

        std::vector<double> wall;
        for (int i = 0; i < benchmark::iterations; ++i) {
            const auto start = std::chrono::steady_clock::now();
            auto result = cpu->fuse(images, {});
            const auto end = std::chrono::steady_clock::now();
            wall.push_back(std::chrono::duration<double, std::milli>(end - start).count());
        }

        std::cout << images.front().width << 'x' << images.front().height << ", "
                  << images.size() << " exposures, "
                  << benchmark::levels_for(images.front().width, images.front().height)
                  << " levels; " << benchmark::warmups << " warmups, "
                  << benchmark::iterations << " trials\n";
        benchmark::report("CPU fuse (wall)", wall);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
