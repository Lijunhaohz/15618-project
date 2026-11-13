#include "fusion/backend.hpp"
#include "fusion/image_io.hpp"

#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

template <typename Function>
bool expect_failure(const char* name, const char* message, Function function) {
    try {
        function();
        std::cerr << name << " unexpectedly succeeded\n";
    } catch (const std::exception& error) {
        if (std::string(error.what()).find(message) != std::string::npos) {
            return true;
        }
        std::cerr << name << " produced an unexpected error: " << error.what() << '\n';
    }
    return false;
}

bool check_backend(fusion::Backend& backend, const char* name) {
    bool ok = true;
    ok &= expect_failure(name, "not implemented", [&] { backend.compute_weights({}); });
    ok &= expect_failure(name, "not implemented", [&] { backend.normalize_weights({}); });
    ok &= expect_failure(name, "not implemented", [&] { backend.build_weight_pyramid({}, 1); });
    ok &= expect_failure(name, "not implemented", [&] { backend.build_laplacian_pyramid({}, 1); });
    ok &= expect_failure(name, "not implemented", [&] { backend.blend({}, {}); });
    ok &= expect_failure(name, "not implemented", [&] { backend.reconstruct({}); });
    ok &= expect_failure(name, "not implemented", [&] { backend.fuse({}, {}); });
    return ok;
}

}  // namespace

int main() {
    bool ok = true;
    auto cpu = fusion::make_cpu_backend();
    ok &= check_backend(*cpu, "CPU");
    ok &= expect_failure("image loading", "not implemented", [] { fusion::load_image(""); });
    ok &= expect_failure("image saving", "not implemented", [] { fusion::save_image("", {}); });

#ifdef FUSION_ENABLE_CUDA
    auto cuda = fusion::make_cuda_backend();
    ok &= check_backend(*cuda, "CUDA");
#else
    ok &= expect_failure("CUDA backend", "not enabled", [] { fusion::make_cuda_backend(); });
#endif

    return ok ? 0 : 1;
}
