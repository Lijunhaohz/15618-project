#include "fusion/backend.hpp"
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

}  // namespace

int main() {
    bool ok = true;
    auto cpu = fusion::make_cpu_backend();
    ok &= cpu != nullptr;
#ifdef FUSION_ENABLE_CUDA
    auto cuda = fusion::make_cuda_backend();
    ok &= cuda != nullptr;
#else
    ok &= expect_failure("CUDA backend", "not enabled", [] { fusion::make_cuda_backend(); });
#endif

    return ok ? 0 : 1;
}
