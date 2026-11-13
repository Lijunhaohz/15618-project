/**
 * Provides a stub for the CUDA backend when CUDA is not enabled.
 */
#include "fusion/backend.hpp"

#include <stdexcept>

namespace fusion {

std::unique_ptr<Backend> make_cuda_backend() {
    throw std::runtime_error(
        "CUDA backend is not enabled; configure with -DFUSION_ENABLE_CUDA=ON");
}

}  // namespace fusion
