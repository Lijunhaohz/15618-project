#pragma once

#include "fusion/backend.hpp"

namespace fusion::detail {

struct CudaStageTimings {
    float quality_ms = 0.0f;
    float normalization_ms = 0.0f;
    float gaussian_ms = 0.0f;
    float laplacian_ms = 0.0f;
    float blend_ms = 0.0f;
    float reconstruction_ms = 0.0f;
    float pipeline_ms = 0.0f;
};

RgbImage profile_cuda_fuse(const std::vector<RgbImage>& images,
                           const FusionOptions& options, CudaStageTimings& timings);

}  // namespace fusion::detail
