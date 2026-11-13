cmake_minimum_required(VERSION 3.18)
project(exposure_fusion LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(FUSION_ENABLE_CUDA "Build the CUDA backend skeleton" OFF)

add_library(fusion_core
    src/cpu_backend.cpp
    src/image_io.cpp
)
target_include_directories(fusion_core PUBLIC include)
target_compile_features(fusion_core PUBLIC cxx_std_17)

if(FUSION_ENABLE_CUDA)
    enable_language(CUDA)
    target_sources(fusion_core PRIVATE src/cuda_backend.cu)
    set_target_properties(fusion_core PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED YES
    )
else()
    target_sources(fusion_core PRIVATE src/cuda_unavailable.cpp)
endif()

add_executable(exposure_fusion src/main.cpp)
target_link_libraries(exposure_fusion PRIVATE fusion_core)

include(CTest)
if(BUILD_TESTING)
    add_executable(skeleton_tests tests/skeleton_tests.cpp)
    target_link_libraries(skeleton_tests PRIVATE fusion_core)
    if(FUSION_ENABLE_CUDA)
        target_compile_definitions(skeleton_tests PRIVATE FUSION_ENABLE_CUDA)
    endif()
    add_test(NAME skeleton_placeholders COMMAND skeleton_tests)
    add_test(NAME cli_help COMMAND exposure_fusion --help)

    add_executable(cpu_weights_tests tests/cpu_weights_tests.cpp)
    target_link_libraries(cpu_weights_tests PRIVATE fusion_core)
    add_test(NAME cpu_weights COMMAND cpu_weights_tests)

    add_executable(cpu_normalization_tests tests/cpu_normalization_tests.cpp)
    target_link_libraries(cpu_normalization_tests PRIVATE fusion_core)
    add_test(NAME cpu_normalization COMMAND cpu_normalization_tests)

    add_executable(cpu_weight_pyramid_tests tests/cpu_weight_pyramid_tests.cpp)
    target_link_libraries(cpu_weight_pyramid_tests PRIVATE fusion_core)
    add_test(NAME cpu_weight_pyramid COMMAND cpu_weight_pyramid_tests)

    add_executable(cpu_laplacian_pyramid_tests tests/cpu_laplacian_pyramid_tests.cpp)
    target_link_libraries(cpu_laplacian_pyramid_tests PRIVATE fusion_core)
    add_test(NAME cpu_laplacian_pyramid COMMAND cpu_laplacian_pyramid_tests)

    add_executable(cpu_blend_tests tests/cpu_blend_tests.cpp)
    target_link_libraries(cpu_blend_tests PRIVATE fusion_core)
    add_test(NAME cpu_blend COMMAND cpu_blend_tests)

    add_executable(cpu_reconstruction_tests tests/cpu_reconstruction_tests.cpp)
    target_link_libraries(cpu_reconstruction_tests PRIVATE fusion_core)
    add_test(NAME cpu_reconstruction COMMAND cpu_reconstruction_tests)

    add_executable(cpu_fusion_tests tests/cpu_fusion_tests.cpp)
    target_link_libraries(cpu_fusion_tests PRIVATE fusion_core)
    add_test(NAME cpu_fusion COMMAND cpu_fusion_tests)
endif()
