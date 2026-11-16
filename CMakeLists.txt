cmake_minimum_required(VERSION 3.18)
project(exposure_fusion LANGUAGES C CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(FUSION_ENABLE_CUDA "Build the CUDA backend" OFF)

find_package(PNG REQUIRED)

if(FUSION_ENABLE_CUDA)
    enable_language(CUDA)
endif()

add_library(fusion_core
    src/cpu_backend.cpp
    src/image_io.cpp
    src/png_io.c
)
target_include_directories(fusion_core PUBLIC include)
target_compile_features(fusion_core PUBLIC cxx_std_17)
set_target_properties(fusion_core PROPERTIES C_STANDARD 99 C_STANDARD_REQUIRED YES)
target_link_libraries(fusion_core PRIVATE PNG::PNG)

if(FUSION_ENABLE_CUDA)
    target_sources(fusion_core PRIVATE src/cuda_backend.cu)
    target_compile_options(fusion_core PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>)
    set_target_properties(fusion_core PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED YES
    )
else()
    target_sources(fusion_core PRIVATE src/cuda_unavailable.cpp)
endif()

add_executable(exposure_fusion src/main.cpp)
target_link_libraries(exposure_fusion PRIVATE fusion_core)

if(FUSION_ENABLE_CUDA)
    add_executable(cuda_benchmark benchmarks/cuda_benchmark.cpp)
    target_include_directories(cuda_benchmark PRIVATE src)
    target_link_libraries(cuda_benchmark PRIVATE fusion_core)
endif()

include(CTest)
if(BUILD_TESTING)
    add_executable(skeleton_tests tests/skeleton_tests.cpp)
    target_link_libraries(skeleton_tests PRIVATE fusion_core)
    if(FUSION_ENABLE_CUDA)
        target_compile_definitions(skeleton_tests PRIVATE FUSION_ENABLE_CUDA)
    endif()
    add_test(NAME skeleton_placeholders COMMAND skeleton_tests)
    add_test(NAME cli_help COMMAND exposure_fusion --help)
    if(UNIX)
        add_test(NAME cuda_runner_script
            COMMAND sh "${CMAKE_CURRENT_SOURCE_DIR}/tests/cuda_runner_test.sh"
                    "${CMAKE_CURRENT_SOURCE_DIR}/scripts/run_cuda.sh")
    endif()
    add_test(NAME cli_integration
        COMMAND ${CMAKE_COMMAND} "-DEXE=$<TARGET_FILE:exposure_fusion>"
                "-DCUDA_ENABLED=${FUSION_ENABLE_CUDA}"
                "-DBRACKET_DIR=${CMAKE_CURRENT_SOURCE_DIR}/data/basic-test-3"
                -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/cli_test.cmake")

    add_executable(cpu_weights_tests tests/cpu_weights_tests.cpp)
    target_link_libraries(cpu_weights_tests PRIVATE fusion_core)
    add_test(NAME cpu_weights COMMAND cpu_weights_tests)

    add_executable(cpu_normalization_tests tests/cpu_normalization_tests.cpp)
    target_link_libraries(cpu_normalization_tests PRIVATE fusion_core)
    add_test(NAME cpu_normalization COMMAND cpu_normalization_tests)

    if(FUSION_ENABLE_CUDA)
        add_executable(cuda_weights_tests tests/cuda_weights_tests.cpp)
        target_link_libraries(cuda_weights_tests PRIVATE fusion_core)
        add_test(NAME cuda_weights COMMAND cuda_weights_tests)
        add_executable(cuda_pyramid_tests tests/cuda_pyramid_tests.cpp)
        target_link_libraries(cuda_pyramid_tests PRIVATE fusion_core)
        add_test(NAME cuda_pyramids COMMAND cuda_pyramid_tests)
        add_executable(cuda_blend_reconstruction_tests tests/cuda_blend_reconstruction_tests.cpp)
        target_link_libraries(cuda_blend_reconstruction_tests PRIVATE fusion_core)
        add_test(NAME cuda_blend_reconstruction COMMAND cuda_blend_reconstruction_tests)
        add_executable(cuda_fusion_tests tests/cuda_fusion_tests.cpp)
        target_include_directories(cuda_fusion_tests PRIVATE src)
        target_link_libraries(cuda_fusion_tests PRIVATE fusion_core)
        add_test(NAME cuda_fusion COMMAND cuda_fusion_tests)
        add_executable(cuda_fusion_validation_tests tests/cuda_fusion_validation_tests.cpp)
        target_link_libraries(cuda_fusion_validation_tests PRIVATE fusion_core)
        add_test(NAME cuda_fusion_validation_synthetic COMMAND cuda_fusion_validation_tests)
        set(validation_images
            "${CMAKE_CURRENT_SOURCE_DIR}/data/basic-test-3/memorial00.png"
            "${CMAKE_CURRENT_SOURCE_DIR}/data/basic-test-3/memorial03.png"
            "${CMAKE_CURRENT_SOURCE_DIR}/data/basic-test-3/memorial06.png")
        if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/data/basic-test-3/memorial00.png" AND
           EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/data/basic-test-3/memorial03.png" AND
           EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/data/basic-test-3/memorial06.png")
            add_test(NAME cuda_fusion_validation_real
                COMMAND cuda_fusion_validation_tests --real ${validation_images})
        else()
            message(STATUS "CUDA real-image validation not registered: memorial bracket is incomplete")
        endif()
    endif()

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

    add_executable(image_io_tests tests/image_io_tests.cpp)
    target_link_libraries(image_io_tests PRIVATE fusion_core)
    add_test(NAME image_io COMMAND image_io_tests)
    file(GLOB basic_test_images CONFIGURE_DEPENDS
         "${CMAKE_CURRENT_SOURCE_DIR}/data/basic-test-3/*.png")
    list(LENGTH basic_test_images basic_test_image_count)
    if(basic_test_image_count EQUAL 3)
        add_test(NAME image_io_real_inputs COMMAND image_io_tests ${basic_test_images})
    endif()
endif()
