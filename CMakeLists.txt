cmake_minimum_required(VERSION 3.21)

project(lbm_cuda_ad
    LANGUAGES CXX CUDA
)

# ------------------------------------------------------------
# Standards
# ------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ------------------------------------------------------------
# CUDA architecture
# ------------------------------------------------------------
set(CMAKE_CUDA_ARCHITECTURES 75 86)

# ------------------------------------------------------------
# Executable
# ------------------------------------------------------------
add_executable(lbm_cuda_ad

    # main
    src/main.cpp

    # core
    src/core/CudaHelpers.cu
    src/core/D3Q19Constants.cu
    src/core/D3Q7Constants.cu
    src/core/Macros.cu

    # collision
    src/collision/BGKCollision.cu
    src/collision/ADBGKCollision.cu

    # boundary (AD)
    src/boundary/ad/ADDirichlet.cu
    src/boundary/ad/ADDirichletTanhPulse.cu
    src/boundary/ad/ADNeumannOutlet.cu

    # boundary (NS)
    src/boundary/ns/BounceBack.cu
    src/boundary/ns/PressureDirichlet.cu
    src/boundary/ns/VelocityDirichlet.cu

    # reaction
    src/reaction/LinearLDF.cu
    src/reaction/LangmuirLDF.cu
)

# ------------------------------------------------------------
# Include directories
# ------------------------------------------------------------
target_include_directories(lbm_cuda_ad
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# ------------------------------------------------------------
# CUDA compile options
# ------------------------------------------------------------
target_compile_options(lbm_cuda_ad
    PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
)

# ------------------------------------------------------------
# CUDA properties
# ------------------------------------------------------------
set_target_properties(lbm_cuda_ad PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# ------------------------------------------------------------
# Warnings
# ------------------------------------------------------------
target_compile_options(lbm_cuda_ad
    PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
)

# ------------------------------------------------------------
# Print config summary
# ------------------------------------------------------------
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "CXX standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "CUDA standard: ${CMAKE_CUDA_STANDARD}")
