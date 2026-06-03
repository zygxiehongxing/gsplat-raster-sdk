cmake_minimum_required(VERSION 3.20)
project(gsplat-raster-sdk VERSION 1.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(GSPLAT_BUILD_SHARED "Build shared SDK library" OFF)

# diff-gaussian-rasterization: must be vendored in this repository.
if(NOT DEFINED GSPLAT_DGR_ROOT)
    set(GSPLAT_DGR_ROOT "${CMAKE_CURRENT_LIST_DIR}/third_party/diff-gaussian-rasterization")
endif()
if(NOT EXISTS "${GSPLAT_DGR_ROOT}/cuda_rasterizer/rasterizer.h")
    message(FATAL_ERROR
        "Missing vendored diff-gaussian-rasterization at ${GSPLAT_DGR_ROOT}. "
        "This project does not use external fallback paths.")
endif()

add_subdirectory(sdk)
