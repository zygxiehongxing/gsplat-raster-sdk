cmake_minimum_required(VERSION 3.20)
project(gsplat-raster-sdk VERSION 1.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(GSPLAT_BUILD_SHARED "Build shared SDK library" OFF)

# diff-gaussian-rasterization: submodule, or sibling 3dgs-osg-viewer copy
if(NOT DEFINED GSPLAT_DGR_ROOT)
    set(GSPLAT_DGR_ROOT "${CMAKE_SOURCE_DIR}/third_party/diff-gaussian-rasterization")
endif()
if(NOT EXISTS "${GSPLAT_DGR_ROOT}/cuda_rasterizer/rasterizer.h")
    set(_fallback "${CMAKE_SOURCE_DIR}/../3dgs-osg-viewer/third_party/diff-gaussian-rasterization")
    if(EXISTS "${_fallback}/cuda_rasterizer/rasterizer.h")
        set(GSPLAT_DGR_ROOT "${_fallback}")
        message(STATUS "Using diff-gaussian-rasterization from ${_fallback}")
    endif()
endif()

add_subdirectory(sdk)
