cmake_minimum_required(VERSION 3.20)
project(gsplat-raster-sdk VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(GSPLAT_BUILD_EXAMPLES "Build render_ply example" ON)
option(GSPLAT_BUILD_SHARED "Build shared library" OFF)

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

include("${CMAKE_SOURCE_DIR}/cmake/CudaRasterizer.cmake")

set(GSPLAT_SDK_SOURCES
    src/gsplat_raster.cpp
    src/ply_loader.cpp
    src/camera_inria.cpp
    src/internal/rasterizer_impl.cpp
    src/png_write.cpp
)

if(GSPLAT_BUILD_SHARED)
    add_library(gsplat_raster SHARED ${GSPLAT_SDK_SOURCES})
else()
    add_library(gsplat_raster STATIC ${GSPLAT_SDK_SOURCES})
endif()

target_include_directories(gsplat_raster
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/third_party
)

target_compile_definitions(gsplat_raster PUBLIC GSPLAT_RASTER_SDK=1)

if(GSPLAT_CUDA_ENABLED)
    target_compile_definitions(gsplat_raster PUBLIC GSPLAT_CUDA_ENABLED=1)
    target_link_libraries(gsplat_raster PUBLIC CudaRasterizer CUDA::cudart)
    if(CUDAToolkit_LIBRARY_DIR)
        target_link_directories(gsplat_raster PUBLIC "${CUDAToolkit_LIBRARY_DIR}")
    elseif(CMAKE_CUDA_COMPILER)
        get_filename_component(_nvcc_dir "${CMAKE_CUDA_COMPILER}" DIRECTORY)
        cmake_path(APPEND _nvcc_dir "../lib/x64" OUTPUT_VARIABLE _nvcc_lib)
        cmake_path(NORMAL_PATH _nvcc_lib)
        if(EXISTS "${_nvcc_lib}")
            target_link_directories(gsplat_raster PUBLIC "${_nvcc_lib}")
        endif()
    endif()
else()
    message(WARNING "CUDA not available — gsplat_raster cannot render")
endif()

if(MSVC)
    target_compile_options(gsplat_raster PRIVATE /W4 /utf-8)
else()
    target_compile_options(gsplat_raster PRIVATE -Wall -Wextra)
endif()

install(TARGETS gsplat_raster
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)
install(DIRECTORY include/gsplat_raster DESTINATION include)

if(GSPLAT_BUILD_EXAMPLES AND GSPLAT_CUDA_ENABLED)
    add_executable(gsplat_render_ply examples/render_ply/main.cpp)
    target_link_libraries(gsplat_render_ply PRIVATE gsplat_raster)
    if(WIN32 AND CUDAToolkit_BIN_DIR)
        file(GLOB _cudart_dll "${CUDAToolkit_BIN_DIR}/cudart64_*.dll")
        if(_cudart_dll)
            add_custom_command(TARGET gsplat_render_ply POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different "${_cudart_dll}" "$<TARGET_FILE_DIR:gsplat_render_ply>"
            )
        endif()
    endif()
endif()
