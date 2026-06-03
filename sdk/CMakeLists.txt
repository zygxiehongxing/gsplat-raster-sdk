include("${PROJECT_SOURCE_DIR}/cmake/CudaRasterizer.cmake")

set(GSPLAT_SDK_SOURCES
    ${PROJECT_SOURCE_DIR}/src/gsplat_raster.cpp
    ${PROJECT_SOURCE_DIR}/src/camera_inria.cpp
    ${PROJECT_SOURCE_DIR}/src/internal/rasterizer_impl.cpp
    ${PROJECT_SOURCE_DIR}/src/png_write.cpp
)

if(GSPLAT_BUILD_SHARED)
    add_library(gsplat_raster SHARED ${GSPLAT_SDK_SOURCES})
else()
    add_library(gsplat_raster STATIC ${GSPLAT_SDK_SOURCES})
endif()

target_include_directories(gsplat_raster
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/third_party
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
install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/gsplat_raster DESTINATION include)
