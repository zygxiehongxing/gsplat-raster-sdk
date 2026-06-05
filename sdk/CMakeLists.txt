include("${PROJECT_SOURCE_DIR}/cmake/CudaRasterizer.cmake")

set(GSPLAT_SDK_SOURCES
    ${PROJECT_SOURCE_DIR}/src/gsplat_raster.cpp
    ${PROJECT_SOURCE_DIR}/src/gaussian_device.cpp
    ${PROJECT_SOURCE_DIR}/src/camera_inria.cpp
    ${PROJECT_SOURCE_DIR}/src/internal/rasterizer_impl.cpp
    ${PROJECT_SOURCE_DIR}/src/internal/gaussian_pool.cpp
    ${PROJECT_SOURCE_DIR}/src/screen_cache.cpp
    ${PROJECT_SOURCE_DIR}/src/screen_ssbo_unpack.cpp
    ${PROJECT_SOURCE_DIR}/src/screen_ssbo_project_debug.cpp
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
    add_library(gsplat_device_filter OBJECT
        ${PROJECT_SOURCE_DIR}/src/internal/device_filter.cu
    )
    add_library(gsplat_screen_cache OBJECT
        ${PROJECT_SOURCE_DIR}/src/internal/screen_cache.cu
    )
    add_library(gsplat_screen_ssbo_unpack OBJECT
        ${PROJECT_SOURCE_DIR}/src/internal/screen_ssbo_unpack.cu
    )
    add_library(gsplat_screen_ssbo_project_debug OBJECT
        ${PROJECT_SOURCE_DIR}/src/internal/screen_ssbo_project_debug.cu
    )
    target_include_directories(gsplat_device_filter PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/third_party
    )
    target_include_directories(gsplat_screen_cache PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/third_party
    )
    target_include_directories(gsplat_device_filter PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    target_link_libraries(gsplat_device_filter PRIVATE CudaRasterizer ${GSPLAT_CUDART_LINK})
    target_include_directories(gsplat_screen_cache PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    target_link_libraries(gsplat_screen_cache PRIVATE ${GSPLAT_CUDART_LINK})
    target_include_directories(gsplat_screen_ssbo_unpack PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/third_party
    )
    target_include_directories(gsplat_screen_ssbo_unpack PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    target_link_libraries(gsplat_screen_ssbo_unpack PRIVATE ${GSPLAT_CUDART_LINK})
    target_include_directories(gsplat_screen_ssbo_project_debug PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/third_party
    )
    target_include_directories(gsplat_screen_ssbo_project_debug PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    target_link_libraries(gsplat_screen_ssbo_project_debug PRIVATE ${GSPLAT_CUDART_LINK})
    set_target_properties(gsplat_screen_ssbo_unpack PROPERTIES
        CUDA_SEPARABLE_COMPILATION OFF
        POSITION_INDEPENDENT_CODE ON
    )
    set_target_properties(gsplat_screen_ssbo_project_debug PROPERTIES
        CUDA_SEPARABLE_COMPILATION OFF
        POSITION_INDEPENDENT_CODE ON
    )
    set_target_properties(gsplat_device_filter PROPERTIES
        CUDA_SEPARABLE_COMPILATION OFF
        POSITION_INDEPENDENT_CODE ON
    )
    set_target_properties(gsplat_screen_cache PROPERTIES
        CUDA_SEPARABLE_COMPILATION OFF
        POSITION_INDEPENDENT_CODE ON
    )
    target_compile_definitions(gsplat_raster PUBLIC GSPLAT_CUDA_ENABLED=1)
    target_include_directories(gsplat_raster PUBLIC ${CUDAToolkit_INCLUDE_DIRS})
    target_link_libraries(gsplat_raster PUBLIC CudaRasterizer ${GSPLAT_CUDART_LINK} gsplat_device_filter
        gsplat_screen_cache gsplat_screen_ssbo_unpack gsplat_screen_ssbo_project_debug)
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
