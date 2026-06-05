# Included before project()/enable_language(CUDA) when -DGSPLAT_USE_CUDA_TOOLCHAIN=ON.
# CUDA 11.8 + VS 2026: skip CMake CompilerId (STL1002); use -ccbin + nvcc host flags on real .cu.

set(_GSPLAT_CUDA_TOOLKIT_ROOT "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8")
if(DEFINED ENV{CUDA_PATH} AND EXISTS "$ENV{CUDA_PATH}/bin/nvcc.exe")
    file(TO_CMAKE_PATH "$ENV{CUDA_PATH}" _GSPLAT_CUDA_TOOLKIT_ROOT)
endif()
if(EXISTS "C:/cuda118/bin/nvcc.exe")
    set(_GSPLAT_CUDA_TOOLKIT_ROOT "C:/cuda118")
endif()

set(CMAKE_CUDA_COMPILER "${_GSPLAT_CUDA_TOOLKIT_ROOT}/bin/nvcc.exe" CACHE FILEPATH "CUDA compiler" FORCE)
set(CUDAToolkit_ROOT "${_GSPLAT_CUDA_TOOLKIT_ROOT}" CACHE PATH "CUDA toolkit root" FORCE)

if(NOT CMAKE_CUDA_HOST_COMPILER)
    set(CMAKE_CUDA_HOST_COMPILER "cl" CACHE FILEPATH "CUDA host compiler" FORCE)
endif()
if(WIN32 AND EXISTS "${CMAKE_CUDA_HOST_COMPILER}")
    file(TO_NATIVE_PATH "${CMAKE_CUDA_HOST_COMPILER}" _gsplat_host_native)
    execute_process(
        COMMAND cmd.exe /c "for %I in (\"${_gsplat_host_native}\") do @echo %~sI"
        OUTPUT_VARIABLE _gsplat_host_short
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(_gsplat_host_short AND EXISTS "${_gsplat_host_short}")
        file(TO_CMAKE_PATH "${_gsplat_host_short}" _gsplat_host_short)
        set(CMAKE_CUDA_HOST_COMPILER "${_gsplat_host_short}" CACHE FILEPATH "CUDA host compiler" FORCE)
    endif()
endif()

# Single CACHE string (no list); semicolon-separated CACHE breaks -Xcompiler=/D... tokens.
set(CMAKE_CUDA_FLAGS
    "-allow-unsupported-compiler -Xcompiler=/D__NV_NO_HOST_COMPILER_CHECK -Xcompiler=/D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH -Xcompiler=/D_DISABLE_EXTENDED_ALIGNED_STORAGE -Xcompiler=/std:c++17"
    CACHE STRING "CUDA flags" FORCE)
set(CMAKE_CUDA_STANDARD 17 CACHE STRING "CUDA standard" FORCE)
set(_GSPLAT_NVCC_LIB_DIR "${_GSPLAT_CUDA_TOOLKIT_ROOT}/lib/x64")
set(_GSPLAT_NVCC_INC_DIR "${_GSPLAT_CUDA_TOOLKIT_ROOT}/include")
# Satisfy cmake_nvcc_parse_implicit_info without CMakeCUDACompilerId.cu (VS2026 STL).
set(CMAKE_CUDA_COMPILER_PRODUCED_OUTPUT
    "#$ PATH=\n#$ INCLUDES=\"-I${_GSPLAT_NVCC_INC_DIR}\"\n#$ LIBRARIES= \"/LIBPATH:${_GSPLAT_NVCC_LIB_DIR}\"\nlink.exe /nologo \"/LIBPATH:${_GSPLAT_NVCC_LIB_DIR}\" cudart.lib\n"
    CACHE INTERNAL "nvcc implicit link info for CMake" FORCE)

# Skip CMakeCUDACompilerId.cu (breaks with VS2026 STL); implicit link dirs set below.
set(CMAKE_CUDA_COMPILER_ID "NVIDIA" CACHE INTERNAL "CUDA compiler vendor" FORCE)
set(CMAKE_CUDA_COMPILER_VERSION "11.8.89" CACHE INTERNAL "CUDA compiler version" FORCE)
set(CMAKE_CUDA_COMPILER_TOOLKIT_ROOT "${_GSPLAT_CUDA_TOOLKIT_ROOT}" CACHE PATH "" FORCE)
set(CMAKE_CUDA_COMPILER_LIBRARY_ROOT "${_GSPLAT_CUDA_TOOLKIT_ROOT}" CACHE PATH "" FORCE)
set(CMAKE_CUDA_COMPILER_TOOLKIT_VERSION "11.8.89" CACHE INTERNAL "" FORCE)
set(CMAKE_CUDA_COMPILER_ID_RUN TRUE CACHE INTERNAL "Skip CUDA compiler-id probe" FORCE)
set(CMAKE_CUDA_COMPILER_FORCED TRUE CACHE INTERNAL "Skip CUDA try_compile probe" FORCE)
set(CMAKE_CUDA_COMPILER_WORKS TRUE CACHE INTERNAL "CUDA compiler works" FORCE)

set(CMAKE_CUDA_IMPLICIT_LINK_LIBRARIES "cudart" CACHE INTERNAL "CUDA implicit libs" FORCE)
set(CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES "${_GSPLAT_CUDA_TOOLKIT_ROOT}/lib/x64" CACHE INTERNAL
    "CUDA implicit lib dirs" FORCE)
set(CMAKE_CUDA_HOST_IMPLICIT_LINK_DIRECTORIES "${_GSPLAT_CUDA_TOOLKIT_ROOT}/lib/x64" CACHE INTERNAL "" FORCE)
set(CMAKE_CUDA_HOST_IMPLICIT_LINK_LIBRARIES "" CACHE INTERNAL "" FORCE)
set(CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES "${_GSPLAT_CUDA_TOOLKIT_ROOT}/include" CACHE INTERNAL "" FORCE)
set(CMAKE_CUDA_ARCHITECTURES "61;70;75;86" CACHE STRING "CUDA arch list" FORCE)

message(STATUS "gsplat cuda cache: nvcc=${CMAKE_CUDA_COMPILER} host=${CMAKE_CUDA_HOST_COMPILER} (skip compiler-id)")
