# Included before enable_language(CUDA) when -DGSPLAT_USE_CUDA_TOOLCHAIN=ON (see CudaRasterizer.cmake)
# CUDA 11.8 + VS 2026 (MSVC 14.44): host_config + STL1002 need explicit nvcc/cl flags.

set(_GSPLAT_CUDA_TOOLKIT_ROOT "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8")
if(DEFINED ENV{CUDA_PATH} AND EXISTS "$ENV{CUDA_PATH}/bin/nvcc.exe")
  file(TO_CMAKE_PATH "$ENV{CUDA_PATH}" _GSPLAT_CUDA_TOOLKIT_ROOT)
endif()

set(CMAKE_CUDA_COMPILER "${_GSPLAT_CUDA_TOOLKIT_ROOT}/bin/nvcc.exe")
if(EXISTS "C:/cuda118/bin/nvcc.exe")
  set(CMAKE_CUDA_COMPILER "C:/cuda118/bin/nvcc.exe")
endif()

set(CMAKE_CUDA_COMPILER_ID "NVIDIA" CACHE INTERNAL "")
set(CMAKE_CUDA_COMPILER_VERSION "11.8.89" CACHE INTERNAL "")
set(CMAKE_CUDA_COMPILER_FORCED ON CACHE INTERNAL "")
set(CMAKE_CUDA_COMPILER_WORKS TRUE CACHE INTERNAL "")

# /D* must use MSVC syntax so nvcc forwards them into *.res for cl (see scripts/probe-nvcc.cmd).
set(_GSPLAT_NVCC_HOST_FLAGS
  "-allow-unsupported-compiler"
  "-Xcompiler=/allow-unsupported-compiler"
  "-Xcompiler=/D__NV_NO_HOST_COMPILER_CHECK"
  "-Xcompiler=/D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH"
)
string(JOIN " " _GSPLAT_NVCC_HOST_FLAGS_STR ${_GSPLAT_NVCC_HOST_FLAGS})

set(CMAKE_CUDA_FLAGS "${_GSPLAT_NVCC_HOST_FLAGS_STR}" CACHE STRING "CUDA flags" FORCE)
set(CMAKE_CUDA_COMPILER_ID_FLAGS "${_GSPLAT_NVCC_HOST_FLAGS_STR}" CACHE STRING
  "Flags passed to nvcc during CMake compiler-id / implicit-link probe" FORCE)

set(CMAKE_CUDA_ARCHITECTURES "61;70;75;86" CACHE STRING "CUDA arch list for 1050 Ti + newer GPUs" FORCE)
