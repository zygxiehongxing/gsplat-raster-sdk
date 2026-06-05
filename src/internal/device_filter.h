#pragma once

#include <gsplat_raster/gsplat_raster.h>

namespace gsplat {
namespace internal {

/// 从 App 侧全量 GPU 高斯中，按相机视锥压缩出可见子集（GaussianDevicePool 调用）。
bool filterVisibleToDevice(int full_count, int sh_degree, int sh_coeffs,
                           const float* d_means3D, const float* d_scales, const float* d_rotations,
                           const float* d_opacities, const float* d_shs, const float* d_colors_precomp,
                           const float view[16], const float proj[16], DeviceGaussianBuffers& out);

}  // namespace internal
}  // namespace gsplat
