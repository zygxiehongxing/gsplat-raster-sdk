#pragma once

#include <gsplat_raster/gsplat_raster.h>

namespace gsplat {
namespace internal {

bool allocScreenBuffers(int W, int H, int sh_degree, int sh_coeffs, DeviceGaussianBuffers& out);
void freeScreenBuffers(DeviceGaussianBuffers& out);
bool clearScreenCache(DeviceGaussianBuffers& buf, float* depth_buf, int pixel_count);
bool scatterPoolToScreen(const DeviceGaussianBuffers& src, const float view[16], const float proj[16], int W,
                         int H, DeviceGaussianBuffers& dst, float* depth_buf, int& filled_out);

}  // namespace internal
}  // namespace gsplat
