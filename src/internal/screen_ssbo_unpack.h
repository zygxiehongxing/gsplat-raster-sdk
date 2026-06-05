#pragma once

#include <gsplat_raster/gsplat_raster.h>

namespace gsplat {
namespace internal {

bool allocScreenSoa(int W, int H, DeviceGaussianBuffers& out);
void freeScreenSoa(DeviceGaussianBuffers& out);

/// 将 GL SSBO 映射后的 AoS 缓冲解压为 Inria SoA（DC，sh_degree=0）。
float screenSsboUnpackScaleMul();
float screenSsboUnpackMaxOpacity();

bool unpackScreenSsboAoS(const GaussianPixelGpu* d_src, int pixel_count, DeviceGaussianBuffers& dst,
                        uint32_t expected_frame_id, float scale_mul, float max_opacity);

int countFilledScreenSoa(const DeviceGaussianBuffers& buf);

/// 仅保留 opacity>0 的屏幕高斯，将 count 设为紧凑后的数量。
bool compactScreenSoa(DeviceGaussianBuffers& buf, int& compact_count);

/// unpack + count + compact；可选 GPU 计时日志（GSPLAT_CUDA_RASTER_TIMING=1）。
bool unpackScreenSsboPipeline(const GaussianPixelGpu* d_src, int width, int height,
                              DeviceGaussianBuffers& out, int& filled_out,
                              uint32_t expected_frame_id, float scale_mul, float max_opacity);

}  // namespace internal
}  // namespace gsplat
