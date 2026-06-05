#include <gsplat_raster/gsplat_raster.h>

#include "internal/screen_ssbo_unpack.h"

namespace gsplat {

#ifndef GSPLAT_CUDA_ENABLED
Status unpackScreenSsboToDevice(const GaussianPixelGpu* /*d_cells*/, int /*grid_w*/, int /*grid_h*/,
                                DeviceGaussianBuffers& /*out*/, int& /*compact_out*/,
                                uint32_t /*expected_frame_id*/) {
    return Status::ErrorNoCuda;
}
#else
Status unpackScreenSsboToDevice(const GaussianPixelGpu* d_cells, int grid_w, int grid_h,
                                DeviceGaussianBuffers& out, int& compact_out,
                                uint32_t expected_frame_id) {
    compact_out = 0;
    if (!d_cells || grid_w <= 0 || grid_h <= 0) return Status::ErrorInvalidArgs;
    const int pixel_count = grid_w * grid_h;
    const float scale_mul = internal::screenSsboUnpackScaleMul();
    const float max_opacity = internal::screenSsboUnpackMaxOpacity();
    if (!internal::unpackScreenSsboPipeline(d_cells, pixel_count, pixel_count, out, compact_out,
                                            expected_frame_id, scale_mul, max_opacity)) {
        return Status::ErrorRenderFailed;
    }
    return Status::Ok;
}
#endif

}  // namespace gsplat
