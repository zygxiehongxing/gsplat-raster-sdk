#include <gsplat_raster/gsplat_raster.h>

#include "internal/screen_ssbo_unpack.h"

namespace gsplat {

#ifndef GSPLAT_CUDA_ENABLED
Status unpackScreenSsboToDevice(const GaussianPixelGpu* /*d_src*/, int /*width*/, int /*height*/,
                                DeviceGaussianBuffers& /*out*/, int& /*filled_out*/,
                                uint32_t /*expected_frame_id*/) {
    return Status::ErrorNoCuda;
}
#else
Status unpackScreenSsboToDevice(const GaussianPixelGpu* d_src, int width, int height,
                                DeviceGaussianBuffers& out, int& filled_out,
                                uint32_t expected_frame_id) {
    filled_out = 0;
    if (!d_src || width <= 0 || height <= 0) return Status::ErrorInvalidArgs;
    const float scale_mul = internal::screenSsboUnpackScaleMul();
    const float max_opacity = internal::screenSsboUnpackMaxOpacity();
    if (!internal::unpackScreenSsboPipeline(d_src, width, height, out, filled_out, expected_frame_id,
                                           scale_mul, max_opacity)) {
        return Status::ErrorRenderFailed;
    }
    return Status::Ok;
}
#endif

}  // namespace gsplat
