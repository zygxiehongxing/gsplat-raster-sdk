#include <gsplat_raster/gsplat_raster.h>

#include "internal/screen_ssbo_project_debug.h"

namespace gsplat {

#ifndef GSPLAT_CUDA_ENABLED
Status renderSsboMeanProjectMap(const Camera& /*cam*/, int /*width*/, int /*height*/,
                                const GaussianPixelGpu* /*d_cells*/, std::vector<uint8_t>& /*out_rgb*/,
                                int& /*marked_pixels*/, uint32_t /*expected_frame_id*/) {
    return Status::ErrorNoCuda;
}
#else
Status renderSsboMeanProjectMap(const Camera& cam, int width, int height, const GaussianPixelGpu* d_cells,
                                std::vector<uint8_t>& out_rgb, int& marked_pixels,
                                uint32_t expected_frame_id) {
    if (!d_cells || width <= 0 || height <= 0) return Status::ErrorInvalidArgs;
    if (!internal::renderSsboMeanProjectMap(cam, width, height, d_cells, out_rgb, marked_pixels,
                                            expected_frame_id)) {
        return Status::ErrorRenderFailed;
    }
    return Status::Ok;
}
#endif

#ifndef GSPLAT_CUDA_ENABLED
Status renderSsboMeanProjectMapGlsl(const ScreenCacheMetaGpu& /*meta*/, int /*width*/, int /*height*/,
                                    const GaussianPixelGpu* /*d_cells*/, std::vector<uint8_t>& /*out_rgb*/,
                                    int& /*marked_pixels*/, uint32_t /*expected_frame_id*/,
                                    bool /*image_top_origin*/) {
    return Status::ErrorNoCuda;
}
#else
Status renderSsboMeanProjectMapGlsl(const ScreenCacheMetaGpu& meta, int width, int height,
                                    const GaussianPixelGpu* d_cells, std::vector<uint8_t>& out_rgb,
                                    int& marked_pixels, uint32_t expected_frame_id, bool image_top_origin) {
    if (!d_cells || width <= 0 || height <= 0) return Status::ErrorInvalidArgs;
    if (!internal::renderSsboMeanProjectMapGlsl(meta, width, height, d_cells, out_rgb, marked_pixels,
                                                expected_frame_id, image_top_origin)) {
        return Status::ErrorRenderFailed;
    }
    return Status::Ok;
}
#endif

#ifndef GSPLAT_CUDA_ENABLED
Status renderSsboColorCellsProjectMapGlsl(const ScreenCacheMetaGpu& /*meta*/, int /*width*/, int /*height*/,
                                          const ScreenColorCellGpu* /*d_cells*/, std::vector<uint8_t>& /*out_rgb*/,
                                          int& /*marked_pixels*/, bool /*image_top_origin*/) {
    return Status::ErrorNoCuda;
}
#else
Status renderSsboColorCellsProjectMapGlsl(const ScreenCacheMetaGpu& meta, int width, int height,
                                          const ScreenColorCellGpu* d_cells, std::vector<uint8_t>& out_rgb,
                                          int& marked_pixels, bool image_top_origin) {
    if (!d_cells || width <= 0 || height <= 0) return Status::ErrorInvalidArgs;
    if (!internal::renderSsboColorCellsProjectMapGlsl(meta, width, height, d_cells, out_rgb, marked_pixels,
                                                      image_top_origin)) {
        return Status::ErrorRenderFailed;
    }
    return Status::Ok;
}
#endif

}  // namespace gsplat
