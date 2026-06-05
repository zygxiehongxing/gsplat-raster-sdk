#pragma once

#include <gsplat_raster/gsplat_raster.h>

namespace gsplat {
namespace internal {

/// W×H RGB8：黑底，将 SSBO mean 经 col-major proj 投到的像素标红。
bool renderSsboMeanProjectMap(const Camera& cam, int width, int height, const GaussianPixelGpu* d_cells,
                              std::vector<uint8_t>& out_rgb, int& marked_pixels,
                              uint32_t expected_frame_id = 0xFFFFFFFFu);

bool renderSsboMeanProjectMapGlsl(const ScreenCacheMetaGpu& meta, int width, int height,
                                  const GaussianPixelGpu* d_cells, std::vector<uint8_t>& out_rgb,
                                  int& marked_pixels, uint32_t expected_frame_id, bool image_top_origin);

bool renderSsboColorCellsProjectMapGlsl(const ScreenCacheMetaGpu& meta, int width, int height,
                                      const ScreenColorCellGpu* d_cells, std::vector<uint8_t>& out_rgb,
                                      int& marked_pixels, bool image_top_origin);

}  // namespace internal
}  // namespace gsplat
