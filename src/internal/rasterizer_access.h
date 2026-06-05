#pragma once

#include <gsplat_raster/gsplat_raster.h>

namespace gsplat {
namespace internal {

class CudaRasterEngine;

CudaRasterEngine& rasterEngine(Rasterizer& raster);
RenderSettings rasterSettings(const Rasterizer& raster);

}  // namespace internal
}  // namespace gsplat
