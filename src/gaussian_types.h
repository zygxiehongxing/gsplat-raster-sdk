#pragma once

#include <gsplat_raster/gsplat_raster.h>

#include <cmath>

namespace gsplat {

inline float sigmoid(float x) { return 1.f / (1.f + std::exp(-x)); }
inline float clamp01(float v) { return v < 0.f ? 0.f : (v > 1.f ? 1.f : v); }

}  // namespace gsplat
