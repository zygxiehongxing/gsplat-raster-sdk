#pragma once

#include <gsplat_raster/gsplat_raster.h>

#include <string>
#include <vector>

namespace gsplat {

bool loadGaussianPly(const std::string& path, std::vector<Gaussian>& out, const PlyLoadOptions& opts);

}  // namespace gsplat
