#include <gsplat_raster/gsplat_raster.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>

static void boundsCenterRadius(const std::vector<gsplat::Gaussian>& g, double c[3], double& radius) {
    if (g.empty()) {
        c[0] = c[1] = c[2] = 0;
        radius = 1;
        return;
    }
    double minx = g[0].x, maxx = g[0].x;
    double miny = g[0].y, maxy = g[0].y;
    double minz = g[0].z, maxz = g[0].z;
    for (const auto& p : g) {
        minx = std::min(minx, static_cast<double>(p.x));
        maxx = std::max(maxx, static_cast<double>(p.x));
        miny = std::min(miny, static_cast<double>(p.y));
        maxy = std::max(maxy, static_cast<double>(p.y));
        minz = std::min(minz, static_cast<double>(p.z));
        maxz = std::max(maxz, static_cast<double>(p.z));
    }
    c[0] = 0.5 * (minx + maxx);
    c[1] = 0.5 * (miny + maxy);
    c[2] = 0.5 * (minz + maxz);
    const double dx = maxx - c[0], dy = maxy - c[1], dz = maxz - c[2];
    radius = std::sqrt(dx * dx + dy * dy + dz * dz);
    radius = std::max(radius, 0.5);
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <input.ply> <output.png> [width] [height] [max_points]\n";
        return 1;
    }
    const std::string ply = argv[1];
    const std::string png = argv[2];
    const int width = (argc > 3) ? std::stoi(argv[3]) : 1280;
    const int height = (argc > 4) ? std::stoi(argv[4]) : 720;
    gsplat::PlyLoadOptions load_opts;
    if (argc > 5) {
        load_opts.max_points = static_cast<size_t>(std::stoull(argv[5]));
        load_opts.drop_sh_rest = true;
    }

    std::vector<gsplat::Gaussian> gaussians;
    std::cout << "Loading " << ply << " ...\n" << std::flush;
    if (!gsplat::loadGaussianPly(ply, gaussians, load_opts)) {
        std::cerr << "PLY load failed\n";
        return 3;
    }
    std::cout << "Gaussians: " << gaussians.size() << "\n";

    if (!gsplat::isCudaAvailable()) {
        std::cerr << "CUDA not available. Build with CUDA toolkit and diff-gaussian-rasterization.\n";
        return 2;
    }

    double center[3], radius;
    boundsCenterRadius(gaussians, center, radius);

    const double dist = radius * 2.5;
    // Match 3dgs-osg-viewer default home camera (Z-up).
    const double eye[3] = {center[0], center[1] - dist, center[2] + dist * 0.35};
    const double up[3] = {0, 0, 1};
    const double aspect = static_cast<double>(width) / static_cast<double>(height);

    gsplat::Rasterizer raster;
    gsplat::RenderSettings settings;
    settings.scale_modifier = 16.f;
    if (load_opts.drop_sh_rest) settings.dc_only = true;
    raster.setSettings(settings);
    if (raster.setGaussians(gaussians) != gsplat::Status::Ok) {
        std::cerr << "Upload failed\n";
        return 4;
    }

    gsplat::Camera cam;
    gsplat::buildCameraLookAt(eye, center, up, 60.0, aspect, 0.01, 10000.0, center, cam, &raster);

    const int frustum = raster.countFrustumPass(cam);
    std::cout << "Frustum pass (sampled): ~" << frustum << "\n";

    std::cout << "Rendering " << width << "x" << height << " -> " << png << " ...\n";
    const gsplat::Status st = raster.renderToPng(cam, width, height, png);
    if (st != gsplat::Status::Ok) {
        std::cerr << "Render failed: " << gsplat::statusString(st) << "\n";
        return 5;
    }
    std::cout << "Visible splats: " << raster.lastVisibleCount() << " / " << raster.numGaussians() << "\n";
    std::cout << "Wrote " << png << "\n";
    return 0;
}
