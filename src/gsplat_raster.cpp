#include <gsplat_raster/gsplat_raster.h>

#include "camera_inria_internal.h"
#include "internal/rasterizer_impl.h"
#include "png_write.h"

#include <cstring>
#include <iostream>

#ifdef GSPLAT_CUDA_ENABLED
#include <cuda_runtime.h>
#endif

namespace gsplat {

const char* statusString(Status s) {
    switch (s) {
    case Status::Ok:
        return "ok";
    case Status::ErrorNoCuda:
        return "cuda not available";
    case Status::ErrorNoGaussians:
        return "no gaussians";
    case Status::ErrorInvalidArgs:
        return "invalid arguments";
    case Status::ErrorRenderFailed:
        return "render failed";
    case Status::ErrorPngWrite:
        return "png write failed";
    default:
        return "unknown";
    }
}

void resetCudaDevice() {
#ifdef GSPLAT_CUDA_ENABLED
    cudaDeviceReset();
#endif
}

bool isCudaAvailable() {
#ifdef GSPLAT_CUDA_ENABLED
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
#else
    return false;
#endif
}

struct Rasterizer::Impl {
    internal::CudaRasterEngine engine;
    RenderSettings settings;
    std::vector<Gaussian> gaussians;
};

Rasterizer::Rasterizer() : impl_(new Impl) {}
Rasterizer::~Rasterizer() { delete impl_; }

void Rasterizer::setSettings(const RenderSettings& s) {
    impl_->settings = s;
    impl_->engine.setSettings(s);
}

RenderSettings Rasterizer::settings() const { return impl_->settings; }

Status Rasterizer::setGaussians(std::vector<Gaussian> gaussians) {
    impl_->gaussians = std::move(gaussians);
    if (!impl_->engine.upload(impl_->gaussians)) return Status::ErrorNoGaussians;
    return Status::Ok;
}

int Rasterizer::numGaussians() const { return impl_->engine.numSplats(); }

int Rasterizer::lastVisibleCount() const { return impl_->engine.lastVisible(); }

int Rasterizer::countFrustumPass(const Camera& cam, int max_samples) const {
    return impl_->engine.countFrustumPass(cam.view, cam.proj, max_samples);
}

int Rasterizer::probeVisibleSplats(const Camera& cam, int width, int height) const {
    if (!isCudaAvailable() || impl_->engine.numSplats() <= 0) return 0;
    return impl_->engine.probeVisibleSplats(width, height, cam.view, cam.proj, cam.cam_pos,
                                            cam.tan_fovx, cam.tan_fovy);
}

Status Rasterizer::render(const Camera& cam, int width, int height, std::vector<uint8_t>& out_rgb) {
    if (!isCudaAvailable()) return Status::ErrorNoCuda;
    if (impl_->engine.numSplats() <= 0) return Status::ErrorNoGaussians;
    if (width <= 0 || height <= 0) return Status::ErrorInvalidArgs;
    if (cam.tan_fovx <= 0.f || cam.tan_fovy <= 0.f) return Status::ErrorInvalidArgs;
    if (!impl_->engine.render(width, height, cam.view, cam.proj, cam.cam_pos, cam.tan_fovx, cam.tan_fovy,
                              out_rgb)) {
        return Status::ErrorRenderFailed;
    }
    return Status::Ok;
}

Status Rasterizer::renderToPng(const Camera& cam, int width, int height, const std::string& png_path) {
    std::vector<uint8_t> rgb;
    const Status st = render(cam, width, height, rgb);
    if (st != Status::Ok) return st;
    if (!writeRgbPng(png_path, width, height, rgb)) return Status::ErrorPngWrite;
    return Status::Ok;
}

void buildCameraLookAt(const double eye[3], const double center[3], const double up[3],
                       double fov_y_deg, double aspect, double znear, double zfar,
                       const double ref_center[3], Camera& out, Rasterizer* raster_for_pick) {
    internal::LookAtMats mats;
    if (raster_for_pick && raster_for_pick->numGaussians() > 0) {
        struct PickCtx {
            const Rasterizer* raster;
        } ctx{raster_for_pick};
        auto score = [](const Camera& cam, void* p) -> int {
            const auto* pc = static_cast<PickCtx*>(p);
            if (!pc || !pc->raster) return 0;
            return pc->raster->countFrustumPass(cam);
        };
        internal::pickMatMode(eye, center, up, fov_y_deg, aspect, znear, zfar, ref_center, score,
                              &ctx, mats);
    } else {
        internal::buildLookAtMats(eye, center, up, fov_y_deg, aspect, znear, zfar, ref_center, mats,
                                  internal::defaultMatMode(), false);
    }
    internal::matsToCamera(mats, out);
}

void buildCameraFromOsg(const double view_osg[16], const double proj_osg[16],
                        const double ref_center[3], Camera& out) {
    internal::LookAtMats mats;
    internal::buildOsgMats(view_osg, proj_osg, ref_center, mats);
    internal::matsToCamera(mats, out);
}

}  // namespace gsplat
