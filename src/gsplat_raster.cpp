#include <gsplat_raster/gsplat_raster.h>

#include <gsplat_raster/gaussian_device.h>

#include "camera_inria_internal.h"
#include "internal/gaussian_pool.h"
#include "internal/rasterizer_impl.h"
#include "png_write.h"

#ifdef GSPLAT_CUDA_ENABLED
#include <cuda_runtime.h>
#endif

namespace gsplat {

#ifndef GSPLAT_CUDA_ENABLED
void freeDeviceGaussianBuffers(DeviceGaussianBuffers& buffers) { buffers = DeviceGaussianBuffers{}; }
#endif

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
    cudaFree(nullptr);
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
#else
    return false;
#endif
}

struct Rasterizer::Impl {
    internal::CudaRasterEngine engine;
    RenderSettings settings;
};

namespace internal {
CudaRasterEngine& rasterEngine(Rasterizer& raster) { return raster.impl_->engine; }
RenderSettings rasterSettings(const Rasterizer& raster) { return raster.impl_->settings; }
}  // namespace internal  // NOLINT: defined in rasterizer_access.h

Rasterizer::Rasterizer() : impl_(new Impl) {}
Rasterizer::~Rasterizer() { delete impl_; }

void Rasterizer::setSettings(const RenderSettings& s) {
    impl_->settings = s;
    impl_->engine.setSettings(s);
}

RenderSettings Rasterizer::settings() const { return impl_->settings; }

int Rasterizer::lastVisibleCount() const { return impl_->engine.lastVisible(); }

Status Rasterizer::render(const Camera& cam, int width, int height, const DeviceGaussianBuffers& gaussians,
                          std::vector<uint8_t>& out_rgb) {
    if (!isCudaAvailable()) return Status::ErrorNoCuda;
    if (gaussians.count <= 0) return Status::ErrorNoGaussians;
    if (width <= 0 || height <= 0) return Status::ErrorInvalidArgs;
    if (cam.tan_fovx <= 0.f || cam.tan_fovy <= 0.f) return Status::ErrorInvalidArgs;
    if (!impl_->engine.renderFromDevice(width, height, cam.view, cam.proj, cam.cam_pos, cam.tan_fovx,
                                        cam.tan_fovy, gaussians, out_rgb)) {
        return Status::ErrorRenderFailed;
    }
    return Status::Ok;
}

Status Rasterizer::renderDevice(const Camera& cam, int width, int height,
                                const DeviceGaussianBuffers& gaussians) {
    if (!isCudaAvailable()) return Status::ErrorNoCuda;
    if (gaussians.count <= 0) return Status::ErrorNoGaussians;
    if (width <= 0 || height <= 0) return Status::ErrorInvalidArgs;
    if (cam.tan_fovx <= 0.f || cam.tan_fovy <= 0.f) return Status::ErrorInvalidArgs;
    if (!impl_->engine.renderFromDeviceGpuOnly(width, height, cam.view, cam.proj, cam.cam_pos, cam.tan_fovx,
                                               cam.tan_fovy, gaussians)) {
        return Status::ErrorRenderFailed;
    }
    return Status::Ok;
}

const float* Rasterizer::deviceRgbPlanar() const { return impl_->engine.deviceRgbPlanar(); }

int Rasterizer::deviceRgbWidth() const { return impl_->engine.deviceRgbWidth(); }

int Rasterizer::deviceRgbHeight() const { return impl_->engine.deviceRgbHeight(); }

Status Rasterizer::renderToPng(const Camera& cam, int width, int height, const DeviceGaussianBuffers& gaussians,
                               const std::string& png_path) {
    std::vector<uint8_t> rgb;
    const Status st = render(cam, width, height, gaussians, rgb);
    if (st != Status::Ok) return st;
    if (!writeRgbPng(png_path, width, height, rgb)) return Status::ErrorPngWrite;
    return Status::Ok;
}

void buildCameraLookAt(const double eye[3], const double center[3], const double up[3], double fov_y_deg,
                       double aspect, double znear, double zfar, const double ref_center[3], Camera& out) {
    internal::LookAtMats mats;
    internal::buildLookAtMats(eye, center, up, fov_y_deg, aspect, znear, zfar, ref_center, mats,
                              internal::defaultMatMode(), false);
    internal::matsToCamera(mats, out);
}

void buildCameraFromOsg(const double view_osg[16], const double proj_osg[16], const double ref_center[3],
                        Camera& out, int* mat_mode_out, const float* probe_xyz, int probe_count) {
    internal::LookAtMats mats;
    internal::buildOsgMats(view_osg, proj_osg, ref_center, mats, mat_mode_out, probe_xyz, probe_count);
    internal::matsToCamera(mats, out);
}

}  // namespace gsplat
