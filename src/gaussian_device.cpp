#include <gsplat_raster/gaussian_device.h>

#include "internal/gaussian_pool.h"
#include "internal/rasterizer_access.h"

namespace gsplat {

struct GaussianDevicePool::Impl {
    internal::CudaGaussianPool pool;
};

GaussianDevicePool::GaussianDevicePool() : impl_(new Impl) {}
GaussianDevicePool::~GaussianDevicePool() { delete impl_; }

void GaussianDevicePool::setDcOnly(bool dc_only) { impl_->pool.setDcOnly(dc_only); }
bool GaussianDevicePool::dcOnly() const { return impl_->pool.dcOnly(); }

Status GaussianDevicePool::upload(std::vector<Gaussian> gaussians) {
    if (!impl_->pool.upload(gaussians)) return Status::ErrorNoGaussians;
    return Status::Ok;
}

int GaussianDevicePool::numSplats() const { return impl_->pool.numSplats(); }

void GaussianDevicePool::bindFullDeviceView(DeviceGaussianBuffers& out) const {
    impl_->pool.bindFullDeviceView(out);
}

Status GaussianDevicePool::filterVisible(const Camera& cam, DeviceGaussianBuffers& out) {
    if (!isCudaAvailable()) return Status::ErrorNoCuda;
    if (impl_->pool.numSplats() <= 0) return Status::ErrorNoGaussians;
    if (!impl_->pool.filterVisible(cam.view, cam.proj, out)) return Status::ErrorRenderFailed;
    return Status::Ok;
}

int GaussianDevicePool::countFrustumPass(const Camera& cam, int max_samples) const {
    return impl_->pool.countFrustumPass(cam.view, cam.proj, max_samples);
}

int GaussianDevicePool::probeVisibleSplats(const Camera& cam, int width, int height, Rasterizer& raster) const {
    if (!isCudaAvailable() || impl_->pool.numSplats() <= 0) return 0;
    return impl_->pool.probeVisibleSplats(width, height, cam.view, cam.proj, cam.cam_pos, cam.tan_fovx,
                                          cam.tan_fovy, internal::rasterEngine(raster),
                                          internal::rasterSettings(raster));
}

}  // namespace gsplat
