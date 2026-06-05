#include <gsplat_raster/screen_cache.h>

#include <gsplat_raster/gaussian_device.h>

#include "internal/gaussian_pool.h"
#include "internal/screen_cache_internal.h"

#ifdef GSPLAT_CUDA_ENABLED
#include <cuda_runtime.h>
#endif

namespace gsplat {

struct ScreenGaussianCache::Impl {
#ifdef GSPLAT_CUDA_ENABLED
    float* d_depth = nullptr;
    int depth_pixels = 0;
#endif
};

ScreenGaussianCache::ScreenGaussianCache() : impl_(new Impl) {}
ScreenGaussianCache::~ScreenGaussianCache() {
#ifndef GSPLAT_CUDA_ENABLED
    buffers_ = DeviceGaussianBuffers{};
#else
    internal::freeScreenBuffers(buffers_);
    if (impl_->d_depth) {
        cudaFree(impl_->d_depth);
        impl_->d_depth = nullptr;
    }
#endif
    delete impl_;
}

#ifdef GSPLAT_CUDA_ENABLED
bool ScreenGaussianCache::ensureDepthBuffer(int pixels) {
    if (pixels <= 0) return false;
    if (impl_->d_depth && impl_->depth_pixels == pixels) return true;
    if (impl_->d_depth) {
        cudaFree(impl_->d_depth);
        impl_->d_depth = nullptr;
    }
    if (cudaMalloc(&impl_->d_depth, static_cast<size_t>(pixels) * sizeof(float)) != cudaSuccess) {
        return false;
    }
    impl_->depth_pixels = pixels;
    return true;
}

Status ScreenGaussianCache::ensureBufferLayout(int sh_degree, int sh_coeffs) {
    if (width_ <= 0 || height_ <= 0) return Status::ErrorInvalidArgs;
    const int n = width_ * height_;
    if (buffers_.means3D && buffers_.count == n && buffers_.sh_degree == sh_degree &&
        buffers_.sh_coeffs == sh_coeffs) {
        return Status::Ok;
    }
    internal::freeScreenBuffers(buffers_);
    if (!internal::allocScreenBuffers(width_, height_, sh_degree, sh_coeffs, buffers_)) {
        return Status::ErrorRenderFailed;
    }
    return Status::Ok;
}
#endif

Status ScreenGaussianCache::resize(int width, int height) {
    if (width <= 0 || height <= 0) return Status::ErrorInvalidArgs;
#ifndef GSPLAT_CUDA_ENABLED
    (void)width;
    (void)height;
    return Status::ErrorNoCuda;
#else
    width_ = width;
    height_ = height;
    if (!ensureDepthBuffer(width * height)) return Status::ErrorRenderFailed;
    return Status::Ok;
#endif
}

void ScreenGaussianCache::clear() {
#ifndef GSPLAT_CUDA_ENABLED
    filled_count_ = 0;
#else
    if (buffers_.count > 0 && impl_->d_depth) {
        internal::clearScreenCache(buffers_, impl_->d_depth, buffers_.count);
    }
    filled_count_ = 0;
#endif
}

Status ScreenGaussianCache::scatterFromPool(const GaussianDevicePool& pool, const Camera& cam) {
    if (!isCudaAvailable()) return Status::ErrorNoCuda;
    if (width_ <= 0 || height_ <= 0) return Status::ErrorInvalidArgs;
    if (pool.numSplats() <= 0) return Status::ErrorNoGaussians;
#ifndef GSPLAT_CUDA_ENABLED
    (void)pool;
    (void)cam;
    return Status::ErrorNoCuda;
#else
    DeviceGaussianBuffers src;
    pool.bindFullDeviceView(src);
    const Status layout_st = ensureBufferLayout(src.sh_degree, src.sh_coeffs);
    if (layout_st != Status::Ok) return layout_st;
    if (!ensureDepthBuffer(width_ * height_)) return Status::ErrorRenderFailed;

    int filled = 0;
    if (!internal::scatterPoolToScreen(src, cam.view, cam.proj, width_, height_, buffers_, impl_->d_depth,
                                       filled)) {
        return Status::ErrorRenderFailed;
    }
    filled_count_ = filled;
    return Status::Ok;
#endif
}

}  // namespace gsplat
