#pragma once

/**
 * 与屏幕等大的 GPU 高斯缓存（W×H 格点，每格最多一颗可见高斯）。
 * 存放源池中的完整高斯属性；同像素深度竞争（更近者保留）。
 * SDK Rasterizer 通过 deviceBuffers() 只读光栅。
 */

#include <gsplat_raster/gsplat_raster.h>

namespace gsplat {

class GaussianDevicePool;

class ScreenGaussianCache {
public:
    ScreenGaussianCache();
    ~ScreenGaussianCache();

    ScreenGaussianCache(const ScreenGaussianCache&) = delete;
    ScreenGaussianCache& operator=(const ScreenGaussianCache&) = delete;

    int width() const { return width_; }
    int height() const { return height_; }

    Status resize(int width, int height);

    void clear();

    /// 从源池投影 scatter：完整高斯属性 + 深度竞争写入对应像素格。
    Status scatterFromPool(const GaussianDevicePool& pool, const Camera& cam);

    const DeviceGaussianBuffers& deviceBuffers() const { return buffers_; }

    /// 本帧有效格点数（opacity>0，≤ W×H）。
    int filledCount() const { return filled_count_; }

private:
#ifdef GSPLAT_CUDA_ENABLED
    bool ensureDepthBuffer(int pixels);
    Status ensureBufferLayout(int sh_degree, int sh_coeffs);
#endif

    struct Impl;
    Impl* impl_;
    int width_ = 0;
    int height_ = 0;
    int filled_count_ = 0;
    DeviceGaussianBuffers buffers_{};
};

}  // namespace gsplat
