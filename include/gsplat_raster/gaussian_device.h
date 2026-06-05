#pragma once

/**
 * App 侧 GPU 高斯池：PLY/全量上传、视锥筛选、压缩子集。
 * SDK Rasterizer 不持有全量数据，只消费 filter 后的 DeviceGaussianBuffers。
 */

#include <gsplat_raster/gsplat_raster.h>

#include <vector>

namespace gsplat {

class Rasterizer;

/// App 持有：全量高斯在 GPU 上，按相机筛出子集供 Rasterizer::render 使用。
class GaussianDevicePool {
public:
    GaussianDevicePool();
    ~GaussianDevicePool();

    GaussianDevicePool(const GaussianDevicePool&) = delete;
    GaussianDevicePool& operator=(const GaussianDevicePool&) = delete;

    /// 上传选项（仅影响 upload：是否只用 SH DC）。
    void setDcOnly(bool dc_only);
    bool dcOnly() const;

    /// CPU 高斯 → GPU 全量（App 数据源，不经过 Rasterizer）。
    Status upload(std::vector<Gaussian> gaussians);

    int numSplats() const;

    /// 非拥有视图：指向池内全量 GPU 缓冲（供 ScreenGaussianCache scatter 等 App Pass 使用）。
    void bindFullDeviceView(DeviceGaussianBuffers& out) const;

    /// 按相机视锥从全量压缩出可见子集（写入 out，由 freeDeviceGaussianBuffers 释放）。
    Status filterVisible(const Camera& cam, DeviceGaussianBuffers& out);

    /// 采样估算视锥内高斯数量（诊断）。
    int countFrustumPass(const Camera& cam, int max_samples = 8192) const;

    /// 低分辨率探测 splat 可见数（需传入已配置好的 Rasterizer）。
    int probeVisibleSplats(const Camera& cam, int width, int height, Rasterizer& raster) const;

private:
    struct Impl;
    Impl* impl_;
};

}  // namespace gsplat
