#pragma once

#include <gsplat_raster/gsplat_raster.h>

#include <vector>

namespace gsplat {
namespace internal {

class CudaRasterEngine;

/// GPU 全量高斯池（上传 + 视锥筛选）；与光栅引擎分离。
class CudaGaussianPool {
public:
    CudaGaussianPool();
    ~CudaGaussianPool();

    CudaGaussianPool(const CudaGaussianPool&) = delete;
    CudaGaussianPool& operator=(const CudaGaussianPool&) = delete;

    void setDcOnly(bool dc_only) { dc_only_ = dc_only; }
    bool dcOnly() const { return dc_only_; }

    bool upload(const std::vector<Gaussian>& gaussians);
    int numSplats() const { return num_splats_; }

    bool filterVisible(const float view[16], const float proj[16], DeviceGaussianBuffers& out);

    int countFrustumPass(const float view[16], const float proj[16], int max_samples) const;

    int probeVisibleSplats(int w, int h, const float view[16], const float proj[16], const float cam_pos[3],
                           float tan_fovx, float tan_fovy, CudaRasterEngine& raster,
                           const RenderSettings& raster_settings);

    void bindFullDeviceView(DeviceGaussianBuffers& out) const;

private:
    void uploadOne(const std::vector<Gaussian>& cloud);
    bool uploadGaussiansToDevice();
    void freeDevice();

    bool dc_only_ = false;
    int num_splats_ = 0;
    int sh_degree_ = 0;
    int sh_coeffs_ = 0;

    std::vector<float> means3D_;
    std::vector<float> scales_;
    std::vector<float> rotations_;
    std::vector<float> opacities_;
    std::vector<float> shs_;
    std::vector<float> colors_precomp_;

#ifdef GSPLAT_CUDA_ENABLED
    float* d_means3D_ = nullptr;
    float* d_scales_ = nullptr;
    float* d_rotations_ = nullptr;
    float* d_opacities_ = nullptr;
    float* d_shs_ = nullptr;
    float* d_colors_precomp_ = nullptr;
#endif
};

}  // namespace internal
}  // namespace gsplat
