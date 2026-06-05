#pragma once

#include <gsplat_raster/gsplat_raster.h>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace gsplat {
namespace internal {

/// CUDA 光栅引擎：仅渲染外部传入的 GPU 子集（无全量高斯存储）。
class CudaRasterEngine {
public:
    CudaRasterEngine();
    ~CudaRasterEngine();

    CudaRasterEngine(const CudaRasterEngine&) = delete;
    CudaRasterEngine& operator=(const CudaRasterEngine&) = delete;

    void setSettings(const RenderSettings& s);
    int lastVisible() const { return last_visible_; }

    bool renderFromDevice(int w, int h, const float view[16], const float proj[16], const float cam_pos[3],
                          float tan_fovx, float tan_fovy, const DeviceGaussianBuffers& src,
                          std::vector<uint8_t>& rgb);
    bool renderFromDeviceGpuOnly(int w, int h, const float view[16], const float proj[16],
                                 const float cam_pos[3], float tan_fovx, float tan_fovy,
                                 const DeviceGaussianBuffers& src);

    const float* deviceRgbPlanar() const;
    int deviceRgbWidth() const { return last_device_w_; }
    int deviceRgbHeight() const { return last_device_h_; }

private:
    void freeDevice();
    bool renderCore(int w, int h, const float view[16], const float proj[16], const float cam_pos[3],
                    float tan_fovx, float tan_fovy, int P, int D, int M, const float* d_means3D,
                    const float* d_shs, const float* d_colors_precomp, const float* d_opacities,
                    const float* d_scales, const float* d_rotations, std::vector<uint8_t>* rgb);

#ifdef GSPLAT_CUDA_ENABLED
    char* resizeDeviceScratch(char*& buf, size_t& capacity_bytes, size_t required_bytes);
#endif

    RenderSettings settings_;
    float scale_modifier_ = 1.f;
    float background_[3] = {0.02f, 0.02f, 0.06f};
    int last_visible_ = 0;
    int last_device_w_ = 0;
    int last_device_h_ = 0;

    std::vector<float> out_color_;
    std::vector<int> radii_;

#ifdef GSPLAT_CUDA_ENABLED
    float* d_background_ = nullptr;
    float* d_view_ = nullptr;
    float* d_proj_ = nullptr;
    float* d_cam_pos_ = nullptr;
    float* d_out_color_ = nullptr;
    int* d_radii_ = nullptr;
    char* d_geom_ = nullptr;
    char* d_binning_ = nullptr;
    char* d_image_ = nullptr;
    size_t d_geom_cap_ = 0;
    size_t d_binning_cap_ = 0;
    size_t d_image_cap_ = 0;
    size_t d_out_color_floats_ = 0;
    size_t d_radii_count_ = 0;
#endif
};

}  // namespace internal
}  // namespace gsplat
