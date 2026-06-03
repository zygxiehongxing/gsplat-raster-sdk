#pragma once

#include <gsplat_raster/gsplat_raster.h>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace gsplat {
namespace internal {

/// CUDA 光栅引擎：封装高斯上传、显存管理与单帧渲染。
class CudaRasterEngine {
public:
    /// 初始化 CUDA 设备与内部状态。
    CudaRasterEngine();
    /// 释放所有设备资源。
    ~CudaRasterEngine();

    CudaRasterEngine(const CudaRasterEngine&) = delete;
    CudaRasterEngine& operator=(const CudaRasterEngine&) = delete;

    /// 写入渲染参数。
    void setSettings(const RenderSettings& s);
    /// 上传高斯数组到 CPU/GPU 缓冲。
    bool upload(const std::vector<Gaussian>& gaussians);

    /// 获取已上传的 splat 数量。
    int numSplats() const { return num_splats_; }
    /// 获取最近一次渲染的可见 splat 数量。
    int lastVisible() const { return last_visible_; }

    /// 执行一次渲染。
    bool render(int w, int h, const float view[16], const float proj[16], const float cam_pos[3],
                float tan_fovx, float tan_fovy, std::vector<uint8_t>& rgb);

    /// 采样估算通过视锥的 splat 数量。
    int countFrustumPass(const float view[16], const float proj[16], int max_samples) const;

    /// 低分辨率探测可见 splat 数量。
    int probeVisibleSplats(int w, int h, const float view[16], const float proj[16],
                           const float cam_pos[3], float tan_fovx, float tan_fovy);

private:
    /// 将输入高斯展开为内部 SoA 缓冲。
    void uploadOne(const std::vector<Gaussian>& gaussians);
    /// 释放设备缓冲。
    void freeDevice();
    /// 把当前 CPU 缓冲复制到 GPU。
    bool uploadGaussiansToDevice();

#ifdef GSPLAT_CUDA_ENABLED
    /// 按需扩容 CUDA scratch 缓冲。
    char* resizeDeviceScratch(char*& buf, size_t& capacity_bytes, size_t required_bytes);
#endif

    /// 当前生效的渲染参数。
    RenderSettings settings_;
    /// splat 总数。
    int num_splats_ = 0;
    /// SH 阶数。
    int sh_degree_ = 0;
    /// 每个点的 SH 系数数。
    int sh_coeffs_ = 0;
    /// 当前 scale_modifier。
    float scale_modifier_ = 1.f;
    /// 背景色缓存。
    float background_[3] = {0.02f, 0.02f, 0.06f};
    /// 最近一次可见数量。
    int last_visible_ = 0;

    /// CPU 端 SoA 数据缓存。
    std::vector<float> means3D_;
    std::vector<float> scales_;
    std::vector<float> rotations_;
    std::vector<float> opacities_;
    std::vector<float> shs_;
    std::vector<float> colors_precomp_;
    std::vector<float> out_color_;
    std::vector<int> radii_;

#ifdef GSPLAT_CUDA_ENABLED
    /// GPU 端核心数据与中间缓冲。
    float* d_means3D_ = nullptr;
    float* d_scales_ = nullptr;
    float* d_rotations_ = nullptr;
    float* d_opacities_ = nullptr;
    float* d_shs_ = nullptr;
    float* d_colors_precomp_ = nullptr;
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
