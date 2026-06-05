#pragma once

/**
 * gsplat-raster-sdk — Inria diff-gaussian-rasterization wrapper (no OSG).
 *
 * 职责分离：
 * - App：GaussianDevicePool 上传全量、按相机筛出 DeviceGaussianBuffers
 * - SDK Rasterizer：仅消费 App 提供的 GPU 子集 + Camera 做光栅
 */

#include <gsplat_raster/screen_gaussian_pixel.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace gsplat {

constexpr int kShCoeffs = 48;

struct Gaussian {
    float x = 0, y = 0, z = 0;
    float opacity_logit = 0.f;
    float scale_log[3] = {0, 0, 0};
    float rot[4] = {1, 0, 0, 0};
    float sh[kShCoeffs] = {};
    bool has_sh_rest = false;
};

struct RenderSettings {
    float background[3] = {0.02f, 0.02f, 0.06f};
    float scale_modifier = 1.f;
    bool dc_only = false;
};

/// App 筛选后的 GPU 高斯子集（Rasterizer 唯一几何输入）。
struct DeviceGaussianBuffers {
    float* means3D = nullptr;
    float* scales = nullptr;
    float* rotations = nullptr;
    float* opacities = nullptr;
    float* shs = nullptr;
    float* colors_precomp = nullptr;
    int count = 0;
    int sh_degree = 0;
    int sh_coeffs = 0;
};

void freeDeviceGaussianBuffers(DeviceGaussianBuffers& buffers);

/// view/proj：GLSL 行主序（与 vec4(p,1)*view*proj 一致；proj 仅为 P）。
struct Camera {
    float view[16] = {};
    float proj[16] = {};
    float cam_pos[3] = {};
    float tan_fovx = 0.f;
    float tan_fovy = 0.f;
};

enum class Status {
    Ok = 0,
    ErrorNoCuda,
    ErrorNoGaussians,
    ErrorInvalidArgs,
    ErrorRenderFailed,
    ErrorPngWrite,
};

const char* statusString(Status s);
bool isCudaAvailable();
void resetCudaDevice();

/// 将 RGB8（行优先）写入 PNG。
bool writeRgbPng(const std::string& path, int width, int height, const std::vector<uint8_t>& rgb);

/// GL SSBO 屏格（cells[grid_w*grid_h]）→ SDK SoA DeviceGaussianBuffers（compact 后供 EWA 光栅）。
Status unpackScreenSsboToDevice(const GaussianPixelGpu* d_cells, int grid_w, int grid_h,
                                DeviceGaussianBuffers& out, int& compact_out,
                                uint32_t expected_frame_id = 0xFFFFFFFFu);

/// 调试图（与 SSBO 同尺寸）：用 Camera.proj（row V*P）将每格 mean 投影到像素并标红。
Status renderSsboMeanProjectMap(const Camera& cam, int width, int height, const GaussianPixelGpu* d_cells,
                                std::vector<uint8_t>& out_rgb, int& marked_pixels,
                                uint32_t expected_frame_id = 0xFFFFFFFFu);

/// 调试图：用 meta.view_glsl/proj_glsl 投影 mean（与 GL 点云一致）。
Status renderSsboMeanProjectMapGlsl(const ScreenCacheMetaGpu& meta, int width, int height,
                                    const GaussianPixelGpu* d_cells, std::vector<uint8_t>& out_rgb,
                                    int& marked_pixels, uint32_t expected_frame_id = 0xFFFFFFFFu,
                                    bool image_top_origin = true);

/// 颜色 SSBO：用 meta.view_glsl/proj_glsl（与 GL 片元路径一致）将 xyz 投到 2D 并写 rgba。
Status renderSsboColorCellsProjectMapGlsl(const ScreenCacheMetaGpu& meta, int width, int height,
                                          const ScreenColorCellGpu* d_cells, std::vector<uint8_t>& out_rgb,
                                          int& marked_pixels, bool image_top_origin = true);

class Rasterizer;

namespace internal {
class CudaRasterEngine;
CudaRasterEngine& rasterEngine(Rasterizer& raster);
RenderSettings rasterSettings(const Rasterizer& raster);
}  // namespace internal

/// 纯光栅器：不持有、不上传全量高斯。
class Rasterizer {
    friend internal::CudaRasterEngine& internal::rasterEngine(Rasterizer&);
    friend RenderSettings internal::rasterSettings(const Rasterizer&);

public:
    Rasterizer();
    ~Rasterizer();

    Rasterizer(const Rasterizer&) = delete;
    Rasterizer& operator=(const Rasterizer&) = delete;

    void setSettings(const RenderSettings& s);
    RenderSettings settings() const;

    /// 使用 App 提供的 GPU 子集 + 相机渲染 RGB8（行优先，含 D2H）。
    Status render(const Camera& cam, int width, int height, const DeviceGaussianBuffers& gaussians,
                  std::vector<uint8_t>& out_rgb);

    /// GPU 光栅，不回读 CPU；结果通过 deviceRgbPlanar() 读取直至下次 render/renderDevice。
    Status renderDevice(const Camera& cam, int width, int height, const DeviceGaussianBuffers& gaussians);

    /// planar float RGB（CHW），与 Inria 光栅输出布局一致。
    const float* deviceRgbPlanar() const;
    int deviceRgbWidth() const;
    int deviceRgbHeight() const;

    Status renderToPng(const Camera& cam, int width, int height, const DeviceGaussianBuffers& gaussians,
                       const std::string& png_path);

    /// 最近一次 render 中 radii>0 的数量（子集上的 splat 统计）。
    int lastVisibleCount() const;

private:
    struct Impl;
    Impl* impl_;
};

void buildCameraLookAt(const double eye[3], const double center[3], const double up[3], double fov_y_deg,
                       double aspect, double znear, double zfar, const double ref_center[3], Camera& out);

void buildCameraFromOsg(const double view_osg[16], const double proj_osg[16], const double ref_center[3],
                      Camera& out, int* mat_mode_out = nullptr, const float* probe_xyz = nullptr,
                      int probe_count = 0);

/// 由 SSBO meta 生成光栅相机：直接拷贝 view_glsl / proj_glsl。
void buildCameraFromSsboMeta(const ScreenCacheMetaGpu& meta, Camera& out);

}  // namespace gsplat
