#pragma once

/**
 * gsplat-raster-sdk — Inria diff-gaussian-rasterization wrapper (no OSG).
 * Input: Gaussian arrays + camera.
 * Output: RGB buffer or PNG file.
 */

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace gsplat {

/// 每个高斯的 SH 系数总长度（DC + 其余阶）。
constexpr int kShCoeffs = 48;

/// SDK 使用的单个高斯点数据结构。
struct Gaussian {
    float x = 0, y = 0, z = 0;
    float opacity_logit = 0.f;
    float scale_log[3] = {0, 0, 0};
    float rot[4] = {1, 0, 0, 0};  // w,x,y,z
    float sh[kShCoeffs] = {};
    bool has_sh_rest = false;
};

/// 渲染参数（背景色、缩放倍率、是否仅用 DC 颜色）。
struct RenderSettings {
    float background[3] = {0.02f, 0.02f, 0.06f};
    float scale_modifier = 1.f;
    bool dc_only = false;
};

/// Inria 光栅器相机输入。
/// - view/proj: 列主序 4x4（与 CUDA kernel 一致）
/// - cam_pos: 世界坐标相机位置
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

/// 状态码转可读字符串。
const char* statusString(Status s);

/// 检查 CUDA 设备与光栅内核是否可用。
bool isCudaAvailable();

/// 重置 CUDA 上下文（CLI 多次运行时可释放显存状态）。
void resetCudaDevice();

/// SDK 光栅器封装类：管理高斯上传与 CUDA 渲染调用。
class Rasterizer {
public:
    /// 构造光栅器实例。
    Rasterizer();
    /// 析构并释放关联资源。
    ~Rasterizer();

    Rasterizer(const Rasterizer&) = delete;
    Rasterizer& operator=(const Rasterizer&) = delete;

    /// 设置渲染参数（背景、scale_modifier、dc_only）。
    void setSettings(const RenderSettings& s);
    /// 获取当前渲染参数副本。
    RenderSettings settings() const;

    /// 上传高斯数组到 GPU。
    Status setGaussians(std::vector<Gaussian> gaussians);
    /// 当前已上传高斯数量。
    int numGaussians() const;
    /// 最近一次渲染统计到的可见高斯数量。
    int lastVisibleCount() const;

    /// 渲染到 RGB8 缓冲区（行优先）。
    Status render(const Camera& cam, int width, int height, std::vector<uint8_t>& out_rgb);

    /// 渲染并直接写出 PNG（8-bit RGB）。
    Status renderToPng(const Camera& cam, int width, int height, const std::string& png_path);

    /// 估算有多少高斯通过当前相机视锥（采样近似）。
    int countFrustumPass(const Camera& cam, int max_samples = 8192) const;

    /// 低分辨率快速渲染并返回可见 splat 数量（调参与诊断用）。
    int probeVisibleSplats(const Camera& cam, int width = 640, int height = 360) const;

private:
    struct Impl;
    Impl* impl_;
};

/// 通过 eye/center/up 构建 Inria 约定相机矩阵。
void buildCameraLookAt(const double eye[3], const double center[3], const double up[3],
                       double fov_y_deg, double aspect, double znear, double zfar,
                       const double ref_center[3], Camera& out,
                       Rasterizer* raster_for_pick = nullptr);

/// 将 OSG 的 view/proj（row-major）转换为 Inria 相机输入。
void buildCameraFromOsg(const double view_osg[16], const double proj_osg[16],
                        const double ref_center[3], Camera& out);

}  // namespace gsplat
