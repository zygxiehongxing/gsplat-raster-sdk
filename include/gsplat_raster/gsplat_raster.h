#pragma once

/**
 * gsplat-raster-sdk — Inria diff-gaussian-rasterization wrapper (no OSG).
 * Input: Gaussian arrays (or PLY) + camera.
 * Output: RGB buffer or PNG file.
 */

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace gsplat {

constexpr int kShCoeffs = 48;

struct PlyLoadOptions {
    size_t max_points = 0;
    float min_opacity = 0.005f;
    bool drop_sh_rest = false;
    /** When max_points > 0: stride-read entire PLY (fast scan). false = reservoir sample (slower, uniform). */
    bool stride_subsample = false;
};

struct Gaussian {
    float x = 0, y = 0, z = 0;
    float opacity_logit = 0.f;
    float scale_log[3] = {0, 0, 0};
    float rot[4] = {1, 0, 0, 0};  // w,x,y,z
    float sh[kShCoeffs] = {};
    bool has_sh_rest = false;
};

/** Background RGB in [0,1]. */
struct RenderSettings {
    float background[3] = {0.02f, 0.02f, 0.06f};
    float scale_modifier = 1.f;
    bool dc_only = false;
};

/**
 * Camera for Inria rasterizer.
 * Use buildCameraLookAt() or fill view/proj/cam_pos/tan_fov yourself (training export).
 *
 * view, proj: column-major 4x4 (same packing as Inria CUDA kernels).
 * cam_pos: world-space camera position (3 floats).
 */
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
    ErrorPlyLoad,
    ErrorPngWrite,
};

const char* statusString(Status s);

/** True if CUDA device and CudaRasterizer were linked. */
bool isCudaAvailable();

bool loadGaussianPly(const std::string& path, std::vector<Gaussian>& out,
                     const PlyLoadOptions& opts = {});

class Rasterizer {
public:
    Rasterizer();
    ~Rasterizer();

    Rasterizer(const Rasterizer&) = delete;
    Rasterizer& operator=(const Rasterizer&) = delete;

    void setSettings(const RenderSettings& s);
    RenderSettings settings() const;

    Status setGaussians(const std::vector<Gaussian>& gaussians);
    Status loadPly(const std::string& path, const PlyLoadOptions& opts = {});

    int numGaussians() const;
    int lastVisibleCount() const;

    /**
     * Render to RGB8 row-major, size = width * height * 3.
     * Returns Ok even when 0 splats visible (background fill).
     */
    Status render(const Camera& cam, int width, int height, std::vector<uint8_t>& out_rgb);

    /** render() then write PNG (8-bit RGB). */
    Status renderToPng(const Camera& cam, int width, int height, const std::string& png_path);

    /** Sample how many gaussians pass Inria frustum (approximate). */
    int countFrustumPass(const Camera& cam, int max_samples = 8192) const;

    /** Quick low-res render to count visible splats (for matrix tuning). */
    int probeVisibleSplats(const Camera& cam, int width = 640, int height = 360) const;

private:
    struct Impl;
    Impl* impl_;
};

/**
 * Build Inria-compatible view/proj (official V*P_inria + auto matrix pick).
 * If raster_for_pick has gaussians, probes packing modes and picks best NDC pass.
 */
void buildCameraLookAt(const double eye[3], const double center[3], const double up[3],
                       double fov_y_deg, double aspect, double znear, double zfar,
                       const double ref_center[3], Camera& out,
                       Rasterizer* raster_for_pick = nullptr);

}  // namespace gsplat
