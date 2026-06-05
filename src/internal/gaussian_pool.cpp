#include "gaussian_pool.h"

#include "device_filter.h"
#include "rasterizer_impl.h"
#include "../gaussian_types.h"

#include <algorithm>
#include <cmath>
#include <iostream>

#ifdef GSPLAT_CUDA_ENABLED
#include <cuda_runtime.h>
#endif

namespace gsplat {
namespace internal {

namespace {
constexpr float kShC0 = 0.28209479177387814f;

#ifdef GSPLAT_CUDA_ENABLED
bool cudaOk(cudaError_t err, const char* what) {
    if (err == cudaSuccess) return true;
    std::cerr << "CUDA " << what << ": " << cudaGetErrorString(err) << "\n";
    return false;
}

template <typename T>
void freeDev(T*& p) {
    if (p) {
        cudaFree(p);
        p = nullptr;
    }
}
#endif
}  // namespace

CudaGaussianPool::CudaGaussianPool() {
#ifdef GSPLAT_CUDA_ENABLED
    cudaSetDevice(0);
    cudaFree(0);
#endif
}

CudaGaussianPool::~CudaGaussianPool() { freeDevice(); }

void CudaGaussianPool::freeDevice() {
#ifdef GSPLAT_CUDA_ENABLED
    freeDev(d_means3D_);
    freeDev(d_scales_);
    freeDev(d_rotations_);
    freeDev(d_opacities_);
    freeDev(d_shs_);
    freeDev(d_colors_precomp_);
#endif
}

bool CudaGaussianPool::upload(const std::vector<Gaussian>& gaussians) {
    uploadOne(gaussians);
    if (num_splats_ <= 0) return false;
    return uploadGaussiansToDevice();
}

bool CudaGaussianPool::uploadGaussiansToDevice() {
#ifdef GSPLAT_CUDA_ENABLED
    auto up = [](float*& dev, const std::vector<float>& host) -> bool {
        freeDev(dev);
        if (host.empty()) return true;
        const size_t bytes = host.size() * sizeof(float);
        if (!cudaOk(cudaMalloc(&dev, bytes), "malloc gaussian attr")) return false;
        return cudaOk(cudaMemcpy(dev, host.data(), bytes, cudaMemcpyHostToDevice), "H2D gaussian attr");
    };

    if (!up(d_means3D_, means3D_)) return false;
    if (!up(d_scales_, scales_)) return false;
    if (!up(d_rotations_, rotations_)) return false;
    if (!up(d_opacities_, opacities_)) return false;

    freeDev(d_shs_);
    freeDev(d_colors_precomp_);
    if (sh_degree_ > 0) {
        if (!up(d_shs_, shs_)) return false;
    } else {
        if (!up(d_colors_precomp_, colors_precomp_)) return false;
    }
    return true;
#else
    return num_splats_ > 0;
#endif
}

void CudaGaussianPool::uploadOne(const std::vector<Gaussian>& cloud) {
    num_splats_ = static_cast<int>(cloud.size());
    if (num_splats_ <= 0) return;

    bool any_rest = false;
    if (!dc_only_) {
        for (const auto& g : cloud) {
            if (g.has_sh_rest) {
                any_rest = true;
                break;
            }
        }
    }
    sh_degree_ = any_rest ? 3 : 0;
    sh_coeffs_ = any_rest ? 16 : 0;

    means3D_.assign(static_cast<size_t>(num_splats_) * 3, 0.f);
    scales_.assign(static_cast<size_t>(num_splats_) * 3, 0.f);
    rotations_.assign(static_cast<size_t>(num_splats_) * 4, 0.f);
    opacities_.resize(static_cast<size_t>(num_splats_));
    shs_.assign(static_cast<size_t>(num_splats_) * static_cast<size_t>(sh_coeffs_) * 3, 0.f);
    colors_precomp_.assign(static_cast<size_t>(num_splats_) * 3, 0.f);

    auto finite = [](float v) { return std::isfinite(v) ? v : 0.f; };

    bool quat_wxyz = true;
    {
        int score_wxyz = 0;
        int score_xyzw = 0;
        const int sample_n = std::min(num_splats_, 2048);
        const int step = std::max(1, num_splats_ / std::max(1, sample_n));
        for (int i = 0; i < num_splats_; i += step) {
            const Gaussian& g = cloud[static_cast<size_t>(i)];
            const float a0 = std::abs(finite(g.rot[0]));
            const float a3 = std::abs(finite(g.rot[3]));
            if (a0 >= a3) ++score_wxyz;
            if (a3 >= a0) ++score_xyzw;
        }
        quat_wxyz = score_wxyz >= score_xyzw;
        std::cout << "Quat layout: " << (quat_wxyz ? "wxyz" : "xyzw")
                  << " (score " << score_wxyz << "/" << score_xyzw << ")\n";
    }

    for (int i = 0; i < num_splats_; ++i) {
        const Gaussian& g = cloud[static_cast<size_t>(i)];
        means3D_[static_cast<size_t>(i) * 3 + 0] = finite(g.x);
        means3D_[static_cast<size_t>(i) * 3 + 1] = finite(g.y);
        means3D_[static_cast<size_t>(i) * 3 + 2] = finite(g.z);

        opacities_[static_cast<size_t>(i)] = clamp01(sigmoid(finite(g.opacity_logit)));
        const float sx = std::clamp(finite(g.scale_log[0]), -20.f, 0.0f);
        const float sy = std::clamp(finite(g.scale_log[1]), -20.f, 0.0f);
        const float sz = std::clamp(finite(g.scale_log[2]), -20.f, 0.0f);
        scales_[static_cast<size_t>(i) * 3 + 0] = std::exp(sx);
        scales_[static_cast<size_t>(i) * 3 + 1] = std::exp(sy);
        scales_[static_cast<size_t>(i) * 3 + 2] = std::exp(sz);

        float qw = 1.f, qx = 0.f, qy = 0.f, qz = 0.f;
        if (quat_wxyz) {
            qw = g.rot[0];
            qx = g.rot[1];
            qy = g.rot[2];
            qz = g.rot[3];
        } else {
            qx = g.rot[0];
            qy = g.rot[1];
            qz = g.rot[2];
            qw = g.rot[3];
        }
        const float qlen = std::sqrt(qw * qw + qx * qx + qy * qy + qz * qz);
        if (qlen > 1e-8f) {
            const float inv = 1.f / qlen;
            qw *= inv;
            qx *= inv;
            qy *= inv;
            qz *= inv;
        } else {
            qw = 1.f;
            qx = qy = qz = 0.f;
        }
        rotations_[static_cast<size_t>(i) * 4 + 0] = qw;
        rotations_[static_cast<size_t>(i) * 4 + 1] = qx;
        rotations_[static_cast<size_t>(i) * 4 + 2] = qy;
        rotations_[static_cast<size_t>(i) * 4 + 3] = qz;

        if (sh_degree_ > 0) {
            for (int c = 0; c < sh_coeffs_; ++c) {
                const size_t dst =
                    static_cast<size_t>(i) * static_cast<size_t>(sh_coeffs_) * 3 + static_cast<size_t>(c) * 3;
                const int src = (c == 0) ? 0 : (3 + (c - 1) * 3);
                shs_[dst + 0] = finite(g.sh[src + 0]);
                shs_[dst + 1] = finite(g.sh[src + 1]);
                shs_[dst + 2] = finite(g.sh[src + 2]);
            }
        } else {
            colors_precomp_[static_cast<size_t>(i) * 3 + 0] = clamp01(0.5f + kShC0 * finite(g.sh[0]));
            colors_precomp_[static_cast<size_t>(i) * 3 + 1] = clamp01(0.5f + kShC0 * finite(g.sh[1]));
            colors_precomp_[static_cast<size_t>(i) * 3 + 2] = clamp01(0.5f + kShC0 * finite(g.sh[2]));
        }
    }
}

void CudaGaussianPool::bindFullDeviceView(DeviceGaussianBuffers& out) const {
    out.count = num_splats_;
    out.sh_degree = sh_degree_;
    out.sh_coeffs = sh_coeffs_;
#ifdef GSPLAT_CUDA_ENABLED
    out.means3D = d_means3D_;
    out.scales = d_scales_;
    out.rotations = d_rotations_;
    out.opacities = d_opacities_;
    out.shs = sh_degree_ > 0 ? d_shs_ : nullptr;
    out.colors_precomp = sh_degree_ > 0 ? nullptr : d_colors_precomp_;
#else
    out.means3D = nullptr;
    out.scales = nullptr;
    out.rotations = nullptr;
    out.opacities = nullptr;
    out.shs = nullptr;
    out.colors_precomp = nullptr;
#endif
}

int CudaGaussianPool::probeVisibleSplats(int w, int h, const float view[16], const float proj[16],
                                         const float cam_pos[3], float tan_fovx, float tan_fovy,
                                         CudaRasterEngine& raster, const RenderSettings& raster_settings) {
    RenderSettings boosted = raster_settings;
    boosted.scale_modifier = std::max(boosted.scale_modifier, 32.f);
    raster.setSettings(boosted);
    DeviceGaussianBuffers full;
    bindFullDeviceView(full);
    std::vector<uint8_t> rgb;
    raster.renderFromDevice(w, h, view, proj, cam_pos, tan_fovx, tan_fovy, full, rgb);
    const int vis = raster.lastVisible();
    raster.setSettings(raster_settings);
    return vis;
}

int CudaGaussianPool::countFrustumPass(const float view[16], const float proj[16], int max_samples) const {
    if (num_splats_ <= 0) return 0;
    const int step = std::max(1, num_splats_ / std::max(1, max_samples));
    int pass = 0;
    for (int i = 0; i < num_splats_; i += step) {
        const size_t o = static_cast<size_t>(i) * 3;
        const float x = means3D_[o], y = means3D_[o + 1], z = means3D_[o + 2];
        const float view_z = view[2] * x + view[6] * y + view[10] * z + view[14];
        if (view_z >= -1e-4f) continue;
        const float hx = proj[0] * x + proj[4] * y + proj[8] * z + proj[12];
        const float hy = proj[1] * x + proj[5] * y + proj[9] * z + proj[13];
        const float hz = proj[2] * x + proj[6] * y + proj[10] * z + proj[14];
        const float hw = proj[3] * x + proj[7] * y + proj[11] * z + proj[15];
        if (hw <= 1e-6f) continue;
        const float invw = 1.f / hw;
        const float nx = hx * invw, ny = hy * invw, nz = hz * invw;
        if (nx >= -1.3f && nx <= 1.3f && ny >= -1.3f && ny <= 1.3f && nz >= -1.0f && nz <= 1.0f) {
            ++pass;
        }
    }
    return pass;
}

bool CudaGaussianPool::filterVisible(const float view[16], const float proj[16], DeviceGaussianBuffers& out) {
#ifndef GSPLAT_CUDA_ENABLED
    (void)view;
    (void)proj;
    (void)out;
    return false;
#else
    if (num_splats_ <= 0) return false;
    if (!d_means3D_ && !uploadGaussiansToDevice()) return false;
    return filterVisibleToDevice(num_splats_, sh_degree_, sh_coeffs_, d_means3D_, d_scales_, d_rotations_,
                                 d_opacities_, sh_degree_ > 0 ? d_shs_ : nullptr,
                                 sh_degree_ > 0 ? nullptr : d_colors_precomp_, view, proj, out);
#endif
}

}  // namespace internal
}  // namespace gsplat
