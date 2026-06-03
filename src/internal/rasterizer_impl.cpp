#include "rasterizer_impl.h"

#include "../gaussian_types.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>

#ifdef GSPLAT_CUDA_ENABLED
#include <cuda_runtime.h>
#include "rasterizer.h"
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

CudaRasterEngine::CudaRasterEngine() {
#ifdef GSPLAT_CUDA_ENABLED
    cudaSetDevice(0);
    cudaFree(0);
#endif
}

CudaRasterEngine::~CudaRasterEngine() { freeDevice(); }

void CudaRasterEngine::freeDevice() {
#ifdef GSPLAT_CUDA_ENABLED
    freeDev(d_means3D_);
    freeDev(d_scales_);
    freeDev(d_rotations_);
    freeDev(d_opacities_);
    freeDev(d_shs_);
    freeDev(d_colors_precomp_);
    freeDev(d_background_);
    freeDev(d_view_);
    freeDev(d_proj_);
    freeDev(d_cam_pos_);
    freeDev(d_out_color_);
    freeDev(d_radii_);
    freeDev(d_geom_);
    freeDev(d_binning_);
    freeDev(d_image_);
    d_geom_cap_ = 0;
    d_binning_cap_ = 0;
    d_image_cap_ = 0;
    d_out_color_floats_ = 0;
    d_radii_count_ = 0;
#endif
}

#ifdef GSPLAT_CUDA_ENABLED
char* CudaRasterEngine::resizeDeviceScratch(char*& buf, size_t& capacity_bytes,
                                            size_t required_bytes) {
    if (required_bytes > capacity_bytes) {
        freeDev(buf);
        if (!cudaOk(cudaMalloc(&buf, required_bytes), "malloc scratch")) {
            capacity_bytes = 0;
            return nullptr;
        }
        capacity_bytes = required_bytes;
    }
    return buf;
}
#endif

void CudaRasterEngine::setSettings(const RenderSettings& s) {
    settings_ = s;
    scale_modifier_ = s.scale_modifier;
    background_[0] = s.background[0];
    background_[1] = s.background[1];
    background_[2] = s.background[2];
}

bool CudaRasterEngine::upload(const std::vector<Gaussian>& gaussians) {
    uploadOne(gaussians);
    if (num_splats_ <= 0) return false;
    return uploadGaussiansToDevice();
}

bool CudaRasterEngine::uploadGaussiansToDevice() {
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

    freeDev(d_background_);
    if (!cudaOk(cudaMalloc(&d_background_, 3 * sizeof(float)), "malloc background")) return false;
    if (!cudaOk(cudaMemcpy(d_background_, background_, 3 * sizeof(float), cudaMemcpyHostToDevice),
                "H2D background")) {
        return false;
    }

    if (!d_view_ && !cudaOk(cudaMalloc(&d_view_, 16 * sizeof(float)), "malloc view")) return false;
    if (!d_proj_ && !cudaOk(cudaMalloc(&d_proj_, 16 * sizeof(float)), "malloc proj")) return false;
    if (!d_cam_pos_ && !cudaOk(cudaMalloc(&d_cam_pos_, 3 * sizeof(float)), "malloc cam_pos")) {
        return false;
    }
    return true;
#else
    return num_splats_ > 0;
#endif
}

void CudaRasterEngine::uploadOne(const std::vector<Gaussian>& cloud) {
    num_splats_ = static_cast<int>(cloud.size());
    if (num_splats_ <= 0) return;

    bool any_rest = false;
    if (!settings_.dc_only) {
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

    // Heuristic: choose quaternion layout that looks like training export.
    // If |rot_0| dominates for most samples, assume wxyz; otherwise assume xyzw.
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
        // Clamp extreme large splats from noisy/exported PLYs.
        // Use a tighter upper bound to preserve scene details (avoid over-blurry large splats).
        const float sx = std::clamp(finite(g.scale_log[0]), -20.f, -4.0f);
        const float sy = std::clamp(finite(g.scale_log[1]), -20.f, -4.0f);
        const float sz = std::clamp(finite(g.scale_log[2]), -20.f, -4.0f);
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

int CudaRasterEngine::probeVisibleSplats(int w, int h, const float view[16], const float proj[16],
                                         const float cam_pos[3], float tan_fovx, float tan_fovy) {
    const RenderSettings saved = settings_;
    RenderSettings boosted = saved;
    boosted.scale_modifier = std::max(boosted.scale_modifier, 32.f);
    setSettings(boosted);
    std::vector<uint8_t> rgb;
    render(w, h, view, proj, cam_pos, tan_fovx, tan_fovy, rgb);
    const int vis = last_visible_;
    setSettings(saved);
    return vis;
}

int CudaRasterEngine::countFrustumPass(const float view[16], const float proj[16],
                                       int max_samples) const {
    if (num_splats_ <= 0) return 0;
    const int step = std::max(1, num_splats_ / std::max(1, max_samples));
    int pass = 0;
    for (int i = 0; i < num_splats_; i += step) {
        const size_t o = static_cast<size_t>(i) * 3;
        const float x = means3D_[o], y = means3D_[o + 1], z = means3D_[o + 2];
        const float pz = view[2] * x + view[6] * y + view[10] * z + view[14];
        if (pz <= 0.2f) continue;
        const float hx = proj[0] * x + proj[4] * y + proj[8] * z + proj[12];
        const float hy = proj[1] * x + proj[5] * y + proj[9] * z + proj[13];
        const float hw = proj[3] * x + proj[7] * y + proj[11] * z + proj[15];
        if (hw <= 1e-6f) continue;
        const float invw = 1.f / hw;
        const float nx = hx * invw, ny = hy * invw;
        if (nx >= -1.3f && nx <= 1.3f && ny >= -1.3f && ny <= 1.3f) ++pass;
    }
    return pass;
}

bool CudaRasterEngine::render(int width, int height, const float view[16], const float proj[16],
                              const float cam_pos[3], float tan_fovx, float tan_fovy,
                              std::vector<uint8_t>& out_rgb) {
    if (num_splats_ <= 0 || width <= 0 || height <= 0) return false;

#ifndef GSPLAT_CUDA_ENABLED
    (void)view;
    (void)proj;
    (void)cam_pos;
    (void)tan_fovx;
    (void)tan_fovy;
    (void)out_rgb;
    return false;
#else
    if (!d_means3D_ && !uploadGaussiansToDevice()) return false;

    const int W = width, H = height, P = num_splats_, D = sh_degree_, M = sh_coeffs_;
    const size_t out_floats = static_cast<size_t>(3 * H * W);

    if (out_floats > d_out_color_floats_) {
        freeDev(d_out_color_);
        if (!cudaOk(cudaMalloc(&d_out_color_, out_floats * sizeof(float)), "malloc out_color")) {
            return false;
        }
        d_out_color_floats_ = out_floats;
    }
    if (static_cast<size_t>(P) > d_radii_count_) {
        freeDev(d_radii_);
        if (!cudaOk(cudaMalloc(&d_radii_, static_cast<size_t>(P) * sizeof(int)), "malloc radii")) {
            return false;
        }
        d_radii_count_ = static_cast<size_t>(P);
    }

    if (!cudaOk(cudaMemcpy(d_view_, view, 16 * sizeof(float), cudaMemcpyHostToDevice), "H2D view")) {
        return false;
    }
    if (!cudaOk(cudaMemcpy(d_proj_, proj, 16 * sizeof(float), cudaMemcpyHostToDevice), "H2D proj")) {
        return false;
    }
    if (!cudaOk(cudaMemcpy(d_cam_pos_, cam_pos, 3 * sizeof(float), cudaMemcpyHostToDevice), "H2D cam")) {
        return false;
    }
    if (!cudaOk(cudaMemcpy(d_background_, background_, 3 * sizeof(float), cudaMemcpyHostToDevice),
                "H2D background")) {
        return false;
    }

    int rendered = 0;
    float mod = scale_modifier_;
    last_visible_ = 0;
    radii_.assign(static_cast<size_t>(P), 0);

    for (int attempt = 0; attempt < 3; ++attempt) {
        try {
            rendered = CudaRasterizer::Rasterizer::forward(
                [&](size_t N) { return resizeDeviceScratch(d_geom_, d_geom_cap_, N); },
                [&](size_t N) { return resizeDeviceScratch(d_binning_, d_binning_cap_, N); },
                [&](size_t N) { return resizeDeviceScratch(d_image_, d_image_cap_, N); },
                P, D, M, d_background_, W, H, d_means3D_, sh_degree_ > 0 ? d_shs_ : nullptr,
                sh_degree_ > 0 ? nullptr : d_colors_precomp_, d_opacities_, d_scales_, mod,
                d_rotations_, nullptr, d_view_, d_proj_, d_cam_pos_, tan_fovx, tan_fovy, false,
                d_out_color_, d_radii_, true);
        } catch (const std::exception& e) {
            std::cerr << "CUDA rasterizer: " << e.what() << " (scale_modifier=" << mod << ")\n";
            rendered = 0;
        }

        if (!cudaOk(cudaMemcpy(radii_.data(), d_radii_, static_cast<size_t>(P) * sizeof(int),
                               cudaMemcpyDeviceToHost),
                    "D2H radii")) {
            return false;
        }

        last_visible_ = 0;
        for (int r : radii_) {
            if (r > 0) ++last_visible_;
        }
        if (last_visible_ > 0) {
            break;
        }
        mod *= 3.f;
    }

    out_rgb.resize(static_cast<size_t>(W * H * 3));
    if (rendered <= 0) {
        const uint8_t bg[3] = {static_cast<uint8_t>(background_[0] * 255.f + 0.5f),
                               static_cast<uint8_t>(background_[1] * 255.f + 0.5f),
                               static_cast<uint8_t>(background_[2] * 255.f + 0.5f)};
        for (size_t i = 0; i < out_rgb.size(); i += 3) {
            out_rgb[i] = bg[0];
            out_rgb[i + 1] = bg[1];
            out_rgb[i + 2] = bg[2];
        }
        return true;
    }

    out_color_.resize(out_floats);
    if (!cudaOk(cudaMemcpy(out_color_.data(), d_out_color_, out_floats * sizeof(float),
                           cudaMemcpyDeviceToHost),
                "D2H out_color")) {
        return false;
    }

    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            const int dst_pix = y * W + x;
            const int src_pix = (H - 1 - y) * W + x;  // Flip Y to match OSG window orientation.
            const size_t dst = static_cast<size_t>(dst_pix) * 3;
            const float r = out_color_[static_cast<size_t>(0 * H * W + src_pix)];
            const float g = out_color_[static_cast<size_t>(1 * H * W + src_pix)];
            const float b = out_color_[static_cast<size_t>(2 * H * W + src_pix)];
            out_rgb[dst + 0] = static_cast<uint8_t>(clamp01(r) * 255.f + 0.5f);
            out_rgb[dst + 1] = static_cast<uint8_t>(clamp01(g) * 255.f + 0.5f);
            out_rgb[dst + 2] = static_cast<uint8_t>(clamp01(b) * 255.f + 0.5f);
        }
    }
    return cudaOk(cudaDeviceSynchronize(), "sync");
#endif
}

}  // namespace internal
}  // namespace gsplat
