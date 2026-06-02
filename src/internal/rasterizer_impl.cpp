#include "rasterizer_impl.h"

#include "../gaussian_types.h"

#include <algorithm>
#include <cmath>
#include <iostream>

#ifdef GSPLAT_CUDA_ENABLED
#include <cuda_runtime.h>
#include "rasterizer.h"
#endif

namespace gsplat {
namespace internal {

namespace {
constexpr float kShC0 = 0.28209479177387814f;
}

CudaRasterEngine::CudaRasterEngine() = default;
CudaRasterEngine::~CudaRasterEngine() = default;

void CudaRasterEngine::setSettings(const RenderSettings& s) {
    settings_ = s;
    scale_modifier_ = s.scale_modifier;
    background_[0] = s.background[0];
    background_[1] = s.background[1];
    background_[2] = s.background[2];
}

bool CudaRasterEngine::upload(const std::vector<Gaussian>& gaussians) {
    uploadOne(gaussians);
    return num_splats_ > 0;
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

    for (int i = 0; i < num_splats_; ++i) {
        const Gaussian& g = cloud[static_cast<size_t>(i)];
        means3D_[static_cast<size_t>(i) * 3 + 0] = g.x;
        means3D_[static_cast<size_t>(i) * 3 + 1] = g.y;
        means3D_[static_cast<size_t>(i) * 3 + 2] = g.z;

        opacities_[static_cast<size_t>(i)] = sigmoid(g.opacity_logit);
        scales_[static_cast<size_t>(i) * 3 + 0] = std::exp(g.scale_log[0]);
        scales_[static_cast<size_t>(i) * 3 + 1] = std::exp(g.scale_log[1]);
        scales_[static_cast<size_t>(i) * 3 + 2] = std::exp(g.scale_log[2]);

        float qw = g.rot[0], qx = g.rot[1], qy = g.rot[2], qz = g.rot[3];
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
                shs_[dst + 0] = g.sh[src + 0];
                shs_[dst + 1] = g.sh[src + 1];
                shs_[dst + 2] = g.sh[src + 2];
            }
        } else {
            colors_precomp_[static_cast<size_t>(i) * 3 + 0] = clamp01(0.5f + kShC0 * g.sh[0]);
            colors_precomp_[static_cast<size_t>(i) * 3 + 1] = clamp01(0.5f + kShC0 * g.sh[1]);
            colors_precomp_[static_cast<size_t>(i) * 3 + 2] = clamp01(0.5f + kShC0 * g.sh[2]);
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
    const int W = width, H = height, P = num_splats_, D = sh_degree_, M = sh_coeffs_;
    out_color_.assign(static_cast<size_t>(3 * H * W), 0.f);
    radii_.assign(static_cast<size_t>(P), 0);

    auto resizeBuf = [](std::vector<char>& buf, size_t N) -> char* {
        buf.resize(N);
        return buf.data();
    };

    int rendered = 0;
    float mod = scale_modifier_;
    last_visible_ = 0;
    for (int attempt = 0; attempt < 3; ++attempt) {
        rendered = CudaRasterizer::Rasterizer::forward(
            [&](size_t N) { return resizeBuf(geom_buffer_, N); },
            [&](size_t N) { return resizeBuf(binning_buffer_, N); },
            [&](size_t N) { return resizeBuf(image_buffer_, N); }, P, D, M, background_, W, H,
            means3D_.data(), sh_degree_ > 0 ? shs_.data() : nullptr,
            sh_degree_ > 0 ? nullptr : colors_precomp_.data(), opacities_.data(), scales_.data(),
            mod, rotations_.data(), nullptr, view, proj, cam_pos, tan_fovx, tan_fovy, false,
            out_color_.data(), radii_.data(), false);
        last_visible_ = 0;
        for (int r : radii_) {
            if (r > 0) ++last_visible_;
        }
        if (last_visible_ > 0) {
            scale_modifier_ = mod;
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
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            const int pix = y * W + x;
            const size_t dst = static_cast<size_t>(pix) * 3;
            float r = out_color_[static_cast<size_t>(0 * H * W + pix)];
            float g = out_color_[static_cast<size_t>(1 * H * W + pix)];
            float b = out_color_[static_cast<size_t>(2 * H * W + pix)];
            out_rgb[dst + 0] = static_cast<uint8_t>(clamp01(r) * 255.f + 0.5f);
            out_rgb[dst + 1] = static_cast<uint8_t>(clamp01(g) * 255.f + 0.5f);
            out_rgb[dst + 2] = static_cast<uint8_t>(clamp01(b) * 255.f + 0.5f);
        }
    }
    cudaDeviceSynchronize();
    return true;
#endif
}

}  // namespace internal
}  // namespace gsplat
