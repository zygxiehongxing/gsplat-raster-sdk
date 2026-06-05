#include "rasterizer_impl.h"

#include "../gaussian_types.h"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>

#ifdef GSPLAT_CUDA_ENABLED
#include <cuda_runtime.h>
#include "rasterizer.h"
#endif

namespace gsplat {
namespace internal {

namespace {
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

bool cudaRasterTimingEnabled() {
    static int cached = -1;
    if (cached < 0) {
        const char* env = std::getenv("GSPLAT_CUDA_RASTER_TIMING");
        cached = (env && env[0] != '\0' && env[0] != '0') ? 1 : 0;
    }
    return cached != 0;
}

float elapsedCudaEventMs(cudaEvent_t start, cudaEvent_t stop) {
    float ms = 0.f;
    if (cudaEventSynchronize(stop) != cudaSuccess) return -1.f;
    if (cudaEventElapsedTime(&ms, start, stop) != cudaSuccess) return -1.f;
    return ms;
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
char* CudaRasterEngine::resizeDeviceScratch(char*& buf, size_t& capacity_bytes, size_t required_bytes) {
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

bool CudaRasterEngine::renderFromDevice(int width, int height, const float view[16], const float proj[16],
                                        const float cam_pos[3], float tan_fovx, float tan_fovy,
                                        const DeviceGaussianBuffers& src, std::vector<uint8_t>& out_rgb) {
    if (src.count <= 0 || width <= 0 || height <= 0 || !src.means3D || !src.scales || !src.rotations ||
        !src.opacities) {
        return false;
    }
#ifndef GSPLAT_CUDA_ENABLED
    (void)view;
    (void)proj;
    (void)cam_pos;
    (void)tan_fovx;
    (void)tan_fovy;
    (void)src;
    (void)out_rgb;
    return false;
#else
    return renderCore(width, height, view, proj, cam_pos, tan_fovx, tan_fovy, src.count, src.sh_degree,
                      src.sh_coeffs, src.means3D, src.sh_degree > 0 ? src.shs : nullptr,
                      src.sh_degree > 0 ? nullptr : src.colors_precomp, src.opacities, src.scales,
                      src.rotations, out_rgb);
#endif
}

bool CudaRasterEngine::renderCore(int width, int height, const float view[16], const float proj[16],
                                  const float cam_pos[3], float tan_fovx, float tan_fovy, int P, int D,
                                  int M, const float* d_means3D, const float* d_shs,
                                  const float* d_colors_precomp, const float* d_opacities,
                                  const float* d_scales, const float* d_rotations,
                                  std::vector<uint8_t>& out_rgb) {
#ifndef GSPLAT_CUDA_ENABLED
    (void)width;
    (void)height;
    (void)view;
    (void)proj;
    (void)cam_pos;
    (void)tan_fovx;
    (void)tan_fovy;
    (void)P;
    (void)D;
    (void)M;
    (void)d_means3D;
    (void)d_shs;
    (void)d_colors_precomp;
    (void)d_opacities;
    (void)d_scales;
    (void)d_rotations;
    (void)out_rgb;
    return false;
#else
    if (P <= 0 || width <= 0 || height <= 0 || !d_means3D || !d_opacities || !d_scales || !d_rotations) {
        return false;
    }

    const int W = width, H = height;
    const size_t out_floats = static_cast<size_t>(3 * H * W);

    if (!d_background_ && !cudaOk(cudaMalloc(&d_background_, 3 * sizeof(float)), "malloc background")) {
        return false;
    }
    if (!d_view_ && !cudaOk(cudaMalloc(&d_view_, 16 * sizeof(float)), "malloc view")) return false;
    if (!d_proj_ && !cudaOk(cudaMalloc(&d_proj_, 16 * sizeof(float)), "malloc proj")) return false;
    if (!d_cam_pos_ && !cudaOk(cudaMalloc(&d_cam_pos_, 3 * sizeof(float)), "malloc cam_pos")) return false;

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

    constexpr int kBlock = 16;
    constexpr uint64_t kMaxEstTiles = 8'000'000ULL;
    // GL SSBO 屏缓存：尺度已钳位，跳过 tile 预算重试（否则 visible 常被误杀为 0）。
    constexpr bool kDisableTileBudgetForDiag = true;

    constexpr bool kInputPrefilteredByGl = false;
    // SSBO-screen path treats current input set as the frame's full source.
    // Do not auto-amplify scale based on "visible count" feedback.
    const bool time_gpu = cudaRasterTimingEnabled();
    cudaEvent_t ev_fwd0 = nullptr;
    cudaEvent_t ev_fwd1 = nullptr;
    cudaEvent_t ev_d2h0 = nullptr;
    cudaEvent_t ev_d2h1 = nullptr;
    if (time_gpu) {
        cudaEventCreate(&ev_fwd0);
        cudaEventCreate(&ev_fwd1);
        cudaEventCreate(&ev_d2h0);
        cudaEventCreate(&ev_d2h1);
    }

    float last_fwd_ms = -1.f;
    float last_d2h_radii_ms = -1.f;
    auto destroyTimingEvents = [&]() {
        if (!time_gpu) return;
        cudaEventDestroy(ev_fwd0);
        cudaEventDestroy(ev_fwd1);
        cudaEventDestroy(ev_d2h0);
        cudaEventDestroy(ev_d2h1);
    };
    static int timing_log_counter = 0;
    auto logTimingLine = [&](const char* d2h_out_note) {
        if (!time_gpu) return;
        if ((timing_log_counter++ % 30) != 0) return;
        std::cout << "[CUDA raster] timing " << W << "x" << H << " P=" << P
                  << " gpu_forward_ms=" << last_fwd_ms << " d2h_radii_ms=" << last_d2h_radii_ms
                  << " d2h_out_color_ms=" << d2h_out_note << " visible=" << last_visible_
                  << " rendered=" << rendered << "\n";
    };

    for (int attempt = 0; attempt < 2; ++attempt) {
        bool forward_failed = false;
        try {
            if (time_gpu) {
                cudaEventRecord(ev_fwd0, 0);
            }
            rendered = CudaRasterizer::Rasterizer::forward(
                [&](size_t N) { return resizeDeviceScratch(d_geom_, d_geom_cap_, N); },
                [&](size_t N) { return resizeDeviceScratch(d_binning_, d_binning_cap_, N); },
                [&](size_t N) { return resizeDeviceScratch(d_image_, d_image_cap_, N); },
                P, D, M, d_background_, W, H, d_means3D, D > 0 ? d_shs : nullptr,
                D > 0 ? nullptr : d_colors_precomp, d_opacities, d_scales, mod, d_rotations, nullptr,
                d_view_, d_proj_, d_cam_pos_, tan_fovx, tan_fovy, kInputPrefilteredByGl, d_out_color_,
                d_radii_, true);
            if (time_gpu) {
                cudaEventRecord(ev_fwd1, 0);
                last_fwd_ms = elapsedCudaEventMs(ev_fwd0, ev_fwd1);
            }
        } catch (const std::exception& e) {
            std::cerr << "CUDA rasterizer: " << e.what() << " (scale_modifier=" << mod << " P=" << P
                      << ")\n";
            rendered = 0;
            forward_failed = true;
            cudaGetLastError();
            freeDev(d_geom_);
            freeDev(d_binning_);
            freeDev(d_image_);
            d_geom_cap_ = 0;
            d_binning_cap_ = 0;
            d_image_cap_ = 0;
        }

        if (time_gpu) {
            cudaEventRecord(ev_d2h0, 0);
        }
        const bool radii_ok =
            cudaOk(cudaMemcpy(radii_.data(), d_radii_, static_cast<size_t>(P) * sizeof(int),
                              cudaMemcpyDeviceToHost),
                   "D2H radii");
        if (time_gpu) {
            cudaEventRecord(ev_d2h1, 0);
            last_d2h_radii_ms = elapsedCudaEventMs(ev_d2h0, ev_d2h1);
        }
        if (!radii_ok) {
            cudaGetLastError();
            last_visible_ = 0;
            if (attempt + 1 < 2) {
                mod *= forward_failed ? 0.5f : 0.7f;
                cudaDeviceReset();
                freeDevice();
                continue;
            }
            logTimingLine("n/a");
            destroyTimingEvents();
            return false;
        }

        last_visible_ = 0;
        int max_r = 0;
        int max_r_idx = -1;
        uint64_t est_tiles = 0;
        for (int i = 0; i < P; ++i) {
            const int r = radii_[static_cast<size_t>(i)];
            if (r <= 0) continue;
            ++last_visible_;
            if (r > max_r) {
                max_r = r;
                max_r_idx = i;
            }
            const int tx = (2 * r + kBlock - 1) / kBlock;
            est_tiles += static_cast<uint64_t>(tx) * static_cast<uint64_t>(tx);
        }

        if (max_r_idx >= 0 && max_r > 4096) {
            static int huge_radius_log_left = 8;
            if (huge_radius_log_left > 0) {
                --huge_radius_log_left;
                float mean[3] = {};
                float scale[3] = {};
                const size_t m_off = static_cast<size_t>(max_r_idx) * 3u;
                cudaMemcpy(mean, d_means3D + m_off, sizeof(mean), cudaMemcpyDeviceToHost);
                cudaMemcpy(scale, d_scales + m_off, sizeof(scale), cudaMemcpyDeviceToHost);
                const float vx = view[0] * mean[0] + view[1] * mean[1] + view[2] * mean[2] + view[3];
                const float vy = view[4] * mean[0] + view[5] * mean[1] + view[6] * mean[2] + view[7];
                const float vz = view[8] * mean[0] + view[9] * mean[1] + view[10] * mean[2] + view[11];
                float clip[4] = {};
                const float v4[4] = {mean[0], mean[1], mean[2], 1.f};
                float t4[4] = {};
                for (int c = 0; c < 4; ++c) {
                    float s = 0.f;
                    for (int r = 0; r < 4; ++r) s += v4[r] * view[r * 4 + c];
                    t4[c] = s;
                }
                for (int c = 0; c < 4; ++c) {
                    float s = 0.f;
                    for (int r = 0; r < 4; ++r) s += t4[r] * proj[r * 4 + c];
                    clip[c] = s;
                }
                const float cx = clip[0];
                const float cy = clip[1];
                const float cz = clip[2];
                const float cw = clip[3];
                std::cerr << "[CUDA raster] huge_radius idx=" << max_r_idx << " r=" << max_r
                          << " mod=" << mod << " mean=(" << mean[0] << "," << mean[1] << "," << mean[2]
                          << ") scale=(" << scale[0] << "," << scale[1] << "," << scale[2]
                          << ") view=(" << vx << "," << vy << "," << vz << ") clip=(" << cx << "," << cy
                          << "," << cz << "," << cw << ")\n";
            }
        }

        if (!kDisableTileBudgetForDiag && last_visible_ > 0 && (est_tiles > kMaxEstTiles || max_r > 256)) {
            static int tile_warn_left = 3;
            if (tile_warn_left > 0) {
                --tile_warn_left;
                std::cerr << "[CUDA raster] tile budget exceeded: visible=" << last_visible_
                          << " max_radius=" << max_r << " est_tiles=" << est_tiles
                          << " scale_modifier=" << mod << " -> retry\n";
            }
            last_visible_ = 0;
            rendered = 0;
            if (attempt + 1 < 2) {
                mod *= 0.5f;
                continue;
            }
        }

        if (rendered > 0) break;
        if (forward_failed && attempt + 1 < 2) {
            mod *= 0.5f;
            continue;
        }
    }

    if (P > 0) {
        static int radii_log_left = 5;
        if (radii_log_left > 0) {
            --radii_log_left;
            int max_r = 0;
            for (int r : radii_) {
                if (r > max_r) max_r = r;
            }
            std::cout << "[CUDA raster] P=" << P << " radii>0=" << last_visible_ << " max_radius=" << max_r
                      << " scale_modifier=" << mod << "\n";
        }
    }

    if (last_visible_ <= 0 && P > 0) {
        static int radii_zero_log_left = 3;
        if (radii_zero_log_left > 0) {
            --radii_zero_log_left;
            std::cerr << "[CUDA raster] radii=0 for all " << P << " splats (scale_modifier up to " << mod
                      << ")\n";
        }
    }

    out_rgb.resize(static_cast<size_t>(W * H * 3));
    if (rendered <= 0) {
        logTimingLine("n/a");
        destroyTimingEvents();
        if (P > 0) {
            std::cerr << "[CUDA raster] forward produced no pixels (visible=" << last_visible_
                      << " P=" << P << " scale_modifier=" << mod << ")\n";
        }
        return false;
    }

    out_color_.resize(out_floats);
    float d2h_out_ms = -1.f;
    if (time_gpu) {
        cudaEventRecord(ev_d2h0, 0);
    }
    if (!cudaOk(cudaMemcpy(out_color_.data(), d_out_color_, out_floats * sizeof(float),
                           cudaMemcpyDeviceToHost),
                "D2H out_color")) {
        destroyTimingEvents();
        return false;
    }
    if (time_gpu) {
        cudaEventRecord(ev_d2h1, 0);
        d2h_out_ms = elapsedCudaEventMs(ev_d2h0, ev_d2h1);
        std::cout << "[CUDA raster] timing " << W << "x" << H << " P=" << P
                  << " gpu_forward_ms=" << last_fwd_ms << " d2h_radii_ms=" << last_d2h_radii_ms
                  << " d2h_out_color_ms=" << d2h_out_ms << " visible=" << last_visible_ << "\n";
        destroyTimingEvents();
    }

    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            const int dst_pix = y * W + x;
            const int src_pix = (H - 1 - y) * W + x;
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
