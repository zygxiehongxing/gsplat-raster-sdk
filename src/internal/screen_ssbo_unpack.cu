#include "screen_ssbo_unpack.h"

#include "screen_cache_internal.h"

#include <cuda_runtime.h>
#include <gsplat_raster/screen_gaussian_pixel.h>

#include <cstdlib>
#include <cmath>
#include <iostream>

namespace gsplat {
namespace internal {

namespace {

__global__ void unpackAoSToSoAKernel(int n, const GaussianPixelGpu* src, float scale_mul, float max_opacity,
                                     float* means, float* scales, float* rotations, float* opacities,
                                     float* colors, uint32_t expected_frame_id) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || !src) return;
    const GaussianPixelGpu& g = src[i];
    const bool frame_ok = (expected_frame_id == 0xFFFFFFFFu) || (g.frame_id == expected_frame_id);
    means[i * 3 + 0] = g.mean_opacity[0];
    means[i * 3 + 1] = g.mean_opacity[1];
    means[i * 3 + 2] = g.mean_opacity[2];
    scales[i * 3 + 0] = g.scale_pad[0] * scale_mul;
    scales[i * 3 + 1] = g.scale_pad[1] * scale_mul;
    scales[i * 3 + 2] = g.scale_pad[2] * scale_mul;
    rotations[i * 4 + 0] = g.rot[0];
    rotations[i * 4 + 1] = g.rot[1];
    rotations[i * 4 + 2] = g.rot[2];
    rotations[i * 4 + 3] = g.rot[3];
    float opa = frame_ok ? g.mean_opacity[3] : 0.f;
    if (opa > max_opacity) {
        opa = max_opacity;
    }
    opacities[i] = opa;
    colors[i * 3 + 0] = g.color_pad[0];
    colors[i * 3 + 1] = g.color_pad[1];
    colors[i * 3 + 2] = g.color_pad[2];
}

__global__ void countFilledKernel(int n, const float* opacities, int* out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (opacities[i] > 0.f) {
        atomicAdd(out, 1);
    }
}

__device__ void copyGaussianSlot(int dst, int src, const float* sm, float* dm, const float* ss, float* ds,
                                 const float* sr, float* dr, const float* so, float* dop, const float* sc,
                                 float* dc) {
    dm[dst * 3 + 0] = sm[src * 3 + 0];
    dm[dst * 3 + 1] = sm[src * 3 + 1];
    dm[dst * 3 + 2] = sm[src * 3 + 2];
    ds[dst * 3 + 0] = ss[src * 3 + 0];
    ds[dst * 3 + 1] = ss[src * 3 + 1];
    ds[dst * 3 + 2] = ss[src * 3 + 2];
    dr[dst * 4 + 0] = sr[src * 4 + 0];
    dr[dst * 4 + 1] = sr[src * 4 + 1];
    dr[dst * 4 + 2] = sr[src * 4 + 2];
    dr[dst * 4 + 3] = sr[src * 4 + 3];
    dop[dst] = so[src];
    dc[dst * 3 + 0] = sc[src * 3 + 0];
    dc[dst * 3 + 1] = sc[src * 3 + 1];
    dc[dst * 3 + 2] = sc[src * 3 + 2];
}

__global__ void compactPositiveOpacityKernel(int n, const float* means, const float* scales, const float* rotations,
                                             const float* opacities, const float* colors, float* out_means,
                                             float* out_scales, float* out_rotations, float* out_opacities,
                                             float* out_colors, int* out_count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (opacities[i] <= 0.f) return;
    const int o = atomicAdd(out_count, 1);
    copyGaussianSlot(o, i, means, out_means, scales, out_scales, rotations, out_rotations, opacities,
                     out_opacities, colors, out_colors);
}

}  // namespace

bool allocScreenSoa(int W, int H, DeviceGaussianBuffers& out) {
    return allocScreenBuffers(W, H, 0, 0, out);
}

void freeScreenSoa(DeviceGaussianBuffers& out) { freeScreenBuffers(out); }

float screenSsboUnpackScaleMul() {
    static float cached = -1.f;
    if (cached < 0.f) {
        const char* env = std::getenv("GSPLAT_SCREEN_SCALE_MUL");
        if (env && env[0] != '\0' && env[0] != '0') {
            cached = std::strtof(env, nullptr);
        } else {
            cached = 0.035f;
        }
        if (cached <= 0.f || !std::isfinite(cached)) {
            cached = 0.035f;
        }
    }
    return cached;
}

float screenSsboUnpackMaxOpacity() {
    static float cached = -1.f;
    if (cached < 0.f) {
        const char* env = std::getenv("GSPLAT_SCREEN_MAX_OPACITY");
        if (env && env[0] != '\0') {
            cached = std::strtof(env, nullptr);
        } else {
            cached = 0.65f;
        }
        if (cached <= 0.f || cached > 1.f || !std::isfinite(cached)) {
            cached = 0.65f;
        }
    }
    return cached;
}

bool unpackScreenSsboAoS(const GaussianPixelGpu* d_src, int pixel_count, DeviceGaussianBuffers& dst,
                         uint32_t expected_frame_id, float scale_mul, float max_opacity) {
    if (pixel_count <= 0 || !d_src || dst.count != pixel_count) return false;
    if (!dst.means3D || !dst.scales || !dst.rotations || !dst.opacities || !dst.colors_precomp) {
        return false;
    }
    const int threads = 256;
    const int blocks = (pixel_count + threads - 1) / threads;
    unpackAoSToSoAKernel<<<blocks, threads>>>(pixel_count, d_src, scale_mul, max_opacity, dst.means3D,
                                              dst.scales, dst.rotations, dst.opacities, dst.colors_precomp,
                                              expected_frame_id);
    return cudaGetLastError() == cudaSuccess;
}

int countFilledScreenSoa(const DeviceGaussianBuffers& buf) {
    if (buf.count <= 0 || !buf.opacities) return 0;
    int* d_count = nullptr;
    if (cudaMalloc(&d_count, sizeof(int)) != cudaSuccess) return 0;
    cudaMemset(d_count, 0, sizeof(int));
    const int threads = 256;
    const int blocks = (buf.count + threads - 1) / threads;
    countFilledKernel<<<blocks, threads>>>(buf.count, buf.opacities, d_count);
    int host = 0;
    cudaMemcpy(&host, d_count, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_count);
    cudaDeviceSynchronize();
    return host;
}

bool compactScreenSoa(DeviceGaussianBuffers& buf, int& compact_count) {
    compact_count = 0;
    if (buf.count <= 0 || !buf.means3D || !buf.scales || !buf.rotations || !buf.opacities || !buf.colors_precomp) {
        return false;
    }
    const int filled = countFilledScreenSoa(buf);
    if (filled <= 0) {
        buf.count = 0;
        return false;
    }
    if (filled >= buf.count) {
        compact_count = filled;
        return true;
    }

    float *d_means = nullptr, *d_scales = nullptr, *d_rotations = nullptr, *d_opacities = nullptr,
          *d_colors = nullptr;
    int* d_count = nullptr;
    const size_t n = static_cast<size_t>(filled);
    if (cudaMalloc(&d_means, n * 3 * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&d_scales, n * 3 * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&d_rotations, n * 4 * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&d_opacities, n * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&d_colors, n * 3 * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&d_count, sizeof(int)) != cudaSuccess) {
        if (d_means) cudaFree(d_means);
        if (d_scales) cudaFree(d_scales);
        if (d_rotations) cudaFree(d_rotations);
        if (d_opacities) cudaFree(d_opacities);
        if (d_colors) cudaFree(d_colors);
        if (d_count) cudaFree(d_count);
        return false;
    }
    cudaMemset(d_count, 0, sizeof(int));
    const int threads = 256;
    const int blocks = (buf.count + threads - 1) / threads;
    compactPositiveOpacityKernel<<<blocks, threads>>>(buf.count, buf.means3D, buf.scales, buf.rotations,
                                                      buf.opacities, buf.colors_precomp, d_means, d_scales,
                                                      d_rotations, d_opacities, d_colors, d_count);
    int host_count = 0;
    cudaMemcpy(&host_count, d_count, sizeof(int), cudaMemcpyDeviceToHost);
    const size_t bytes3 = static_cast<size_t>(host_count) * 3 * sizeof(float);
    const size_t bytes4 = static_cast<size_t>(host_count) * 4 * sizeof(float);
    const size_t bytes1 = static_cast<size_t>(host_count) * sizeof(float);
    cudaMemcpy(buf.means3D, d_means, bytes3, cudaMemcpyDeviceToDevice);
    cudaMemcpy(buf.scales, d_scales, bytes3, cudaMemcpyDeviceToDevice);
    cudaMemcpy(buf.rotations, d_rotations, bytes4, cudaMemcpyDeviceToDevice);
    cudaMemcpy(buf.opacities, d_opacities, bytes1, cudaMemcpyDeviceToDevice);
    cudaMemcpy(buf.colors_precomp, d_colors, bytes3, cudaMemcpyDeviceToDevice);
    cudaFree(d_means);
    cudaFree(d_scales);
    cudaFree(d_rotations);
    cudaFree(d_opacities);
    cudaFree(d_colors);
    cudaFree(d_count);
    if (cudaGetLastError() != cudaSuccess || host_count <= 0) {
        buf.count = 0;
        return false;
    }
    compact_count = host_count;
    buf.count = compact_count;
    return true;
}

namespace {

bool cudaRasterTimingEnabled() {
    static int cached = -1;
    if (cached < 0) {
        const char* env = std::getenv("GSPLAT_CUDA_RASTER_TIMING");
        cached = (env && env[0] != '\0' && env[0] != '0') ? 1 : 0;
    }
    return cached != 0;
}

}  // namespace

bool unpackScreenSsboPipeline(const GaussianPixelGpu* d_src, int width, int height,
                              DeviceGaussianBuffers& out, int& filled_out,
                              uint32_t expected_frame_id, float scale_mul, float max_opacity) {
    filled_out = 0;
    if (!d_src || width <= 0 || height <= 0) return false;
    const int n = width * height;
    const bool time_gpu = cudaRasterTimingEnabled();
    cudaEvent_t ev0 = nullptr;
    cudaEvent_t ev1 = nullptr;
    if (time_gpu) {
        cudaEventCreate(&ev0);
        cudaEventCreate(&ev1);
        cudaEventRecord(ev0, 0);
    }
    if (out.count != n) {
        freeScreenSoa(out);
        if (!allocScreenSoa(width, height, out)) return false;
    }
    if (!unpackScreenSsboAoS(d_src, n, out, expected_frame_id, scale_mul, max_opacity)) return false;
    {
        static int log_left = 2;
        if (log_left > 0) {
            --log_left;
            std::cout << "[CUDA unpack] screen scale_mul=" << scale_mul << " max_opacity=" << max_opacity
                      << "\n";
        }
    }
    filled_out = countFilledScreenSoa(out);
    int compact_count = 0;
    if (!compactScreenSoa(out, compact_count)) return false;
    filled_out = compact_count;
    if (time_gpu) {
        cudaEventRecord(ev1, 0);
        cudaEventSynchronize(ev1);
        float ms = 0.f;
        cudaEventElapsedTime(&ms, ev0, ev1);
        static int unpack_log_counter = 0;
        if ((unpack_log_counter++ % 30) == 0) {
            std::cout << "[CUDA unpack] " << width << "x" << height << " cells=" << n
                      << " compact=" << compact_count << " gpu_ms=" << ms << "\n";
        }
        cudaEventDestroy(ev0);
        cudaEventDestroy(ev1);
    }
    return true;
}

}  // namespace internal
}  // namespace gsplat
