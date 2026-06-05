#include "screen_cache_internal.h"

#include <cuda_runtime.h>

namespace gsplat {
namespace internal {
namespace {

__device__ bool projectToScreenPixel(const float view[16], const float proj[16], float x, float y, float z,
                                     int W, int H, int& px, int& py, float& depth_key) {
    const float view_z = view[2] * x + view[6] * y + view[10] * z + view[14];
    if (view_z >= -1e-4f) return false;
    depth_key = -view_z;

    const float hx = proj[0] * x + proj[4] * y + proj[8] * z + proj[12];
    const float hy = proj[1] * x + proj[5] * y + proj[9] * z + proj[13];
    const float hz = proj[2] * x + proj[6] * y + proj[10] * z + proj[14];
    const float hw = proj[3] * x + proj[7] * y + proj[11] * z + proj[15];
    if (hw <= 1e-6f) return false;

    const float invw = 1.f / hw;
    const float nx = hx * invw;
    const float ny = hy * invw;
    const float nz = hz * invw;
    if (nx < -1.f || nx > 1.f || ny < -1.f || ny > 1.f || nz < -1.f || nz > 1.f) return false;

    px = static_cast<int>((nx * 0.5f + 0.5f) * static_cast<float>(W));
    py = static_cast<int>((ny * 0.5f + 0.5f) * static_cast<float>(H));
    py = H - 1 - py;
    return px >= 0 && px < W && py >= 0 && py < H;
}

__device__ void copyGaussianToSlot(int o, int i, bool use_sh, int sh_coeffs, const float* src_means,
                                   const float* src_scales, const float* src_rotations, const float* src_opacities,
                                   const float* src_shs, const float* src_colors, float* dst_means, float* dst_scales,
                                   float* dst_rotations, float* dst_opacities, float* dst_shs, float* dst_colors) {
    dst_means[o * 3 + 0] = src_means[i * 3 + 0];
    dst_means[o * 3 + 1] = src_means[i * 3 + 1];
    dst_means[o * 3 + 2] = src_means[i * 3 + 2];

    dst_scales[o * 3 + 0] = src_scales[i * 3 + 0];
    dst_scales[o * 3 + 1] = src_scales[i * 3 + 1];
    dst_scales[o * 3 + 2] = src_scales[i * 3 + 2];

    dst_rotations[o * 4 + 0] = src_rotations[i * 4 + 0];
    dst_rotations[o * 4 + 1] = src_rotations[i * 4 + 1];
    dst_rotations[o * 4 + 2] = src_rotations[i * 4 + 2];
    dst_rotations[o * 4 + 3] = src_rotations[i * 4 + 3];

    dst_opacities[o] = src_opacities[i];

    if (use_sh) {
        const int n = sh_coeffs * 3;
        const int src_base = i * n;
        const int dst_base = o * n;
        for (int k = 0; k < n; ++k) {
            dst_shs[dst_base + k] = src_shs[src_base + k];
        }
    } else {
        dst_colors[o * 3 + 0] = src_colors[i * 3 + 0];
        dst_colors[o * 3 + 1] = src_colors[i * 3 + 1];
        dst_colors[o * 3 + 2] = src_colors[i * 3 + 2];
    }
}

__global__ void clearScreenKernel(int n, float* opacities, float* colors, float* shs, int sh_floats,
                                  float* depth_buf) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    opacities[i] = 0.f;
    if (colors) {
        colors[i * 3 + 0] = 0.f;
        colors[i * 3 + 1] = 0.f;
        colors[i * 3 + 2] = 0.f;
    }
    if (shs && sh_floats > 0) {
        const int base = i * sh_floats;
        for (int k = 0; k < sh_floats; ++k) {
            shs[base + k] = 0.f;
        }
    }
    if (depth_buf) {
        depth_buf[i] = 0.f;
    }
}

__global__ void countFilledKernel(int n, const float* opacities, int* count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (opacities[i] > 0.f) {
        atomicAdd(count, 1);
    }
}

__global__ void scatterWithDepthKernel(int P, int W, int H, const float* view, const float* proj,
                                       const float* src_means, const float* src_scales, const float* src_rotations,
                                       const float* src_opacities, const float* src_shs, const float* src_colors,
                                       bool use_sh, int sh_coeffs, float* dst_means, float* dst_scales,
                                       float* dst_rotations, float* dst_opacities, float* dst_shs, float* dst_colors,
                                       float* depth_buf) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= P) return;

    const float x = src_means[i * 3 + 0];
    const float y = src_means[i * 3 + 1];
    const float z = src_means[i * 3 + 2];

    int px = 0;
    int py = 0;
    float depth_key = 0.f;
    if (!projectToScreenPixel(view, proj, x, y, z, W, H, px, py, depth_key)) return;

    const int o = py * W + px;
    unsigned int* depth_u = reinterpret_cast<unsigned int*>(depth_buf + o);
    const unsigned int new_u = __float_as_uint(depth_key);
    unsigned int assumed = *depth_u;
    while (assumed < new_u) {
        const unsigned int old = atomicCAS(depth_u, assumed, new_u);
        if (old == assumed) {
            copyGaussianToSlot(o, i, use_sh, sh_coeffs, src_means, src_scales, src_rotations, src_opacities,
                               src_shs, src_colors, dst_means, dst_scales, dst_rotations, dst_opacities, dst_shs,
                               dst_colors);
            return;
        }
        assumed = old;
    }
}

bool cudaOk(cudaError_t err) { return err == cudaSuccess; }

void freeDev(float*& p) {
    if (p) {
        cudaFree(p);
        p = nullptr;
    }
}

}  // namespace

bool allocScreenBuffers(int W, int H, int sh_degree, int sh_coeffs, DeviceGaussianBuffers& out) {
    freeScreenBuffers(out);
    if (W <= 0 || H <= 0) return false;
    const size_t n = static_cast<size_t>(W) * static_cast<size_t>(H);
    if (!cudaOk(cudaMalloc(&out.means3D, n * 3 * sizeof(float))) ||
        !cudaOk(cudaMalloc(&out.scales, n * 3 * sizeof(float))) ||
        !cudaOk(cudaMalloc(&out.rotations, n * 4 * sizeof(float))) ||
        !cudaOk(cudaMalloc(&out.opacities, n * sizeof(float)))) {
        freeScreenBuffers(out);
        return false;
    }
    const bool use_sh = sh_degree > 0;
    if (use_sh) {
        if (!cudaOk(cudaMalloc(&out.shs, n * static_cast<size_t>(sh_coeffs) * 3 * sizeof(float)))) {
            freeScreenBuffers(out);
            return false;
        }
        out.colors_precomp = nullptr;
    } else if (!cudaOk(cudaMalloc(&out.colors_precomp, n * 3 * sizeof(float)))) {
        freeScreenBuffers(out);
        return false;
    }
    out.count = static_cast<int>(n);
    out.sh_degree = sh_degree;
    out.sh_coeffs = sh_coeffs;
    return true;
}

void freeScreenBuffers(DeviceGaussianBuffers& out) {
    freeDev(out.means3D);
    freeDev(out.scales);
    freeDev(out.rotations);
    freeDev(out.opacities);
    freeDev(out.shs);
    freeDev(out.colors_precomp);
    out.count = 0;
    out.sh_degree = 0;
    out.sh_coeffs = 0;
}

bool clearScreenCache(DeviceGaussianBuffers& buf, float* depth_buf, int pixel_count) {
    if (buf.count <= 0 || !buf.opacities || pixel_count <= 0) return false;
    const int threads = 256;
    const int blocks = (pixel_count + threads - 1) / threads;
    const bool use_sh = buf.sh_degree > 0;
    const int sh_floats = use_sh ? buf.sh_coeffs * 3 : 0;
    clearScreenKernel<<<blocks, threads>>>(pixel_count, buf.opacities, use_sh ? nullptr : buf.colors_precomp,
                                           use_sh ? buf.shs : nullptr, sh_floats, depth_buf);
    return cudaOk(cudaDeviceSynchronize());
}

bool scatterPoolToScreen(const DeviceGaussianBuffers& src, const float view[16], const float proj[16], int W,
                         int H, DeviceGaussianBuffers& dst, float* depth_buf, int& filled_out) {
    filled_out = 0;
    if (src.count <= 0 || W <= 0 || H <= 0 || dst.count != W * H) return false;
    if (!src.means3D || !src.scales || !src.rotations || !src.opacities) return false;
    if (!dst.means3D || !dst.scales || !dst.rotations || !dst.opacities) return false;
    if (!depth_buf) return false;

    const bool use_sh = src.sh_degree > 0;
    if (use_sh) {
        if (!src.shs || !dst.shs || dst.sh_degree != src.sh_degree || dst.sh_coeffs != src.sh_coeffs) return false;
    } else if (!src.colors_precomp || !dst.colors_precomp) {
        return false;
    }

    float* d_view = nullptr;
    float* d_proj = nullptr;
    int* d_filled = nullptr;
    auto fail = [&]() -> bool {
        if (d_view) cudaFree(d_view);
        if (d_proj) cudaFree(d_proj);
        if (d_filled) cudaFree(d_filled);
        return false;
    };

    if (!cudaOk(cudaMalloc(&d_view, 16 * sizeof(float))) || !cudaOk(cudaMalloc(&d_proj, 16 * sizeof(float))) ||
        !cudaOk(cudaMalloc(&d_filled, sizeof(int)))) {
        return fail();
    }
    if (!cudaOk(cudaMemcpy(d_view, view, 16 * sizeof(float), cudaMemcpyHostToDevice)) ||
        !cudaOk(cudaMemcpy(d_proj, proj, 16 * sizeof(float), cudaMemcpyHostToDevice)) ||
        !cudaOk(cudaMemset(d_filled, 0, sizeof(int)))) {
        return fail();
    }

    if (!clearScreenCache(dst, depth_buf, W * H)) return fail();

    const int threads = 256;
    const int blocks = (src.count + threads - 1) / threads;
    scatterWithDepthKernel<<<blocks, threads>>>(
        src.count, W, H, d_view, d_proj, src.means3D, src.scales, src.rotations, src.opacities,
        use_sh ? src.shs : nullptr, use_sh ? nullptr : src.colors_precomp, use_sh, src.sh_coeffs, dst.means3D,
        dst.scales, dst.rotations, dst.opacities, use_sh ? dst.shs : nullptr,
        use_sh ? nullptr : dst.colors_precomp, depth_buf);

    const int count_blocks = (dst.count + threads - 1) / threads;
    countFilledKernel<<<count_blocks, threads>>>(dst.count, dst.opacities, d_filled);

    int filled = 0;
    if (!cudaOk(cudaMemcpy(&filled, d_filled, sizeof(int), cudaMemcpyDeviceToHost))) {
        return fail();
    }
    cudaFree(d_view);
    cudaFree(d_proj);
    cudaFree(d_filled);
    if (!cudaOk(cudaDeviceSynchronize())) return false;
    filled_out = filled;
    return true;
}

}  // namespace internal
}  // namespace gsplat
