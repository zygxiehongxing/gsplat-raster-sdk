#include "device_filter.h"

#include <cuda_runtime.h>
#include "rasterizer.h"

namespace gsplat {
namespace {

__global__ void compactGaussianKernel(int P, int sh_coeffs, const bool* present, const float* means3D,
                                        const float* scales, const float* rotations, const float* opacities,
                                        const float* shs, const float* colors_precomp, bool use_sh,
                                        float* out_means3D, float* out_scales, float* out_rotations,
                                        float* out_opacities, float* out_shs, float* out_colors_precomp,
                                        int* counter) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= P || !present[i]) return;
    const int o = atomicAdd(counter, 1);

    out_means3D[o * 3 + 0] = means3D[i * 3 + 0];
    out_means3D[o * 3 + 1] = means3D[i * 3 + 1];
    out_means3D[o * 3 + 2] = means3D[i * 3 + 2];

    out_scales[o * 3 + 0] = scales[i * 3 + 0];
    out_scales[o * 3 + 1] = scales[i * 3 + 1];
    out_scales[o * 3 + 2] = scales[i * 3 + 2];

    out_rotations[o * 4 + 0] = rotations[i * 4 + 0];
    out_rotations[o * 4 + 1] = rotations[i * 4 + 1];
    out_rotations[o * 4 + 2] = rotations[i * 4 + 2];
    out_rotations[o * 4 + 3] = rotations[i * 4 + 3];

    out_opacities[o] = opacities[i];

    if (use_sh) {
        const int n = sh_coeffs * 3;
        const int src_base = i * n;
        const int dst_base = o * n;
        for (int k = 0; k < n; ++k) {
            out_shs[dst_base + k] = shs[src_base + k];
        }
    } else {
        out_colors_precomp[o * 3 + 0] = colors_precomp[i * 3 + 0];
        out_colors_precomp[o * 3 + 1] = colors_precomp[i * 3 + 1];
        out_colors_precomp[o * 3 + 2] = colors_precomp[i * 3 + 2];
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

void freeDeviceGaussianBuffers(DeviceGaussianBuffers& buffers) {
    freeDev(buffers.means3D);
    freeDev(buffers.scales);
    freeDev(buffers.rotations);
    freeDev(buffers.opacities);
    freeDev(buffers.shs);
    freeDev(buffers.colors_precomp);
    buffers.count = 0;
    buffers.sh_degree = 0;
    buffers.sh_coeffs = 0;
}

namespace internal {

bool filterVisibleToDevice(int full_count, int sh_degree, int sh_coeffs,
                           const float* d_means3D, const float* d_scales, const float* d_rotations,
                           const float* d_opacities, const float* d_shs, const float* d_colors_precomp,
                           const float view[16], const float proj[16], DeviceGaussianBuffers& out) {
    freeDeviceGaussianBuffers(out);
    if (full_count <= 0 || !d_means3D || !d_scales || !d_rotations || !d_opacities) {
        return false;
    }
    if (sh_degree > 0 && !d_shs) return false;
    if (sh_degree == 0 && !d_colors_precomp) return false;

    float* d_view = nullptr;
    float* d_proj = nullptr;
    bool* d_present = nullptr;
    int* d_counter = nullptr;

    auto fail = [&]() -> bool {
        if (d_view) cudaFree(d_view);
        if (d_proj) cudaFree(d_proj);
        if (d_present) cudaFree(d_present);
        if (d_counter) cudaFree(d_counter);
        freeDeviceGaussianBuffers(out);
        return false;
    };

    if (!cudaOk(cudaMalloc(&d_view, 16 * sizeof(float))) || !cudaOk(cudaMalloc(&d_proj, 16 * sizeof(float))) ||
        !cudaOk(cudaMalloc(&d_present, static_cast<size_t>(full_count) * sizeof(bool))) ||
        !cudaOk(cudaMalloc(&d_counter, sizeof(int)))) {
        return fail();
    }

    if (!cudaOk(cudaMemcpy(d_view, view, 16 * sizeof(float), cudaMemcpyHostToDevice)) ||
        !cudaOk(cudaMemcpy(d_proj, proj, 16 * sizeof(float), cudaMemcpyHostToDevice))) {
        return fail();
    }

    CudaRasterizer::Rasterizer::markVisible(full_count, const_cast<float*>(d_means3D), d_view, d_proj,
                                            d_present);

    const size_t cap = static_cast<size_t>(full_count);
    if (!cudaOk(cudaMalloc(&out.means3D, cap * 3 * sizeof(float))) ||
        !cudaOk(cudaMalloc(&out.scales, cap * 3 * sizeof(float))) ||
        !cudaOk(cudaMalloc(&out.rotations, cap * 4 * sizeof(float))) ||
        !cudaOk(cudaMalloc(&out.opacities, cap * sizeof(float)))) {
        return fail();
    }
    const bool use_sh = sh_degree > 0;
    if (use_sh) {
        if (!cudaOk(cudaMalloc(&out.shs, cap * static_cast<size_t>(sh_coeffs) * 3 * sizeof(float)))) {
            return fail();
        }
    } else if (!cudaOk(cudaMalloc(&out.colors_precomp, cap * 3 * sizeof(float)))) {
        return fail();
    }

    if (!cudaOk(cudaMemset(d_counter, 0, sizeof(int)))) {
        return fail();
    }

    const int threads = 256;
    const int blocks = (full_count + threads - 1) / threads;
    compactGaussianKernel<<<blocks, threads>>>(
        full_count, sh_coeffs, d_present, d_means3D, d_scales, d_rotations, d_opacities, d_shs,
        d_colors_precomp, use_sh, out.means3D, out.scales, out.rotations, out.opacities, out.shs,
        out.colors_precomp, d_counter);

    int visible_count = 0;
    if (!cudaOk(cudaMemcpy(&visible_count, d_counter, sizeof(int), cudaMemcpyDeviceToHost))) {
        return fail();
    }

    cudaFree(d_view);
    cudaFree(d_proj);
    cudaFree(d_present);
    cudaFree(d_counter);

    if (!cudaOk(cudaDeviceSynchronize())) {
        return fail();
    }

    out.count = visible_count;
    out.sh_degree = sh_degree;
    out.sh_coeffs = sh_coeffs;
    return visible_count > 0;
}

}  // namespace internal
}  // namespace gsplat
