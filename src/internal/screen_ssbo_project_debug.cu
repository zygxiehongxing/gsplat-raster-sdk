#include "screen_ssbo_project_debug.h"

#include <gsplat_raster/screen_gaussian_pixel.h>

#include <cuda_runtime.h>

namespace gsplat {
namespace internal {
namespace {

__device__ void mulRowMat4(const float* M, float x, float y, float z, float w, float out[4]) {
    const float in[4] = {x, y, z, w};
    for (int c = 0; c < 4; ++c) {
        float s = 0.f;
        for (int r = 0; r < 4; ++r) {
            s += in[r] * M[r * 4 + c];
        }
        out[c] = s;
    }
}

/// GLSL: vec4(pos,1) * u_view * u_proj（与 gl_screen_cache clipGlslRowOsg 相同布局）
__device__ bool worldToPixelGlslRow(const float* view_glsl, const float* proj_glsl, float x, float y, float z,
                                    int w, int h, int& px, int& py) {
    float t[4] = {};
    float c[4] = {};
    mulRowMat4(view_glsl, x, y, z, 1.f, t);
    mulRowMat4(proj_glsl, t[0], t[1], t[2], t[3], c);
    const float cw = c[3];
    if (cw <= 1e-7f) return false;
    const float iw = 1.f / cw;
    const float nx = c[0] * iw;
    const float ny = c[1] * iw;
    const float fx = ((nx + 1.f) * static_cast<float>(w) - 1.f) * 0.5f;
    const float fy = ((ny + 1.f) * static_cast<float>(h) - 1.f) * 0.5f;
    px = static_cast<int>(fx + 0.5f);
    py = static_cast<int>(fy + 0.5f);
    return px >= 0 && py >= 0 && px < w && py < h;
}

__device__ bool worldToPixelRowMajor(const float* proj, float x, float y, float z, int w, int h, int& px,
                                     int& py) {
    const float cw = proj[12] * x + proj[13] * y + proj[14] * z + proj[15];
    if (cw <= 1e-7f) return false;
    const float iw = 1.f / cw;
    const float nx = (proj[0] * x + proj[1] * y + proj[2] * z + proj[3]) * iw;
    const float ny = (proj[4] * x + proj[5] * y + proj[6] * z + proj[7]) * iw;
    const float fx = ((nx + 1.f) * static_cast<float>(w) - 1.f) * 0.5f;
    const float fy = ((ny + 1.f) * static_cast<float>(h) - 1.f) * 0.5f;
    px = static_cast<int>(fx + 0.5f);
    py = static_cast<int>(fy + 0.5f);
    return px >= 0 && py >= 0 && px < w && py < h;
}

__global__ void clearDarkKernel(int n, uint8_t* rgb) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    rgb[i * 3 + 0] = 24;
    rgb[i * 3 + 1] = 24;
    rgb[i * 3 + 2] = 28;
}

__global__ void markProjectedMeansGlslKernel(int n, const GaussianPixelGpu* cells, const float* view_glsl,
                                             const float* proj_glsl, int w, int h, uint8_t* rgb,
                                             uint32_t expected_frame_id, bool image_top_origin, int* marked_out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const GaussianPixelGpu& g = cells[i];
    if (g.mean_opacity[3] <= 1e-4f) return;
    if (expected_frame_id != 0xFFFFFFFFu && g.frame_id != expected_frame_id) return;
    int px = 0;
    int py = 0;
    if (!worldToPixelGlslRow(view_glsl, proj_glsl, g.mean_opacity[0], g.mean_opacity[1], g.mean_opacity[2], w, h,
                             px, py)) {
        return;
    }
    if (image_top_origin) py = h - 1 - py;
    const size_t off = (static_cast<size_t>(py) * static_cast<size_t>(w) + static_cast<size_t>(px)) * 3u;
    rgb[off + 0] = 255;
    rgb[off + 1] = 0;
    rgb[off + 2] = 0;
    if (marked_out) atomicAdd(marked_out, 1);
}

__global__ void markProjectedMeansKernel(int n, const GaussianPixelGpu* cells, const float* view, const float* proj,
                                         int w, int h, uint8_t* rgb, uint32_t expected_frame_id, int* marked_out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const GaussianPixelGpu& g = cells[i];
    if (g.mean_opacity[3] <= 1e-4f) return;
    if (expected_frame_id != 0xFFFFFFFFu && g.frame_id != expected_frame_id) return;
    int px = 0;
    int py = 0;
    if (!worldToPixelGlslRow(view, proj, g.mean_opacity[0], g.mean_opacity[1], g.mean_opacity[2], w, h, px, py)) {
        return;
    }
    const size_t off = (static_cast<size_t>(py) * static_cast<size_t>(w) + static_cast<size_t>(px)) * 3u;
    rgb[off + 0] = 255;
    rgb[off + 1] = 0;
    rgb[off + 2] = 0;
    if (marked_out) atomicAdd(marked_out, 1);
}

__device__ uint8_t colorToU8(float v) {
    if (v < 0.f) v = 0.f;
    if (v > 1.f) v = 1.f;
    return static_cast<uint8_t>(v * 255.f + 0.5f);
}

__global__ void projectColorCellsKernel(int n, const ScreenColorCellGpu* cells, const float* view_glsl,
                                        const float* proj_glsl, int w, int h, uint8_t* rgb, bool image_top_origin,
                                        int* marked_out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const ScreenColorCellGpu& c = cells[i];
    if (c.rgba[3] <= 1e-4f && c.rgba[0] <= 1e-4f && c.rgba[1] <= 1e-4f && c.rgba[2] <= 1e-4f) return;
    int px = 0;
    int py = 0;
    if (!worldToPixelGlslRow(view_glsl, proj_glsl, c.xyz[0], c.xyz[1], c.xyz[2], w, h, px, py)) return;
    if (image_top_origin) py = h - 1 - py;
    const size_t off = (static_cast<size_t>(py) * static_cast<size_t>(w) + static_cast<size_t>(px)) * 3u;
    rgb[off + 0] = colorToU8(c.rgba[0]);
    rgb[off + 1] = colorToU8(c.rgba[1]);
    rgb[off + 2] = colorToU8(c.rgba[2]);
    if (marked_out) atomicAdd(marked_out, 1);
}

}  // namespace

bool renderSsboMeanProjectMap(const Camera& cam, int width, int height, const GaussianPixelGpu* d_cells,
                              std::vector<uint8_t>& out_rgb, int& marked_pixels,
                              uint32_t expected_frame_id) {
    marked_pixels = 0;
    if (width <= 0 || height <= 0 || !d_cells) return false;
    const int n = width * height;
    const size_t rgb_bytes = static_cast<size_t>(n) * 3u;
    out_rgb.assign(rgb_bytes, 0);

    uint8_t* d_rgb = nullptr;
    float* d_view = nullptr;
    float* d_proj = nullptr;
    int* d_marked = nullptr;
    if (cudaMalloc(&d_rgb, rgb_bytes) != cudaSuccess) return false;
    if (cudaMalloc(&d_view, 16 * sizeof(float)) != cudaSuccess) {
        cudaFree(d_rgb);
        return false;
    }
    if (cudaMalloc(&d_proj, 16 * sizeof(float)) != cudaSuccess) {
        cudaFree(d_rgb);
        cudaFree(d_view);
        return false;
    }
    if (cudaMalloc(&d_marked, sizeof(int)) != cudaSuccess) {
        cudaFree(d_rgb);
        cudaFree(d_view);
        cudaFree(d_proj);
        return false;
    }

    const int threads = 256;
    const int blocks_clear = (n + threads - 1) / threads;
    clearDarkKernel<<<blocks_clear, threads>>>(n, d_rgb);
    cudaMemcpy(d_view, cam.view, 16 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_proj, cam.proj, 16 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_marked, 0, sizeof(int));
    markProjectedMeansKernel<<<blocks_clear, threads>>>(n, d_cells, d_view, d_proj, width, height, d_rgb,
                                                      expected_frame_id, d_marked);
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(d_rgb);
        cudaFree(d_view);
        cudaFree(d_proj);
        cudaFree(d_marked);
        return false;
    }
    cudaMemcpy(out_rgb.data(), d_rgb, rgb_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(&marked_pixels, d_marked, sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaFree(d_rgb);
    cudaFree(d_view);
    cudaFree(d_proj);
    cudaFree(d_marked);
    return true;
}

bool renderSsboMeanProjectMapGlsl(const ScreenCacheMetaGpu& meta, int width, int height,
                                  const GaussianPixelGpu* d_cells, std::vector<uint8_t>& out_rgb,
                                  int& marked_pixels, uint32_t expected_frame_id, bool image_top_origin) {
    marked_pixels = 0;
    if (width <= 0 || height <= 0 || !d_cells) return false;
    const int n = width * height;
    const size_t rgb_bytes = static_cast<size_t>(n) * 3u;
    out_rgb.assign(rgb_bytes, 0);

    uint8_t* d_rgb = nullptr;
    float* d_view = nullptr;
    float* d_proj = nullptr;
    int* d_marked = nullptr;
    if (cudaMalloc(&d_rgb, rgb_bytes) != cudaSuccess) return false;
    if (cudaMalloc(&d_view, 16 * sizeof(float)) != cudaSuccess) {
        cudaFree(d_rgb);
        return false;
    }
    if (cudaMalloc(&d_proj, 16 * sizeof(float)) != cudaSuccess) {
        cudaFree(d_rgb);
        cudaFree(d_view);
        return false;
    }
    if (cudaMalloc(&d_marked, sizeof(int)) != cudaSuccess) {
        cudaFree(d_rgb);
        cudaFree(d_view);
        cudaFree(d_proj);
        return false;
    }

    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    clearDarkKernel<<<blocks, threads>>>(n, d_rgb);
    cudaMemcpy(d_view, meta.view_glsl, 16 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_proj, meta.proj_glsl, 16 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_marked, 0, sizeof(int));
    markProjectedMeansGlslKernel<<<blocks, threads>>>(n, d_cells, d_view, d_proj, width, height, d_rgb,
                                                      expected_frame_id, image_top_origin, d_marked);
    if (cudaGetLastError() != cudaSuccess) {
        cudaFree(d_rgb);
        cudaFree(d_view);
        cudaFree(d_proj);
        cudaFree(d_marked);
        return false;
    }
    cudaMemcpy(out_rgb.data(), d_rgb, rgb_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(&marked_pixels, d_marked, sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaFree(d_rgb);
    cudaFree(d_view);
    cudaFree(d_proj);
    cudaFree(d_marked);
    return true;
}

bool renderSsboColorCellsProjectMapGlsl(const ScreenCacheMetaGpu& meta, int width, int height,
                                        const ScreenColorCellGpu* d_cells, std::vector<uint8_t>& out_rgb,
                                        int& marked_pixels, bool image_top_origin) {
    marked_pixels = 0;
    if (width <= 0 || height <= 0 || !d_cells) return false;
    const int meta_w = static_cast<int>(meta.info[1]);
    const int meta_h = static_cast<int>(meta.info[2]);
    if (meta_w > 0 && meta_h > 0 && (meta_w != width || meta_h != height)) return false;
    const int n = width * height;
    const size_t rgb_bytes = static_cast<size_t>(n) * 3u;
    out_rgb.assign(rgb_bytes, 0);

    uint8_t* d_rgb = nullptr;
    float* d_view = nullptr;
    float* d_proj = nullptr;
    int* d_marked = nullptr;
    if (cudaMalloc(&d_rgb, rgb_bytes) != cudaSuccess) return false;
    if (cudaMalloc(&d_view, 16 * sizeof(float)) != cudaSuccess) {
        cudaFree(d_rgb);
        return false;
    }
    if (cudaMalloc(&d_proj, 16 * sizeof(float)) != cudaSuccess) {
        cudaFree(d_rgb);
        cudaFree(d_view);
        return false;
    }
    if (cudaMalloc(&d_marked, sizeof(int)) != cudaSuccess) {
        cudaFree(d_rgb);
        cudaFree(d_view);
        cudaFree(d_proj);
        return false;
    }

    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    clearDarkKernel<<<blocks, threads>>>(n, d_rgb);
    cudaMemcpy(d_view, meta.view_glsl, 16 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_proj, meta.proj_glsl, 16 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_marked, 0, sizeof(int));
    projectColorCellsKernel<<<blocks, threads>>>(n, d_cells, d_view, d_proj, width, height, d_rgb, image_top_origin,
                                               d_marked);
    if (cudaGetLastError() != cudaSuccess) {
        cudaFree(d_rgb);
        cudaFree(d_view);
        cudaFree(d_proj);
        cudaFree(d_marked);
        return false;
    }
    cudaMemcpy(out_rgb.data(), d_rgb, rgb_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(&marked_pixels, d_marked, sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaFree(d_rgb);
    cudaFree(d_view);
    cudaFree(d_proj);
    cudaFree(d_marked);
    return true;
}

}  // namespace internal
}  // namespace gsplat
