#pragma once

#include <cstdint>

namespace gsplat {

/// SSBO 头部元数据：与该帧 GL 绘制同源（与 uniform u_view/u_proj 一致）。
struct ScreenCacheMetaGpu {
    float view_glsl[16];  // row: vec4(pos,1)*u_view（glUniformMatrix4fv GL_TRUE）
    float proj_glsl[16];  // row: vec4*t*u_proj（仅 P，非 P*V）
    float cam_pos_tan[4]; // cam_pos.xyz, tan_fovx
    uint32_t info[4];     // [0]=frame_id [1]=capacity [2]=filled [3]=viewport_xy packed
};

static_assert(sizeof(ScreenCacheMetaGpu) == 160, "ScreenCacheMetaGpu must be 160 bytes");

/// App SSBO 列表默认容量（视锥内 3D 高斯槽位数）。
constexpr uint32_t kDefaultScreenSsboCapacity = 2000000u;

/// GL SSBO 头：meta + GPU atomic 追加计数（后接 GaussianPixelGpu cells[capacity]）。
struct ScreenSsboListHeaderGpu {
    ScreenCacheMetaGpu meta;
    uint32_t append_count;
    uint32_t _pad[3];
};

static_assert(sizeof(ScreenSsboListHeaderGpu) == 176, "ScreenSsboListHeaderGpu must be 176 bytes");

/// Pack GL viewport origin (vp_x, vp_y) into info[3]; each component must fit uint16.
inline uint32_t packScreenCacheViewport(int vp_x, int vp_y) {
    const uint32_t x = static_cast<uint32_t>(vp_x) & 0xFFFFu;
    const uint32_t y = static_cast<uint32_t>(vp_y) & 0xFFFFu;
    return x | (y << 16u);
}

inline void unpackScreenCacheViewport(uint32_t packed, int& vp_x, int& vp_y) {
    vp_x = static_cast<int>(packed & 0xFFFFu);
    vp_y = static_cast<int>((packed >> 16u) & 0xFFFFu);
}

/// 单 SSBO AoS：一颗 3D 高斯（GL std430 / CUDA 共用布局，DC 模式）。
struct GaussianPixelGpu {
    float mean_opacity[4];  // xyz, opacity
    float scale_pad[4];     // scale xyz
    float rot[4];           // quaternion wxyz
    float color_pad[4];     // rgb
    uint32_t frame_id;      // frame stamp written by GL pass
    uint32_t _pad0;
    uint32_t _pad1;
    uint32_t _pad2;
};

static_assert(sizeof(GaussianPixelGpu) == 80, "GaussianPixelGpu must be 80 bytes (std430 stride)");

/// 片元颜色 SSBO（GL binding=1）：与 Gaussian SSBO 同索引，存赢家片元的 rgb 与世界坐标 xyz。
struct ScreenColorCellGpu {
    float rgba[4];  // rgb + alpha(1)
    float xyz[4];   // world xyz, w unused
};

static_assert(sizeof(ScreenColorCellGpu) == 32, "ScreenColorCellGpu must be 32 bytes (std430)");

}  // namespace gsplat
