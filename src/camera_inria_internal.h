#pragma once

#include <gsplat_raster/gsplat_raster.h>

namespace gsplat {
namespace internal {

/// 相机矩阵打包结果：直接可写入 Camera。
struct LookAtMats {
    float view[16];
    float proj[16];
    float cam_pos[3];
    float tan_fovx = 0.f;
    float tan_fovy = 0.f;
};

/// 基于 eye/center/up 生成相机矩阵（支持多打包模式）。
void buildLookAtMats(const double eye[3], const double center[3], const double up[3], double fov_y_deg,
                     double aspect, double znear, double zfar, const double ref_center[3],
                     LookAtMats& out, int view_proj_mode, bool proj_mul_pv);

/// 默认打包模式编码（ViewPack*16+ProjPack）。
int defaultMatMode();

/// 根据评分函数在候选模式里选择最优矩阵打包。
void pickMatMode(const double eye[3], const double center[3], const double up[3], double fov_y_deg,
                 double aspect, double znear, double zfar, const double ref_center[3],
                 int (*score_fn)(const Camera& cam, void* ctx), void* ctx, LookAtMats& out);

/// 将 OSG row-major view/proj 转成 Inria 光栅可用矩阵。
void buildOsgMats(const double view_osg[16], const double proj_osg[16], const double ref_center[3],
                  LookAtMats& out);

/// 把 LookAtMats 拷贝到公开 Camera 结构。
void matsToCamera(const LookAtMats& mats, Camera& out);

}  // namespace internal
}  // namespace gsplat
