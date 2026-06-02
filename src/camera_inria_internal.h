#pragma once

#include <gsplat_raster/gsplat_raster.h>

namespace gsplat {
namespace internal {

struct LookAtMats {
    float view[16];
    float proj[16];
    float cam_pos[3];
    float tan_fovx = 0.f;
    float tan_fovy = 0.f;
};

void buildLookAtMats(const double eye[3], const double center[3], const double up[3], double fov_y_deg,
                     double aspect, double znear, double zfar, const double ref_center[3],
                     LookAtMats& out, int view_proj_mode, bool proj_mul_pv);

// view_proj_mode: encoded ViewPack*16+ProjPack (see camera_inria.cpp)
int defaultMatMode();
int matModeInriaGlClip();
int matModeInriaFullProj();

void pickMatMode(const double eye[3], const double center[3], const double up[3], double fov_y_deg,
                 double aspect, double znear, double zfar, const double ref_center[3],
                 int (*score_fn)(const Camera& cam, void* ctx), void* ctx, LookAtMats& out);

void matsToCamera(const LookAtMats& mats, Camera& out);

}  // namespace internal
}  // namespace gsplat
