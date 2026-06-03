#include "camera_inria_internal.h"

#include <cmath>
#include <cstring>
#include <iostream>

namespace gsplat {
namespace internal {
namespace {

struct Mat4 {
    double m[16] = {};

    static Mat4 identity() {
        Mat4 I;
        I.m[0] = I.m[5] = I.m[10] = I.m[15] = 1.0;
        return I;
    }

    double operator()(int r, int c) const { return m[r * 4 + c]; }
    double& operator()(int r, int c) { return m[r * 4 + c]; }

    Mat4 operator*(const Mat4& b) const {
        Mat4 o;
        for (int r = 0; r < 4; ++r) {
            for (int c = 0; c < 4; ++c) {
                double s = 0;
                for (int k = 0; k < 4; ++k) s += (*this)(r, k) * b(k, c);
                o(r, c) = s;
            }
        }
        return o;
    }

    bool invert(Mat4& inv) const {
        double a[16], b[16];
        for (int i = 0; i < 16; ++i) a[i] = m[i];
        const double det =
            a[0] * (a[5] * (a[10] * a[15] - a[11] * a[14]) - a[6] * (a[9] * a[15] - a[11] * a[13]) +
                    a[7] * (a[9] * a[14] - a[10] * a[13])) -
            a[1] * (a[4] * (a[10] * a[15] - a[11] * a[14]) - a[6] * (a[8] * a[15] - a[11] * a[12]) +
                    a[7] * (a[8] * a[14] - a[10] * a[12])) +
            a[2] * (a[4] * (a[9] * a[15] - a[11] * a[13]) - a[5] * (a[8] * a[15] - a[11] * a[12]) +
                    a[7] * (a[8] * a[13] - a[9] * a[12])) -
            a[3] * (a[4] * (a[9] * a[14] - a[10] * a[13]) - a[5] * (a[8] * a[14] - a[10] * a[12]) +
                    a[6] * (a[8] * a[13] - a[9] * a[12]));
        if (std::fabs(det) < 1e-12) return false;
        const double invDet = 1.0 / det;
        b[0] = invDet * (a[5] * (a[10] * a[15] - a[11] * a[14]) - a[6] * (a[9] * a[15] - a[11] * a[13]) +
                         a[7] * (a[9] * a[14] - a[10] * a[13]));
        b[1] = invDet * -(a[1] * (a[10] * a[15] - a[11] * a[14]) - a[2] * (a[9] * a[15] - a[11] * a[13]) +
                          a[3] * (a[9] * a[14] - a[10] * a[13]));
        b[2] = invDet * (a[1] * (a[6] * a[15] - a[7] * a[14]) - a[2] * (a[5] * a[15] - a[7] * a[13]) +
                         a[3] * (a[5] * a[14] - a[6] * a[13]));
        b[3] = invDet * -(a[1] * (a[6] * a[11] - a[7] * a[10]) - a[2] * (a[5] * a[11] - a[7] * a[9]) +
                          a[3] * (a[5] * a[10] - a[6] * a[9]));
        b[4] = invDet * -(a[4] * (a[10] * a[15] - a[11] * a[14]) - a[6] * (a[8] * a[15] - a[11] * a[12]) +
                          a[7] * (a[8] * a[14] - a[10] * a[12]));
        b[5] = invDet * (a[0] * (a[10] * a[15] - a[11] * a[14]) - a[2] * (a[8] * a[15] - a[11] * a[12]) +
                         a[3] * (a[8] * a[14] - a[10] * a[12]));
        b[6] = invDet * -(a[0] * (a[6] * a[15] - a[7] * a[14]) - a[2] * (a[4] * a[15] - a[7] * a[12]) +
                          a[3] * (a[4] * a[14] - a[6] * a[12]));
        b[7] = invDet * (a[0] * (a[6] * a[11] - a[7] * a[10]) - a[2] * (a[4] * a[11] - a[7] * a[8]) +
                         a[3] * (a[4] * a[10] - a[6] * a[8]));
        b[8] = invDet * (a[4] * (a[9] * a[15] - a[11] * a[13]) - a[5] * (a[8] * a[15] - a[11] * a[12]) +
                         a[7] * (a[8] * a[13] - a[9] * a[12]));
        b[9] = invDet * -(a[0] * (a[9] * a[15] - a[11] * a[13]) - a[1] * (a[8] * a[15] - a[11] * a[12]) +
                          a[3] * (a[8] * a[13] - a[9] * a[12]));
        b[10] = invDet * (a[0] * (a[5] * a[15] - a[7] * a[13]) - a[1] * (a[4] * a[15] - a[7] * a[12]) +
                          a[3] * (a[4] * a[13] - a[5] * a[12]));
        b[11] = invDet * -(a[0] * (a[5] * a[11] - a[7] * a[9]) - a[1] * (a[4] * a[11] - a[7] * a[8]) +
                           a[3] * (a[4] * a[9] - a[5] * a[8]));
        b[12] = invDet * -(a[4] * (a[9] * a[14] - a[10] * a[13]) - a[5] * (a[8] * a[14] - a[10] * a[12]) +
                           a[6] * (a[8] * a[13] - a[9] * a[12]));
        b[13] = invDet * (a[0] * (a[9] * a[14] - a[10] * a[13]) - a[1] * (a[8] * a[14] - a[10] * a[12]) +
                          a[2] * (a[8] * a[13] - a[9] * a[12]));
        b[14] = invDet * -(a[0] * (a[5] * a[14] - a[6] * a[13]) - a[1] * (a[4] * a[14] - a[6] * a[12]) +
                           a[2] * (a[4] * a[13] - a[5] * a[12]));
        b[15] = invDet * (a[0] * (a[5] * a[10] - a[6] * a[9]) - a[1] * (a[4] * a[10] - a[6] * a[8]) +
                          a[2] * (a[4] * a[9] - a[5] * a[8]));
        inv = Mat4();
        std::memcpy(inv.m, b, sizeof(b));
        return true;
    }
};

enum class ViewPack { Inria = 0, Gl = 1 };
enum class ProjPack { InriaClip = 0, GlClip = 1, InriaVP = 2, GlPV = 3, GlSep = 4 };

int encodeMode(ViewPack v, ProjPack p) { return static_cast<int>(v) * 16 + static_cast<int>(p); }
void decodeMode(int code, ViewPack& v, ProjPack& p) {
    v = static_cast<ViewPack>(code / 16);
    p = static_cast<ProjPack>(code % 16);
}

const char* viewName(ViewPack v) { return v == ViewPack::Inria ? "inria_view" : "gl_view"; }
const char* projName(ProjPack p) {
    switch (p) {
    case ProjPack::InriaClip:
        return "inria_pack(proj*V)";
    case ProjPack::GlClip:
        return "gl_pack(proj*V)";
    case ProjPack::InriaVP:
        return "inria_pack(V*P)";
    case ProjPack::GlPV:
        return "gl_pack(P*V)";
    case ProjPack::GlSep:
        return "gl_pack(P)";
    default:
        return "?";
    }
}

Mat4 lookAt(const double eye[3], const double center[3], const double up[3]) {
    const double f[3] = {center[0] - eye[0], center[1] - eye[1], center[2] - eye[2]};
    double flen = std::sqrt(f[0] * f[0] + f[1] * f[1] + f[2] * f[2]);
    if (flen < 1e-12) flen = 1;
    const double fn[3] = {f[0] / flen, f[1] / flen, f[2] / flen};

    double s[3] = {fn[1] * up[2] - fn[2] * up[1], fn[2] * up[0] - fn[0] * up[2],
                   fn[0] * up[1] - fn[1] * up[0]};
    double slen = std::sqrt(s[0] * s[0] + s[1] * s[1] + s[2] * s[2]);
    if (slen < 1e-12) {
        s[0] = 1;
        s[1] = s[2] = 0;
        slen = 1;
    }
    s[0] /= slen;
    s[1] /= slen;
    s[2] /= slen;

    const double u[3] = {s[1] * fn[2] - s[2] * fn[1], s[2] * fn[0] - s[0] * fn[2],
                         s[0] * fn[1] - s[1] * fn[0]};

    Mat4 M = Mat4::identity();
    M(0, 0) = s[0];
    M(0, 1) = s[1];
    M(0, 2) = s[2];
    M(1, 0) = u[0];
    M(1, 1) = u[1];
    M(1, 2) = u[2];
    M(2, 0) = -fn[0];
    M(2, 1) = -fn[1];
    M(2, 2) = -fn[2];
    M(0, 3) = -(s[0] * eye[0] + s[1] * eye[1] + s[2] * eye[2]);
    M(1, 3) = -(u[0] * eye[0] + u[1] * eye[1] + u[2] * eye[2]);
    M(2, 3) = fn[0] * eye[0] + fn[1] * eye[1] + fn[2] * eye[2];
    return M;
}

Mat4 perspectiveGl(double fovY_rad, double aspect, double znear, double zfar) {
    const double t = std::tan(fovY_rad * 0.5) * znear;
    const double b = -t;
    const double r = t * aspect;
    const double l = -r;
    Mat4 P = Mat4::identity();
    P(0, 0) = 2.0 * znear / (r - l);
    P(1, 1) = 2.0 * znear / (t - b);
    P(0, 2) = (r + l) / (r - l);
    P(1, 2) = (t + b) / (t - b);
    P(2, 2) = -(zfar + znear) / (zfar - znear);
    P(2, 3) = -(2.0 * zfar * znear) / (zfar - znear);
    P(3, 2) = -1.0;
    P(3, 3) = 0.0;
    return P;
}

Mat4 projectionInria(double znear, double zfar, double fovX, double fovY) {
    const double tanHalfFovY = std::tan(fovY * 0.5);
    const double tanHalfFovX = std::tan(fovX * 0.5);
    const double top = tanHalfFovY * znear;
    const double bottom = -top;
    const double right = tanHalfFovX * znear;
    const double left = -right;

    Mat4 P = Mat4::identity();
    P(0, 0) = 2.0 * znear / (right - left);
    P(1, 1) = 2.0 * znear / (top - bottom);
    P(0, 2) = (right + left) / (right - left);
    P(1, 2) = (top + bottom) / (top - bottom);
    P(3, 2) = 1.0;
    P(2, 2) = zfar / (zfar - znear);
    P(2, 3) = -(zfar * znear) / (zfar - znear);
    return P;
}

void packInria(const Mat4& M, float out[16]) {
    for (int c = 0; c < 4; ++c) {
        for (int r = 0; r < 4; ++r) {
            out[c * 4 + r] = static_cast<float>(M(c, r));
        }
    }
}

void packColMajor(const Mat4& M, float out[16]) {
    for (int c = 0; c < 4; ++c) {
        for (int r = 0; r < 4; ++r) {
            out[c * 4 + r] = static_cast<float>(M(r, c));
        }
    }
}

void mulColMajor4x4(float* out, const float* a, const float* b) {
    for (int col = 0; col < 4; ++col) {
        for (int row = 0; row < 4; ++row) {
            double sum = 0.0;
            for (int k = 0; k < 4; ++k) {
                sum += static_cast<double>(a[k * 4 + row]) * static_cast<double>(b[col * 4 + k]);
            }
            out[col * 4 + row] = static_cast<float>(sum);
        }
    }
}

float cudaViewZ(const float view[16], double x, double y, double z) {
    return view[2] * static_cast<float>(x) + view[6] * static_cast<float>(y) +
           view[10] * static_cast<float>(z) + view[14];
}

float mvRow2Z(const Mat4& V, double x, double y, double z) {
    return static_cast<float>(V(2, 0) * x + V(2, 1) * y + V(2, 2) * z + V(2, 3));
}

void alignViewZ(Mat4& V, const Mat4& V_osg, double cx, double cy, double cz) {
    if (mvRow2Z(V_osg, cx, cy, cz) < 0.f) {
        for (int r = 0; r < 4; ++r) V(r, 2) = -V(r, 2);
    }
    float test[16];
    packInria(V, test);
    if (cudaViewZ(test, cx, cy, cz) <= 0.2f) {
        for (int r = 0; r < 4; ++r) V(r, 2) = -V(r, 2);
    }
}

void eyeWorldFromOsgView(const Mat4& V_raw, float out[3]) {
    Mat4 inv;
    if (!V_raw.invert(inv)) {
        out[0] = out[1] = out[2] = 0.f;
        return;
    }
    // OSG stores matrices row-major; camera world position is inv row 3 (cols 0..2).
    out[0] = static_cast<float>(inv(3, 0));
    out[1] = static_cast<float>(inv(3, 1));
    out[2] = static_cast<float>(inv(3, 2));
}

void camPosFromView(const float view_cm[16], float out[3]) {
    Mat4 M;
    for (int c = 0; c < 4; ++c) {
        for (int r = 0; r < 4; ++r) M(r, c) = view_cm[c * 4 + r];
    }
    Mat4 inv;
    if (!M.invert(inv)) {
        out[0] = out[1] = out[2] = 0.f;
        return;
    }
    out[0] = static_cast<float>(inv(0, 3));
    out[1] = static_cast<float>(inv(1, 3));
    out[2] = static_cast<float>(inv(2, 3));
}

struct SceneMats {
    Mat4 V;
    Mat4 proj_gl;
    Mat4 P_inria;
    double fovX;
    double fovY;
};

SceneMats makeScene(const double eye[3], const double center[3], const double up[3], double fov_y_deg,
                   double aspect, double znear, double zfar, const double ref_center[3]) {
    SceneMats s;
    const double fovY = fov_y_deg * 3.14159265358979323846 / 180.0;
    s.fovY = fovY;
    s.fovX = 2.0 * std::atan(std::tan(fovY * 0.5) * aspect);
    Mat4 V = lookAt(eye, center, up);
    (void)ref_center;
    // LookAt 路径直接使用几何定义的 V，避免启发式 Z 翻转造成镜像/拉伸。
    s.V = V;
    s.proj_gl = perspectiveGl(fovY, aspect, znear, zfar);
    s.P_inria = projectionInria(znear, zfar, s.fovX, s.fovY);
    return s;
}

void buildViewMatrix(float out[16], ViewPack mode, const Mat4& V) {
    if (mode == ViewPack::Inria) {
        packInria(V, out);
    } else {
        packColMajor(V, out);
    }
}

void buildProjMatrix(float out[16], ProjPack mode, const SceneMats& scene, bool proj_mul_pv) {
    const Mat4 clip = scene.proj_gl * scene.V;
    float v[16];
    float p[16];
    switch (mode) {
    case ProjPack::InriaClip:
        packInria(clip, out);
        break;
    case ProjPack::GlClip:
        packColMajor(clip, out);
        break;
    case ProjPack::InriaVP:
        if (proj_mul_pv) {
            packInria(scene.P_inria * scene.V, out);
        } else {
            packInria(scene.V * scene.P_inria, out);
        }
        break;
    case ProjPack::GlPV:
        packColMajor(scene.V, v);
        packColMajor(scene.proj_gl, p);
        mulColMajor4x4(out, p, v);
        break;
    case ProjPack::GlSep:
        packColMajor(scene.proj_gl, out);
        break;
    }
}

void applyMode(const SceneMats& scene, ViewPack vp, ProjPack pp, bool proj_mul_pv, LookAtMats& out) {
    buildViewMatrix(out.view, vp, scene.V);
    buildProjMatrix(out.proj, pp, scene, proj_mul_pv);
    camPosFromView(out.view, out.cam_pos);
    out.tan_fovx = static_cast<float>(std::tan(scene.fovX * 0.5));
    out.tan_fovy = static_cast<float>(std::tan(scene.fovY * 0.5));
}

Mat4 mat4FromArray(const double m[16]) {
    Mat4 M;
    std::memcpy(M.m, m, sizeof(M.m));
    return M;
}

// Match 3dgs-osg-viewer fovFromOsgProjection (OSG uses OpenGL perspective: P(3,2)=-1).
bool fovFromOsgPerspective(const Mat4& proj, double& fovX, double& fovY, double& znear, double& zfar) {
    znear = 0.01;
    zfar = 10000.0;
    const double sy = std::abs(proj(1, 1));
    if (sy < 1e-9) return false;
    fovY = 2.0 * std::atan(1.0 / sy);
    const double sx = std::abs(proj(0, 0));
    if (sx < 1e-9) return false;
    // OpenGL perspective: m00=1/(aspect*tan(fovy/2)), m11=1/tan(fovy/2)
    // => tan(fovx/2) = aspect*tan(fovy/2) = (m11/m00)*tan(fovy/2).
    fovX = 2.0 * std::atan((sy / sx) * std::tan(fovY * 0.5));
    const double m22 = proj(2, 2);
    const double m23 = proj(2, 3);
    const double m32 = proj(3, 2);
    if (std::abs(m32 + 1.0) < 0.05) {
        zfar = m23 / (m22 + 1.0);
        znear = m23 / (m22 - 1.0);
        if (znear > zfar) std::swap(znear, zfar);
        if (znear <= 0 || zfar <= znear || !std::isfinite(znear) || !std::isfinite(zfar)) {
            znear = 0.01;
            zfar = 10000.0;
        }
    }
    return std::isfinite(fovX) && std::isfinite(fovY) && fovX > 0 && fovY > 0;
}

struct OsgScene {
    Mat4 V;
    Mat4 V_raw;
    Mat4 proj_gl;
    double fovX = 0;
    double fovY = 0;
    double znear = 0.01;
    double zfar = 10000.0;
};

OsgScene makeOsgScene(const double view_osg[16], const double proj_osg[16], const double ref_center[3]) {
    OsgScene s;
    s.V_raw = mat4FromArray(view_osg);
    s.proj_gl = mat4FromArray(proj_osg);
    (void)ref_center;
    // OSG 路径直接使用当前帧原始 view，避免额外启发式改写。
    s.V = s.V_raw;
    if (!fovFromOsgPerspective(s.proj_gl, s.fovX, s.fovY, s.znear, s.zfar)) {
        s.fovY = 60.0 * 3.14159265358979323846 / 180.0;
        s.fovX = s.fovY;
        s.znear = 0.01;
        s.zfar = 10000.0;
    }
    return s;
}

/** OpenGL unified path: view=world->view, proj=clip<-world with NDC z in [-1,1]. */
void buildOsgCameraGl(const OsgScene& scene, LookAtMats& out) {
    packColMajor(scene.V, out.view);
    const Mat4 clip = scene.proj_gl * scene.V;
    packColMajor(clip, out.proj);
    eyeWorldFromOsgView(scene.V_raw, out.cam_pos);
    if (std::fabs(out.cam_pos[0]) < 1e-6f && std::fabs(out.cam_pos[1]) < 1e-6f &&
        std::fabs(out.cam_pos[2]) < 1e-6f) {
        std::cout << "[CUDA camera] eye from osg view is near-zero, fallback to SDK view inversion\n";
        camPosFromView(out.view, out.cam_pos);
    }
    out.tan_fovx = static_cast<float>(std::tan(scene.fovX * 0.5));
    out.tan_fovy = static_cast<float>(std::tan(scene.fovY * 0.5));
}

}  // namespace

int defaultMatMode() { return encodeMode(ViewPack::Gl, ProjPack::GlClip); }

void matsToCamera(const LookAtMats& mats, Camera& out) {
    std::memcpy(out.view, mats.view, sizeof(out.view));
    std::memcpy(out.proj, mats.proj, sizeof(out.proj));
    std::memcpy(out.cam_pos, mats.cam_pos, sizeof(out.cam_pos));
    out.tan_fovx = mats.tan_fovx;
    out.tan_fovy = mats.tan_fovy;
}

void buildLookAtMats(const double eye[3], const double center[3], const double up[3], double fov_y_deg,
                     double aspect, double znear, double zfar, const double ref_center[3],
                     LookAtMats& out, int mode_code, bool proj_mul_pv) {
    const SceneMats scene = makeScene(eye, center, up, fov_y_deg, aspect, znear, zfar, ref_center);
    ViewPack vp;
    ProjPack pp;
    decodeMode(mode_code, vp, pp);
    applyMode(scene, vp, pp, proj_mul_pv, out);
}

void pickMatMode(const double eye[3], const double center[3], const double up[3], double fov_y_deg,
                 double aspect, double znear, double zfar, const double ref_center[3],
                 int (*score_fn)(const Camera& cam, void* ctx), void* ctx, LookAtMats& out) {
    const SceneMats scene = makeScene(eye, center, up, fov_y_deg, aspect, znear, zfar, ref_center);

    struct Cand {
        ViewPack vp;
        ProjPack pp;
        bool mul_pv;
    };
    // Match 3dgs-osg-viewer probing order: try GL view + GL clip first.
    const Cand cands[] = {
        {ViewPack::Gl, ProjPack::GlClip, false},
        {ViewPack::Inria, ProjPack::GlClip, false},
        {ViewPack::Inria, ProjPack::InriaVP, false},
        {ViewPack::Inria, ProjPack::InriaClip, false},
        {ViewPack::Inria, ProjPack::InriaVP, true},
        {ViewPack::Gl, ProjPack::InriaVP, false},
    };

    int best = -1;
    int best_mode = encodeMode(ViewPack::Gl, ProjPack::GlClip);
    bool best_mul = false;

    for (const Cand& c : cands) {
        LookAtMats tmp;
        applyMode(scene, c.vp, c.pp, c.mul_pv, tmp);
        Camera cam;
        std::memcpy(cam.view, tmp.view, sizeof(cam.view));
        std::memcpy(cam.proj, tmp.proj, sizeof(cam.proj));
        std::memcpy(cam.cam_pos, tmp.cam_pos, sizeof(cam.cam_pos));
        cam.tan_fovx = tmp.tan_fovx;
        cam.tan_fovy = tmp.tan_fovy;
        const int score = score_fn ? score_fn(cam, ctx) : 0;
        std::cout << "  CUDA mat " << viewName(c.vp) << " + " << projName(c.pp)
                  << (c.mul_pv ? " (P*V)" : "") << " score~" << score << "\n";
        if (score > best) {
            best = score;
            best_mode = encodeMode(c.vp, c.pp);
            best_mul = c.mul_pv;
        }
    }

    applyMode(scene, static_cast<ViewPack>(best_mode / 16), static_cast<ProjPack>(best_mode % 16),
              best_mul, out);
    std::cout << "CUDA camera: picked " << viewName(static_cast<ViewPack>(best_mode / 16)) << " + "
              << projName(static_cast<ProjPack>(best_mode % 16)) << " (NDC~=" << best << ")\n";
}

void buildOsgMats(const double view_osg[16], const double proj_osg[16], const double ref_center[3],
                  LookAtMats& out) {
    const OsgScene scene = makeOsgScene(view_osg, proj_osg, ref_center);
    buildOsgCameraGl(scene, out);
}

}  // namespace internal
}  // namespace gsplat
