#pragma once

#include <gsplat_raster/gsplat_raster.h>

#include <cstdint>
#include <string>
#include <vector>

namespace gsplat {
namespace internal {

class CudaRasterEngine {
public:
    CudaRasterEngine();
    ~CudaRasterEngine();

    void setSettings(const RenderSettings& s);
    bool upload(const std::vector<Gaussian>& gaussians);

    int numSplats() const { return num_splats_; }
    int lastVisible() const { return last_visible_; }

    bool render(int w, int h, const float view[16], const float proj[16], const float cam_pos[3],
                float tan_fovx, float tan_fovy, std::vector<uint8_t>& rgb);

    int countFrustumPass(const float view[16], const float proj[16], int max_samples) const;

    /** Low-res forward to count splats with radii > 0 (for camera matrix pick). */
    int probeVisibleSplats(int w, int h, const float view[16], const float proj[16],
                           const float cam_pos[3], float tan_fovx, float tan_fovy);

private:
    void uploadOne(const std::vector<Gaussian>& gaussians);

    RenderSettings settings_;
    int num_splats_ = 0;
    int sh_degree_ = 0;
    int sh_coeffs_ = 0;
    float scale_modifier_ = 1.f;
    float background_[3] = {0.02f, 0.02f, 0.06f};
    int last_visible_ = 0;

    std::vector<float> means3D_;
    std::vector<float> scales_;
    std::vector<float> rotations_;
    std::vector<float> opacities_;
    std::vector<float> shs_;
    std::vector<float> colors_precomp_;
    std::vector<char> geom_buffer_;
    std::vector<char> binning_buffer_;
    std::vector<char> image_buffer_;
    std::vector<float> out_color_;
    std::vector<int> radii_;
};

}  // namespace internal
}  // namespace gsplat
