#include "png_write.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stbi_image_write.h"

namespace gsplat {

bool writeRgbPng(const std::string& path, int width, int height, const std::vector<uint8_t>& rgb) {
    if (width <= 0 || height <= 0) return false;
    const size_t need = static_cast<size_t>(width) * static_cast<size_t>(height) * 3;
    if (rgb.size() < need) return false;
    return stbi_write_png(path.c_str(), width, height, 3, rgb.data(), width * 3) != 0;
}

}  // namespace gsplat
