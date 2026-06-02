#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace gsplat {

bool writeRgbPng(const std::string& path, int width, int height, const std::vector<uint8_t>& rgb);

}
