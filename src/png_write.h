#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace gsplat {

/// 将 RGB8 缓冲（行优先）编码并写入 PNG 文件。
bool writeRgbPng(const std::string& path, int width, int height, const std::vector<uint8_t>& rgb);

}
