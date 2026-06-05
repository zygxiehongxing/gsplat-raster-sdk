# WIP 分支说明：`wip/ssbo-device-pool-pipeline`

本分支保存 **2026-06** SDK 工作区快照。与 App 仓库同名分支配套使用。

## 相对 `c9d587e`（效果较好的基线）的主要变化

### 公开 API

| 基线 `c9d587e` | 本分支 |
|----------------|--------|
| `Rasterizer::setGaussians` + `render(cam,w,h,rgb)` | **移除** `setGaussians` |
| Rasterizer 持有全量 GPU 数据 | App 侧 **`GaussianDevicePool`** 上传；Rasterizer 只消费 **`DeviceGaussianBuffers` 子集** |
| — | `filterVisible`（`markVisible` + GPU compact） |
| — | `unpackScreenSsboToDevice`（GL SSBO → SoA，供 App 屏缓存路径） |

新头文件：`gaussian_device.h`、`screen_gaussian_pixel.h`、`screen_cache.h`。

### CUDA preprocess（`forward.cu` / `auxiliary.h`）

**片元级 EWA 混合未改**；变化在预处理与调度：

| 项 | `c9d587e` | 本分支 |
|----|-----------|--------|
| 矩阵约定 | 设备端列主序风格 | **GLSL 行主序** `vec4*V*P` |
| 2D cov blur | `+0.3` | `+0.15` |
| tile 半径 | `ceil(3√max(λ))` 正方形 | 各向异性 `rx/ry`，cap **14px** |
| `radii` | `int` | `int2` + `copyMaxRadii` |
| 极小 splat | rect 空则丢弃 | 屏内中心强制占 1 tile |

### `rasterizer_impl.cpp`

- `kDisableTileBudgetForDiag = true`：SSBO 子集路径下关闭 tile 预算重试，避免 visible 被误杀为 0。
- 输入按「App 提供的子集即本帧全部源」处理，不再根据 visible 反馈自动放大 scale。

### SSBO 解包（`screen_ssbo_unpack.cu`）

- 默认 `scale_mul = 0.035`（`GSPLAT_SCREEN_SCALE_MUL`）
- 默认 `max_opacity = 0.65`（`GSPLAT_SCREEN_MAX_OPACITY`）

## 推荐使用方式

**画质优先（对齐好版）** — App 侧：

```cpp
GaussianDevicePool pool;
pool.upload(std::move(gaussians));
DeviceGaussianBuffers subset;
pool.filterVisible(cam, subset);
raster.render(cam, w, h, subset, rgb);
freeDeviceGaussianBuffers(subset);
```

参考：`gsplat-raster-app/src/render_ply_main.cpp`。

**不要用 SSBO 解包结果作为主渲染输入**（仅诊断/对齐实验）。

## 构建

与 App 一起通过 `gsplat-raster-app/scripts/configure_and_build.bat` 构建（子目录 `gsplat-raster-sdk`）。

## 对比好版

```cmd
git diff c9d587e -- third_party/diff-gaussian-rasterization/cuda_rasterizer/forward.cu
git diff c9d587e -- include/gsplat_raster/gsplat_raster.h
```
