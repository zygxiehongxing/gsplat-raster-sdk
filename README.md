# gsplat-raster-sdk

**SDK 层**：根据 App（或 CLI）传入的**相机参数**与**高斯点云**，执行 Inria CUDA 光栅化并输出图像。本仓库**不包含** OSG、窗口或交互逻辑。

Standalone CUDA raster SDK wrapping [Inria diff-gaussian-rasterization](https://github.com/graphdeco-inria/diff-gaussian-rasterization).

Application-side viewer/CLI lives in a sibling project:

- `../gsplat-raster-app` — PLY 加载、OSG 可视化、交互时把当前帧相机 + 高斯传给本 SDK

## 职责划分

| 层 | 仓库 | 做什么 |
|----|------|--------|
| App | `gsplat-raster-app` | PLY / OSG / 交互；组装 `Camera` + `Gaussian` 并调用 SDK |
| SDK | `gsplat-raster-sdk`（本仓库） | `Rasterizer::render` / `renderToPng`；矩阵打包与 CUDA 内核 |

```text
  App ── Camera, width, height, RenderSettings ──► Rasterizer
        Gaussians (setGaussians / loadPly)
                      │
                      ▼
              diff-gaussian-rasterization (CUDA)
                      │
                      ▼
              RGB8 buffer 或 PNG 文件
```

**Input:** `Gaussian` 数组（或 PLY 经 `loadPly`）+ `Camera`（`view` / `proj` / `cam_pos` / `tan_fov`）  
**Output:** RGB buffer or PNG  

No OSG dependency in this repository.

## Dependencies

- CUDA Toolkit (tested with 11.8)
- MSVC 2019+ or compatible C++17 compiler
- `diff-gaussian-rasterization` (+ glm submodule)

### Get Inria rasterizer

Either copy from sibling project:

```text
3dgs-osg-viewer/third_party/diff-gaussian-rasterization
```

Or submodule:

```bash
git submodule add https://github.com/graphdeco-inria/diff-gaussian-rasterization third_party/diff-gaussian-rasterization
cd third_party/diff-gaussian-rasterization
git submodule update --init third_party/glm
```

Configure with:

```bash
cmake -B build -DGSPLAT_DGR_ROOT=/path/to/diff-gaussian-rasterization
```

## Build SDK (Windows)

```cmd
cd gsplat-raster-sdk
scripts\build.cmd
```

Produces:

- `build/sdk/gsplat_raster.lib` (SDK static library)

Use `../gsplat-raster-app` for CLI rendering and OSG roaming app.

## API overview

Header: `include/gsplat_raster/gsplat_raster.h`  
Namespace: `gsplat`

### 1. Check CUDA

```cpp
if (!gsplat::isCudaAvailable()) { /* no GPU */ }
```

### 2. Prepare gaussians

```cpp
std::vector<gsplat::Gaussian> G;
gsplat::Rasterizer raster;
gsplat::Status st = raster.setGaussians(std::move(G));
```

> SDK 仅负责光栅化，不再包含 PLY 读取逻辑；PLY 解析由 app 侧负责。

### 3. Camera

**Option A — look-at helper (OpenGL-style, packed for Inria):**

```cpp
double eye[3]    = {cx, cy - 10, cz + 3};
double center[3] = {cx, cy, cz};
double up[3]     = {0, 0, 1};
gsplat::Camera cam;
gsplat::buildCameraLookAt(eye, center, up,
    60.0,           // fov_y degrees
    1280.0 / 720.0, // aspect
    0.01, 10000.0,  // znear, zfar
    center,         // ref point for view-Z alignment
    cam);
```

**Option B — training / COLMAP matrices (column-major 4×4):**

```cpp
gsplat::Camera cam;
std::memcpy(cam.view, your_w2c_column_major, 16 * sizeof(float));
std::memcpy(cam.proj, your_full_proj_column_major, 16 * sizeof(float));
std::memcpy(cam.cam_pos, camera_position_world, 3 * sizeof(float));
cam.tan_fovx = ...;
cam.tan_fovy = ...;
```

Matrix layout matches Inria CUDA (`viewmatrix` / `projmatrix` in `Rasterizer::forward`).

**Option C — OSG `view` / `proj` matrices (row-major 16 doubles, `osg::Matrix` layout):**

```cpp
gsplat::buildCameraFromOsg(view_osg, proj_osg, ref_center, cam);
```

Used by `gsplat-raster-app` so SDK output matches the OSG camera.

### 4. Render settings

```cpp
gsplat::RenderSettings settings;
settings.background[0] = 0.02f;
settings.background[1] = 0.02f;
settings.background[2] = 0.06f;
settings.scale_modifier = 1.0f;  // increase if splats too small (e.g. 8)
settings.dc_only = false;
raster.setSettings(settings);
```

### 5. Render → RGB or PNG

```cpp
std::vector<uint8_t> rgb;
gsplat::Status st = raster.render(cam, 1280, 720, rgb);
// rgb size = width * height * 3, row-major RGB

st = raster.renderToPng(cam, 1280, 720, "frame.png");
```

### 6. Diagnostics

```cpp
int pass = raster.countFrustumPass(cam);      // approximate in-frustum count
int vis  = raster.lastVisibleCount();         // after last render (radii > 0)
const char* msg = gsplat::statusString(st);
```

### Status codes

| `Status` | Meaning |
|----------|---------|
| `Ok` | Success (note: 0 visible splats still returns Ok with background) |
| `ErrorNoCuda` | Built without CUDA or no device |
| `ErrorNoGaussians` | Empty cloud |
| `ErrorInvalidArgs` | Bad width/height or FOV |
| `ErrorRenderFailed` | CUDA forward failed |
| `ErrorPlyLoad` | PLY parse failed |
| `ErrorPngWrite` | stb PNG write failed |

## Link in your project (CMake)

```cmake
add_subdirectory(../gsplat-raster-sdk ${CMAKE_BINARY_DIR}/gsplat-raster-sdk)
target_link_libraries(your_app PRIVATE gsplat_raster)
target_include_directories(your_app PRIVATE ${CMAKE_SOURCE_DIR}/../gsplat-raster-sdk/include)
```

Define `GSPLAT_CUDA_ENABLED=1` is set automatically when CUDA build succeeds.

## `Gaussian` fields (training PLY)

| Field | PLY / meaning |
|-------|----------------|
| `x,y,z` | Position |
| `opacity_logit` | Raw opacity (sigmoid applied internally) |
| `scale_log[3]` | log scale → exp |
| `rot[4]` | Quaternion w,x,y,z |
| `sh[48]` | f_dc + f_rest (interleaved RGB per SH band) |
| `has_sh_rest` | Use SH degree 3 when true |

## Limits

- Full forward cost scales with **uploaded** splat count; use `max_points` or your own frustum cull before `setGaussians`.
- Large scenes need **spatial tiling** (not included in this SDK).
- Camera must match Inria conventions; wrong matrices → empty or background-only images.

## License

SDK wrapper: use same terms as your project.  
`diff-gaussian-rasterization`: see Inria LICENSE in that repository.
