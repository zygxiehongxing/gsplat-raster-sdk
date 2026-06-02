# gsplat-raster-sdk

Standalone SDK wrapping [Inria diff-gaussian-rasterization](https://github.com/graphdeco-inria/diff-gaussian-rasterization).

**Input:** 3D Gaussian Splatting PLY (or `Gaussian` arrays) + camera  
**Output:** RGB buffer or PNG  

No OpenSceneGraph. Suitable as a shared library for other apps (viewers, pipelines, services).

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

## Build (Windows)

```cmd
cd gsplat-raster-sdk
scripts\build.cmd
```

Produces:

- `build/Release/gsplat_raster.lib` (static library)
- `build/Release/gsplat_render_ply.exe` (example)

## Quick start (CLI)

```cmd
gsplat_render_ply point_cloud.ply out.png 1280 720 500000
```

Arguments: `ply`, `png`, optional `width`, `height`, `max_points` (0 = all).

## API overview

Header: `include/gsplat_raster/gsplat_raster.h`  
Namespace: `gsplat`

### 1. Check CUDA

```cpp
if (!gsplat::isCudaAvailable()) { /* no GPU */ }
```

### 2. Load gaussians

```cpp
std::vector<gsplat::Gaussian> G;
gsplat::PlyLoadOptions opts;
opts.max_points = 500000;   // subsample large clouds
opts.drop_sh_rest = false;  // true = DC-only (faster, lower quality)

gsplat::Rasterizer raster;
gsplat::Status st = raster.loadPly("scene.ply", opts);
// or: raster.setGaussians(G);
```

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
Default helper uses **Inria-packed view** + **GL clip `proj * view`** (same as `3dgs-osg-viewer` stable path).

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
