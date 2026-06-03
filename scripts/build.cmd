@echo off
setlocal EnableExtensions
cd /d %~dp0\..

set "DGR=%~dp0..\3dgs-osg-viewer\third_party\diff-gaussian-rasterization"
if not exist "%DGR%\cuda_rasterizer\rasterizer.h" set "DGR=%~dp0..\..\3dgs-osg-viewer\third_party\diff-gaussian-rasterization"
if not exist "%DGR%\cuda_rasterizer\rasterizer.h" (
    echo ERROR: diff-gaussian-rasterization not found
    exit /b 1
)

set "CUDA118=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8"
if exist "%CUDA118%\bin\nvcc.exe" (
    set "CUDA_PATH=%CUDA118%"
    if exist "C:\cuda118\bin\nvcc.exe" (
        set "GSPLAT_CUDA_BIN=C:\cuda118\bin;%CUDA118%\bin"
    ) else (
        set "GSPLAT_CUDA_BIN=%CUDA118%\bin"
    )
)

set "VSROOT=C:\Program Files\Microsoft Visual Studio\18"
set "VSCOMM="
for %%E in (BuildTools Community Professional Enterprise) do (
    if exist "%VSROOT%\%%E\VC\Auxiliary\Build\vcvarsall.bat" (
        set "VSCOMM=%VSROOT%\%%E"
        goto :vs_ok
    )
)
echo ERROR: Visual Studio not found
exit /b 1

:vs_ok
call "%VSCOMM%\VC\Auxiliary\Build\vcvarsall.bat" amd64 -vcvars_ver=14.44
if errorlevel 1 exit /b 1
if defined GSPLAT_CUDA_BIN set "PATH=%GSPLAT_CUDA_BIN%;%PATH%"

set "NINJA=%VSCOMM%\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe"
if not exist "%NINJA%" (
    for /f "delims=" %%N in ('where ninja 2^>nul') do set "NINJA=%%N" & goto :ninja_ok
    echo ERROR: ninja not found
    exit /b 1
)
:ninja_ok

if not exist build mkdir build
cd build
if "%GSPLAT_CLEAN_BUILD%"=="1" (
    if exist CMakeCache.txt del /f CMakeCache.txt
)

cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release ^
  -DCMAKE_MAKE_PROGRAM="%NINJA%" ^
  -DGSPLAT_DGR_ROOT="%DGR%" ^
  -DGSPLAT_USE_CUDA_TOOLCHAIN=ON
if errorlevel 1 exit /b 1

"%NINJA%" gsplat_raster
if errorlevel 1 exit /b 1

echo.
echo OK: %CD%\sdk\gsplat_raster.lib
endlocal
