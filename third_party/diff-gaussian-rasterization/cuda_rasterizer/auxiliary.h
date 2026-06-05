/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#ifndef CUDA_RASTERIZER_AUXILIARY_H_INCLUDED
#define CUDA_RASTERIZER_AUXILIARY_H_INCLUDED

#include "config.h"
#include "stdio.h"

#define BLOCK_SIZE (BLOCK_X * BLOCK_Y)
#define NUM_WARPS (BLOCK_SIZE/32)

// Spherical harmonics coefficients
__device__ const float SH_C0 = 0.28209479177387814f;
__device__ const float SH_C1 = 0.4886025119029199f;
__device__ const float SH_C2[] = {
	1.0925484305920792f,
	-1.0925484305920792f,
	0.31539156525252005f,
	-1.0925484305920792f,
	0.5462742152960396f
};
__device__ const float SH_C3[] = {
	-0.5900435899266435f,
	2.890611442640554f,
	-0.4570457994644658f,
	0.3731763325901154f,
	-0.4570457994644658f,
	1.445305721320277f,
	-0.5900435899266435f
};

__forceinline__ __device__ float ndc2Pix(float v, int S)
{
	return ((v + 1.0) * S - 1.0) * 0.5;
}

/// Axis-aligned tile bounds for an ellipse: separate half-extents along screen x/y (pixels).
__forceinline__ __device__ void getRect(const float2 p, int radius_x, int radius_y, uint2& rect_min, uint2& rect_max,
                                        dim3 grid)
{
	rect_min = {
		min(grid.x, max((int)0, (int)((p.x - radius_x) / BLOCK_X))),
		min(grid.y, max((int)0, (int)((p.y - radius_y) / BLOCK_Y)))
	};
	rect_max = {
		min(grid.x, max((int)0, (int)((p.x + radius_x + BLOCK_X - 1) / BLOCK_X))),
		min(grid.y, max((int)0, (int)((p.y + radius_y + BLOCK_Y - 1) / BLOCK_Y)))
	};
}

__forceinline__ __device__ void getRect(const float2 p, int max_radius, uint2& rect_min, uint2& rect_max, dim3 grid)
{
	getRect(p, max_radius, max_radius, rect_min, rect_max, grid);
}

/// 3-sigma AABB half-extents (pixels) from 2D covariance eigenvalues and orientation.
__forceinline__ __device__ float2 anisoRadiiFromCov(const float3& cov, float lambda1, float lambda2)
{
	float2 u1;
	const float b = cov.y;
	const float a = cov.x;
	if (fabsf(b) > 1e-6f) {
		u1 = make_float2(b, lambda1 - a);
	} else {
		u1 = (a >= cov.z) ? make_float2(1.f, 0.f) : make_float2(0.f, 1.f);
	}
	const float inv_len = rsqrtf(u1.x * u1.x + u1.y * u1.y);
	u1.x *= inv_len;
	u1.y *= inv_len;
	const float2 u2 = make_float2(-u1.y, u1.x);
	float rx = ceilf(3.f * sqrtf(fmaxf(0.1f, lambda1 * u1.x * u1.x + lambda2 * u2.x * u2.x)));
	float ry = ceilf(3.f * sqrtf(fmaxf(0.1f, lambda1 * u1.y * u1.y + lambda2 * u2.y * u2.y)));
	// Screen-cache / dense views: smaller cap reduces solid color blocks (was 32).
	const float kMaxSplatRadius = 14.f;
	rx = fminf(fmaxf(rx, 1.f), kMaxSplatRadius);
	ry = fminf(fmaxf(ry, 1.f), kMaxSplatRadius);
	return make_float2(rx, ry);
}

// GLSL row-major: vec4(p,1) * M, M[r*4+c] = M(r,c).

/// OSG/GL row view：相机朝 -Z，在前的点 view.z < 0；沿视线深度 = -view.z（>0）。
__forceinline__ __device__ bool viewInFrontGlRow(float3 t_view)
{
	return t_view.z < 0.0f;
}

__forceinline__ __device__ float viewDepthGlRow(float3 t_view)
{
	return -t_view.z;
}
/// GLSL row-major: vec4(p,1) * M, element M(r,c) at matrix[r*4+c].
__forceinline__ __device__ float3 transformPoint4x3(const float3& p, const float* matrix)
{
	const float v[4] = { p.x, p.y, p.z, 1.0f };
	float3 transformed = {};
	for (int c = 0; c < 3; ++c) {
		float s = 0.0f;
		for (int r = 0; r < 4; ++r) {
			s += v[r] * matrix[r * 4 + c];
		}
		((float*)&transformed)[c] = s;
	}
	return transformed;
}

__forceinline__ __device__ float4 transformPoint4x4(const float3& p, const float* matrix)
{
	const float v[4] = { p.x, p.y, p.z, 1.0f };
	float4 transformed = {};
	for (int c = 0; c < 4; ++c) {
		float s = 0.0f;
		for (int r = 0; r < 4; ++r) {
			s += v[r] * matrix[r * 4 + c];
		}
		((float*)&transformed)[c] = s;
	}
	return transformed;
}

/// GLSL: vec4(p,1) * view * proj（view/proj 均为行主序，proj 仅为 P）。
__forceinline__ __device__ float4 worldToClipRow(const float3& p, const float* view, const float* proj)
{
	const float v[4] = { p.x, p.y, p.z, 1.0f };
	float t[4] = {};
	for (int c = 0; c < 4; ++c) {
		float s = 0.0f;
		for (int r = 0; r < 4; ++r) {
			s += v[r] * view[r * 4 + c];
		}
		t[c] = s;
	}
	float4 clip = {};
	clip.x = t[0] * proj[0] + t[1] * proj[4] + t[2] * proj[8] + t[3] * proj[12];
	clip.y = t[0] * proj[1] + t[1] * proj[5] + t[2] * proj[9] + t[3] * proj[13];
	clip.z = t[0] * proj[2] + t[1] * proj[6] + t[2] * proj[10] + t[3] * proj[14];
	clip.w = t[0] * proj[3] + t[1] * proj[7] + t[2] * proj[11] + t[3] * proj[15];
	return clip;
}

__forceinline__ __device__ float3 transformVec4x3(const float3& p, const float* matrix)
{
	const float v[3] = { p.x, p.y, p.z };
	float3 transformed = {};
	for (int c = 0; c < 3; ++c) {
		float s = 0.0f;
		for (int r = 0; r < 3; ++r) {
			s += v[r] * matrix[r * 4 + c];
		}
		((float*)&transformed)[c] = s;
	}
	return transformed;
}

__forceinline__ __device__ float3 transformVec4x3Transpose(const float3& p, const float* matrix)
{
	float3 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z,
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z,
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z,
	};
	return transformed;
}

__forceinline__ __device__ float dnormvdz(float3 v, float3 dv)
{
	float sum2 = v.x * v.x + v.y * v.y + v.z * v.z;
	float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);
	float dnormvdz = (-v.x * v.z * dv.x - v.y * v.z * dv.y + (sum2 - v.z * v.z) * dv.z) * invsum32;
	return dnormvdz;
}

__forceinline__ __device__ float3 dnormvdv(float3 v, float3 dv)
{
	float sum2 = v.x * v.x + v.y * v.y + v.z * v.z;
	float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);

	float3 dnormvdv;
	dnormvdv.x = ((+sum2 - v.x * v.x) * dv.x - v.y * v.x * dv.y - v.z * v.x * dv.z) * invsum32;
	dnormvdv.y = (-v.x * v.y * dv.x + (sum2 - v.y * v.y) * dv.y - v.z * v.y * dv.z) * invsum32;
	dnormvdv.z = (-v.x * v.z * dv.x - v.y * v.z * dv.y + (sum2 - v.z * v.z) * dv.z) * invsum32;
	return dnormvdv;
}

__forceinline__ __device__ float4 dnormvdv(float4 v, float4 dv)
{
	float sum2 = v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
	float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);

	float4 vdv = { v.x * dv.x, v.y * dv.y, v.z * dv.z, v.w * dv.w };
	float vdv_sum = vdv.x + vdv.y + vdv.z + vdv.w;
	float4 dnormvdv;
	dnormvdv.x = ((sum2 - v.x * v.x) * dv.x - v.x * (vdv_sum - vdv.x)) * invsum32;
	dnormvdv.y = ((sum2 - v.y * v.y) * dv.y - v.y * (vdv_sum - vdv.y)) * invsum32;
	dnormvdv.z = ((sum2 - v.z * v.z) * dv.z - v.z * (vdv_sum - vdv.z)) * invsum32;
	dnormvdv.w = ((sum2 - v.w * v.w) * dv.w - v.w * (vdv_sum - vdv.w)) * invsum32;
	return dnormvdv;
}

__forceinline__ __device__ float sigmoid(float x)
{
	return 1.0f / (1.0f + expf(-x));
}

__forceinline__ __device__ bool in_frustum(int idx,
	const float* orig_points,
	const float* viewmatrix,
	const float* projmatrix,
	bool prefiltered,
	float3& p_view)
{
	float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };
	p_view = transformPoint4x3(p_orig, viewmatrix);
	(void)prefiltered;

	float4 p_hom = worldToClipRow(p_orig, viewmatrix, projmatrix);
	if (p_hom.w <= 0.0000001f)
	{
		return false;
	}
	return true;
}

#define CHECK_CUDA(A, debug) \
A; if(debug) { \
auto ret = cudaDeviceSynchronize(); \
if (ret != cudaSuccess) { \
std::cerr << "\n[CUDA ERROR] in " << __FILE__ << "\nLine " << __LINE__ << ": " << cudaGetErrorString(ret); \
throw std::runtime_error(cudaGetErrorString(ret)); \
} \
}

#endif