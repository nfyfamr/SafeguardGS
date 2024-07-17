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

#ifndef CUDA_RASTERIZER_FORWARD_H_INCLUDED
#define CUDA_RASTERIZER_FORWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

namespace FORWARD
{
	// Perform initial steps for each Gaussian prior to rasterization.
	void preprocess(int P, int D, int M,
		const float* orig_points,
		const glm::vec3* scales,
		const float scale_modifier,
		const glm::vec4* rotations,
		const float* opacities,
		const float* shs,
		bool* clamped,
		const float* cov3D_precomp,
		const float* colors_precomp,
		const float* viewmatrix,
		const float* projmatrix,
		const glm::vec3* cam_pos,
		const int W, int H,
		const float focal_x, float focal_y,
		const float tan_fovx, float tan_fovy,
		int* radii,
		float2* points_xy_image,
		float* depths,
		float* cov3Ds,
		float* colors,
		float4* conic_opacity,
		const dim3 grid,
		uint32_t* tiles_touched,
		bool prefiltered);

	// Main rasterization method.
	void render(
		const dim3 grid, dim3 block,
		const uint2* ranges,
		const uint32_t* point_list,
		int W, int H,
		const float2* points_xy_image,
		const float* features,
		const float4* conic_opacity,
		float* final_T,
		uint32_t* n_contrib,
		const float* bg_color,
		float* out_color);

	void count_gaussian(
		const dim3 grid, dim3 block,
		const uint2* ranges,
		const uint32_t* point_list,
		int W, int H,
		const float2* points_xy_image,
		const float* features,
		const float4* conic_opacity,
		float* final_T,
		uint32_t* n_contrib,
		const float* bg_color,
		int* gaussians_count,
		float* important_score,
		float* out_color);
		
	void bw_score_gaussian(
		const dim3 grid, dim3 block,
		const uint2* ranges,
		const uint32_t* point_list,
		int W, int H,
		const float2* points_xy_image,
		const float* features,
		const float4* conic_opacity,
		float* final_T,
		uint32_t* n_contrib,
		const float* bg_color,
		int* gaussians_count,
		int* accum_max_count,
		float* blending_weight_score,
		float* out_color);

	void mw_score_gaussian(
		const dim3 grid, dim3 block,
		const uint2* ranges,
		const uint32_t* point_list,
		int W, int H,
		const float2* points_xy_image,
		const float* features,
		const float4* conic_opacity,
		float* final_T,
		uint32_t* n_contrib,
		const float* bg_color,
		float* max_weight_score,
		float* out_color);

	void topk_gaussian(
		const dim3 grid, dim3 block,
		const uint2* ranges,
		const uint32_t* point_list,
		int W, int H,
		const float2* points_xy_image,
		const float* features,
		const float4* conic_opacity,
		float* final_T,
		uint32_t* n_contrib,
		const float* bg_color,
		const int score_function,
		const float p_dist_activation_coef,
		const float c_dist_activation_coef,
		uint64_t* gaussian_keys_unsorted,
		uint32_t* gaussian_values_unsorted,
		float* out_color);

	void topk_color_gaussian(
		const dim3 grid, dim3 block,
		const uint2* ranges,
		const uint32_t* point_list,
		int W, int H,
		const float2* points_xy_image,
		const float* features,
		const float4* conic_opacity,
		float* final_T,
		uint32_t* n_contrib,
		const float* bg_color,
		const float* image_gt,
		const int score_function,
		const float p_dist_activation_coef,
		const float c_dist_activation_coef,
		uint64_t* gaussian_keys_unsorted,
		uint32_t* gaussian_values_unsorted,
		float* out_color);

	void topk_weight_gaussian(
		const dim3 grid, dim3 block,
		const uint2* ranges,
		const uint32_t* point_list,
		int W, int H,
		const float2* points_xy_image,
		const float* features,
		const float4* conic_opacity,
		float* final_T,
		uint32_t* n_contrib,
		const float* bg_color,
		uint64_t* gaussian_keys_unsorted,
		uint32_t* gaussian_values_unsorted,
		float* out_color);
}


#endif