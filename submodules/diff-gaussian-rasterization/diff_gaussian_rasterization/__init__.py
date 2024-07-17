#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from typing import NamedTuple
import torch.nn as nn
import torch
from . import _C

def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [item.cpu().clone() if isinstance(item, torch.Tensor) else item for item in input_tuple]
    return tuple(copied_tensors)

def rasterize_gaussians(
    means3D,
    means2D,
    sh,
    colors_precomp,
    opacities,
    scales,
    rotations,
    cov3Ds_precomp,
    raster_settings,
):
    return _RasterizeGaussians.apply(
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
    )

class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
    ):

        # Restructure arguments the way that the C++ lib expects them
        args = (
            raster_settings.bg, 
            means3D,
            colors_precomp,
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            sh,
            raster_settings.sh_degree,
            raster_settings.campos,
            raster_settings.prefiltered,
            raster_settings.debug
        )

        gaussians_count, accum_max_count, important_score = None, None, None

        # Invoke C++/CUDA rasterizer
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                if raster_settings.f_count is not None:
                    num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer, gaussians_count, important_score = _C.count_gaussians(*args)
                elif raster_settings.bw_score is not None:
                    num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer, gaussians_count, accum_max_count, important_score = _C.bw_score_gaussians(*args)
                elif raster_settings.mw_score is not None:
                    num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer, important_score = _C.mw_score_gaussians(*args)
                elif (raster_settings.safeguard_gs_topk is not None) and (raster_settings.safeguard_gs_score_function & 0xf0 == 0):
                    args += (raster_settings.safeguard_gs_topk, raster_settings.safeguard_gs_score_function, raster_settings.p_dist_activation_coef, raster_settings.c_dist_activation_coef)
                    num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer, important_score = _C.topk_gaussians(*args)
                elif (raster_settings.safeguard_gs_topk is not None) and (raster_settings.safeguard_gs_score_function & 0xf0 >= 0):
                    args += (raster_settings.safeguard_gs_topk, raster_settings.safeguard_gs_score_function, raster_settings.image_gt, raster_settings.p_dist_activation_coef, raster_settings.c_dist_activation_coef)
                    num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer, important_score = _C.topk_color_gaussians(*args)
                elif raster_settings.efficient_gs_topk is not None:
                    args += (raster_settings.efficient_gs_topk,)
                    num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer, important_score = _C.topk_weight_gaussians(*args)
                else:
                    num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer = _C.rasterize_gaussians(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_fw.dump")
                print("\nAn error occured in forward. Please forward snapshot_fw.dump for debugging.")
                raise ex
        else:
            if raster_settings.f_count is not None:
                num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer, gaussians_count, important_score = _C.count_gaussians(*args)
            elif raster_settings.bw_score is not None:
                num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer, gaussians_count, accum_max_count, important_score = _C.bw_score_gaussians(*args)
            elif raster_settings.mw_score is not None:
                num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer, important_score = _C.mw_score_gaussians(*args)
            elif (raster_settings.safeguard_gs_topk is not None) and (raster_settings.safeguard_gs_score_function & 0xf0 == 0):
                args += (raster_settings.safeguard_gs_topk, raster_settings.safeguard_gs_score_function, raster_settings.p_dist_activation_coef, raster_settings.c_dist_activation_coef)
                num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer, important_score = _C.topk_gaussians(*args)
            elif (raster_settings.safeguard_gs_topk is not None) and (raster_settings.safeguard_gs_score_function & 0xf0 >= 0):
                args += (raster_settings.safeguard_gs_topk, raster_settings.safeguard_gs_score_function, raster_settings.image_gt, raster_settings.p_dist_activation_coef, raster_settings.c_dist_activation_coef)
                num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer, important_score = _C.topk_color_gaussians(*args)
            elif raster_settings.efficient_gs_topk is not None:
                args += (raster_settings.efficient_gs_topk,)
                num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer, important_score = _C.topk_weight_gaussians(*args)
            else:
                num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer = _C.rasterize_gaussians(*args)

        # Keep relevant tensors for backward
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, geomBuffer, binningBuffer, imgBuffer)
        return color, radii, gaussians_count, accum_max_count, important_score

    @staticmethod
    def backward(ctx, grad_out_color, *_):

        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, geomBuffer, binningBuffer, imgBuffer = ctx.saved_tensors

        # Restructure args as C++ method expects them
        args = (raster_settings.bg,
                means3D, 
                radii, 
                colors_precomp, 
                scales, 
                rotations, 
                raster_settings.scale_modifier, 
                cov3Ds_precomp, 
                raster_settings.viewmatrix, 
                raster_settings.projmatrix, 
                raster_settings.tanfovx, 
                raster_settings.tanfovy, 
                grad_out_color, 
                sh, 
                raster_settings.sh_degree, 
                raster_settings.campos,
                geomBuffer,
                num_rendered,
                binningBuffer,
                imgBuffer,
                raster_settings.debug)

        # Compute gradients for relevant tensors by invoking backward method
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations = _C.rasterize_gaussians_backward(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_bw.dump")
                print("\nAn error occured in backward. Writing snapshot_bw.dump for debugging.\n")
                raise ex
        else:
             grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations = _C.rasterize_gaussians_backward(*args)

        grads = (
            grad_means3D,
            grad_means2D,
            grad_sh,
            grad_colors_precomp,
            grad_opacities,
            grad_scales,
            grad_rotations,
            grad_cov3Ds_precomp,
            None,
        )

        return grads

class GaussianRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int 
    tanfovx : float
    tanfovy : float
    bg : torch.Tensor
    scale_modifier : float
    viewmatrix : torch.Tensor
    projmatrix : torch.Tensor
    sh_degree : int
    campos : torch.Tensor
    prefiltered : bool
    debug : bool
    f_count : bool
    bw_score : bool
    mw_score : bool
    efficient_gs_topk : int
    safeguard_gs_topk : int
    safeguard_gs_score_function : int
    image_gt : torch.Tensor
    p_dist_activation_coef : float
    c_dist_activation_coef : float

class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings

    def markVisible(self, positions):
        # Mark visible points (based on frustum culling for camera) with a boolean 
        with torch.no_grad():
            raster_settings = self.raster_settings
            visible = _C.mark_visible(
                positions,
                raster_settings.viewmatrix,
                raster_settings.projmatrix)
            
        return visible

    def forward(self, means3D, means2D, opacities, shs = None, colors_precomp = None, scales = None, rotations = None, cov3D_precomp = None):
        
        raster_settings = self.raster_settings

        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            raise Exception('Please provide excatly one of either SHs or precomputed colors!')
        
        if ((scales is None or rotations is None) and cov3D_precomp is None) or ((scales is not None or rotations is not None) and cov3D_precomp is not None):
            raise Exception('Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!')
        
        if shs is None:
            shs = torch.Tensor([])
        if colors_precomp is None:
            colors_precomp = torch.Tensor([])

        if scales is None:
            scales = torch.Tensor([])
        if rotations is None:
            rotations = torch.Tensor([])
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([])

        # Invoke C++/CUDA rasterization routine
        return rasterize_gaussians(
            means3D,
            means2D,
            shs,
            colors_precomp,
            opacities,
            scales, 
            rotations,
            cov3D_precomp,
            raster_settings, 
        )

