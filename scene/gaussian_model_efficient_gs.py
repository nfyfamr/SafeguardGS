# This is our implementation of EfficientGS

import torch
from scene.gaussian_model import GaussianModel
import gc

class EfficientGS(GaussianModel):

    def prune_after_densify(self, opt, iteration, scene=None, pipe=None, background=None, render=None):
        if iteration in opt.efficient_gs_prune_iterations:
            prune_mask = self.topk_gaussians(scene, pipe, background, render, K=opt.efficient_gs_prune_topk)
            self.prune_points(prune_mask)
    
            torch.cuda.empty_cache()
            gc.collect()

    @torch.no_grad()
    def topk_gaussians(gaussians, scene, pipe, background, render, K=1):
        viewpoint_stack = scene.getTrainCameras().copy()
        valid_prune_mask = torch.zeros((gaussians.get_xyz.shape[0]), device="cuda", dtype=torch.bool)

        for viewpoint_cam in viewpoint_stack:
            valid_prune_mask = torch.logical_or(valid_prune_mask, render(viewpoint_cam, gaussians, pipe, background, efficient_gs_topk=K)["important_score"])

        return ~valid_prune_mask