# This is our implementation of RadSplat

import torch
from scene.gaussian_model import GaussianModel
import gc

class RadSplat(GaussianModel):

    def prune_after_densify(self, opt, iteration, scene=None, pipe=None, background=None, render=None):
        if iteration in opt.rad_splat_prune_iterations:
            imp_list = self.prune_list(scene, pipe, background, render)
            prune_mask = imp_list < opt.rad_splat_prune_threshold
            self.prune_points(prune_mask)

            torch.cuda.empty_cache()
            gc.collect()

    @torch.no_grad()
    def prune_list(self, scene, pipe, background, count_render):
        viewpoint_stack = scene.getTrainCameras().copy()
        imp_list = torch.zeros(self._xyz.shape[0], device="cuda")

        for viewpoint_cam in viewpoint_stack:
            imp_list = torch.maximum(imp_list, count_render(viewpoint_cam, self, pipe, background, mw_score=True)["important_score"])
            gc.collect()

        return imp_list
