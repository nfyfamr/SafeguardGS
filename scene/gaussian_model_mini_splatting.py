# This is our implementation of Mini-Splatting

import torch
from scene.gaussian_model import GaussianModel
import gc

class MiniSplatting(GaussianModel):

    def oneupSHdegree(self, opt, iteration):
        if iteration > opt.densify_until_iter and self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def sample_gaussians(self, sample_ratio, imp_list):
        probability_list = imp_list / imp_list.sum()
        target_samples = int(self._xyz.shape[0] * sample_ratio)
        valid_samples = torch.count_nonzero(probability_list)
        target_samples = target_samples if target_samples <= valid_samples else valid_samples
        sample_idx = torch.multinomial(probability_list, target_samples)
        prune_mask = torch.zeros(self._xyz.shape[0], device="cuda").scatter_(0, sample_idx, 1.).bool()
        self.prune_points(~prune_mask)

    def prune_gaussians(self, percent, imp_list):
        sorted_tensor, _ = torch.sort(imp_list, dim=0)
        index_nth_percentile = int(percent * (sorted_tensor.shape[0] - 1))
        value_nth_percentile = sorted_tensor[index_nth_percentile]
        prune_mask = (imp_list <= value_nth_percentile).squeeze()
        self.prune_points(prune_mask)

    def prune_after_densify(self, opt, iteration, scene=None, pipe=None, background=None, render=None):
        if iteration in opt.mini_splatting_prune_iterations:
            # sampling
            imp_list = self.prune_list(scene, pipe, background, render)
            if not opt.mini_splatting_deterministic_prune:
                self.sample_gaussians(opt.mini_splatting_preserving_ratio, imp_list)
            else:
                self.prune_gaussians(1 - opt.mini_splatting_preserving_ratio, imp_list)
            
        torch.cuda.empty_cache()
        gc.collect()

    @torch.no_grad()
    def prune_list(self, scene, pipe, background, count_render):
        viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop()
        imp_list = count_render(viewpoint_cam, self, pipe, background, bw_score=True)["important_score"]

        for _ in range(len(viewpoint_stack)):
            viewpoint_cam = viewpoint_stack.pop()
            imp_list += count_render(viewpoint_cam, self, pipe, background, bw_score=True)["important_score"]
            gc.collect()

        return imp_list
