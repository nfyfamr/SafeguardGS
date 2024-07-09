# This is our implementation of random pruning

import torch
from scene.gaussian_model import GaussianModel
import gc

class RandomPrune(GaussianModel):

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, n_split):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        torch.cuda.empty_cache()

    def prune_after_densify(self, opt, iteration, scene=None, pipe=None, background=None, render=None):
        if iteration in opt.random_prune_iterations:
            num_gauss = self._xyz.shape[0]
            prune_mask = torch.zeros(num_gauss, dtype=torch.bool)
            prune_mask[:int(num_gauss*opt.random_prune_ratio)] = True
            self.prune_points(prune_mask[torch.randperm(num_gauss)])

            torch.cuda.empty_cache()
            gc.collect()