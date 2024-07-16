# This is our implementation of random pruning

import torch
from scene.gaussian_model import GaussianModel
import gc

class RandomPrune(GaussianModel):

    def prune_after_densify(self, opt, iteration, scene=None, pipe=None, background=None, render=None):
        if iteration in opt.random_prune_iterations:
            num_gauss = self._xyz.shape[0]
            prune_mask = torch.zeros(num_gauss, dtype=torch.bool)
            prune_mask[:int(num_gauss*opt.random_prune_ratio)] = True
            self.prune_points(prune_mask[torch.randperm(num_gauss)])

            torch.cuda.empty_cache()
            gc.collect()