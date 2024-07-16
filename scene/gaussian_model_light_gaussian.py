# Codes from LightGaussian paper, https://github.com/VITA-Group/LightGaussian

import torch
from utils.general_utils import get_expon_lr_func
from torch import nn
import os
from scene.gaussian_model import GaussianModel
from torch.optim.lr_scheduler import ExponentialLR
import gc

class LightGaussian(GaussianModel):

    def prune_gaussians(self, percent, import_score: list):
        sorted_tensor, _ = torch.sort(import_score, dim=0)
        index_nth_percentile = int(percent * (sorted_tensor.shape[0] - 1))
        value_nth_percentile = sorted_tensor[index_nth_percentile]
        prune_mask = (import_score <= value_nth_percentile).squeeze()
        self.prune_points(prune_mask)

    def prune_after_densify(self, opt, iteration, scene=None, pipe=None, background=None, render=None):
        if iteration in opt.light_gaussian_prune_iterations:
            # TODO Add prunning types
            gaussian_list, imp_list = self.prune_list(scene, pipe, background, render)
            i = opt.light_gaussian_prune_iterations.index(iteration)
            v_list = self.calculate_v_imp_score(imp_list, opt.light_gaussian_v_pow)
            self.prune_gaussians((opt.light_gaussian_prune_decay**i) * opt.light_gaussian_prune_percent, v_list)

            torch.cuda.empty_cache()
            gc.collect()

    def calculate_v_imp_score(self, imp_list, v_pow):
        """
        :param gaussians: A data structure containing Gaussian components with a get_scaling method.
        :param imp_list: The importance scores for each Gaussian component.
        :param v_pow: The power to which the volume ratios are raised.
        :return: A list of adjusted values (v_list) used for pruning.
        """
        # Calculate the volume of each Gaussian component
        volume = torch.prod(self.get_scaling, dim=1)
        # Determine the kth_percent_largest value
        index = int(len(volume) * 0.9)
        sorted_volume, _ = torch.sort(volume, descending=True)
        kth_percent_largest = sorted_volume[index]
        # Calculate v_list
        v_list = torch.pow(volume / kth_percent_largest, v_pow)
        v_list = v_list * imp_list
        return v_list

    @torch.no_grad()
    def prune_list(self, scene, pipe, background, count_render):
        viewpoint_stack = scene.getTrainCameras().copy()
        gaussian_list, imp_list = None, None
        viewpoint_cam = viewpoint_stack.pop()
        render_pkg = count_render(viewpoint_cam, self, pipe, background, f_count=True)
        gaussian_list, imp_list = (
            render_pkg["gaussians_count"],
            render_pkg["important_score"],
        )

        for iteration in range(len(viewpoint_stack)):
            # Pick a random Camera
            # prunning
            viewpoint_cam = viewpoint_stack.pop()
            render_pkg = count_render(viewpoint_cam, self, pipe, background, f_count=True)
            gaussians_count, important_score = (
                render_pkg["gaussians_count"].detach(),
                render_pkg["important_score"].detach(),
            )
            gaussian_list += gaussians_count
            imp_list += important_score
            gc.collect()
        return gaussian_list, imp_list
