# Codes from LightGaussian paper, https://github.com/VITA-Group/LightGaussian

import torch
from utils.general_utils import get_expon_lr_func
from torch import nn
import os
from scene.gaussian_model import GaussianModel
from torch.optim.lr_scheduler import ExponentialLR
import gc

class LightGaussian(GaussianModel):

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        self.scheduler = ExponentialLR(self.optimizer, gamma=0.97)
        

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        if iteration % 1000 == 0:
            self.scheduler.step()

        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

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
