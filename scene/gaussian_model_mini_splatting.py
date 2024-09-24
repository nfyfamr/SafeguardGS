# Codes from Mini-Splatting paper (https://github.com/fatPeter/mini-splatting)

import torch
from scene.gaussian_model import GaussianModel
import gc
import numpy as np

class MiniSplatting(GaussianModel):

    ### SafeguardGS author's implementation ###
    # def sample_gaussians(self, sample_ratio, imp_list):
    #     probability_list = imp_list / imp_list.sum()
    #     target_samples = int(self._xyz.shape[0] * sample_ratio)
    #     valid_samples = torch.count_nonzero(probability_list)
    #     target_samples = target_samples if target_samples <= valid_samples else valid_samples
    #     sample_idx = torch.multinomial(probability_list, target_samples)
    #     prune_mask = torch.zeros(self._xyz.shape[0], device="cuda").scatter_(0, sample_idx, 1.).bool()
    #     self.prune_points(~prune_mask)

    # def prune_gaussians(self, percent, imp_list):
    #     sorted_tensor, _ = torch.sort(imp_list, dim=0)
    #     index_nth_percentile = int(percent * (sorted_tensor.shape[0] - 1))
    #     value_nth_percentile = sorted_tensor[index_nth_percentile]
    #     prune_mask = (imp_list <= value_nth_percentile).squeeze()
    #     self.prune_points(prune_mask)

    # def prune_after_densify(self, opt, iteration, scene=None, pipe=None, background=None, render=None):
    #     if iteration in opt.mini_splatting_prune_iterations:
    #         # sampling
    #         imp_list = self.prune_list(scene, pipe, background, render)
    #         if not opt.mini_splatting_deterministic_prune:
    #             self.sample_gaussians(opt.mini_splatting_preserving_ratio, imp_list)
    #         else:
    #             self.prune_gaussians(1 - opt.mini_splatting_preserving_ratio, imp_list)
            
    #     torch.cuda.empty_cache()
    #     gc.collect()

    # @torch.no_grad()
    # def prune_list(self, scene, pipe, background, count_render):
    #     viewpoint_stack = scene.getTrainCameras().copy()
    #     viewpoint_cam = viewpoint_stack.pop()
    #     imp_list = count_render(viewpoint_cam, self, pipe, background, bw_score=True)["important_score"]

    #     for _ in range(len(viewpoint_stack)):
    #         viewpoint_cam = viewpoint_stack.pop()
    #         imp_list += count_render(viewpoint_cam, self, pipe, background, bw_score=True)["important_score"]
    #         gc.collect()

    #     return imp_list
    ###########################################

    def init_cdf_mask(self, importance, thres=1.0):
        importance = importance.flatten()   
        if thres != 1.0:
            percent_sum = thres
            vals, idx = torch.sort(importance + (1e-6))
            cumsum_val = torch.cumsum(vals, dim=0)
            split_index = ((cumsum_val / vals.sum()) > (1 - percent_sum)).nonzero().min()
            split_val_nonprune = vals[split_index]

            non_prune_mask = importance > split_val_nonprune 
        else: 
            non_prune_mask = torch.ones_like(importance).bool()
        
        return non_prune_mask

    def prune_after_densify(self, opt, iteration, scene=None, pipe=None, background=None, render=None):
        # simp_iteration1
        if iteration == opt.mini_splatting_prune_iterations[0]:
            imp_score = self.prune_list(scene, pipe, background, render, imp_metric=opt.mini_splatting_imp_metric)

            prob = imp_score / imp_score.sum()
            prob = prob.cpu().numpy()

            factor = opt.mini_splatting_preserving_ratio
            N_xyz = self._xyz.shape[0]
            num_sampled = int(N_xyz * factor * ((prob != 0).sum() / prob.shape[0]))
            indices = np.random.choice(N_xyz, size=num_sampled, p=prob, replace=False)

            mask = np.zeros(N_xyz, dtype=bool)
            mask[indices] = True
                 
            self.prune_points(mask == False)
            
            ### We don't do reinitialization because our concern is the effectiveness of pruning function
            # self.max_sh_degree=dataset.sh_degree
            # self.reinitial_pts(self._xyz, SH2RGB(self._features_dc+0)[:,0])
            # self.training_setup(opt)
            
            torch.cuda.empty_cache()
            gc.collect()

        # simp_iteration2
        if len(opt.mini_splatting_prune_iterations) > 1 and iteration == opt.mini_splatting_prune_iterations[1]:
            imp_score = self.prune_list(scene, pipe, background, render, imp_metric=opt.mini_splatting_imp_metric)

            non_prune_mask = self.init_cdf_mask(importance=imp_score, thres=0.99)
            
            self.prune_points(non_prune_mask == False)
            # self.training_setup(opt)
            
            torch.cuda.empty_cache()
            gc.collect()

    @torch.no_grad()
    def prune_list(gaussians, scene, pipe, background, render_imp, imp_metric='indoor'):
        imp_score = torch.zeros(gaussians._xyz.shape[0]).cuda()
        accum_area_max = torch.zeros(gaussians._xyz.shape[0]).cuda()
        views = scene.getTrainCameras().copy()
        for view in views:
            render_pkg = render_imp(view, gaussians, pipe, background, bw_score=True)
            accum_weights = render_pkg["important_score"] # accum_weights
            area_proj = render_pkg["gaussians_count"] # area_proj
            area_max = render_pkg["area_max"] # area_max
            accum_area_max += area_max

            if imp_metric == 'outdoor':
                mask_t = area_max != 0
                temp = imp_score + accum_weights / area_proj
                imp_score[mask_t] = temp[mask_t]
            else:
                imp_score += accum_weights

            gc.collect()

        imp_score[accum_area_max == 0] = 0
        return imp_score
