# SafeguardGS implementation

import torch
from scene.gaussian_model import GaussianModel
import gc

class SafeguardGaussian(GaussianModel):

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, n_split):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent, n_split)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def prune_after_densify(self, opt, iteration, scene=None, pipe=None, background=None, render=None):
        if iteration in opt.safeguard_gs_prune_iterations:
            prune_mask = self.topk_gaussians(scene, pipe, background, render, K=opt.safeguard_gs_purne_topk, safeguard_gs_score_function=opt.safeguard_gs_score_function, p_dist_activation_coef=opt.safeguard_gs_p_dist_activation_coef, c_dist_activation_coef=opt.safeguard_gs_c_dist_activation_coef)
            self.prune_points(prune_mask)
                    
            torch.cuda.empty_cache()
            gc.collect()

    @torch.no_grad()
    def topk_gaussians(gaussians, scene, pipe, background, render, K=10, safeguard_gs_score_function=0, p_dist_activation_coef=1.0, c_dist_activation_coef=1.0):
        viewpoint_stack = scene.getTrainCameras().copy()
        valid_prune_mask = torch.zeros((gaussians.get_xyz.shape[0]), device="cuda", dtype=torch.bool)

        use_color_error = safeguard_gs_score_function & 0xf0 >= 0
        for viewpoint_cam in viewpoint_stack:
            if use_color_error:
                valid_prune_mask = torch.logical_or(valid_prune_mask, render(viewpoint_cam, gaussians, pipe, background, safeguard_gs_topk=K, safeguard_gs_score_function=safeguard_gs_score_function, p_dist_activation_coef=p_dist_activation_coef, c_dist_activation_coef=c_dist_activation_coef, image_gt=viewpoint_cam.original_image.cuda())["important_score"])
            else:
                valid_prune_mask = torch.logical_or(valid_prune_mask, render(viewpoint_cam, gaussians, pipe, background, safeguard_gs_topk=K, safeguard_gs_score_function=safeguard_gs_score_function, p_dist_activation_coef=p_dist_activation_coef, c_dist_activation_coef=c_dist_activation_coef)["important_score"])

        return ~valid_prune_mask