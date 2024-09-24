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

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, ModelPool
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
import wandb
from time import time
from copy import deepcopy
from render import render_set, render_sets
from metrics import evaluate

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = ModelPool[dataset.prune_method](dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):        
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer, test=True)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image)) + gaussians.addtional_loss(opt)
        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()
            gaussians.prune_before_densify(opt, iteration)

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background), render_pkg)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, opt.n_split)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()
            gaussians.prune_after_densify(opt, iteration, scene=scene, pipe=pipe, background=bg, render=render)

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, render_pkg):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    wandb.log({
        "train/total_points": scene.gaussians.get_xyz.shape[0],
        "train/num_rendered": torch.count_nonzero(render_pkg['visibility_filter']).item(),
        "train/memory_(GiB)": torch.cuda.memory_reserved()/1024**3,
        }, 
        commit=True if iteration % 500 == 0 else False,
        step=iteration - 1)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs, test=True)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    # For rendering
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    run = wandb.init(config=None, project="test")
    iter_end = args.iterations
    wandb.define_metric("train/memory_(GiB)", summary="max")
    if run.sweep_id:
        args.model_path += f"{wandb.config.prune_method}/{wandb.config.scene}"
        args.source_path += wandb.config.scene
        if getattr(wandb.config, 'prune_iterations', None): 
            setattr(args, f'{wandb.config.prune_method}_prune_iterations', wandb.config.prune_iterations)
            args.model_path += '/prn_' + str('_'.join(map(str, wandb.config.prune_iterations)))
        if getattr(wandb.config, 'prune_method', None): args.prune_method = wandb.config.prune_method
        ### compact_3dgs
        if getattr(wandb.config, 'compact_3dgs_mask_lr', None): args.compact_3dgs_mask_lr = wandb.config.compact_3dgs_mask_lr
        if getattr(wandb.config, 'compact_3dgs_prune_iter', None): args.compact_3dgs_prune_iter = wandb.config.compact_3dgs_prune_iter
        if getattr(wandb.config, 'compact_3dgs_lambda_mask', None): args.compact_3dgs_lambda_mask = wandb.config.compact_3dgs_lambda_mask
        ### light_gaussian
        if getattr(wandb.config, 'light_gaussian_prune_percent', None): 
            args.light_gaussian_prune_percent = wandb.config.light_gaussian_prune_percent
            args.model_path += '/pratio_' + str(args.light_gaussian_prune_percent)
        if getattr(wandb.config, 'light_gaussian_prune_decay', None): args.light_gaussian_prune_decay = wandb.config.light_gaussian_prune_decay
        if getattr(wandb.config, 'light_gaussian_v_pow', None): args.light_gaussian_v_pow = wandb.config.light_gaussian_v_pow
        ### random
        if getattr(wandb.config, 'random_prune_ratio', None): 
            args.random_prune_ratio = wandb.config.random_prune_ratio
            args.model_path += '/pratio_' + str(args.random_prune_ratio)
        ### mini_splatting
        if getattr(wandb.config, 'mini_splatting_preserving_ratio', None): 
            args.mini_splatting_preserving_ratio = wandb.config.mini_splatting_preserving_ratio
            args.model_path += '/pratio_' + str(args.mini_splatting_preserving_ratio)
        if getattr(wandb.config, 'mini_splatting_imp_metric', None) is None:
            args.mini_splatting_imp_metric = 'outdoor' if wandb.config.scene.split('/')[-1] in ['train', 'truck', 'bicycle', 'flowers', 'garden', 'stump', 'treehill'] else 'indoor'
            print(args.mini_splatting_imp_metric)
        ### rad_splat
        if getattr(wandb.config, 'rad_splat_prune_threshold', None): 
            args.rad_splat_prune_threshold = wandb.config.rad_splat_prune_threshold
            args.model_path += '/pth_' + str(args.rad_splat_prune_threshold)
        ### efficient_gs
        if getattr(wandb.config, 'efficient_gs_prune_topk', None): 
            args.efficient_gs_prune_topk = wandb.config.efficient_gs_prune_topk
            args.model_path += '/topk_' + str(args.efficient_gs_prune_topk)
        ### safeguard_gs
        if getattr(wandb.config, 'safeguard_gs_purne_topk', None):
            args.safeguard_gs_purne_topk = wandb.config.safeguard_gs_purne_topk
            args.model_path += '/top_' + str(args.safeguard_gs_purne_topk)
        if getattr(wandb.config, 'safeguard_gs_score_function', None):
            args.safeguard_gs_score_function = wandb.config.safeguard_gs_score_function
            args.model_path += '/f_' + str(args.safeguard_gs_score_function)
        if getattr(wandb.config, 'safeguard_gs_p_dist_activation_coef', None): args.safeguard_gs_p_dist_activation_coef = wandb.config.safeguard_gs_p_dist_activation_coef
        if getattr(wandb.config, 'safeguard_gs_c_dist_activation_coef', None): args.safeguard_gs_c_dist_activation_coef = wandb.config.safeguard_gs_c_dist_activation_coef
        if getattr(wandb.config, 'n_split', None): args.n_split = wandb.config.n_split

        args.port = randint(6000, 9000)

    # check prune_method
    if args.prune_method not in list(ModelPool.keys()):
        print(f"Possible args.prune_method: {list(ModelPool.keys())}")
        exit(0)
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    t0 = time()
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)
    training_duration = time() - t0

    # All done
    print("\nTraining complete.")

    ########## render.py ##########
    args_r = deepcopy(args)
    args_r.iteration = -1

    print("Rendering " + args_r.model_path)
    
    fps = render_sets(lp.extract(args_r), args_r.iteration, pp.extract(args_r), args_r.skip_train, args_r.skip_test)


    ########## eval.py ##########
    args_e = deepcopy(args_r)

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    print("Evaluating " + args_e.model_path)

    # Set up command line argument parser
    args_e.model_paths = [args_e.model_path]
    full_dict = evaluate(args_e.model_paths)
    
    wandb.log({
        "train/training_duration": training_duration,
        "train/fps": fps.get('train', None),
        "test/fps": fps.get('test', None),
        "test/ssim": full_dict[args_e.model_path][f'ours_{args.iterations}']['SSIM'],
        "test/psnr": full_dict[args_e.model_path][f'ours_{args.iterations}']['PSNR'],
        "test/lpips": full_dict[args_e.model_path][f'ours_{args.iterations}']['LPIPS'],
    }, commit=True)