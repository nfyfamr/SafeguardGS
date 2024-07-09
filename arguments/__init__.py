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

from argparse import ArgumentParser, Namespace
import sys
import os

def list_of_ints(arg):
    return list(map(int, arg.split(',')))

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                elif t == list:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=list_of_ints)
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                elif t == list:
                    group.add_argument("--" + key, default=value, type=list_of_ints)
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda"
        self.eval = False
        self.prune_method = '3dgs'
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002
        self.random_background = False
        self.n_split = 2
        ### compact_3dgs
        self.compact_3dgs_mask_lr = 0.01
        self.compact_3dgs_lambda_mask = 0.0005
        self.compact_3dgs_prune_iter = 1_000
        ### light_gaussian
        self.light_gaussian_prune_iterations = [20_000]
        self.light_gaussian_prune_percent = 0.6
        self.light_gaussian_prune_decay = 0.6
        self.light_gaussian_v_pow = 0.1
        ### random
        self.random_prune_iterations = [15_000]
        self.random_prune_ratio = 0.1
        ### mini_splatting
        self.mini_splatting_prune_iterations = [15_000]
        self.mini_splatting_preserving_ratio = 0.1
        self.mini_splatting_deterministic_prune = True # if False, use important score as probability distribution for sampling as the Mini-Splatting paper.
        ### rad_splat
        self.rad_splat_prune_threshold = 0.01 # 0.25 for light-weight model
        self.rad_splat_prune_iterations = [16_000, 24_000]
        ### efficient_gs
        self.efficient_gs_prune_iterations = [15_500]
        self.efficient_gs_prune_topk = 1
        ### safeguard_gs
        self.safeguard_gs_purne_topk = 10
        self.safeguard_gs_prune_iterations = [15_000]
        self.safeguard_gs_score_function = 0x01
        # Function IDs are defined using bitmasking. For example, `safeguard_gs_score_function=0x38` outputs `exp_color_error * dist_error * opacity * transmittance`.
        # First 2 bytes:
        #   0x00. score = 1
        #   0x01. score = opacity
        #   0x02. score = alpha
        #   0x03. score = opacity * transmittance
        #   0x04. score = alpha * transmittance (EfficientGS)
        #   0x05. score = dist error
        #   0x06. score = dist error * opacity
        #   0x07. score = dist error * alpha
        #   0x08. score = dist error * opacity * transmittance
        #   0x09. score = dist error * alpha * transmittance
        # Last 2 bytes:
        #   0x10. score = color error (Cosine similarity)
        #   0x20. score = color error (Manhattan distance)
        #   0x30. score = exp color error (Manhattan distance)
        self.safeguard_gs_p_dist_activation_coef = 1.0
        self.safeguard_gs_c_dist_activation_coef = 1.0
        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
