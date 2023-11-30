# -*- coding: utf-8 -*-
import random
import torch
import numpy as np
import os
import sys, random
from multiprocessing import Pool
from os import path as osp
from tqdm import tqdm
from math import log10, sqrt
import torch.nn.functional as F

root_path = os.path.abspath('.')
sys.path.append(root_path)
from degradation.ESR.degradations_functionality import *
from degradation.ESR.diffjpeg import *
from degradation.ESR.utils import filter2D
from opt import opt


def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr



def common_degradation(out, opt, kernels, verbose = False):
    jpeger = DiffJPEG(differentiable=False).cuda()
    kernel1, kernel2 = kernels

    
    order = [i for i in range(7)]
    random.shuffle(order)

    for idx in order:
        
        if idx == 0:
            # Element 0: Bluring kernel
            out = filter2D(out, kernel1)
            if verbose: print(f"(1st) blur noise")

        elif idx == 1:
            # Element 1:Resize with different mode
            updown_type = random.choices(['up', 'down', 'keep'], opt['resize_prob'])[0]
            if updown_type == 'up':
                scale = np.random.uniform(1, opt['resize_range'][1])
            elif updown_type == 'down':
                scale = np.random.uniform(opt['resize_range'][0], 1)
            else:
                scale = 1
            mode = random.choice(opt['resize_options'])
            out = F.interpolate(out, scale_factor=scale, mode=mode)
            if verbose: print(f"(1st) resize scale is {scale} and resize mode is {mode}")

        elif idx == 2:
            # Element 2: Noise effect (gaussian / poisson)
            gray_noise_prob = opt['gray_noise_prob']
            if np.random.uniform() < opt['gaussian_noise_prob']:
                # gaussian noise
                out = random_add_gaussian_noise_pt(
                    out, sigma_range=opt['noise_range'], clip=True, rounds=False, gray_prob=gray_noise_prob)
                name = "gaussian_noise"
            else:
                # poisson noise
                out = random_add_poisson_noise_pt(
                    out,
                    scale_range=opt['poisson_scale_range'],
                    gray_prob=gray_noise_prob,
                    clip=True,
                    rounds=False)
                name = "poisson_noise"
            if verbose: print("(1st) " + str(name))

        elif idx == 3:
            # Element 3: JPEG compression (并没有resize back)
            # jpeg_p = random.randint(*opt['jpeg_range'])
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*opt['jpeg_range'])
            out = torch.clamp(out, 0, 1)  # clamp to [0, 1], otherwise JPEGer will result in unpleasant artifacts
            if verbose: print("(1st) compressed img with quality " + str(jpeg_p) ) # 这个要放在out前面，不然值会很奇怪
            out = jpeger(out, quality=jpeg_p)

        elif idx == 4:
            # Element 4: Add blur 2nd time
            if np.random.uniform() < opt['second_blur_prob']:
                # 这个bluring不是必定触发的
                if verbose: print("(2nd) blur noise")
                out = filter2D(out, kernel2)

        elif idx == 5:
            # Element 5: Second Resize for 4x scaling
            updown_type = random.choices(['up', 'down', 'keep'], opt['resize_prob2'])[0]
            if updown_type == 'up':
                scale = np.random.uniform(1, opt['resize_range2'][1])
            elif updown_type == 'down':
                scale = np.random.uniform(opt['resize_range2'][0], 1)
            else:
                scale = 1
            mode = random.choice(opt['resize_options'])
            out = F.interpolate(out, scale_factor=scale, mode=mode)

        elif idx == 6:
            # Element 6: Add noise 2nd time
            gray_noise_prob = opt['gray_noise_prob2']
            if np.random.uniform() < opt['gaussian_noise_prob2']:
                # gaussian noise
                if verbose: print("(2nd) gaussian noise")
                out = random_add_gaussian_noise_pt(
                    out, sigma_range=opt['noise_range2'], clip=True, rounds=False, gray_prob=gray_noise_prob)
                name = "gaussian_noise"
            else:
                # poisson noise
                if verbose: print("(2nd) poisson noise")
                out = random_add_poisson_noise_pt(
                    out, scale_range=opt['poisson_scale_range2'], gray_prob=gray_noise_prob, clip=True, rounds=False)
                name = "poisson_noise"

        else:
            raise ValueError


        # Element 7: 就是JPEG/VC
        return out