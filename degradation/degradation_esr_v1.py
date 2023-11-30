# -*- coding: utf-8 -*-

import argparse
import cv2
import torch
import numpy as np
import gc
import os, shutil, time
import sys, random
from os import path as osp
from tqdm import tqdm
from math import log10, sqrt
import torch.nn.functional as F

root_path = os.path.abspath('.')
sys.path.append(root_path)
# import sample folder files
from degradation.ESR.utils import *
from degradation.ESR.degradations_functionality import *
from degradation.ESR.diffjpeg import *
from degradation.ESR.degradation_esr_shared import common_degradation as regular_common_degradation
from degradation.ESR.degradation_esr_shuffled import common_degradation as shuffled_common_degradation
from opt import opt





class degradation_v1:
    def __init__(self):
        self.kernel1, self.kernel2, self.sinc_kernel = None, None, None
        self.queue_size = 160


    def reset_kernels(self, opt):
        kernel1, kernel2, sinc_kernel = generate_kernels(opt)
        self.kernel1 = kernel1.unsqueeze(0).cuda()
        self.kernel2 = kernel2.unsqueeze(0).cuda()
        self.sinc_kernel = sinc_kernel.unsqueeze(0).cuda()
        

    @torch.no_grad()
    def degradate_process(self, out, opt, store_lists, verbose = False, use_shuffled=False):
        ''' ESR Degradation V1 mode (Same as the original paper)
        Args:
            out (tensor):           BxCxHxW All input images as tensor
            opt (dict):             All configuration we need to process 
            store_lists ([str]):    List of paths we used to store all images
            verbose (bool):         Whether print some information for auxiliary log (default: False)
            use_shuffled (bool):     Whether the common degradation use shuffled version (default: False)
        '''

        batch_size, _, ori_h, ori_w= out.size()

        # Shared degradation until the last step
        if use_shuffled:
            out = shuffled_common_degradation(out, opt, [self.kernel1, self.kernel2], verbose=verbose)
        else:
            out = regular_common_degradation(out, opt, [self.kernel1, self.kernel2], verbose=verbose)

        jpeger = DiffJPEG(differentiable=False).cuda()
        


        # Add sinc & JPEG compression
        # Group [resize back + sinc filter] together as one operation.
        # We consider two orders:
        #   1. [resize back + sinc filter] + JPEG compression
        #   2. JPEG compression + [resize back + sinc filter]
        # Empirically, we find other combinations (sinc + JPEG + Resize) will introduce twisted lines.
        mode = random.choice(opt['resize_options'])
        jpeg_p = out.new_zeros(out.size(0)).uniform_(*opt['jpeg_range2'])
        if np.random.uniform() < 0.5:
            # Resize back + sinc filter
            out = F.interpolate(out, size=(ori_h // opt['scale'], ori_w // opt['scale']), mode = mode)
            out = filter2D(out, self.sinc_kernel)
            if verbose: print(f"(2nd) Sinc Filter process with resize back mode {mode}")

            # JPEG compression, 论文说跟cv2的压缩方式还有点不一样，这个要研究一下
            out = torch.clamp(out, 0, 1)  # clamp to [0, 1], otherwise JPEGer will result in unpleasant artifacts
            if verbose: print("(2nd) compressed img with quality " +  str(jpeg_p))
            out = jpeger(out, quality=jpeg_p)

        else:
            # JPEG compression, 论文说跟cv2的压缩方式还有点不一样，这个要研究一下
            out = torch.clamp(out, 0, 1)  # clamp to [0, 1], otherwise JPEGer will result in unpleasant artifacts
            if verbose: print("(2nd) compressed img with quality " +  str( jpeg_p))
            out = jpeger(out, quality=jpeg_p)
            

            # Resize back + sinc filter
            out = F.interpolate(out, size=(ori_h // opt['scale'], ori_w // opt['scale']), mode = mode)
            out = filter2D(out, self.sinc_kernel)
            if verbose: print(f"(2nd) Sinc Filter process with resize back mode {mode}")

        # Clamp and Round
        out = torch.clamp((out * 255.0).round(), 0, 255) / 255.


        # Save all files down
        for idx in range(batch_size):
            output_name = store_lists[idx]
            single_frame = tensor2np(out[idx])
            # Store the image
            cv2.imwrite(output_name, single_frame)            



