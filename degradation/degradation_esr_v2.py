# -*- coding: utf-8 -*-

import argparse
import cv2
import torch
import numpy as np
import os, shutil, time
import sys, random
import gc
from os import path as osp
from tqdm import tqdm
from math import log10, sqrt
import torch.nn.functional as F


# Import files from the local folder
root_path = os.path.abspath('.')
sys.path.append(root_path)
from degradation.ESR.utils import *
from degradation.ESR.degradations_functionality import *
from degradation.ESR.diffjpeg import *
from degradation.ESR.degradation_esr_shared import common_degradation as regular_common_degradation
from degradation.ESR.degradation_esr_shuffled import common_degradation as shuffled_common_degradation
from degradation.video_compression import video_compression_model
from opt import opt



class degradation_v2:
    def __init__(self):
        self.kernel1, self.kernel2, self.sinc_kernel = None, None, None


    def reset_kernels(self, opt, debug=False):

        kernel1, kernel2, sinc_kernel = generate_kernels(opt)
        if debug:
            print("kernel1 is ", kernel1) 
            print("sinc_kernel is ", sinc_kernel)

        self.kernel1 = kernel1.unsqueeze(0).cuda()
        self.kernel2 = kernel2.unsqueeze(0).cuda()
        self.sinc_kernel = sinc_kernel.unsqueeze(0).cuda()


    @torch.no_grad()
    def degradate_process(self, out, opt, store_lists, verbose = False, use_shuffled=False):
        ''' ESR Degradation V2 mode (with Video Compression Model)
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



        # Add sinc & Video compression: [resize back + sinc filter] + Video compression]
        mode = random.choice(opt['resize_options'])
        # Resize back + Sinc filter
        out = F.interpolate(out, size=(ori_h // opt['scale'], ori_w // opt['scale']), mode = mode)
        # out = filter2D(out, self.sinc_kernel) # we drop sinc kernel in video compression module
        out = torch.clamp(out, 0, 1)


        # Video Compression model API
        video_compression_model(out, store_lists, opt, verbose)
        


        

