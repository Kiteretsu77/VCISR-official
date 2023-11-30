# -*- coding: utf-8 -*-
import argparse
import cv2, time
import torch
import os, shutil, time
import sys, random
from os import path as osp
from tqdm import tqdm
import copy
import warnings
import gc

warnings.filterwarnings("ignore")

# import same folder files #
root_path = os.path.abspath('.')
sys.path.append(root_path)
from degradation.ESR.utils import np2tensor
from degradation.ESR.degradations_functionality import *
from degradation.ESR.diffjpeg import *
from degradation.degradation_esr_v2 import degradation_v2
from degradation.degradation_esr_v1 import degradation_v1
from opt import opt
os.environ['CUDA_VISIBLE_DEVICES'] = opt['CUDA_VISIBLE_DEVICES']  #'0,1'




@torch.no_grad()
def generate_low_res_esr(org_opt, verbose=False):
    ''' Generate LR dataset from HR ones by ESR degradation
    Args:
        org_opt (dict):     The setting we will use
        verbose (bool): Whether we print out some information
    '''

    # Prepare folders
    input_folder = org_opt['input_folder']
    save_folder = org_opt['save_folder']
    if osp.exists(save_folder):
        shutil.rmtree(save_folder)
    os.makedirs(save_folder)

    # Scan all images
    input_img_lists, output_img_lists = [], []
    for file in sorted(os.listdir(input_folder)):      
        input_img_lists.append(osp.join(input_folder, file))
        output_img_lists.append(osp.join(save_folder, file))
        

    # Setting
    batch_size = org_opt["degradation_batch_size"]  
    img_length = len(input_img_lists)

    obj_img = degradation_v1()
    obj_vc = degradation_v2()
    

    # Remove log file
    if os.path.exists("datasets/degradation_log.txt"):
        os.remove("datasets/degradation_log.txt")


    # Extract image to torch batch
    iter_lists = []
    first_iter_length = min(random.randint(batch_size // 4, batch_size-1), img_length)
    iter_lists.append(first_iter_length)

    middle_batches_num = (img_length - first_iter_length) // batch_size
    for _ in range(middle_batches_num):
        iter_lists.append(batch_size)

    last_iter_length = img_length - first_iter_length - middle_batches_num * batch_size
    if last_iter_length == 0:
        total_range = middle_batches_num + 1
    else:
        total_range = middle_batches_num + 2
        iter_lists.append(last_iter_length)
    
    assert(sum(iter_lists) == len(input_img_lists))
    
    
    

    # Iterate all batches
    for batch_idx in tqdm(range(0, total_range), desc="Degradation"):
        # Make a copy of the org_opt hyperparameter
        opt = copy.deepcopy(org_opt)


        # Reset kernels in every degradation batch for ESR
        obj_img.reset_kernels(opt)
        obj_vc.reset_kernels(opt)

        # Find the needed img lists
        iter_length = iter_lists.pop(0)
        needed_img_lists = []
        store_img_lists = []
        for _ in range(iter_length):
            needed_img_lists.append(input_img_lists.pop(0))
            store_img_lists.append(output_img_lists.pop(0))


        # Read all images and transform them to tensor
        out = None
        for idx in range(len(needed_img_lists)):

            input_path = needed_img_lists[idx]

            img_bgr = cv2.imread(input_path)


            if out is None:
                out = np2tensor(img_bgr) # tensor
            else:
                out = torch.cat((out, np2tensor(img_bgr)), 0)
        try:
            _, _, _, _ = out.size()
        except Exception:
            print(batch_idx, first_iter_length, last_iter_length, total_range)
            print(out)
            os._exit(0)
        
        

        if opt['degradation_mode'] == 'V1':
            # ESR V1 execuation
            obj_img.degradate_process(out, opt, store_img_lists, verbose = False, use_shuffled=False)
        elif opt['degradation_mode'] == 'V2':
            if random.random() < opt['v1_proportion']:
                # V1 skip mode
                obj_img.degradate_process(out, opt, store_img_lists, verbose = False, use_shuffled=False)
            else:
                obj_vc.degradate_process(out, opt, store_img_lists, verbose = False, use_shuffled=False)
        else:
            raise NotImplementedError

        # I think that we need to clean memory here
        del out
        gc.collect()
        if batch_idx != 0 and batch_idx%4 == 0:
            # empty the torch cache at certain iteration of each epoch
            torch.cuda.empty_cache()

    assert(len(input_img_lists) == 0)
        
        
def main(args):
    opt['input_folder'] = args.input
    opt['save_folder'] = args.output

    generate_low_res_esr(opt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default= os.path.join(opt['dataset_path'], opt["degrade_hr_dataset_name"]), help='Input folder')
    parser.add_argument('--output', type=str, default= os.path.join(opt['dataset_path'], opt["lr_dataset_name"]), help='Output folder')
    args = parser.parse_args()

    main(args)