import cv2
import torch
import numpy as np
import os, shutil, time
import sys, random
import gc
from math import log10, sqrt
import torch.nn.functional as F


# Import files from the local folder
root_path = os.path.abspath('.')
sys.path.append(root_path)
from degradation.ESR.utils import tensor2np
from opt import opt


def video_compression_model(out, store_lists, opt, verbose = False):
    ''' Video Compression Degradation model
    Args:
        out (tensor):           BxCxHxW All input images as tensor
        store_lists ([str]):    List of paths we used to store all images
        opt (dict):             All configuration we need to process
        verbose (bool):         Whether print some information for auxiliary log (default: False)
    '''


    # Preparation
    batch_size, _, cur_h, cur_w= out.size()
    store_path = "datasets/batch_output"
    if os.path.exists(store_path):
        shutil.rmtree(store_path)
    os.mkdir(store_path)


    # Store frames to png images
    for idx in range(batch_size):
        single_frame = tensor2np(out[idx])  # numpy format

        prefix = ""
        if idx < 100:
            prefix += "0" 
        if idx < 10:
            prefix += "0"
        prefix += str(idx)  # 从0-31
        cv2.imwrite(os.path.join(store_path, prefix + ".png"), single_frame)


    ########################## SetUp Hyper-parameter #########################################
    # Video codec setting
    video_codec = random.choices(opt['video_codec'], opt['video_codec_prob'])[0]  
    
    # FPS setting
    fps = str(random.randint(*opt['fps_range']))            #only integer for FPS
    if video_codec == "mpeg2video":
        fps = str(25) # It's said that mpeg2 can only either be 25 or 29.97 fps

    # Encode Preset setting (determines speed of encoding, will utilize different search algorithm)
    preset = random.choices(opt['encode_preset'], opt['encode_preset_prob'])[0]
    
    # CRF setting (for x264 and x265)
    assert(len(opt['video_codec']) == len(opt['crf_offset']))
    codec_idx = opt['video_codec'].index(video_codec)
    assert(codec_idx != -1)
    crf_value = random.randint(*opt['crf_range'])
    crf_value = str(crf_value + opt['crf_offset'][codec_idx]) # adjust crf for some worse situation
    # continues .... (hard code section)
    if video_codec == "mpeg2video" or video_codec == "libxvid":
        # mpeg2 and mpeg4 don't use crf to control bitrate, instead directly use bitrate controller (b:v) in ffmpeg to control it!
        bitrate = str(random.randint(*opt['mpeg2_4_bitrate_range']))  # only currently supports three choices
        crf = " -b:v " + bitrate + "k "
        crf_value = bitrate + "k "
    else:
        crf = " -crf " + crf_value + " "

    # Ratio Scaling setting
    ratio_type = random.choices(['shrink', 'expand', 'keep'], opt['ratio_prob'])[0]
    if ratio_type == 'expand':
        ratio_scale = np.random.uniform(1, opt['ratio_range'][1])
    elif ratio_type == 'shrink':
        ratio_scale = np.random.uniform(opt['ratio_range'][0], 1)
    else:
        ratio_scale = 1
    if video_codec == "mpeg2video":
        # For older mpeg2, we are more lenient to it on the ratio scaling, only scale expand(don't shrink)
        # *************** I recommend only shrink ratio scaling on mpeg4, h264, and h265 ***************
        ratio_scale = max(ratio_scale, 1)
    encode_scale = str(int(  ((ratio_scale*cur_w)//2) * 2 ) ) + ":" + str(cur_h)  #only even bug; else, there will be a bug
    decode_scale = str(cur_w) + ":" + str(cur_h)
    

    # Finish settting and print information out
    if verbose: 
        print(f"(1st) Video compression with codec {video_codec}, fps {fps}, crf {crf_value}, preset {preset}, scale {encode_scale}")
        f = open("datasets/degradation_log.txt", "a")
        f.write(f"Video compression with codec {video_codec}, fps {fps}, crf {crf_value}, preset {preset}, scale {encode_scale} \n\n")
        f.close()
    ############################################################################################


    ########################### Encode Frames to Video ###########################
    middle = " -x265-params log-level=error "
    additional = " "
    if not verbose:
        additional += "-loglevel 0"  # loglevel 0 is the quiest version


    video_store_dir = os.path.join(store_path, "merge.mp4") 
    if os.path.exists(video_store_dir):
        shutil.rmtree(video_store_dir)
        
    # Cannot use hardware encode here
    ffmpeg_encode_cmd = "ffmpeg -r " + fps + " -i " + store_path + "/%03d.png -vcodec " + video_codec + middle + crf + " -vf scale=" + encode_scale + " -preset " + preset + " -pix_fmt yuv420p " + video_store_dir + additional
    os.system(ffmpeg_encode_cmd)

    
    ############################## Decode Video to Frames ########################
    # output order is 1-128
    ffmpeg_decode_cmd = "ffmpeg -i " + video_store_dir + " -vf scale=" + decode_scale + ",fps=" + fps + " " + store_path + "/output_%03d.png " + additional
    os.system(ffmpeg_decode_cmd)

    if verbose: print(f"(1st) Video output with scale {decode_scale}")

    
    # Iterate all result and move to correct places
    for idx in range(len(out)):
        # intput dir setup
        prefix = ""
        if idx+1 < 100:
            prefix += "0" 
        if idx+1 < 10:
            prefix += "0"
        prefix += str(idx+1)
        input_path = os.path.join(store_path, "output_" + prefix + ".png")

        # output dir setup
        output_path = store_lists[idx]

        try:
            shutil.move(input_path, output_path)  
        except Exception as e:
            print(e)
            print(f"It is trying to move {input_path} to {output_path}")
            print("The following is information related to this bugs' codec, please have a look:")
            print(f"(1st) Video compression with codec {video_codec}, crf {crf}, preset {preset}, scale {encode_scale}")
            os._exit(0)


    ############################################################################################