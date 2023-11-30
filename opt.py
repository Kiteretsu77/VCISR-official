# -*- coding: utf-8 -*-
import os


opt = {}
##################################################### Global Setting ###########################################################
opt['description'] = ""             # Description for model, e.g. 4x_GRL_official

opt['architecture'] = "GRL"                         # "GRL" || "GRLGAN"
opt['base_degradation_model'] = "ESR"               # "ESR" (Real-ESRGAN Backbone)
opt['degradation_mode'] = "V2"                      # "V1" (Only Image Compression degradation) || "V2" (With Video compression Degradation)
################################################################################################################################

# GPU setting
opt['CUDA_VISIBLE_DEVICES'] = '0'           #   '0/1/2/3/4' depends on your GPU
os.environ['CUDA_VISIBLE_DEVICES'] = opt['CUDA_VISIBLE_DEVICES']  


##################################################### Setting for General Training #############################################
# Essential setting
opt['scale'] = 4                         # default scale is 4 in our paper
opt['degradate_generation_freq'] = 1     # How frequent we degradat HR to LR (1: means Real-Time Degrade)
opt['train_dataloader_workers'] = 5      # Number of workers for DataLoader
opt['checkpoints_freq'] = 50             # frequency to store checkpoints in the folder

# Dataset Path
opt['dataset_path'] = "datasets/"                      # Dataset path (usually it should be under the same folder, and must have train_lr or train_hr in the folder)
opt["degrade_hr_dataset_name"] = "train_hr"            # degradate start dataset：
opt["train_hr_dataset_name"] = "train_hr_usm"          # Usually, this place is to decide whether you use usm dataset
opt["lr_dataset_name"] = "train_lr"                    # LR store name

# Shared loss
opt['pixel_loss'] = "L1"                                # "L1" || "L1_Charbonnier_loss" || "L1_MS-SSIM"

# Adam optimizer setting
opt["adam_beta1"] = 0.9
opt["adam_beta2"] = 0.99
opt['decay_gamma'] = 0.5                                # Decay the learning rate per decay_iteration
opt['MS-SSIM_alpha'] = 0.2                              # The alpha weight for MS-SSIM and L1 loss will be 1-alpha weight

#################################################################################################################################

########## Model Architecture Setting ######################3
if opt['architecture'] == "GRL":
    # Setting for GRL Training 
    opt['train_iterations'] = 700000            # Training Iterations
    opt['train_batch_size'] = 12                # Large resolution (574x574) batch size = 8; Smaller resolution (360x360) batch size = 12
    opt["start_learning_rate"] = 0.0002         # Training Epoch, use the as Real-ESRGAN: 0.0001 - 0.0002 is ok, based on your need

    opt['decay_iteration'] = 50000            # Decay iteration  
    opt['double_milestones'] = [opt['decay_iteration']*4, opt['decay_iteration']*8, opt['decay_iteration']*11]         # Iteration based time you double your learning rate


elif opt['architecture'] == "GRLGAN":
    # Settomg fpr GRL GAN Traning
    opt['train_iterations'] = 280000         # Training Iterations
    opt['train_batch_size'] = 6             # For Large resolution (574x574), batch size = 6; For Smaller resolution (360x360), batch size = 12
    opt["start_learning_rate"] = 0.0001      # Training Epoch, use the as Real-ESRGAN: 0.0001 - 0.0002 is ok, based on your need

    opt["perceptual_loss_weight"] = 1.0
    opt['train_perceptual_vgg_type'] = 'vgg19'
    opt['train_perceptual_layer_weights'] = {'conv1_2': 0.1, 'conv2_2': 0.1, 'conv3_4': 1, 'conv4_4': 1, 'conv5_4': 1}
    opt["gan_loss_weight"] = 0.1   

    opt['decay_iteration'] = 20000                              # Decay iteration  
    opt['double_milestones'] = [opt['decay_iteration']*4, opt['decay_iteration']*8, opt['decay_iteration']*11]         # Iteration based time you double your learning rate


else:
    raise NotImplementedError("Please check you architecture option setting!")


# Basic setting for degradation
opt["degradation_batch_size"] = 128                 # Degradation batch size
opt["augment_prob"] = 0.5                           # Probability of augmenting (Flip, Rotate) the HR and LR dataset in dataset loading part                                


if opt['architecture'] in ["GRL", "GRLGAN"]:       

    # Blur kernel1
    opt['kernel_range'] = [3, 11]      
    opt['kernel_list'] = ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso']
    opt['kernel_prob'] = [0.45, 0.25, 0.12, 0.03, 0.12, 0.03]  
    opt['sinc_prob'] = 0.1             
    opt['blur_sigma'] = [0.2, 3]      
    opt['betag_range'] = [0.5, 4]       
    opt['betap_range'] = [1, 2]      

    # Blur kernel2 
    opt['kernel_list2'] = ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso']
    opt['kernel_prob2'] = [0.45, 0.25, 0.12, 0.03, 0.12, 0.03]   
    opt['sinc_prob2'] = 0.1            
    opt['blur_sigma2'] = [0.2, 1.5]    
    opt['betag_range2'] = [0.5, 4]      
    opt['betap_range2'] = [1, 2]        

    # The first degradation process
    opt['resize_prob'] = [0.2, 0.7, 0.1]       
    opt['resize_range'] = [0.15, 1.5]               
    opt['gaussian_noise_prob'] = 0.5            
    opt['noise_range'] =  [1, 30]               
    opt['poisson_scale_range'] = [0.05, 3]    
    opt['gray_noise_prob'] =  0.4               
    opt['jpeg_range'] = [30, 95]              

    # The second degradation process
    opt['second_blur_prob'] =  0.8             
    opt['resize_prob2'] = [0.3, 0.4, 0.3]           # [up, down, keep] Resize Probability
    opt['resize_range2'] = [0.3, 1.2]               
    opt['gaussian_noise_prob2'] = 0.5          
    opt['noise_range2'] = [1, 25]               
    opt['poisson_scale_range2'] = [0.05, 2.5]    
    opt['gray_noise_prob2'] = 0.4           
    
    opt['final_sinc_prob'] = 0.8                    

    # Other common settings
    opt['resize_options'] = ['area', 'bilinear', 'bicubic']     


# This only changes the last image compression of V1 to video compression
if opt['degradation_mode'] == "V2":         
    # Setting for Degradation with Video Compression (V2)

    # V1 Skip setting
    opt['v1_proportion'] = 0.05                     # [~0.05] 
    opt['jpeg_range2'] = [30, 95]                   # V1 JPEG proportion
    
    # Codec
    opt['video_codec'] = ["mpeg2video", "libxvid", "libx264", "libx265"]     # codec
    opt['video_codec_prob'] = [0.2, 0.2, 0.4, 0.2]  

    # CRF
    opt['crf_range'] = [20, 32]                                                                
    opt['crf_offset'] = [0, 0, 0, 5]                            # CRF=23: AVC's default value; CRF=8: HEVC's default value
    opt['mpeg2_4_bitrate_range'] = [3800, 5800]                 

    # Preset
    opt['encode_preset'] = ["slow", "medium", "fast", "faster", "superfast"]         
    opt['encode_preset_prob'] = [0.1, 0.5, 0.25, 0.12, 0.03]                        

    # Auxiliary (Ratio Scaling + FPS)    
    opt['ratio_prob'] = [0.2, 0.4, 0.4]                         # shrink, expand, keep: just width adjust prob
    opt['ratio_range'] = [0.8, 1.35]                            # bottom, ceil 
    opt['fps_range'] = [16, 30]                                 

elif opt['degradation_mode'] == "V1":
    if opt['architecture'] in ["ESRNET", "ESRGAN", "GRL", "GRLGAN"]:
        opt['jpeg_range2'] = [30, 95]

else:
    raise NotImplementedError("Please check you architecture option setting!")


    
