# -*- coding: utf-8 -*-
import sys
import os
import torch


# import important files
root_path = os.path.abspath('.')
sys.path.append(root_path)
from architecture.grl import GRL               # This place need to adjust for different models
from train_code.train_master import train_master



# Mixed precision training
scaler = torch.cuda.amp.GradScaler()


class train_grl(train_master):
    def __init__(self, options, args) -> None:
        super().__init__(options, args, "grl") 

    def loss_init(self):
        # Prepare pixel loss
        self.pixel_loss_load()
        

    def call_model(self):
        patch_size = 144
        window_size = 8

        # GRL Small
        self.generator = GRL(
            upscale=4,
            img_size=patch_size,
            window_size=8,
            depths=[4, 4, 4, 4],
            embed_dim=128,
            num_heads_window=[2, 2, 2, 2],
            num_heads_stripe=[2, 2, 2, 2],
            mlp_ratio=2,
            qkv_proj_type="linear",
            anchor_proj_type="avgpool",
            anchor_window_down_factor=2,
            out_proj_type="linear",
            conv_type="1conv",
            upsampler="pixelshuffle",
        ).cuda()

        # GRL Base
        # self.generator = GRL(
        #     upscale=4,
        #     img_size=patch_size,
        #     window_size=window_size,
        #     depths=[4, 4, 8, 8, 8, 4, 4],
        #     embed_dim=180,
        #     num_heads_window=[3, 3, 3, 3, 3, 3, 3],
        #     num_heads_stripe=[3, 3, 3, 3, 3, 3, 3],
        #     mlp_ratio=2,
        #     qkv_proj_type="linear",
        #     anchor_proj_type="avgpool",
        #     anchor_window_down_factor=2,
        #     out_proj_type="linear",
        #     conv_type="1conv",
        #     upsampler="pixelshuffle",
        #     local_connection=True,
        # ).cuda()
        
        # GRL Base for BSR
        # self.generator = GRL(
        #     upscale=4,
        #     in_channels=3,
        #     img_size=patch_size,
        #     img_range = 1.,
        #     window_size=16,
        #     depths=[4, 4, 8, 8, 8, 4, 4],
        #     embed_dim=180,
        #     num_heads_window=[3, 3, 3, 3, 3, 3, 3],
        #     num_heads_stripe=[3, 3, 3, 3, 3, 3, 3],
        #     stripe_size = [32, 64],
        #     stripe_groups = [None, None],
        #     stripe_shift = True,
        #     mlp_ratio=2,
        #     qkv_proj_type="linear",
        #     anchor_proj_type="avgpool",
        #     anchor_one_stage = True,
        #     anchor_window_down_factor = 4,
        #     out_proj_type="linear",
        #     conv_type="1conv",
        #     upsampler="nearest+conv",
        #     init_method = "n",
        #     fairscale_checkpoint = False,
        #     offload_to_cpu = False,
        #     double_window = False,
        #     stripe_square = False,
        #     separable_conv_act = True,
        #     local_connection = True,
        #     # use_buffer = True, 
        #     # use_efficient_buffer = True,
        # ).cuda()
        
        self.generator = torch.compile(self.generator).cuda()
        self.generator.train()

    
    def run(self):
        self.master_run()
                        

    
    def calculate_loss(self, gen_hr, imgs_hr):

        # Generator pixel loss (l1 loss):  generated vs. GT
        l_g_pix = self.cri_pix(gen_hr, imgs_hr, self.batch_idx)
        self.weight_store["pixel_loss"] = l_g_pix
        self.generator_loss += l_g_pix


    def tensorboard_report(self, iteration):
        # self.writer.add_scalar('Loss/train-Generator_Loss-Iteration', self.generator_loss, iteration)
        self.writer.add_scalar('Loss/train-Pixel_Loss-Iteration', self.weight_store["pixel_loss"], iteration)
