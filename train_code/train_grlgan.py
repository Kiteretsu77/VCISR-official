# -*- coding: utf-8 -*-

import  sys
import os
import torch

# import important files
root_path = os.path.abspath('.')
sys.path.append(root_path)
from architecture.grl import GRL
from architecture.discriminator import UNetDiscriminatorSN
from train_code.train_master import train_master



class train_grlgan(train_master):
    def __init__(self, options, args) -> None:
        super().__init__(options, args, "grlgan", True)


    def loss_init(self):

        # prepare pixel loss (Generator)
        self.pixel_loss_load()

        # prepare perceptual loss
        self.GAN_loss_load()


    def call_model(self):
        # Generator: GRL Small
        patch_size = 144
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
        self.generator = torch.compile(self.generator).cuda()

        # Discriminator
        self.discriminator = torch.compile(UNetDiscriminatorSN(3)).cuda()
        
        self.generator.train(); self.discriminator.train()

    def run(self):
        self.master_run()
                        


    def calculate_loss(self, gen_hr, imgs_hr):

        ###########  Real CUGAN has 3 losses on Generator  ###########
        # Generator pixel loss (l1 loss):  generated vs. GT
        l_g_pix = self.cri_pix(gen_hr, imgs_hr)
        self.generator_loss += l_g_pix
        self.weight_store["pixel_loss"] = l_g_pix

        # Generator perceptual loss:        generated vs. perceptual
        l_g_percep = self.cri_perceptual(gen_hr, imgs_hr)
        self.generator_loss += l_g_percep
        self.weight_store["perceptual_loss"] = l_g_percep

        # Generator GAN loss               label correction
        fake_g_preds = self.discriminator(gen_hr)
        l_g_gan = self.cri_gan(fake_g_preds, True, is_disc=False) 
        self.generator_loss += l_g_gan
        self.weight_store["gan_loss"] = l_g_gan 


    def tensorboard_report(self, iteration):
        self.writer.add_scalar('Loss/train-Generator_Loss-Iteration', self.generator_loss, iteration)
        self.writer.add_scalar('Loss/train-Pixel_Loss-Iteration', self.weight_store["pixel_loss"], iteration)
        self.writer.add_scalar('Loss/train-Perceptual_Loss-Iteration', self.weight_store["perceptual_loss"], iteration)
        self.writer.add_scalar('Loss/train-Discriminator_Loss-Iteration', self.weight_store["gan_loss"], iteration)

