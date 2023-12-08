import os, sys
import torch

# Import files from same folder
root_path = os.path.abspath('.')
sys.path.append(root_path)
from architecture.rrdb import RRDBNet
from architecture.grl import GRL
from architecture.swinir import SwinIR


def load_grl(generator_weight_PATH, print_options=True):
    ''' A simpler API to load GRL model
    Args:
        generator_weight_PATH (str): The path to the weight
        print_options (bool): whether to print options to show what kinds of setting is used
    Returns:
        generator (torch): the generator instance of the model
    '''

    # Load the checkpoint
    checkpoint_g = torch.load(generator_weight_PATH)

     # Find the generator weight
    if 'model_state_dict' in checkpoint_g:
        weight = checkpoint_g['model_state_dict']

        # GRL Small
        generator = GRL(
            upscale=4,
            img_size=144,
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


    else:
        print("This weight is not supported")
        os._exit(0)

    # torch.compile weight key rename
    old_keys = [key for key in weight]
    for old_key in old_keys:
        if old_key[:10] == "_orig_mod.":
            new_key = old_key[10:]
            weight[new_key] = weight[old_key]
            del weight[old_key]

    generator.load_state_dict(weight)
    generator = generator.eval().cuda()

    # Print options to show what kinds of setting is used
    if print_options:
        if 'opt' in checkpoint_g:
            for key in checkpoint_g['opt']:
                value = checkpoint_g['opt'][key]
                print(f'{key} : {value}')

    return generator
