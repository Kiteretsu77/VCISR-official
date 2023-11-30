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
        loss = checkpoint_g["lowest_generator_weight"]
        if "iteration" in checkpoint_g:
            iteration = checkpoint_g["iteration"]
        else:
            iteration = "NAN"

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

        print(f"the generator weight is {loss} at iteration {iteration}")

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


def load_grl_base(generator_weight_PATH, print_options=True):
    ''' A simpler API to load GRL model
    Args:
        generator_weight_PATH (str): The path to the weight
        print_options (bool): whether to print options to show what kinds of setting is used
    Returns:
        generator (torch): the generator instance of the model
    '''
    # generator_weight_PATH = "saved_models/bsr_grl_base.ckpt"

    # GRL-BASE for Real-World SR configuration based on yaml setting
    generator = GRL(
        upscale=4,
        in_channels=3,
        img_size=128,
        img_range = 1.,
        window_size=16,
        depths=[4, 4, 8, 8, 8, 4, 4],
        embed_dim=180,
        num_heads_window=[3, 3, 3, 3, 3, 3, 3],
        num_heads_stripe=[3, 3, 3, 3, 3, 3, 3],
        stripe_size = [32, 64],
        stripe_groups = [None, None],
        stripe_shift = True,
        mlp_ratio=2,
        qkv_proj_type="linear",
        anchor_proj_type="avgpool",
        anchor_one_stage = True,
        anchor_window_down_factor = 4,
        out_proj_type="linear",
        conv_type="1conv",
        upsampler="nearest+conv",
        init_method = "n",
        fairscale_checkpoint = False,
        offload_to_cpu = False,
        double_window = False,
        stripe_square = False,
        separable_conv_act = True,
        local_connection = True,
        # use_buffer = True, 
        # use_efficient_buffer = True,
    ).cuda()


    # Load weights
    checkpoint = torch.load(generator_weight_PATH)
    weight = checkpoint['state_dict']
    old_keys = [key for key in weight]
    key_word = "model_g."
    for old_key in old_keys:
        if old_key[:len(key_word)] == key_word:
            new_key = old_key[len(key_word):]
            weight[new_key] = weight[old_key]
            del weight[old_key]

    generator.load_state_dict(weight, strict=False)


    return generator


def load_swinir(generator_weight_PATH, scale, large_model=False):
    ''' A simpler API to load SwinIR model
    Args:
        generator_weight_PATH (str): The path to the weight
        scale (int): the scaling factor
        large_model (bool): whether we use SwinIR large model setting
    Returns:
        generator (torch): the generator instance of the model
    '''  

    # Load the generator (The followin is only the setting for Real-World SR, other task of SwinIR may use differently)
    if not large_model:
        # use 'nearest+conv' to avoid block artifacts
        generator = SwinIR(upscale=scale, in_chans=3, img_size=64, window_size=8,
                    img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                    mlp_ratio=2, upsampler='nearest+conv', resi_connection='1conv').cuda()
    else:
        # larger model size; use '3conv' to save parameters and memory; use ema for GAN training
        generator = SwinIR(upscale=scale, in_chans=3, img_size=64, window_size=8,
                    img_range=1., depths=[6, 6, 6, 6, 6, 6, 6, 6, 6], embed_dim=240,
                    num_heads=[8, 8, 8, 8, 8, 8, 8, 8, 8],
                    mlp_ratio=2, upsampler='nearest+conv', resi_connection='3conv').cuda()
    param_key_g = 'params_ema'


    # Load the weight
    pretrained_model = torch.load(generator_weight_PATH)
    generator.load_state_dict(pretrained_model[param_key_g] if param_key_g in pretrained_model.keys() else pretrained_model, strict=True)

    return generator


def load_rrdb_bsr(generator_weight_PATH, scale):  
    ''' A simpler API to load BSR/RealSR version RRDB model (The model param naming may be a little bit different)
    Args:
        generator_weight_PATH (str): The path to the weight
        scale (int): the scaling factor
        print_options (bool): whether to print options to show what kinds of setting is used
    Returns:
        generator (torch): the generator instance of the model
    '''  

    # Prepare the generator
    generator = RRDBNet(3, 3, scale=scale)  # The other setting should be ok with the default setting

    # Load the weight from the checkpoint
    weight = torch.load(generator_weight_PATH)

    # Rename the checkpoint weight
    old_keys = [key for key in weight]
    for old_key in old_keys:
        elements = old_key.split('.')
        
        # Map the old key to the new key (May need hard coding here)
        if elements[0] == "RRDB_trunk":
            new_key = "body." + str(elements[1]) + "." + elements[2].lower() + "." + elements[3] + "." + elements[4]

        elif elements[0] == "trunk_conv":
            new_key = "conv_body." + str(elements[1])

        elif elements[0] == "upconv1":
            new_key = "conv_up1." + str(elements[1])

        elif elements[0] == "upconv2":
            new_key = "conv_up2." + str(elements[1])

        elif elements[0] == "HRconv":
            new_key = "conv_hr." + str(elements[1])

        else:
            # With the same name, we need to change
            continue

        weight[new_key] = weight[old_key]
        del weight[old_key]
         

    # Load 
    generator.load_state_dict(weight)
    generator = generator.eval().cuda()

    return generator


def load_rrdb(generator_weight_PATH, scale, print_options=False):  
    ''' A simpler API to load RRDB model
    Args:
        generator_weight_PATH (str): The path to the weight
        scale (int): the scaling factor
        print_options (bool): whether to print options to show what kinds of setting is used
    Returns:
        generator (torch): the generator instance of the model
    '''  

    # Load the checkpoint
    checkpoint_g = torch.load(generator_weight_PATH)

    # Find the generator weight
    if 'params_ema' in checkpoint_g:
        # For official ESRNET/ESRGAN weight
        weight = checkpoint_g['params_ema']
        generator = RRDBNet(3, 3, scale=scale)

    elif 'params' in checkpoint_g:
        # For official ESRNET/ESRGAN weight
        weight = checkpoint_g['params']
        generator = RRDBNet(3, 3, scale=scale)

    elif 'model_state_dict' in checkpoint_g:
        # For my personal trained weight
        weight = checkpoint_g['model_state_dict']
        loss = checkpoint_g["lowest_generator_weight"]
        if "iteration" in checkpoint_g:
            iteration = checkpoint_g["iteration"]
        else:
            iteration = "NAN"
        generator = RRDBNet(3, 3, scale=scale)  
        # generator = torch.compile(generator)# torch.compile
        print(f"the generator weight is {loss} at iteration {iteration}")

    else:
        print("This weight is not supported")
        os._exit(0)


    # Handle torch.compile weight key rename
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