import os
import torch
import pyiqa


class NIQE_metric(object):
    def __init__(self) -> None:
        device = torch.device("cuda")
        self.niqe_metric = pyiqa.create_metric('niqe', device=device)
        
    def __call__(self, img_path):
        ''' calculate NIQE value
        Args:
            img_path (str): image dir
        '''
        niqe_value = self.niqe_metric(img_path) # crop_border=0
        # niqe_value = calculate_niqe(img, crop_border=0) 
        return niqe_value.item()


