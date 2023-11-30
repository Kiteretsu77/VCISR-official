import os
import torch
import pyiqa


class CLIPIQA_metric(object):
    def __init__(self) -> None:
        device = torch.device("cuda")
        self.clipiqa_metric = pyiqa.create_metric('clipiqa', device=device)
        
    def __call__(self, img_path):
        ''' calculate NIQE value
        Args:
            img_path (str): image dir
        '''
        value = self.clipiqa_metric(img_path) 
        return value.item()

