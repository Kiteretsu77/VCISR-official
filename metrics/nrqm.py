import os
import torch
import pyiqa


class NRQM_metric(object):
    def __init__(self) -> None:
        device = torch.device("cuda")
        self.nrqm_metric = pyiqa.create_metric('nrqm', device=device)
        
    def __call__(self, img_path):
        ''' calculate NRQM value
        Args:
            img_path (str): image dir
        '''
        nrqm_value = self.nrqm_metric(img_path) # crop_border=0
        return nrqm_value.item()

