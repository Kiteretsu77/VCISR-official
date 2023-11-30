import os
import torch
import pyiqa

'''
    In the implementation of Brisque, we only considers non-zero value; however, for a fair comparison, the paper result of Brisque takes the average of all.
'''
class BRISQUE_metric(object):
    def __init__(self) -> None:
        device = torch.device("cuda")
        self.brisque_metric = pyiqa.create_metric('brisque', device=device)
        
    def __call__(self, img_path):
        ''' calculate BRISQUE value
        Args:
            img_path (str): image dir
        '''
        brisque_value = self.brisque_metric(img_path) # crop_border=0
        return brisque_value[0].item()

