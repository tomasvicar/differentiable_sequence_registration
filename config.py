import numpy as np
import torch

from utils import create_masks_hamming
from utils import create_masks_crop

class Config:
    

    # device = torch.device('cpu')
    device = torch.device('cuda:0')
    
    
    np_dtype = np.float32
    torch_dtype = torch.float32


    iterations = [80,120,140,150]
    
    gamma = 0.1
    
    # num_batches = 1
    num_batches = 10
    
    interp_mode = 'bilinear'
    # interp_mode = 'bicubic'
    
    align_corners = False


    # create_masks = create_masks_hamming(crop_fraction=0.1, np_dtype=np_dtype)
    create_masks = create_masks_crop(crop_fraction=0.1, np_dtype=np_dtype) 
    
    pad = 0
    

    regularization_factor = 1e-5


    scales = np.array([np.sqrt(128),np.sqrt(64),np.sqrt(32), np.sqrt(16), np.sqrt(8), np.sqrt(4), np.sqrt(2), 1])
    sigmas = scales - 1

    init_lr = 0.0002

    