import numpy as np
import torch
import os
import cv2
from glob import glob

def toTensor4d(inp):
    '''adapt channel layout from (..., H, W, C) to (..., C, H, W) and convert to pytorch tensor'''
    return torch.from_numpy(np.moveaxis(inp, -1, -3)).contiguous().div(256)                             # TODO: Consider if we want to normalize to [0,1] or just convert to float in range [0, 255]

class FocalDataset(torch.utils.data.Dataset):
    def __init__(self, path='./integrals', augment=False):
        self.path = path
        self.num_files = len(glob(os.path.join(self.path, '*_integral.tiff')))
        self.augment = augment
        
    def __len__(self):
        return self.num_files
    
    def __getitem__(self, index):
        ok, focal_stack = cv2.imreadmulti(os.path.join(self.path, f'{index}_integral.tiff'))
        if not ok:
            raise IOError(f'Failed to load index: {index}')
        focal_stack = np.stack(focal_stack)
        ground_truth = cv2.imread(os.path.join(self.path, f'{index}_gt.png'))
        
        focal_stack, ground_truth = toTensor4d(focal_stack), toTensor4d(ground_truth)
        
        if self.augment:
            raise NotImplementedError
        
        return focal_stack, ground_truth