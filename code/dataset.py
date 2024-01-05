import numpy as np
import torch
import os
import cv2
from glob import glob

def toTensor(inp):
    '''adapt channel layout from (..., H, W, C) to (..., C, H, W) and convert to pytorch tensor'''
    return torch.from_numpy(np.moveaxis(inp, -1, -3)).contiguous().float()

class FocalDataset(torch.utils.data.Dataset):
    def __init__(self, path='./integrals', used_focal_lengths_idx=None,
                 input_channels=3, output_channels=3, augment=False, normalize=True, seed=42):
        '''
        Dataset class to load generated integral focal stacks

        used_focal_lengths_idx: list with idx-values from 0-9 are allowed, O = focal-length with 0m, 1 = focal-length with 0.2m,
            2 = 0.4m, 3 = 0.6m, 4 = 0.8m, ...

        path: path to directory filled with integral focal stacks and ground truths
        grayscale: if True, return integrals and ground truths with a single color channel. If False, repeats that single channel three times for "rbg"
        augment: randomly flip and rotate
        normalize: If true values are in the range [0,1], else in [0, 255]
        seed: random seed for augmentation
        '''
        self.path = path
        self.num_files = len(glob(os.path.join(self.path, '*_integral.tiff')))
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.augment = augment
        self.normalize = normalize
        self.rng = np.random.default_rng(seed)

        if used_focal_lengths_idx is None:
            self.used_focal_lengths_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        else:
            self.used_focal_lengths_idx = used_focal_lengths_idx
        
    def __len__(self):
        return self.num_files

    def __getitem__(self, index):
        ok, focal_stack = cv2.imreadmulti(os.path.join(self.path, f'{index}_integral.tiff'))
        if not ok:
            raise IOError(f'Failed to load index: {index}')
            
        focal_stack = np.stack(focal_stack)[...,None]   # shape (num_focal_lengths, w, h, 1)
        focal_stack = focal_stack[self.used_focal_lengths_idx]

        ground_truth = cv2.imread(os.path.join(self.path, f'{index}_gt.png'))[...,[0]]   # shape (w, h, 1)
        
        if self.augment:
            # augment the input and target images with random horizontal and vertical flips, and a random number of 90 degree rotations
            
            flip1, flip2 = self.rng.random(2)
            if flip1 > 0.5:
                focal_stack = np.flip(focal_stack, -2)
                ground_truth = np.flip(ground_truth, -2)
            if flip2 > 0.5:
                focal_stack = np.flip(focal_stack, -3)
                ground_truth = np.flip(ground_truth, -3)
            
            num_rot = self.rng.integers(4)
            focal_stack = np.rot90(focal_stack, num_rot, (-3, -2))
            ground_truth = np.rot90(ground_truth, num_rot, (-3, -2))
            
        if self.normalize:
            focal_stack = focal_stack / np.float32(256)
            ground_truth = ground_truth / np.float32(256)
            
        focal_stack = np.repeat(focal_stack, self.input_channels, -1)
        ground_truth = np.repeat(ground_truth, self.output_channels, -1)
        
        return toTensor(focal_stack), toTensor(ground_truth)