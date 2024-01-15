import numpy as np
import torch
import os
import cv2
from glob import glob

def toTensor(inp):
    '''adapt channel layout from (..., H, W, C) to (..., C, H, W) and convert to pytorch tensor'''
    return torch.from_numpy(np.moveaxis(inp, -1, -3)).contiguous().float()
    
class PositionDataset(torch.utils.data.Dataset):
    def __init__(self, path='./integrals', used_focal_lengths_idx=slice(None),  # use all available
                 augment=False, normalize=True, seed=42):
        '''
        Dataset class to load generated integral focal stacks

        used_focal_lengths_idx: list with idx-values from 0-3 are allowed, O = focal-length with 0m, 1 = focal-length with 0.4m, 

        path: path to directory filled with integral focal stacks and ground truths
        grayscale: if True, return integrals and ground truths with a single color channel. If False, repeats that single channel three times for "rbg"
        augment: randomly flip and rotate
        normalize: If true values are in the range [0,1], else in [0, 255]
        seed: random seed for augmentation
        '''
        self.path = path
        self.files = glob(os.path.join(self.path, '*_integral.tiff'))
        self.augment = augment
        self.normalize = normalize
        self.rng = np.random.default_rng(seed)
        self.used_focal_lengths_idx = used_focal_lengths_idx
        
    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        int_file = self.files[index]
        prefix = os.path.basename(int_file).rsplit('_', 1)[0]
        gt_file = os.path.join(self.path, prefix + '_gt.png')
        
        ok, focal_stack = cv2.imreadmulti(int_file)
        if not ok:
            raise IOError(f'Failed to load index: {int_file}')
            
        focal_stack = np.stack(focal_stack)   # shape (num_focal_lengths, w, h)
        focal_stack = focal_stack[self.used_focal_lengths_idx]
        focal_stack = np.moveaxis(focal_stack, 0, -1)  # shape (w, h, num_focal_lengths)
        
        xy = prefix.split('_')[-2:]
        if xy[0] == 'N' or xy[1] == 'N':
            person_position = np.array([0,0])
        else:
            person_position = np.array(list(map(int, xy)))

        ground_truth = cv2.imread(gt_file)[...,[0]]   # shape (w, h, 1)
        
        if self.augment:
            # augment the input and target images with random horizontal and vertical flips, and a random number of 90 degree rotations
            
            flip1, flip2 = self.rng.random(2)
            if flip1 > 0.5:  # x-flip
                focal_stack = np.flip(focal_stack, -2)
                ground_truth = np.flip(ground_truth, -2)
                person_position *= np.array([1, -1])
            if flip2 > 0.5:   # y-flip
                focal_stack = np.flip(focal_stack, -3)
                person_position *= np.array([-1, 1])
                ground_truth = np.flip(ground_truth, -3)
            
            num_rot = self.rng.integers(4)
            focal_stack = np.rot90(focal_stack, num_rot, (-3, -2))
            for _ in range(num_rot):
                person_position *= np.array([1, -1])
                person_position = person_position[[1, 0]]
            ground_truth = np.rot90(ground_truth, num_rot, (-3, -2))
            
        if self.normalize:
            focal_stack = focal_stack / np.float32(256)
            ground_truth = ground_truth / np.float32(256)
            
        person_position = ((-person_position/16 + 1) * 256).astype(int)  # may or may not be the correct formula

        return toTensor(focal_stack), toTensor(ground_truth), person_position
    