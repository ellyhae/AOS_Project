from pathlib import Path
from utils import calculate_psnr_tensor, calculate_ssim_tensor
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import os
import cv2
import json
from glob import glob
from tqdm import tqdm

import numpy as np
import torch.nn.functional as F

from swin2sr import Swin2SR as Swin
from dataset import PositionDataset

def load_model(path: str):
    model = Swin(img_size=512,
                 in_chans=1,
                 window_size=8,
                 depths=[2, 2, 2, 2],
                 num_heads=[4, 4, 4, 4],
                 embed_dim=32,
                 mlp_ratio=4,
                 img_range=1.,
                 ape=True,
                 use_checkpoint=True).cuda()
    
    existing_model_state = Path(path)
    if not existing_model_state.exists():
        print(f"Model state not found at {existing_model_state.absolute()}")
        exit(1)
    
    print(f"Using the existing model state from {existing_model_state.absolute()}")
    model.load_state_dict(torch.load(existing_model_state))

    model.eval()

    return model

def tens2img(tensor: torch.Tensor):
    '''Move the axes of a 3D Tensor such that it can be plotted as an image'''
    return np.moveaxis(tensor.detach().cpu().numpy(), 0,-1)

def load_tiff(path: str, focal_idx: list):
    ok, focal_stack = cv2.imreadmulti(path)
    if not ok:
        raise IOError(f'Failed to load index: {path}')
        
    focal_stack = np.stack(focal_stack)   # shape (num_focal_lengths, w, h)
    focal_stack = focal_stack[focal_idx]
    #focal_stack = np.moveaxis(focal_stack, 0, -1)
    focal_stack = torch.from_numpy(focal_stack).contiguous().div(256)
    return focal_stack

def load_image(path: str, focal_idx: list):
    _, ext = os.path.splitext(path)

    if ext.lower() in ['.tiff', '.tif']:
        ok, focal_stack = cv2.imreadmulti(path, flags=cv2.IMREAD_ANYDEPTH)
        if not ok:
            raise IOError(f'Failed to load TIFF: {path}')
        focal_stack = np.stack(focal_stack)
    elif ext.lower() in ['.png', '.jpg', '.jpeg']:
        img = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
        if img is None:
            raise IOError(f'Failed to load image: {path}')
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=-1)
        focal_stack = np.array([img])
    else:
        raise ValueError(f'Unsupported file format: {ext}')

    # Select specified focal indices
    focal_stack = focal_stack[focal_idx]

    # Convert to PyTorch tensor and scale
    focal_stack = torch.from_numpy(focal_stack).float().div(256)  # Ensure float type for division

    # Ensure consistent dimension order for PyTorch (B, C, H, W)
    focal_stack = focal_stack.permute(0, 3, 1, 2)  # Change dimension order
    
    if ext.lower() in ['.png', '.jpg', '.jpeg']:
        focal_stack = focal_stack[0]
    
    if focal_stack.shape != [1, 255, 255]:
        focal_stack = focal_stack.unsqueeze(1)
        focal_stack = F.interpolate(focal_stack, size=(512, 512), mode='bilinear')
        focal_stack = focal_stack.squeeze(1)
    
    return focal_stack


@torch.no_grad()
@torch.autocast(device_type='cuda', dtype=torch.float16)
def self_ensemble(model: Swin, stack: torch.Tensor, second_pass: bool):
    get_rots = lambda x: [torch.rot90(x, i, (-2, -1)) for i in range(4)]
    undo_rots = lambda x: [torch.rot90(j, -i, (-2, -1)) for i, j in enumerate(x)]

    ### maybe also include flipped versions
    versions = torch.stack(get_rots(stack)).cuda()
    
    preds = model(versions)
    
    preds_2 = model(preds)

    no_ensemble_denoised = preds[0]
    no_ensemble_denoised_2 = preds_2[0]

    fixed_preds = torch.stack(undo_rots(preds))
    denoised = fixed_preds.median(0).values
    
    fixed_preds_2 = torch.stack(undo_rots(preds_2))
    denoised_2 = fixed_preds_2.median(0).values
    
    return no_ensemble_denoised, denoised, no_ensemble_denoised_2, denoised_2

def postprocess(stack: torch.Tensor):
    #med = stack.median() #.values()
    #mask = stack < med
    #scale = 100
    #stack[mask] = stack[mask].mul(scale).ceil().div(scale)
    #stack[~mask] = stack[~mask].mul(scale).floor().div(scale)
    #stack = stack.round(decimals=2)
    return stack.mul(256).clip(0, 255).int()

def calculate_metrics(denoised, ground_truth):
    dn = denoised[None,:].float()
    gt = ground_truth[None,:].float()
    loss = nn.functional.l1_loss(dn, gt).item()
    psnr = calculate_psnr_tensor(gt, dn, 255.)
    ssim = calculate_ssim_tensor(gt, dn, 255.)
    return [loss, psnr, ssim]

model = load_model('tmp/model_3199.pth')
second_pass = True
# input_image_path = os.path.join('real_integrals', 'focal_stack')
file_ending = 'png'

if not os.path.exists('real_integrals_pred'):
    os.makedirs('real_integrals_pred')
    
if not os.path.exists(os.path.join('real_integrals_pred', 'focal_stack')):
    os.makedirs(os.path.join('real_integrals_pred', 'focal_stack'))

def make_predictions(input_image_path='real_integrals'):
    
    focal_idx = [0]
    if os.path.isdir(input_image_path):
        print('Detected folder as input path, loading all files')
        image_files = glob(os.path.join(input_image_path, f'*.{file_ending}'))
    else:
        image_files = [input_image_path]

    ## make predictions
    metrics, ensemble_metrics = [], []
    for i, f in enumerate(tqdm(image_files, desc='Calculating output(s)')):
        stack = load_image(f, focal_idx)
        # print(np.shape(stack))

        no_ensemble_denoised, denoised, no_ensemble_denoised_2, denoised_2 = map(
            postprocess, self_ensemble(model, stack, second_pass))
        print(f)

        # plot and print every iteration
        if i%1 == 0:
            fig, axes = plt.subplots(1, 5, figsize=(18,8), sharey=True, sharex=True)

            a1 = axes[0].imshow(tens2img(stack), cmap='gray', vmin=0, vmax=1)
            #plt.colorbar(a1, shrink=0.5)
            a2 = axes[1].imshow(tens2img(no_ensemble_denoised), cmap='gray', vmin=0, vmax=255)
            #plt.colorbar(a2, shrink=0.5)
            a3 = axes[2].imshow(tens2img(denoised), cmap='gray', vmin=0, vmax=255)
            #a3 = axes[2].imshow(tens2img(g), cmap='gray', vmin=0, vmax=1)
            #plt.colorbar(a3, shrink=0.5)
            a4 = axes[3].imshow(tens2img(no_ensemble_denoised_2), cmap='gray', vmin=0, vmax=255)
            #plt.colorbar(a2, shrink=0.5)
            a5 = axes[4].imshow(tens2img(denoised_2), cmap='gray', vmin=0, vmax=255)

            axes[0].set_title('Input')
            axes[1].set_title('Denoised')
            axes[2].set_title('Self Ensemble Denoised')

            axes[3].set_title('Denoised Multi-Pass\nExt. Train')

            # axes[4].set_title('Self Ensemble Denoised\nMulti-Pass Ext. Train')


            a5 = axes[4].imshow(tens2img(denoised_2), cmap='gray', vmin=0, vmax=255)
            axes[4].set_title('Self Ensemble Denoised\nMulti-Pass Ext. Train')

            # Now you can safely use a5 since it's defined
            image_data = a5.get_array().data

            # Splitting the path 'f' and replacing the first component
            path_components = f.split(os.path.sep)
            path_components[0] = 'real_integrals_pred'
            new_path = os.path.join(*path_components)

            # Ensure the directory exists before saving the image
            save_dir = os.path.dirname(new_path)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            # Save the image data to the new path
            plt.imsave(new_path, image_data, cmap='gray', vmin=0, vmax=255)

            # plt.show()


if __name__ == "__main__":
    make_predictions(input_image_path='real_integrals')
    make_predictions(input_image_path=os.path.join('real_integrals', 'focal_stack'))