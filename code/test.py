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

@torch.no_grad()
@torch.autocast(device_type='cuda', dtype=torch.float16)
def self_ensemble(model: Swin, stack: torch.Tensor):
    get_rots = lambda x: [torch.rot90(x, i, (-2, -1)) for i in range(4)]
    undo_rots = lambda x: [torch.rot90(j, -i, (-2, -1)) for i, j in enumerate(x)]

    ### maybe also include flipped versions
    versions = torch.stack(get_rots(stack)).cuda()

    preds = model(versions)

    no_ensemble_denoised = preds[0]

    fixed_preds = torch.stack(undo_rots(preds))
    denoised = fixed_preds.median(0).values
    return no_ensemble_denoised, denoised

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

def main():
    model = load_model('tmp/model_72000.pth')

    focal_idx = [0]
    input_image_path = 'test'#\\0_45_0_2_integral.tiff'
    if os.path.isdir(input_image_path):
        print('Detected folder as input path, loading all files')
        image_files = glob(os.path.join(input_image_path, '*_integral.tiff'))
    else:
        image_files = [input_image_path]

    
    ## make predictions
    metrics, ensemble_metrics = [], []
    for f in tqdm(image_files, desc='Calculating output(s)'):
        stack = load_tiff(f, focal_idx)
        no_ensemble_denoised, denoised = map(postprocess, self_ensemble(model, stack))

        gt_path = f.removesuffix('integral.tiff') + 'gt.png'
        gt = os.path.exists(gt_path)
        if gt:
            ground_truth = torch.from_numpy(np.moveaxis(cv2.imread(gt_path)[...,[0]], -1, 0)).int().cuda()
            metrics.append(calculate_metrics(no_ensemble_denoised, ground_truth))
            ensemble_metrics.append(calculate_metrics(denoised, ground_truth))
    loss, psnr, ssim = np.mean(metrics, 0)
    ensemble_loss, ensemble_psnr, ensemble_ssim = np.mean(ensemble_metrics, 0)
    
    if gt:
        print('\n                   L1 Loss / PSNR / SSIM:')
        print(f'Simple denoised:   {loss:.3f} / {psnr:.3f} / {ssim:.3f}')
        print(f'Ensemble Denoised: {ensemble_loss:.3f} / {ensemble_psnr:.3f} / {ensemble_ssim:.3f}')


    fig, axes = plt.subplots(1, 4 if gt else 3, figsize=(18,8), sharey=True, sharex=True)

    a1 = axes[0].imshow(tens2img(stack), cmap='gray', vmin=0, vmax=1)
    #plt.colorbar(a1, shrink=0.5)
    a2 = axes[1].imshow(tens2img(no_ensemble_denoised), cmap='gray', vmin=0, vmax=255)
    #plt.colorbar(a2, shrink=0.5)
    a3 = axes[2].imshow(tens2img(denoised), cmap='gray', vmin=0, vmax=255)
    #a3 = axes[2].imshow(tens2img(g), cmap='gray', vmin=0, vmax=1)
    #plt.colorbar(a3, shrink=0.5)

    axes[0].set_title('Input')
    axes[1].set_title('Denoised')
    axes[2].set_title('Self Ensemble Denoised')

    if gt:
        a4 = axes[3].imshow(tens2img(ground_truth), cmap='gray', vmin=0, vmax=255)
        axes[3].set_title('Ground Truth')

    plt.show()

if __name__ == "__main__":
    main()