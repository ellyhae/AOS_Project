from pathlib import Path
from utils import calculate_psnr_tensor, calculate_ssim_tensor
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import json

from swin2sr import Swin2SR as Swin
from dataset import PositionDataset

def validate_dataset(validation_dl, model, loss_fn, double_pass=False):
    val_loss, val_psnr, val_ssim = 0.0, 0.0, 0.0
    with torch.no_grad():
        for stack, gt, pos in validation_dl:
            stack = stack.cuda(non_blocking=True)  # move data to the gpu. non_blocking=True in combination with pin_memory in the dataloader leads to async data transfer, which can speed it up
            gt = gt.cuda(non_blocking=True)

            with torch.autocast(device_type='cuda', dtype=torch.float16):
                denoised = model(stack)   # feed the input to the network
                
                if double_pass:
                    denoised = model(denoised)
                
                loss = loss_fn(denoised, gt, pos)

            val_loss += loss.item()
            val_psnr += calculate_psnr_tensor(gt, denoised)
            val_ssim += calculate_ssim_tensor(gt, denoised)

    val_loss /= len(validation_dl)
    val_psnr /= len(validation_dl)
    val_ssim /= len(validation_dl)
    
    #stack, gt, pos = next(iter(validation_dl))
    #stack = stack.cuda(non_blocking=True)  # move data to the gpu. non_blocking=True in combination with pin_memory in the dataloader leads to async data transfer, which can speed it up
    #gt = gt.cuda(non_blocking=True)
    #with torch.autocast(device_type='cuda', dtype=torch.float16):
    #    denoised = model(stack)
    
    return val_loss, val_psnr, val_ssim, stack, gt, denoised
