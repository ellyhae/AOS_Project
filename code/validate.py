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

# from https://github.com/styler00dollar/pytorch-loss-functions/blob/main/vic/loss.py#L27
class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""
    def __init__(self, eps=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        b, c, h, w = y.size()
        loss = torch.sum(torch.sqrt((x - y).pow(2) + self.eps**2))
        return loss/(c*b*h*w)


def tens2img(tensor):
    '''Move the axes of a 3D Tensor such that it can be plotted as an image'''
    return np.moveaxis(tensor.detach().cpu().numpy(), 0,-1)


def validate_dataset(validation_dl, model, loss_fn):
    val_loss, val_psnr, val_ssim = 0.0, 0.0, 0.0
    with torch.no_grad():
        for stack, gt, pos in validation_dl:
            stack = stack.cuda(non_blocking=True)  # move data to the gpu. non_blocking=True in combination with pin_memory in the dataloader leads to async data transfer, which can speed it up
            gt = gt.cuda(non_blocking=True)

            with torch.autocast(device_type='cuda', dtype=torch.float16):
                denoised = model(stack)   # feed the input to the network
            
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


def main(): # for testing the model
    torch.backends.cudnn.benchmark = True

    torch.manual_seed(43)
    np.random.seed(43)

    focal_idx = [0]
    ds = PositionDataset(path='C:\\Users\\chris\\Documents\\JKU\\ComputerVision\\val', used_focal_lengths_idx=focal_idx, augment=False)

    batch_size = 3
    validation_dl = DataLoader(ds,
                               batch_size=batch_size,
                               shuffle=False,   # False for validation set
                               num_workers=4,    # could be useful to prefetch data while the GPU is working
                               pin_memory=True)  # should speed up CPU to GPU data transfer

    
    model = Swin(img_size=512,
                 in_chans=len(focal_idx),
                 window_size=8,
                 depths=[2, 2, 2, 2],
                 num_heads=[4, 4, 4, 4],
                 embed_dim=32,
                 mlp_ratio=4,
                 img_range=1.,
                 ape=True,
                 use_checkpoint=True).cuda()
    

    existing_model_state = Path('tmp/last_model.pth')
    if not existing_model_state.exists():
        print(f"Model state not found at {existing_model_state.absolute}")
        exit(1)
    
    print(f"Using the existing model state from {existing_model_state.absolute}")
    model.load_state_dict(torch.load(existing_model_state))

    model.eval()
    #loss_fn = CharbonnierLoss().cuda()
    from loss import PositionEnhancedLoss
    loss_fn = PositionEnhancedLoss().cuda()
    #loss_fn = torch.nn.L1Loss().cuda()

    stack, gt, pos = next(iter(validation_dl))
    stack = stack.cuda(non_blocking=True)  # move data to the gpu. non_blocking=True in combination with pin_memory in the dataloader leads to async data transfer, which can speed it up
    gt = gt.cuda(non_blocking=True)
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        denoised = model(stack)

    #val_loss, val_psnr, val_ssim, stack, gt, denoised = validate_dataset(validation_dl, model, loss_fn)

    stats: dict = json.loads(Path('tmp/stats.json').read_text())
    batches = [int(key) for key in stats.keys()]
    train_losses = [value[0] for _, value in stats.items()]
    val_losses = [value[1] for _, value in stats.items()]
    psnr = [value[2] for _, value in stats.items()]
    ssim = [value[3] for _, value in stats.items()] 

    print(f"Validation Loss: {val_losses[-1]}, Validation PSNR: {psnr[-1]}, Validation SSIM: {ssim[-1]}")

    # draw chart for losses
    
    plt.plot(batches, train_losses, label='Training')
    plt.plot(batches, val_losses, label='Validation')
    #plt.plot(batches, ssim, label='SSIM')
    plt.xlabel('n samples')
    plt.ylabel('Loss')
    #plt.ylim(0,1)
    plt.legend()
    plt.show()

    plt.plot(batches, psnr, label='PSNR')
    #plt.plot(batches, ssim, label='SSIM')
    plt.xlabel('n samples')
    plt.ylabel('PSNR')
    #plt.ylim(0,1)
    plt.legend()
    plt.show()

    fig, axess = plt.subplots(batch_size, 3, figsize=(18,8), sharey=True, sharex=True)

    for axes, s, d, g in zip(axess, stack, denoised, gt):

        a1 = axes[0].imshow(tens2img(s[[0]]), cmap='gray', vmin=0, vmax=1)
        #plt.colorbar(a1, shrink=0.5)
        a2 = axes[1].imshow(tens2img(d.clamp(0,1)), cmap='gray', vmin=0, vmax=1)
        #plt.colorbar(a2, shrink=0.5)
        a3 = axes[2].imshow(tens2img(g), cmap='gray', vmin=0, vmax=1)
        #plt.colorbar(a3, shrink=0.5)

    axess[0][0].set_title('Input')
    axess[0][1].set_title('Denoised')
    axess[0][2].set_title('Ground Truth')
    plt.show()


if __name__ == "__main__":
    main()