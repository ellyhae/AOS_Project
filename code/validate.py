from pathlib import Path
from utils import calculate_psnr_tensor, calculate_ssim_tensor
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import json

from model import FusionDenoiser
from dataset import FocalDataset

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
        for stack, gt in validation_dl:
            stack = stack.cuda(non_blocking=True)  # move data to the gpu. non_blocking=True in combination with pin_memory in the dataloader leads to async data transfer, which can speed it up
            gt = gt.cuda(non_blocking=True)

            with torch.autocast(device_type='cuda', dtype=torch.float16):
                fused, denoised = model(stack)   # feed the input to the network
            
                loss = loss_fn(denoised, gt)

            val_loss += loss.item()
            val_psnr += calculate_psnr_tensor(gt, denoised)
            val_ssim += calculate_ssim_tensor(gt, denoised)

    val_loss /= len(validation_dl)
    val_psnr /= len(validation_dl)
    val_ssim /= len(validation_dl)
    
    return val_loss, val_psnr, val_ssim, stack, gt, fused, denoised


def main(): # for testing the model
    torch.backends.cudnn.benchmark = True

    torch.manual_seed(43)
    np.random.seed(43)

    ds = FocalDataset(path='./integrals_validation', used_focal_lengths_idx=[0, 1, 3, 5, 7, 8, 9], augment=True)

    batch_size = 2
    validation_dl = DataLoader(ds,
                               batch_size=batch_size,
                               shuffle=False,   # False for validation set
                               num_workers=0,    # could be useful to prefetch data while the GPU is working
                               pin_memory=True)  # should speed up CPU to GPU data transfer

    model = FusionDenoiser(use_checkpoint=True).cuda()

    existing_model_state = Path('tmp/model.pth')
    if not existing_model_state.exists():
        print(f"Model state not found at {existing_model_state.absolute}")
        exit(1)
    
    print(f"Using the existing model state from {existing_model_state.absolute}")
    model.load_state_dict(torch.load(existing_model_state))

    model.eval()
    loss_fn = CharbonnierLoss().cuda()

    val_loss, val_psnr, val_ssim, stack, gt, fused, denoised = validate_dataset(validation_dl, model, loss_fn)

    print(f"Validation Loss: {val_loss}, Validation PSNR: {val_psnr}, Validation SSIM: {val_ssim}")

    # draw chart for losses
    # stats: dict = json.loads(Path('tmp/stats').read_text())
    # batches = [int(key) for key in stats.keys()]
    # psnr = [value[2] for _, value in stats.items()]
    # ssim = [value[3] for _, value in stats.items()]

    # plt.plot(batches, psnr, label='PSNR')
    # plt.plot(batches, ssim, label='SSIM')
    # plt.xlabel('n samples')
    # plt.ylabel('SSIM')
    # plt.ylim(0,1)
    # plt.legend()
    # plt.show()

    plot_index = 1

    fig, axes = plt.subplots(1, 4, figsize=(18,8), sharey=True)

    a1 = axes[0].imshow(tens2img(stack[plot_index,0]), cmap='gray', vmin=0, vmax=1)
    plt.colorbar(a1, shrink=0.5)
    a2 = axes[1].imshow(tens2img(fused[plot_index]), cmap='gray')
    plt.colorbar(a2, shrink=0.5)
    a3 = axes[2].imshow(tens2img(denoised[plot_index]), cmap='gray')
    plt.colorbar(a3, shrink=0.5)
    a4 = axes[3].imshow(tens2img(gt[plot_index]), cmap='gray', vmin=0, vmax=1)
    plt.colorbar(a4, shrink=0.5)

    axes[0].set_title('One of the inputs')
    axes[1].set_title('Fused')
    axes[2].set_title('Denoised')
    axes[3].set_title('Ground Truth')
    plt.show()


if __name__ == "__main__":
    main()