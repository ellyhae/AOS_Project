from pathlib import Path
from utils import calculate_psnr_tensor, calculate_ssim_tensor
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from model import FusionDenoiser
from dataset import FocalDataset


def validate_dataset(validation_dl, model, loss_fn):
    val_loss, val_psnr, val_ssim = 0.0, 0.0, 0.0
    with torch.no_grad():
        for stack, gt in validation_dl:
            stack = stack.cuda(non_blocking=True)  # move data to the gpu. non_blocking=True in combination with pin_memory in the dataloader leads to async data transfer, which can speed it up
            gt = gt.cuda(non_blocking=True)

            with torch.autocast(device_type='cuda', dtype=torch.float16):
                _, denoised = model(stack)   # feed the input to the network
            
                loss = loss_fn(denoised, gt)

            val_loss += loss.item()
            val_psnr += calculate_psnr_tensor(gt, denoised)
            val_ssim += calculate_ssim_tensor(gt, denoised)

    val_loss /= len(validation_dl)
    val_psnr /= len(validation_dl)
    val_ssim /= len(validation_dl)
    
    return val_loss, val_psnr, val_ssim


def main(): # for testing the model
    torch.backends.cudnn.benchmark = True

    torch.manual_seed(43)
    np.random.seed(43)

    ds = FocalDataset(path='./integrals_sample', used_focal_lengths_idx=list(range(8)), augment=True)

    batch_size = 4
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
    loss_fn = torch.nn.L1Loss().cuda()

    val_loss, val_psnr, val_ssim = validate_dataset(validation_dl, model, loss_fn)

    print(f"Validation Loss: {val_loss}, Validation PSNR: {val_psnr}, Validation SSIM: {val_ssim}")


if __name__ == "__main__":
    main()