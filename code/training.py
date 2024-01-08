import json
from pathlib import Path
from validate import validate_dataset
from utils import end_timer_and_print, start_timer
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from model import FusionDenoiser
from dataset import FocalDataset
from tqdm import trange

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

def get_batch(data_iterator, train_dl):
    '''convenience function to infinitely sample from the dataset'''
    res = next(data_iterator, None)
    if res is None:
        data_iterator = iter(train_dl)
        res = next(data_iterator)
    stack, gt = res
    return data_iterator, stack.cuda(non_blocking=True), gt.cuda(non_blocking=True)

def main():
    torch.backends.cudnn.benchmark = True

    torch.manual_seed(43)
    np.random.seed(43)

    train_ds = FocalDataset(path='./integrals', used_focal_lengths_idx=[0, 1, 3, 5, 7, 8, 9], augment=True)

    batch_size = 1
    train_dl = DataLoader(train_ds,
                          batch_size=batch_size,
                          shuffle=True,     # only for training set!
                          num_workers=2,    # could be useful to prefetch data while the GPU is working
                          pin_memory=True)  # should speed up CPU to GPU data transfer
    ds_iter, stack, gt = get_batch(iter(train_dl), train_dl)

    val_ds = FocalDataset(path='./integrals_validation', used_focal_lengths_idx=[0, 1, 3, 5, 7, 8, 9], augment=True)
    val_dl = DataLoader(val_ds,
                        batch_size=batch_size,
                        shuffle=False,     # only for training set!
                        num_workers=2,    # could be useful to prefetch data while the GPU is working
                        pin_memory=True)  # should speed up CPU to GPU data transfer

    num_updates = 430          # the number of gradient updates we want to do
    samples_per_update = 24    # number of samples we want to use for a single update (higher means better gradient estimation, but also higher computational cost)
    accumulate_batches = samples_per_update // batch_size    # the number of batches we need to compute to do a single gradient update
    num_batches = num_updates * accumulate_batches           # the number of batches we need to reach our goal of updates and samples_per_update
    checkpoint_every = 2  # e.g. run validation set, save model, print some logs every n updates

    model = FusionDenoiser(use_checkpoint=True).cuda()
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
    loss_fn = CharbonnierLoss().cuda()

    scaler = torch.cuda.amp.GradScaler()  # used to scale the loss/gradients to prevent float underflow

    temp_losses = []
    train_losses = []
    best_val_loss = None
    stats = dict()
    
    model_path = Path('tmp/model.pth')
    model_path.parent.mkdir(exist_ok=True)

    start_timer()

    loop = trange(num_batches)
    for batch_count in loop:
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            _, denoised = model(stack)   # feed the input to the network
        
            loss = loss_fn(denoised, gt)
            acc_loss = loss / accumulate_batches
            
        ds_iter, stack, gt = get_batch(ds_iter, train_dl)  # load batch during logging and back-propagation
        
        temp_losses.append(loss.item())
        
        scaler.scale(acc_loss).backward()  # scale loss and calculate gradients
        
        if (batch_count + 1) % accumulate_batches == 0:
            scaler.step(optimizer)         # update the weights
            scaler.update()                # update scaling parameters
            optimizer.zero_grad()          # reset gradients
            train_losses.append(np.mean(temp_losses))
            temp_losses = []
            
            if ((batch_count + 1) // accumulate_batches) % checkpoint_every == 0:
                # every n-th update
                val_loss, val_psnr, val_ssim, _, _, _, _ = validate_dataset(val_dl, model, loss_fn)
                model.train()

                print(f'\nBatch {batch_count+1} Training Loss: {train_losses[-1]:.4f} Validation Loss/PSNR/SSIM: {val_loss:.4f} {val_psnr:.4f} {val_ssim:.4f}')
                stats[batch_count + 1] = [train_losses[-1], val_loss, val_psnr, val_ssim]
                Path('tmp/stats').write_text(json.dumps(stats))

                if best_val_loss == None or val_loss <= best_val_loss:
                    print(f'Best validation loss found -> saving the model to {model_path.absolute}')
                    torch.save(model.state_dict(), model_path)
                
                lr_scheduler.step(val_loss)
                loop.set_postfix(dict(train_loss=train_losses[-1], val_loss=val_loss, val_psnr=val_psnr, val_ssim=val_ssim))
    
    torch.save(model.state_dict(), 'tmp/last_model.pth')

    end_timer_and_print(f"Finished training for {num_batches} batches with batch size {batch_size}")


if __name__ == "__main__":
    main()