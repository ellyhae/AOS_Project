import json
from pathlib import Path
from validate import validate_dataset
from utils import end_timer_and_print, start_timer
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from swin2sr import Swin2SR as Swin
from dataset import PositionDataset
from loss import PositionEnhancedLoss
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
    stack, gt, pos = res
    return data_iterator, stack.cuda(non_blocking=True), gt.cuda(non_blocking=True), pos.cuda(non_blocking=True)

def main():
    torch.backends.cudnn.benchmark = True

    torch.manual_seed(43)
    np.random.seed(43)
    
    focal_idx = [0]
    train_ds = PositionDataset(path='train', used_focal_lengths_idx=focal_idx, augment=True)

    batch_size = 2
    train_dl = DataLoader(train_ds,
                          batch_size=batch_size,
                          shuffle=True,     # only for training set!
                          num_workers=4,    # could be useful to prefetch data while the GPU is working
                          pin_memory=True)  # should speed up CPU to GPU data transfer
    ds_iter, stack, gt, pos = get_batch(iter(train_dl), train_dl)

    val_ds = PositionDataset(path='val', used_focal_lengths_idx=focal_idx, augment=False)
    val_dl = DataLoader(val_ds,
                        batch_size=batch_size,
                        shuffle=False,     # only for training set!
                        num_workers=4,    # could be useful to prefetch data while the GPU is working
                        pin_memory=True)  # should speed up CPU to GPU data transfer




    ################ Mostly change parameters here (plus the dataset paths above) ##################

    load_last = False    # True: resume training, False: start new training and overwrite old files
    num_updates = 5000          # the number of gradient updates we want to do (would advise this to be a multiple of checkpoint_every)
    samples_per_update = 8    # number of samples we want to use for a single update (higher means better gradient estimation, but also higher computational cost)
    accumulate_batches = samples_per_update // batch_size    # the number of batches we need to compute to do a single gradient update
    num_batches = num_updates * accumulate_batches           # the number of batches we need to reach our goal of updates and samples_per_update
    checkpoint_every = 500  # e.g. run validation set, save model, print some logs every n updates  /  keep pretty high, validation "wastes" time, but also not too high, as the lr_scheduler needs the result as input to do anything

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
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, verbose=True)

    loss_fn = PositionEnhancedLoss(length=96, factor=.5).cuda()

    ####################################################################





    scaler = torch.cuda.amp.GradScaler()  # used to scale the loss/gradients to prevent float underflow

    temp_losses = []
    train_losses = []
    best_val_loss = None
    stats = dict()
    
    model_path = Path('tmp/model.pth')
    model_path.parent.mkdir(exist_ok=True)
    
    last_batch_num = 0
    
    if load_last:
        model.load_state_dict(torch.load('tmp/model.pth'))
        optimizer.load_state_dict(torch.load('tmp/last_optimizer.pth'))
        lr_scheduler.load_state_dict(torch.load('tmp/last_scheduler.pth'))
        scaler.load_state_dict(torch.load('tmp/last_scaler.pth'))
        for group, lr in zip(optimizer.param_groups, lr_scheduler._last_lr):
            print('Using last Learning Rate:', lr)
            group['lr'] = lr
        
        stats: dict = json.loads(Path('tmp/stats.json').read_text())
        last_batch_num = max(map(int, stats.keys()))
        train_losses, val_losses, _, _ = zip(*stats.values())
        train_losses = list(train_losses)
        print('Last training losses',train_losses)
        best_val_loss = min(val_losses)
        print('Best val loss:', best_val_loss)
        del val_losses
        

    start_timer()

    loop = trange(num_batches)
    for batch_count in loop:
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            denoised = model(stack)   # feed the input to the network
        
            loss = loss_fn(denoised, gt, pos)
            acc_loss = loss / accumulate_batches
            
        ds_iter, stack, gt, pos = get_batch(ds_iter, train_dl)  # load batch during logging and back-propagation
        
        temp_losses.append(loss.item())
        
        scaler.scale(acc_loss).backward()  # scale loss and calculate gradients
        
        if (batch_count + 1) % accumulate_batches == 0:
            scaler.step(optimizer)         # update the weights
            scaler.update()                # update scaling parameters
            optimizer.zero_grad()          # reset gradients
            
            if ((batch_count + 1) // accumulate_batches) % checkpoint_every == 0:
                # every n-th update
                train_losses.append(np.mean(temp_losses[-20*accumulate_batches:]))
                temp_losses = []

                val_loss, val_psnr, val_ssim, _, _, _ = validate_dataset(val_dl, model, loss_fn)
                model.train()

                print(f'\nBatch {batch_count+1+last_batch_num} Training Loss: {train_losses[-1]:.4f} Validation Loss/PSNR/SSIM: {val_loss:.4f} {val_psnr:.4f} {val_ssim:.4f}')
                stats[batch_count + 1+last_batch_num] = [train_losses[-1], val_loss, val_psnr, val_ssim]
                Path('tmp/stats.json').write_text(json.dumps(stats))

                if best_val_loss == None or val_loss <= best_val_loss:
                    best_val_loss = val_loss
                    print(f'Best validation loss found -> saving the model to {model_path.absolute()}')
                    torch.save(model.state_dict(), model_path)
                    best_val_loss = val_loss
                
                lr_scheduler.step(val_loss)
                loop.set_postfix(dict(train_loss=train_losses[-1], val_loss=val_loss, val_psnr=val_psnr, val_ssim=val_ssim))
    
    torch.save(model.state_dict(), 'tmp/last_model.pth')
    torch.save(optimizer.state_dict(), 'tmp/last_optimizer.pth')
    torch.save(lr_scheduler.state_dict(), 'tmp/last_scheduler.pth')
    torch.save(scaler.state_dict(), 'tmp/last_scaler.pth')

    end_timer_and_print(f"Finished training for {num_batches} batches with batch size {batch_size}")


if __name__ == "__main__":
    main()