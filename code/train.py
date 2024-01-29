import json
from argparse import ArgumentParser, BooleanOptionalAction
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import trange

from dataset import PositionDataset
from loss import PositionEnhancedLoss
from swin2sr import Swin2SR as Swin
from utils import end_timer_and_print, start_timer
from validate_2 import validate_dataset


def get_batch(data_iterator, train_dl):
    """convenience function to infinitely sample from the dataset"""
    res = next(data_iterator, None)
    if res is None:
        data_iterator = iter(train_dl)
        res = next(data_iterator)
    stack, gt, pos = res
    return data_iterator, stack.cuda(non_blocking=True), gt.cuda(non_blocking=True), pos.cuda(non_blocking=True)


def main(train_path, val_path, model_path, samples_per_update, checkpoint_every, multi_pass):
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(43)
    np.random.seed(43)

    tmp_path = Path('tmp')
    tmp_path.mkdir(exist_ok=True)

    focal_idx = [0]  # the focal length used, 0 means ground level
    batch_size = 2  # the number of samples passed through the model at once
    load_last = False  # True: resume training, False: start new training and overwrite old files
    num_updates = 5000  # the number of gradient updates we want to do (would advise this to be a multiple of checkpoint_every)
    accumulate_batches = samples_per_update // batch_size  # the number of batches we need to compute to do a single gradient update
    num_batches = num_updates * accumulate_batches  # the number of batches we need to reach our goal of updates and samples_per_update

    train_ds = PositionDataset(path=train_path, used_focal_lengths_idx=focal_idx, augment=True)
    train_dl = DataLoader(train_ds,
                          batch_size=batch_size,
                          shuffle=True,  # only for training set!
                          num_workers=4,  # could be useful to prefetch data while the GPU is working
                          pin_memory=True)  # should speed up CPU to GPU data transfer
    ds_iter, stack, gt, pos = get_batch(iter(train_dl), train_dl)

    val_ds = PositionDataset(path=val_path, used_focal_lengths_idx=focal_idx, augment=False)
    val_dl = DataLoader(val_ds,
                        batch_size=batch_size,
                        shuffle=False,  # only for training set!
                        num_workers=4,  # could be useful to prefetch data while the GPU is working
                        pin_memory=True)  # should speed up CPU to GPU data transfer

    model = Swin(img_size=512,
                 in_chans=len(focal_idx),
                 window_size=8,
                 depths=[2, 2, 2, 2],
                 num_heads=[4, 4, 4, 4],
                 embed_dim=32,
                 mlp_ratio=4,
                 img_range=1.0,
                 ape=True,
                 use_checkpoint=True).cuda()
    if model_path:
        model.load_state_dict(torch.load(model_path))
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, verbose=True)
    loss_fn = PositionEnhancedLoss(length=96, factor=0.5).cuda()
    scaler = torch.cuda.amp.GradScaler()  # used to scale the loss/gradients to prevent float underflow

    tmp_losses = []
    train_losses = []
    best_val_loss = np.inf
    stats = dict()
    last_batch_num = 0

    if load_last:
        model.load_state_dict(torch.load('tmp/last_model.pth'))
        optimizer.load_state_dict(torch.load('tmp/last_optimizer.pth'))
        lr_scheduler.load_state_dict(torch.load('tmp/last_scheduler.pth'))
        scaler.load_state_dict(torch.load('tmp/last_scaler.pth'))
        for group, lr in zip(optimizer.param_groups, lr_scheduler._last_lr):
            print('Using last learning rate:', lr)
            group['lr'] = lr

        stats = json.loads(Path('tmp/stats.json').read_text())
        last_batch_num = max(map(int, stats.keys()))
        train_losses, val_losses, _, _ = zip(*stats.values())
        train_losses = list(train_losses)
        print('Last training losses', train_losses)
        best_val_loss = min(val_losses)
        print('Best validation loss:', best_val_loss)
        del val_losses

    start_timer()

    loop = trange(num_batches)
    single_pass = True
    save_idx = 0

    for batch_count in loop:
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            denoised = model(stack)
            if single_pass:
                loss = loss_fn(denoised, gt, pos)
            else:
                loss_fn(model(denoised), gt, pos)
            if multi_pass:
                single_pass = not single_pass  # toggle switch if multi-pass finetuning is enabled
            acc_loss = loss / accumulate_batches
            scaler.scale(acc_loss).backward(retain_graph=True)  # scale loss and calculate gradients

        ds_iter, stack, gt, pos = get_batch(ds_iter, train_dl)  # load batch during logging and back-propagation
        tmp_losses.append(loss.item())
        scaler.scale(acc_loss).backward()  # scale loss and calculate gradients

        if (batch_count + 1) % accumulate_batches == 0:
            scaler.step(optimizer)  # update the weights
            scaler.update()  # update scaling parameters
            optimizer.zero_grad()  # reset gradients

            if ((batch_count + 1) // accumulate_batches) % checkpoint_every == 0:  # every n-th update
                train_losses.append(np.mean(tmp_losses[-20 * accumulate_batches:]))
                tmp_losses = []

                val_loss, val_psnr, val_ssim, _, _, _ = validate_dataset(val_dl, model, loss_fn, double_pass=True)
                model.train()

                print(f'\nBatch {batch_count + 1 + last_batch_num}',
                      f'Training loss: {train_losses[-1]:.4f}',
                      f'Validation loss/PSNR/SSIM: {val_loss:.4f} {val_psnr:.4f} {val_ssim:.4f}')
                stats[batch_count + 1 + last_batch_num] = [train_losses[-1], val_loss, val_psnr, val_ssim]
                Path('tmp/stats.json').write_text(json.dumps(stats))

                curr_path = f'tmp/model_{save_idx}.pth'
                if val_loss < best_val_loss:
                    print(f'Best validation loss found -> saving the model to {curr_path}')
                    best_val_loss = val_loss

                torch.save(model.state_dict(), curr_path)

                lr_scheduler.step(val_loss)
                loop.set_postfix(dict(train_loss=train_losses[-1], val_loss=val_loss, val_psnr=val_psnr, val_ssim=val_ssim))

            save_idx += 1

    torch.save(model.state_dict(), 'tmp/last_model.pth')
    torch.save(optimizer.state_dict(), 'tmp/last_optimizer.pth')
    torch.save(lr_scheduler.state_dict(), 'tmp/last_scheduler.pth')
    torch.save(scaler.state_dict(), 'tmp/last_scaler.pth')

    end_timer_and_print(f'Finished training for {num_batches} batches with batch size {batch_size}')


if __name__ == "__main__":
    parser = ArgumentParser(description="Training interface for SwinSR")
    parser.add_argument("--train_path", type=str, default="train")
    parser.add_argument("--val_path", type=str, default="val")
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--samples_per_update", type=int, default=16)
    parser.add_argument("--checkpoint_every", type=int, default=200)
    parser.add_argument('--multi_pass', action=BooleanOptionalAction, default=False)
    args = parser.parse_args()

    main(args.train_path,
         args.val_path,
         args.model_path,
         args.samples_per_update,
         args.checkpoint_every,
         args.multi_pass)
