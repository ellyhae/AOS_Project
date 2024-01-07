# from https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html
import torch, time, gc
import numpy as np
from pytorch_msssim import ssim

def tens2img(tensor):
    '''Move the axes of a 3D Tensor such that it can be plotted as an image'''
    return np.moveaxis(tensor.detach().cpu().numpy(), 0,-1)

# Timing utilities
start_time = None

def start_timer():
    global start_time
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start_time = time.time()

def end_timer_and_print(local_msg):
    torch.cuda.synchronize()
    end_time = time.time()
    print("\n" + local_msg)
    print("Total execution time = {:.3f} sec".format(end_time - start_time))
    print("Max memory used by tensors = {:.3f} GB".format(torch.cuda.max_memory_allocated() / 1024**3))

def calculate_psnr_tensor(target, prediction):
    mse = torch.mean((target - prediction) ** 2)
    psnr = 20 * torch.log10(torch.tensor(255.0)) - 10 * torch.log10(mse)
    return psnr.item()  # Convert to Python scalar

def calculate_ssim_tensor(target, prediction):
    # ensure the input tensors are float32, as required by pytorch_msssim
    target = target.type(torch.float32)
    prediction = prediction.type(torch.float32)

    # the ssim function expects tensors in the shape of (batch, channel, height, width)
    # ensure your tensors are correctly shaped
    ssim_value = ssim(target, prediction, data_range=255, size_average=True)  # size_average to return the average of SSIM
    return ssim_value.item()  # make it python scalar
