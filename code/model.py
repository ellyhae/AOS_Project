import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

from ifcnn import IFCNN

def _rgb_to_L(tensor):
    # adapted from https://github.com/python-pillow/Pillow/blob/66c244af3233b1cc6cc2c424e9714420aca109ad/src/libImaging/Convert.c#L226
    # adapted to accept float rgb in the range [0, 1]
    # return (tensor[..., 0, :, :] * 19595 + tensor[..., 1, :, :] * 38470 + tensor[..., 2, :, :] * 7471 + 2**7)[...,None, :, :] / 2**8
    return (tensor[..., 0, :, :] * 19595 + tensor[..., 1, :, :] * 38470 + tensor[..., 2, :, :] * 7471)[...,None, :, :] / 2**8

class FusionDenoiser(nn.Module):
    r""" FusionDenoiser
        A PyTorch Module combining IFCNN for image fusion and SwinIR for image restoration / denoising.
        For now restricted to gray-scale images

    Args:
        fuse_scheme (int): IFCNN element-wise fusion scheme: 0=MAX, 1=SUM, 2=MEAN. Default 0
        img_size (int | tuple(int)): Input image size. Default 64
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        swinir_grayscale (bool): Whether to convert fusion output to grayscale or pass on 'rgb'
        pretrained (bool): Whether to load pretrained weights for IFCNN and SwinIR
    """
    
    def __init__(self, fuse_scheme=0, img_size=512, swin_version='V1', window_size=8, use_checkpoint=False,
                 depths=[6]*6, num_heads=[6]*6, embed_dim=180, mlp_ratio=2,
        super(FusionDenoiser, self).__init__()
        
        self.swinir_grayscale = swinir_grayscale
        
        self.fusion = IFCNN(fuse_scheme=fuse_scheme)
        
        self.use_checkpoint = use_checkpoint
        
        self.swin_version = swin_version
        if self.swin_version == 'V1':
            from swinir import SwinIR as Swin
        elif self.swin_version == 'V2':
            from swin2sr import Swin2SR as Swin
        
        # static arguments taken from https://github.com/cszn/KAIR/blob/master/options/swinir/train_swinir_denoising_gray.json lines 42-57
        self.denoiser = Swin(img_size=img_size, window_size=window_size, use_checkpoint=use_checkpoint,
                             depths=depths, num_heads=num_heads, embed_dim=embed_dim, mlp_ratio=mlp_ratio,
                             upscale=1, in_chans=1 if self.swinir_grayscale else 3, img_range=255.0, upsampler=None, resi_connection="1conv")
        
            
        
    def _grayscale(self, tensor):
        if self.pretrained:
            # de-norm using the means and stds (probably) used during training
            # not sure if this is even needed
            tensor = tensor.mul(self.fusion_std).add(self.fusion_mean)  # .clamp(0, 1)  probably not needed
        return _rgb_to_L(tensor)
        
    def forward(self, x):
        if self.use_checkpoint:
            fused = checkpoint.checkpoint(self.fusion, x, use_reentrant=False)
        else:
            fused = self.fusion(x)
            
        if self.swinir_grayscale:
            fused = self._grayscale(fused)
        
        return fused, self.denoiser(fused)