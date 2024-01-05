import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

from ifcnn import IFCNN

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
    
    def __init__(self, fuse_scheme=0, img_size=512, swin_version='V2', window_size=8, use_checkpoint=False,
                 depths=[2, 2, 6, 2], num_heads=[6, 6, 6, 6], embed_dim=60, mlp_ratio=4,
                 use_rgb_to_L=True, swin_img_range=1., swin_in_chans=1, swin_patch_size=4):
        super(FusionDenoiser, self).__init__()
        
        self.fusion = IFCNN(fuse_scheme=fuse_scheme)
        
        self.use_checkpoint = use_checkpoint
        
        self.swin_version = swin_version
        if self.swin_version == 'V1':
            from swinir import SwinIR as Swin
        elif self.swin_version == 'V2':
            from swin2sr import Swin2SR as Swin
        
        # static arguments taken from https://huggingface.co/docs/transformers/v4.27.0/model_doc/swinv2
        self.denoiser = Swin(img_size=img_size, window_size=window_size, use_checkpoint=use_checkpoint,
                             depths=depths, num_heads=num_heads, embed_dim=embed_dim, mlp_ratio=mlp_ratio,
                             upscale=1, in_chans=swin_in_chans, img_range=swin_img_range, upsampler=None, resi_connection="1conv")
        
        self.use_rgb_to_L = use_rgb_to_L
        if self.use_rgb_to_L:
            assert swin_in_chans == 1, "If rbg_to_L is used for grayscale conversion, the following swin denoiser has to accept single-channel images"
            
            # adapted from https://github.com/python-pillow/Pillow/blob/66c244af3233b1cc6cc2c424e9714420aca109ad/src/libImaging/Convert.c#L226
            self.register_buffer('gray_constants', torch.tensor([19595, 38470, 7471]).div(2**16)[:, None, None])
        
    def forward(self, x):
        if self.use_checkpoint:
            fused = checkpoint.checkpoint(self.fusion, x, use_reentrant=False)
        else:
            fused = self.fusion(x)
        
        if self.use_rgb_to_L:
            fused = (fused * self.gray_constants).sum(-3, keepdim=True)
        
        return fused, self.denoiser(fused)