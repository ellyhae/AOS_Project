# Source: https://github.com/uzeful/IFCNN/blob/master/Code/model.py
# Modified by: Ellena, Nate

'''---------------------------------------------------------------------------
IFCNN: A General Image Fusion Framework Based on Convolutional Neural Network
----------------------------------------------------------------------------'''
import torch
from torch import nn
from torchvision import models


class Conv2dBlock(nn.Module):
    """Convolutional block applicable to image batches of size N×C×H×W"""

    def __init__(self, in_channels: int, out_channels: int, *args, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, *args, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, /) -> torch.Tensor:
        y = self.conv(x)
        y = self.bn(y)
        y = self.relu(y)
        return y


class Fusor:
    """Fusor applicable to image batches of size N×F×C×H×W"""

    def __init__(self, fusion_mode: str = "max"):
        self.fusion_mode = fusion_mode

        match self.fusion_mode:
            case "max":
                self.fuse = Fusor.max
            case "amax":
                self.fuse = Fusor.amax
            case "sum":
                self.fuse = Fusor.sum
            case "mean":
                self.fuse = Fusor.mean
            case _:
                raise ValueError(f"Invalid fusion mode '{self.fusion_mode}'")

    def __call__(self, x: torch.Tensor, /) -> torch.Tensor:
        return self.fuse(x)

    @staticmethod
    def max(x: torch.Tensor, /) -> torch.Tensor:
        # Propagate gradient selectively in case of equal values
        return torch.max(x, 1).values

    @staticmethod
    def amax(x: torch.Tensor, /) -> torch.Tensor:
        # Distribute gradient evenly in case of equal values
        return torch.amax(x, 1)

    @staticmethod
    def sum(x: torch.Tensor, /) -> torch.Tensor:
        return torch.sum(x, 1)

    @staticmethod
    def mean(x: torch.Tensor, /) -> torch.Tensor:
        return torch.mean(x, 1)


class IFCNN(nn.Module):
    """Image-fusion convolutional neural network applicable to image batches of size N×F×C×H×W"""

    def __init__(self, in_channels: int = 1, fusion_mode: str = "max", use_resnet: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.fusion_mode = fusion_mode
        self.use_resnet = use_resnet

        self.extractor = nn.Sequential(
            nn.Conv2d(self.in_channels, 64, 7, padding=3, padding_mode="replicate", bias=False),
            Conv2dBlock(64, 64, 3, padding=1, padding_mode="replicate", bias=False)
        )
        self.fusor = Fusor(self.fusion_mode)
        self.reconstructor = nn.Sequential(
            Conv2dBlock(64, 64, 3, padding=1, padding_mode="replicate", bias=False),
            nn.Conv2d(64, self.in_channels, 1)
        )
        self.apply(self.init_weight)

    @torch.no_grad()
    def init_weight(self, module: nn.Module, /):
        if not isinstance(module, nn.Conv2d):
            return
        if self.use_resnet and module.in_channels == self.in_channels:
            weight = models.resnet101(weights=models.ResNet101_Weights.DEFAULT).conv1.weight
            if module.in_channels == 1:
                weight = weight.sum(1, keepdim=True)
            module.weight = nn.Parameter(weight, requires_grad=False)
            return
        # Apply Kaiming initialization
        fan_out = module.out_channels
        for kernel_size in module.kernel_size:
            fan_out *= kernel_size
        module.weight.normal_(0.0, (2.0 / fan_out) ** 0.5)

    def forward(self, x: torch.Tensor, /) -> torch.Tensor:
        n, f = x.size(0), x.size(1)
        y = x.flatten(0, 1)  # Size (n*f, c, h, w)
        y = self.extractor(y)  # Size (n*f, 64, h, w)
        y = y.unflatten(0, (n, f))  # Size (n, f, 64, h, w)
        y = self.fusor(y)  # Size (n, 64, h, w)
        y = self.reconstructor(y)  # Size (n, c, h, w)
        return y