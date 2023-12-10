# Source: https://github.com/uzeful/IFCNN/blob/master/Code/model.py
# Modified by: Ellena Pfleger

'''---------------------------------------------------------------------------
IFCNN: A General Image Fusion Framework Based on Convolutional Neural Network
----------------------------------------------------------------------------'''
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# My Convolution Block
class ConvBlock(nn.Module):
    def __init__(self, inplane, outplane):
        super(ConvBlock, self).__init__()
        self.padding = (1, 1, 1, 1)
        self.conv = nn.Conv2d(inplane, outplane, kernel_size=3, padding=0, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(outplane)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = F.pad(x, self.padding, 'replicate')
        out = self.conv(out)
        out = self.bn(out)
        out = self.relu(out)
        return out

    
class IFCNN(nn.Module):
    def __init__(self, fuse_scheme=0):
        super(IFCNN, self).__init__()
        self.fuse_scheme = fuse_scheme # MAX, MEAN, SUM
        self.conv2 = ConvBlock(64, 64)
        self.conv3 = ConvBlock(64, 64)
        self.conv4 = nn.Conv2d(64, 3, kernel_size=1, padding=0, stride=1, bias=True)

        # Initialize parameters for other parameters
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

        # Initialize conv1 with the pretrained resnet101 and freeze its parameters
        resnet = models.resnet101(weights=models.resnet.ResNet101_Weights.DEFAULT)
        for p in resnet.parameters():
            p.requires_grad = False
        self.conv1 = resnet.conv1
        self.conv1.stride = 1
        self.conv1.padding = (0, 0)

    def forward(self, tensor):
        # stacking the input changes the values slightly, seemingly due to floating point reasons
        
        # Feature extraction
        outs = tensor.flatten(0, 1)
        outs = F.pad(outs, (3, 3, 3, 3), mode='replicate')
        outs = self.conv1(outs)
        outs = self.conv2(outs)
        outs = outs.unflatten(0, tensor.shape[:2])
        
        # Feature fusion
        if self.fuse_scheme == 0: # MAX
            out = outs.max(-4)[0]
        elif self.fuse_scheme == 1: # SUM
            out = outs.sum(-4)
        elif self.fuse_scheme == 2: # MEAN
            out = outs.mean(-4)
        else: # Default: MAX
            out = outs.max(-4)[0]
        
        # Feature reconstruction
        out = self.conv3(out)
        out = self.conv4(out)
        return out