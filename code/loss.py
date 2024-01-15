import torch
from torch import nn
class PositionEnhancedLoss(nn.Module):
    '''
    Calculates the two losses, one for within a path around the target position and one for all other pixels.
    With factor=0.5 the two losses are considered equally important, closer to 1. makes the target positions more important
    '''
    def __init__(self, length=96, factor=.5):#length=128, factor=.8):
        super(PositionEnhancedLoss, self).__init__()
        self.length = length
        self.half_length = self.length // 2
        self.factor = factor
        self.loss_fn = nn.L1Loss(reduction='none')
    
    def crop_dim(self, d):
        top = d - self.half_length
        top_rest = torch.maximum(-top, torch.tensor(0))
        top = torch.maximum(top, torch.tensor(0))
        
        bottom = d + self.half_length
        bottom_rest = torch.maximum(bottom - 511, torch.tensor(0))
        bottom = torch.minimum(bottom, torch.tensor(511))
        
        top -= bottom_rest
        bottom += top_rest
        return torch.stack([top, bottom], 1)
        
    def forward(self, x, y, position):
        loss = self.loss_fn(x, y)
        yys, xxs = self.crop_dim(position[:,0]), self.crop_dim(position[:,1])
        patch_sum = 0.
        for i, (yy, xx) in enumerate(zip(yys, xxs)):
            #loss[i, :, yy[0]:yy[1], xx[0]:xx[1]] *= self.factor
            patch_sum += loss[i, :, yy[0]:yy[1], xx[0]:xx[1]].sum()
            loss[i, :, yy[0]:yy[1], xx[0]:xx[1]] = 0
        patch_numel = (loss.size(0) * loss.size(1) * self.length ** 2)
        patch_loss = patch_sum / patch_numel
        rest_loss = loss.sum() / (loss.numel() - patch_numel)
        return rest_loss * (1 - self.factor) + self.factor * patch_loss