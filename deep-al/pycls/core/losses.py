# Code is originally from Prateek Munjal et al. (https://arxiv.org/abs/2002.09564)
# from https://github.com/PrateekMunjal/TorchAL by Prateek Munjal which is licensed under MIT license
# You may obtain a copy of the License at
#
# https://github.com/PrateekMunjal/TorchAL/blob/master/LICENSE
#
####################################################################################

"""Loss functions."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.loss import LabelSmoothingCrossEntropy


class LabelSmoothingCrossEntropyWithEnt(nn.Module):
    def __init__(self, smoothing=0.1, ent_weight=0.0):
        super(LabelSmoothingCrossEntropyWithEnt, self).__init__()
        self.label_smoothing_ce = LabelSmoothingCrossEntropy(smoothing=smoothing)
        self.ent_weight = ent_weight

    def forward(self, input, target):
        label_smooth_ce = self.label_smoothing_ce(input, target)

        softmax_output = F.softmax(input, dim=1)
        log_softmax_output = F.log_softmax(input, dim=1)
        entropy = -torch.sum(softmax_output * log_softmax_output, dim=1).mean()

        # Total loss = Cross Entropy Loss + Entropy Regularization
        total_loss = label_smooth_ce + self.ent_weight * entropy
        return total_loss

class CrossEntropyLossWithEnt(nn.CrossEntropyLoss):
    def __init__(self, ent_weight=0.0, temp=1.0, weight=None, size_average=None, ignore_index=-100,
                 reduce=None, reduction='mean'):
        super(CrossEntropyLossWithEnt, self).__init__(weight, size_average, ignore_index, reduce, reduction)
        self.ent_weight = ent_weight
        self.temp = temp
        print(f'Temperature for CE is set: {self.temp}')

    def forward(self, input, target):
        ce_loss = F.cross_entropy(input / self.temp, target, weight=self.weight,
                                   ignore_index=self.ignore_index, reduction=self.reduction)
        # Calculate entropy regularization
        softmax_output = F.softmax(input / self.temp, dim=1)
        log_softmax_output = F.log_softmax(input / self.temp, dim=1)
        entropy = -torch.sum(softmax_output * log_softmax_output, dim=1).mean()

        # Total loss = Cross Entropy Loss + Entropy Regularization
        total_loss = ce_loss + self.ent_weight * entropy

        return total_loss


class RescaleMSELoss(nn.MSELoss):
    def __init__(self, m=1, k=5, weight=None, size_average=None, ignore_index=-100,
                 reduce=None, reduction='mean'):
        super(RescaleMSELoss, self).__init__(size_average, reduce, reduction)
        self.m = m
        self.k = k
        print(f'm: {m}, k: {k} for Rescaling MSE loss')

    def forward(self, input, target):
        target = torch.nonzero(target)[:,1].long()
        num_classes = input.shape[-1]
        logit_c = input[torch.arange(input.shape[0]), target]

        mse_losses = (torch.sum(input ** 2, dim=-1) + (self.k - 1) * logit_c ** 2 - 2 * self.k * self.m * logit_c + self.k * self.m ** 2) / num_classes
        mse_loss = torch.mean(mse_losses)
        return mse_loss



class FocalLoss(nn.Module):
    def __init__(self, gamma=0, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = logpt.exp()

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


# Supported loss functions
_loss_funs = {
    'ce': CrossEntropyLossWithEnt,
    'focal_loss': FocalLoss,
    'mse': nn.MSELoss,
    'rmse': RescaleMSELoss,
    'label_smoothing': LabelSmoothingCrossEntropyWithEnt # LabelSmoothingCrossEntropy
}

def get_loss_fun(cfg):
    """Retrieves the loss function."""
    assert cfg.MODEL.LOSS_FUN in _loss_funs.keys(), \
        'Loss function \'{}\' not supported'.format(cfg.TRAIN.LOSS)

    if cfg.MODEL.GAMMA > 0:
        loss_fun = _loss_funs[cfg.MODEL.LOSS_FUN](ent_weight=cfg.MODEL.GAMMA).cuda()
    elif cfg.MODEL.ENT_WEIGHT > 0:
        try:
            loss_fun = _loss_funs[cfg.MODEL.LOSS_FUN](ent_weight=cfg.MODEL.ENT_WEIGHT).cuda()
        except:
            import IPython; IPython.embed()
    elif cfg.MODEL.LOSS_FUN == 'label_smoothing':
        loss_fun = _loss_funs[cfg.MODEL.LOSS_FUN](smoothing=0.1).cuda()
    else:
        loss_fun = _loss_funs[cfg.MODEL.LOSS_FUN]().cuda()
    return loss_fun


def register_loss_fun(name, ctor):
    """Registers a loss function dynamically."""
    _loss_funs[name] = ctor
