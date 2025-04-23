# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license
# You may obtain a copy of the License at
#
# https://github.com/facebookresearch/pycls/blob/main/LICENSE
#
####################################################################################

"""Optimizer."""

import numpy as np
import torch

def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]

def construct_optimizer(cfg, model):
    """Constructs the optimizer.

    Note that the momentum update in PyTorch differs from the one in Caffe2.
    In particular,

        Caffe2:
            V := mu * V + lr * g
            p := p - V

        PyTorch:
            V := mu * V + g
            p := p - lr * V

    where V is the velocity, mu is the momentum factor, lr is the learning rate,
    g is the gradient and p are the parameters.

    Since V is defined independently of the learning rate in PyTorch,
    when the learning rate is changed there is no need to perform the
    momentum correction by scaling V (unlike in the Caffe2 case).
    """
    if cfg.BN.USE_CUSTOM_WEIGHT_DECAY:
        # Apply different weight decay to Batchnorm and non-batchnorm parameters.
        p_bn = [p for n, p in model.named_parameters() if "bn" in n]
        p_non_bn = [p for n, p in model.named_parameters() if "bn" not in n]
        optim_params = [
            {"params": p_bn, 'lr': cfg.OPTIM.BASE_LR, "weight_decay": cfg.BN.CUSTOM_WEIGHT_DECAY},
            {"params": p_non_bn, 'lr': cfg.OPTIM.BASE_LR, "weight_decay": cfg.OPTIM.WEIGHT_DECAY},
        ]
    elif cfg.MODEL.TYPE.startswith('deit_'):
        optim_params = []
        tmp_optim_params = add_weight_decay(model, cfg.OPTIM.WEIGHT_DECAY)
        for i in range(len(tmp_optim_params)):
            optim_dict = tmp_optim_params[i]
            optim_dict['lr'] = cfg.OPTIM.BASE_LR
            optim_params.append(optim_dict)

    elif cfg.MODEL.TYPE.startswith('resnet') and cfg.MODEL.FINETUNE:
        lr_earlier = cfg.OPTIM.BASE_LR
        lr_later = cfg.OPTIM.BASE_LR
        optim_params = [
            {'params': model.conv1.parameters(), 'lr': lr_earlier},
            {'params': model.bn1.parameters(), 'lr': lr_earlier},
            {'params': model.layer1.parameters(), 'lr': lr_earlier},
            {'params': model.layer2.parameters(), 'lr': lr_earlier},
            {'params': model.layer3.parameters(), 'lr': lr_earlier},
            {'params': model.layer4.parameters(), 'lr': lr_later},
            {'params': model.fc.parameters(), 'lr': lr_later},
        ]
    else:
        optim_params = [
            {'params': model.parameters(), 'lr': cfg.OPTIM.BASE_LR}
        ]

    if cfg.OPTIM.TYPE == 'sgd':
        optimizer =  torch.optim.SGD(
            optim_params,
            momentum=cfg.OPTIM.MOMENTUM,
            weight_decay=cfg.OPTIM.WEIGHT_DECAY,
            dampening=cfg.OPTIM.DAMPENING,
            nesterov=cfg.OPTIM.NESTEROV
        )
    elif cfg.OPTIM.TYPE == 'adam':
        optimizer = torch.optim.Adam(
            optim_params,
            weight_decay=cfg.OPTIM.WEIGHT_DECAY
        )
    else:
        raise NotImplementedError

    return optimizer


def lr_fun_steps(cfg, cur_epoch):
    """Steps schedule (cfg.OPTIM.LR_POLICY = 'steps')."""
    ind = [i for i, s in enumerate(cfg.OPTIM.STEPS) if cur_epoch >= s][-1]
    return cfg.OPTIM.LR_MULT ** ind


def lr_fun_exp(cfg, cur_epoch):
    """Exponential schedule (cfg.OPTIM.LR_POLICY = 'exp')."""
    return cfg.OPTIM.MIN_LR ** (cur_epoch / cfg.OPTIM.MAX_EPOCH)


def lr_fun_cos(cfg, cur_epoch):
    """Cosine schedule (cfg.OPTIM.LR_POLICY = 'cos')."""
    lr = 0.5 * (1.0 + np.cos(np.pi * cur_epoch / cfg.OPTIM.MAX_EPOCH))
    return (1.0 - cfg.OPTIM.MIN_LR) * lr + cfg.OPTIM.MIN_LR


def lr_fun_lin(cfg, cur_epoch):
    """Linear schedule (cfg.OPTIM.LR_POLICY = 'lin')."""
    lr = 1.0 - cur_epoch / cfg.OPTIM.MAX_EPOCH
    return (1.0 - cfg.OPTIM.MIN_LR) * lr + cfg.OPTIM.MIN_LR


def lr_fun_none(cfg, cur_epoch):
    """No schedule (cfg.OPTIM.LR_POLICY = 'none')."""
    return 1


def get_lr_fun(cfg):
    """Retrieves the specified lr policy function"""
    lr_fun = "lr_fun_" + cfg.OPTIM.LR_POLICY
    assert lr_fun in globals(), "Unknown LR policy: " + cfg.OPTIM.LR_POLICY
    err_str = "exp lr policy requires OPTIM.MIN_LR to be greater than 0."
    assert cfg.OPTIM.LR_POLICY != "exp" or cfg.OPTIM.MIN_LR > 0, err_str
    return globals()[lr_fun]


def get_epoch_lr(cfg, cur_epoch):
    """Retrieves the lr for the given epoch according to the policy."""
    # Get lr and scale by by BASE_LR
    lr = get_lr_fun(cfg)(cfg, cur_epoch) * cfg.OPTIM.BASE_LR
    # Linear warmup
    if cur_epoch < cfg.OPTIM.WARMUP_EPOCHS and 'none' not in cfg.OPTIM.LR_POLICY:
        alpha = cur_epoch / cfg.OPTIM.WARMUP_EPOCHS
        warmup_factor = cfg.OPTIM.WARMUP_FACTOR * (1.0 - alpha) + alpha
        lr *= warmup_factor
    return lr


def set_lr(optimizer, new_lr):
    """Sets the optimizer lr to the specified value."""
    largest_lr = -np.inf
    for param_group in optimizer.param_groups:
        if param_group["lr"] > largest_lr:
            largest_lr = param_group["lr"]

    for param_group in optimizer.param_groups:
        ratio = param_group["lr"] / largest_lr
        param_group["lr"] = ratio * new_lr


def plot_lr_fun():
    """Visualizes lr function."""
    epochs = list(range(cfg.OPTIM.MAX_EPOCH))
    lrs = [get_epoch_lr(epoch) for epoch in epochs]
