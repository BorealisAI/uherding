# Code is originally from Prateek Munjal et al. (https://arxiv.org/abs/2002.09564)
# from https://github.com/PrateekMunjal/TorchAL by Prateek Munjal which is licensed under MIT license
# You may obtain a copy of the License at
#
# https://github.com/PrateekMunjal/TorchAL/blob/master/LICENSE
#
####################################################################################

"""Model and loss construction functions."""

from pycls.models.resnet import *
from pycls.models.deit import *

import torch
from torch import nn
from torch.nn import functional as F
from timm.models import create_model


# Supported models
_models = {
    # ResNet style archiectures
    'resnet18': resnet18,
    'resnet34': resnet34,
    'resnet50': resnet50,
    'resnet101': resnet101,
    'resnet152': resnet152,
    'resnext50_32x4d': resnext50_32x4d,
    'resnext101_32x8d': resnext101_32x8d,
    'wide_resnet50_2': wide_resnet50_2,
    'wide_resnet101_2': wide_resnet101_2
}

class FeaturesNet(nn.Module):
    def __init__(self, in_layers, out_layers, use_mlp=True, penultimate_active=False):
        super().__init__()
        self.use_mlp = use_mlp
        self.penultimate_active = penultimate_active
        if use_mlp:
            self.lin1 = nn.Linear(in_layers, in_layers)
            self.lin2 = nn.Linear(in_layers, in_layers)
        self.fc_in_dim = in_layers
        self.num_classes = out_layers
        self.fc = nn.Linear(in_layers, out_layers, bias=False)

    def forward(self, x, y=None):
        feats = x
        if self.use_mlp:
            x = F.relu(self.lin1(x))
            x = F.relu((self.lin2(x)))
        out = self.fc(x)

        output_dict = {'preds': out, 'features': feats, 'labels': y}
        return output_dict

    def forward_classifier(self, x, y=None):
        feats = x
        if self.use_mlp:
            x = F.relu(self.lin1(x))
            x = F.relu((self.lin2(x)))
        out = self.fc(x)

        output_dict = {'preds': out, 'features': feats, 'labels': y}
        return output_dict

class LinearKernel(object):
    def __init__(self, device):
        self.device = device

    def compute_kernel(self, x1, x2, h=None, batch_size=784):
        x1, x2 = x1.to(self.device), x2.to(self.device) # n x d, n' x d
        dist_matrix = []
        batch_round = x2.shape[0] // batch_size + int(x2.shape[0] % batch_size > 0)
        for i in range(batch_round):
            # distance comparisons are done in batches to reduce memory consumption
            x2_subset = x2[i * batch_size: (i + 1) * batch_size]
            dotproduct = torch.matmul(x1, x2_subset.transpose(1,0))
            dist_matrix.append(dotproduct.cpu())

        dist_matrix = torch.cat(dist_matrix, dim=-1).squeeze(0)
        return dist_matrix

class NNNet(nn.Module):
    def __init__(self, num_classes, device="cuda"):
        super().__init__()
        self.device = device
        self.num_classes = num_classes

    def compute_norm(self, x1, x2, batch_size=512):
        x1, x2 = x1.unsqueeze(0).to(self.device), x2.unsqueeze(0).to(self.device) # 1 x n x d, 1 x n' x d
        dist_matrix = []
        batch_round = x2.shape[1] // batch_size + int(x2.shape[1] % batch_size > 0)
        for i in range(batch_round):
            # distance comparisons are done in batches to reduce memory consumption
            x2_subset = x2[:, i * batch_size: (i + 1) * batch_size]
            dist = torch.cdist(x1, x2_subset, p=2.0)
            dist_matrix.append(dist.cpu())

        dist_matrix = torch.cat(dist_matrix, dim=-1).squeeze(0)
        return dist_matrix

    def forward(self, x, y, x_test, return_logits=False):
        x, x_test = F.normalize(x, dim=1), F.normalize(x_test, dim=1)

        dist_matrix = self.compute_norm(x_test, x)
        if return_logits:
            try:
                topk = torch.topk(-dist_matrix, k=self.num_classes, largest=True, dim=1) # N_t x N
            except:
                print('RuntimeError: selected index k out of range')
            preds = topk.values
        else:
            nn_indices = torch.argmin(dist_matrix, dim=1)
            nn_labels = y[nn_indices]
            preds = F.one_hot(nn_labels, num_classes=self.num_classes)

        output_dict = {'preds': preds}
        return output_dict


def get_model(cfg):
    """Gets the model class specified in the config."""

    if cfg.MODEL.TYPE.startswith('deit_'):
        model = create_model(
            cfg.MODEL.TYPE,
            pretrained=False,
            num_classes=cfg.MODEL.NUM_CLASSES,
            drop_rate=0.0,
            drop_path_rate=0.1,
            drop_block_rate=None)
    else:
        model = _models[cfg.MODEL.TYPE](
            num_classes=cfg.MODEL.NUM_CLASSES, use_dropout=True, last_layer=cfg.MODEL.LAST_LAYER, cfg=cfg)
        if cfg.DATASET.NAME == 'MNIST':
            model.conv1 =  torch.nn.Conv2d(
                1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    return model

def build_model(cfg):
    """Builds the model."""
    if cfg.MODEL.LINEAR_FROM_FEATURES:
        if cfg.MODEL.NN:
            return NNNet(cfg.MODEL.NUM_CLASSES)
        else:
            num_features = 384 if (
                cfg.DATASET.NAME in ['IMAGENET', 'IMBALANCED_IMAGENET',
                                     'IMAGENET100', 'IMBALANCED_IMAGENET100'] or cfg.ACTIVE_LEARNING.FEATURE == 'dino') else 512
            return FeaturesNet(num_features, cfg.MODEL.NUM_CLASSES)

    model = get_model(cfg)

    if cfg.MODEL.FINETUNE and cfg.MODEL.CHECKPOINT_PATH:
        checkpoint = torch.load(cfg.MODEL.CHECKPOINT_PATH, map_location='cpu')

        if cfg.MODEL.TYPE.startswith('deit_small'):
            if 'fc.weight' in checkpoint and 'fc.bias' in checkpoint:
                if checkpoint['fc.weight'].shape[0] != model.fc.weight.shape[0]:
                    del checkpoint['fc.weight']
                    del checkpoint['fc.bias']
        elif cfg.MODEL.TYPE.startswith('deit_base'):
            checkpoint = checkpoint['model']
            if 'head.weight' in checkpoint and 'head.bias' in checkpoint:
                if checkpoint['head.weight'].shape[0] != model.fc.weight.shape[0]:
                    del checkpoint['head.weight']
                    del checkpoint['head.bias']

        model.load_state_dict(checkpoint, strict=False)
        print(f'Loaded a checkpoint from {cfg.MODEL.CHECKPOINT_PATH}')

    return model


def register_model(name, ctor):
    """Registers a model dynamically."""
    _models[name] = ctor


def register_loss_fun(name, ctor):
    """Registers a loss function dynamically."""
    _loss_funs[name] = ctor
