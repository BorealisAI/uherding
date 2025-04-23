# Code is originally from the Typiclust (https://arxiv.org/abs/2202.02794) and ProbCover (https://arxiv.org/abs/2205.11320) implementation
# from https://github.com/avihu111/TypiClust by Avihu Dekel and Guy Hacohen which is licensed under MIT license
# You may obtain a copy of the License at
#
# https://github.com/avihu111/TypiClust/blob/main/LICENSE
#
####################################################################################

import torch
import numpy as np

NUM_CLASSESS = {
    'CIFAR10': 10,
    'CIFAR10_scan': 10,
    'CIFAR10_dino': 10,
    'CIFAR100': 100,
    'TINYIMAGENET': 200,
    'DOMAINNET': 345,
    'IMAGENET': 1000,
    'IMAGENET50': 50,
    'IMAGENET100': 100,
    'IMAGENET200': 200,
    'STL10': 10,
}

DATASET_TARGETS_DICT = {
    'train':
        {
            'CIFAR10':'../../scan/results/cifar-10/pretext/targets_seed{seed}.npy',
            'CIFAR100':'../../scan/results/cifar-100/pretext/targets_seed{seed}.npy',
            'TINYIMAGENET': '../../scan/results/tiny-imagenet/pretext/targets_seed{seed}.npy',
            'IMAGENET': '../../dino/runs/imagenet/trainfeat.pth',
            'IMAGENET100': '../../dino/runs/imagenet100/trainfeat.pth',
            'STL10':'../../scan/results/stl-10/pretext/targets_seed{seed}.npy',
            'SVHN':'../../scan/results/svhn-32/pretext/targets_seed{seed}.npy',
            'MNIST':'../../scan/results/mnist-32/pretext/targets_seed{seed}.npy',
        },
    'test':
        {
            'CIFAR10': '../../scan/results/cifar-10/pretext/test_targets_seed{seed}.npy',
            'CIFAR100': '../../scan/results/cifar-100/pretext/test_targets_seed{seed}.npy',
            'TINYIMAGENET': '../../scan/results/tiny-imagenet/pretext/test_targets_seed{seed}.npy',
            'IMAGENET': '../../dino/runs/imagenet/testfeat.pth',
            'IMAGENET100': '../../dino/runs/imagenet100/testfeat.pth',
            'STL10':'../../scan/results/stl-10/pretext/test_targets_seed{seed}.npy',
            'SVHN':'../../scan/results/svhn-32/pretext/test_targets_seed{seed}.npy',
            'MNIST':'../../scan/results/mnist-32/pretext/test_targets_seed{seed}.npy',
        }
}

DATASET_FEATURES_DICT = {
    'train':
        {
            'CIFAR10':'../../scan/results/cifar-10/pretext/features_seed{seed}.npy',
            'CIFAR10_scan':'../../scan/results/cifar-10/scan/features_seed{seed}_clusters10.npy',
            'CIFAR100_scan':'../../scan/results/cifar-100/scan/features_seed{seed}_clusters100.npy',
            'CIFAR10_dino':'../../dino/runs/cifar-10/trainfeat.pth',
            'CIFAR100_dino':'../../dino/runs/cifar-100/trainfeat.pth',
            'CIFAR100':'../../scan/results/cifar-100/pretext/features_seed{seed}.npy',
            'TINYIMAGENET': '../../scan/results/tiny-imagenet/pretext/features_seed{seed}.npy',
            'DOMAINNET': '../../dino/runs/domainnet/trainfeat.pth',
            'IMAGENET': '../../dino/runs/imagenet/trainfeat.pth',
            'IMAGENET100': '../../dino/runs/imagenet100/trainfeat.pth',
            'STL10':'../../scan/results/stl-10/pretext/features_seed{seed}.npy',
            'SVHN':'../../scan/results/svhn-32/pretext/features_seed{seed}.npy',
            'MNIST':'../../scan/results/mnist-32/pretext/features_seed{seed}.npy',
        },
    'test':
        {
            'CIFAR10': '../../scan/results/cifar-10/pretext/test_features_seed{seed}.npy',
            'CIFAR10_scan':'../../scan/results/cifar-10/scan/test_features_seed{seed}_clusters10.npy',
            'CIFAR100_scan':'../../scan/results/cifar-100/scan/test_features_seed{seed}_clusters100.npy',
            'CIFAR10_dino':'../../dino/runs/cifar-10/testfeat.pth',
            'CIFAR100_dino':'../../dino/runs/cifar-100/testfeat.pth',
            'CIFAR100': '../../scan/results/cifar-100/pretext/test_features_seed{seed}.npy',
            'TINYIMAGENET': '../../scan/results/tiny-imagenet/pretext/test_features_seed{seed}.npy',
            'DOMAINNET': '../../dino/runs/domainnet/testfeat.pth',
            'IMAGENET': '../../dino/runs/imagenet/testfeat.pth',
            'IMAGENET100': '../../dino/runs/imagenet100/testfeat.pth',
            'STL10':'../../scan/results/stl-10/pretext/test_features_seed{seed}.npy',
            'SVHN':'../../scan/results/svhn-32/pretext/test_features_seed{seed}.npy',
            'MNIST':'../../scan/results/mnist-32/pretext/test_features_seed{seed}.npy',
        }
}

def load_features(ds_name, seed=1, train=True, normalized=True, is_diffusion=False,
                  feature_type='simclr', dataset=None):
    split = "train" if train else "test"

    num_classes = NUM_CLASSESS[ds_name]
    if ds_name.lower() in ['cifar10', 'cifar100'] and feature_type not in  ['simclr', 'classifier']:
        ds_name = ds_name + '_' + feature_type
    print(f'Loading {ds_name} feature')

    if dataset is not None:
        features = dataset.features

        if isinstance(features, torch.Tensor):
            features = features.cpu().detach().numpy()
        features = features / np.linalg.norm(features, axis=1, keepdims=True)
        print(f'Loaded features from dataset: {ds_name}')
        return features

    try:
        fname = DATASET_FEATURES_DICT[split][ds_name].format(seed=seed)

        if fname.endswith('.npy'):
            features = np.load(fname)
        elif fname.endswith('.pth'):
            features = torch.load(fname)
        else:
            raise Exception("Unsupported filetype")

    except:
        fname = DATASET_FEATURES_DICT[split][ds_name].format(seed=1)

        if fname.endswith('.npy'):
            features = np.load(fname)
        elif fname.endswith('.pth'):
            features = torch.load(fname)
        else:
            raise Exception("Unsupported filetype")

    if isinstance(features, torch.Tensor):
        features = features.cpu().detach().numpy()

    if normalized:
        features = features / np.linalg.norm(features, axis=1, keepdims=True)
    return features


def load_targets(ds_name, seed=1, train=True):
    " load pretrained features for a dataset "
    split = "train" if train else "test"

    try:
        fname = DATASET_TARGETS_DICT[split][ds_name].format(seed=seed)

        if fname.endswith('.npy'):
            targets = np.load(fname)
        elif fname.endswith('.pth'):
            targets = torch.load(fname)
        else:
            raise Exception("Unsupported filetype")

    except:
        fname = DATASET_TARGETS_DICT[split][ds_name].format(seed=1)

        if fname.endswith('.npy'):
            targets = np.load(fname)
        elif fname.endswith('.pth'):
            targets = torch.load(fname)
        else:
            raise Exception("Unsupported filetype")

    return targets
