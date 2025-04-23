# Code is originally from the Typiclust (https://arxiv.org/abs/2202.02794) and ProbCover (https://arxiv.org/abs/2205.11320) implementation
# from https://github.com/avihu111/TypiClust by Avihu Dekel and Guy Hacohen which is licensed under MIT license

# You may obtain a copy of the License at
#
# https://github.com/avihu111/TypiClust/blob/main/LICENSE
#
####################################################################################

import torchvision
from PIL import Image
import pycls.datasets.utils as ds_utils

class CIFAR10(torchvision.datasets.CIFAR10):
    def __init__(self, root, cfg, train, transform, test_transform, download=True, only_features=False):
        super(CIFAR10, self).__init__(root, train, transform=transform, download=download)
        self.test_transform = test_transform
        self.no_aug = False
        self.only_features = only_features

        feature_type = cfg.ACTIVE_LEARNING.FEATURE
        self.features = ds_utils.load_features(cfg.DATASET['NAME'], train=train, normalized=False,
                                               feature_type=feature_type)

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)
        if self.only_features:
            img = self.features[index]
        else:
            if self.no_aug:
                if self.test_transform is not None:
                    img = self.test_transform(img)
            else:
                if self.transform is not None:
                    img = self.transform(img)

        return img, target


class CIFAR100(torchvision.datasets.CIFAR100):
    def __init__(self, root, cfg, train, transform, test_transform, download=True, only_features= False):
        super(CIFAR100, self).__init__(root, train, transform=transform, download=download)
        self.test_transform = test_transform
        self.no_aug = False
        self.only_features = only_features

        feature_type = cfg.ACTIVE_LEARNING.FEATURE
        self.features = ds_utils.load_features("CIFAR100", train=train, normalized=False,
                                               feature_type=feature_type)

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)
        if self.only_features:
            img = self.features[index]
        else:
            if self.no_aug:
                if self.test_transform is not None:
                    img = self.test_transform(img)
            else:
                if self.transform is not None:
                    img = self.transform(img)

        return img, target


class STL10(torchvision.datasets.STL10):
    def __init__(self, root, split, transform, test_transform, download=True, only_features= False):
        super(STL10, self).__init__(root, split, transform=transform, download=download)
        self.test_transform = test_transform
        self.no_aug = False
        self.only_features = only_features
        self.targets = self.labels
        self.features = ds_utils.load_features("STL10", train=True if split=='train' else False, normalized=False)

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.transpose(1,2,0))
        if self.only_features:
            img = self.features[index]
        else:
            if self.no_aug:
                if self.test_transform is not None:
                    img = self.test_transform(img)
            else:
                if self.transform is not None:
                    img = self.transform(img)

        return img, target


class MNIST(torchvision.datasets.MNIST):
    def __init__(self, root, train, transform, test_transform, download=True):
        super(MNIST, self).__init__(root, train, transform=transform, download=download)
        self.test_transform = test_transform
        self.no_aug = False

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.no_aug:
            if self.test_transform is not None:
                img = self.test_transform(img)
        else:
            if self.transform is not None:
                img = self.transform(img)


        return img, target


class SVHN(torchvision.datasets.SVHN):
    def __init__(self, root, train, transform, test_transform, download=True):
        super(SVHN, self).__init__(root, train, transform=transform, download=download)
        self.test_transform = test_transform
        self.no_aug = False

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.no_aug:
            if self.test_transform is not None:
                img = self.test_transform(img)
        else:
            if self.transform is not None:
                img = self.transform(img)

        return img, target
