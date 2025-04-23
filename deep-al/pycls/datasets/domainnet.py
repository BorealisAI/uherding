# Copyright (c) 2025-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from PIL import Image
from torch.utils.data import Dataset
import pycls.datasets.utils as ds_utils

class DomainNet(Dataset):
    def __init__(self, root, split='train', transform=None, test_transform=None,
                only_features=False):

        super(DomainNet, self).__init__()
        self.root = root 
        self.transform = transform
        self.split = split  
        self.no_aug = False

        data_type = split
        self.input_paths, self.labels = [], []
        with open(os.path.join(self.root, f'real_{data_type}.txt'), 'r') as f:
            for item in f.readlines():
                feilds = item.strip()
                name, label = feilds.split(' ')
                self.input_paths.append(os.path.join(self.root, name))
                self.labels.append(int(label))

        self.test_transform = test_transform
        self.only_features = only_features
        self.features = ds_utils.load_features("DOMAINNET", train=split == 'train', normalized=False)


    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            dict: {'image': image, 'target': index of target class, 'meta': dict}
        """
        img_path, label = self.input_paths[index], self.labels[index]

        sample = Image.open(img_path)

        if self.only_features:
            sample = self.features[index]
        else:
            if self.no_aug:
                if self.test_transform is not None:
                    sample = self.test_transform(sample)
            else:
                if self.transform is not None:
                    sample = self.transform(sample)

        return sample, label

    def __len__(self):
        return len(self.input_paths)
    