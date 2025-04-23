# Copyright (c) 2025-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from PIL import Image
from torch.utils.data import Dataset

class DomainNet(Dataset):
    def __init__(self, root, split='train', transform=None):

        super(DomainNet, self).__init__()
        self.root = os.path.join(root, 'domainnet')
        self.transform = transform
        self.split = split  # training set or test set

        data_type = split
        self.input_paths, self.labels = [], []
        with open(os.path.join(self.root, f'real_{data_type}.txt'), 'r') as f:
            for item in f.readlines():
                feilds = item.strip()
                name, label = feilds.split(' ')
                self.input_paths.append(os.path.join(self.root, name))
                self.labels.append(int(label))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            dict: {'image': image, 'target': index of target class, 'meta': dict}
        """
        img_path, label = self.input_paths[index], self.labels[index]

        img = Image.open(img_path)
        img_size = (img.size[1], img.size[0])

        if self.transform is not None:
            img = self.transform(img)

        out = {'image': img, 'target': label, 'meta': {
                'im_size': img_size, 'index': index, 'class_name': index}}
        return out

    def __len__(self):
        return len(self.input_paths)
