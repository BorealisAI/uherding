# Copyright (c) 2025-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os.path as osp
import pickle
from PIL import Image
import lmdb
import six
import torch.utils.data as data
import pycls.datasets.utils as ds_utils


def loads_data(buf):
    """
    Args:
        buf: the output of `dumps`.
    """
    return pickle.loads(buf)


class ImageNet(data.Dataset):
    def __init__(self, data_dir, split, transform=None, test_transform=None, num_classes=1000, only_features=False):
        dataset_name = data_dir.split('/')[-1]
        self.db_path = osp.join(data_dir, 'imagenet_train.lmdb' if split == 'train' else 'imagenet_val.lmdb')
        self.env = lmdb.open(self.db_path, subdir=osp.isdir(self.db_path),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = loads_data(txn.get(b'__len__'))
            self.keys = loads_data(txn.get(b'__keys__'))

        self.transform = transform
        self.test_transform = test_transform
        self.only_features = only_features

        self.features = ds_utils.load_features(
            dataset_name.upper(), train=True if split=='train' else False, normalized=False)
        self.no_aug = False


    def __getitem__(self, index):
        env = self.env
        with env.begin(write=False) as txn:
            byteflow = txn.get(self.keys[index])

        unpacked = loads_data(byteflow)

        # load img
        imgbuf = unpacked[0]
        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        img = Image.open(buf).convert('RGB')

        # load label
        target = unpacked[1]

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

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'
