# Copyright (c) 2025-present, Royal Bank of Canada. 
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import shutil
import pickle
import lmdb

import numpy as np
from PIL import Image

import pycls.datasets.utils as ds_utils
from torch.utils.data import Dataset


def normalize_tin_val_folder_structure(path,
                                       images_folder='images',
                                       annotations_file='val_annotations.txt'):
    # Check if files/annotations are still there to see
    # if we already run reorganize the folder structure.
    images_folder = os.path.join(path, images_folder)
    annotations_file = os.path.join(path, annotations_file)

    # Exists
    if not os.path.exists(images_folder) \
       and not os.path.exists(annotations_file):
        if not os.listdir(path):
            raise RuntimeError('Validation folder is empty.')
        return

    # Parse the annotations
    with open(annotations_file) as f:
        for line in f:
            values = line.split()
            img = values[0]
            label = values[1]
            img_file = os.path.join(images_folder, values[0])
            label_folder = os.path.join(path, label)
            os.makedirs(label_folder, exist_ok=True)
            try:
                shutil.move(img_file, os.path.join(label_folder, img))
            except FileNotFoundError:
                continue

    os.sync()
    assert not os.listdir(images_folder)
    shutil.rmtree(images_folder)
    os.remove(annotations_file)
    os.sync()


class TinyImageNet(Dataset):
    """Dataset for TinyImageNet-200"""
    base_folder = 'tiny-imagenet-200'
    zip_md5 = '90528d7ca1a48142e341f4ef8d21d0de'
    splits = ('train', 'val')
    filename = 'tiny-imagenet-200.zip'
    url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'

    def __init__(self, root, split='train', download=False, transform=None,
                 test_transform=None, only_features=False):
        self.root = root
        self.no_aug = False
        self.lmdb_path = os.path.join(root, f'{split}.lmdb')
        self.split = split
        self.env = lmdb.open(self.lmdb_path, subdir=os.path.isdir(self.lmdb_path),
                             readonly=True, lock=False, readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            # Count the number of images in the dataset
            self.length = txn.stat()['entries'] // 2

        self.transform = transform
        self.test_transform = test_transform

        self.only_features = only_features
        self.features = ds_utils.load_features("TINYIMAGENET", train=split == 'train', normalized=False)

    def __len__(self):
        return self.length

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        with self.env.begin(write=False) as txn:
            image_key = f'{self.split}_image_{index:08d}'.encode('ascii')
            label_key = f'{self.split}_label_{index:08d}'.encode('ascii')

            image_bytes = txn.get(image_key)
            label_bytes = txn.get(label_key)

            sample = pickle.loads(image_bytes)
            target = pickle.loads(label_bytes).squeeze(0)

            sample = Image.fromarray(sample.astype(np.uint8))

        if self.only_features:
            sample = self.features[index]
        else:
            if self.no_aug:
                if self.test_transform is not None:
                    sample = self.test_transform(sample)
            else:
                if self.transform is not None:
                    sample = self.transform(sample)

        return sample, target
