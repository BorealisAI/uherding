# Copyright (c) 2025-present, Royal Bank of Canada.
# Copyright (c) 2023 Yichen Xie
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#####################################################################################
# Code is based on the ActiveFT (https://arxiv.org/abs/2303.14382) implementation
# from https://github.com/yichen928/ActiveFT?tab=readme-ov-file by Yichen Xie which is licensed under Apache-2.0 license.
# You may obtain a copy of the License at
#
# https://github.com/yichen928/ActiveFT/blob/main/LICENSE
#
####################################################################################


import copy
import time
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
from tqdm import tqdm

torch.autograd.set_detect_anomaly(True)
eps = 1e-10
infty = 1e10


class SampleModel(nn.Module):
    def __init__(
        self, lSet_features, uSet_features, sample_num, temperature=0.07,
        init='random', distance='euclidean', balance=1.0,
        slice=None, batch_size=100000):

        super(SampleModel, self).__init__()
        self.features = uSet_features
        self.lSet_num = lSet_features.shape[0]

        self.sample_ids = list(range(uSet_features.shape[0]))

        self.temperature = temperature
        self.sample_num = sample_num
        self.balance = balance
        self.slice = slice
        if slice is None:
            self.slice = int(self.features.shape[0])

        self.batch_size = batch_size
        self.init = init
        self.distance = distance

        if self.lSet_num > 0:
            self.lSet_centroid_vals = lSet_features.clone()

        centroids = self.init_centroids().cuda()
        self.centroids = nn.Parameter(centroids)

    def init_centroids(self):
        if self.init == "random":
            sample_ids = random.sample(self.sample_ids, self.sample_num)
        else:
            raise NotImplementedError(f'init centroid: {self.init} is not implemented')

        if self.lSet_num > 0:
            centroids = torch.cat(
                [self.lSet_centroid_vals, self.features[sample_ids].clone()], dim=0)
        else:
            centroids = self.features[sample_ids].clone()
        return centroids

    def correct_centroids(self):
        if self.lSet_num > 0:
            self.centroids.data[:self.lSet_num] = self.lSet_centroid_vals

    def get_loss(self):
        centroids = F.normalize(self.centroids, dim=1)

        if self.batch_size < len(self.sample_ids):
            sample_ids = random.sample(self.sample_ids, self.batch_size)
        else:
            sample_ids = copy.deepcopy(self.sample_ids)
        features = self.features[sample_ids]

        sample_slice_num = math.ceil(1.0 * self.sample_num / self.slice)
        batch_slice_num = math.ceil(1.0 * self.batch_size / self.slice)

        prod_exp_pos = []
        pos_k = []
        for sid in range(batch_slice_num):
            start = sid * self.slice
            end = (sid + 1) * self.slice
            prod = torch.matmul(features[start: end], centroids.transpose(1, 0))  # (slice_num, k)
            prod = prod / self.temperature
            prod_exp = torch.exp(prod)
            prod_exp_pos_cur, pos_k_cur = torch.max(prod_exp, dim=1)  # (slice_num, )
            prod_exp_pos.append(prod_exp_pos_cur)
            pos_k.append(pos_k_cur)
        pos_k = torch.cat(pos_k, dim=0)
        prod_exp_pos = torch.cat(prod_exp_pos, dim=0)

        cent_prob_exp_sum = []
        for sid in range(sample_slice_num):
            start = sid * self.slice
            end = (sid + 1) * self.slice
            cent_prod = torch.matmul(centroids.detach(), centroids[start:end].transpose(1, 0))  # (k, slice_num)
            cent_prod = cent_prod / self.temperature
            cent_prod_exp = torch.exp(cent_prod)
            cent_prob_exp_sum_cur = torch.sum(cent_prod_exp, dim=0)  # (slice_num, )
            cent_prob_exp_sum.append(cent_prob_exp_sum_cur)
        cent_prob_exp_sum = torch.cat(cent_prob_exp_sum, dim=0)

        J = torch.log(prod_exp_pos) - torch.log(prod_exp_pos + cent_prob_exp_sum[pos_k] * self.balance)
        J = -torch.mean(J)

        return J


class ActiveFT:
    def __init__(self, cfg, lSet, uSet, budgetSize, clf_model, dataset, dataObj, device='cuda', permute=True):
        self.cfg = cfg
        self.ds_name = self.cfg['DATASET']['NAME']
        self.device = device
        self.budgetSize = budgetSize

        self.lSet = lSet
        self.total_uSet = copy.deepcopy(uSet)
        subset_size = 100000
        print(f'Subset size: {subset_size}')

        if permute:
            self.uSet = np.random.permutation(self.total_uSet)[:subset_size]
        else:
            self.uSet = self.total_uSet

        self.dataset = dataset
        self.dataObj = dataObj

        self.clf_model = clf_model

        self.relevant_indices = np.concatenate([self.lSet, self.uSet]).astype(int)

        self.relevant_features = torch.from_numpy(self.get_representation(
            self.clf_model, self.relevant_indices, self.dataset))

        self.lr = 0.001
        self.max_iter = 300 if self.ds_name not in ['IMAGENET', 'IMBALANCED_IMAGENET'] else 100


    @torch.no_grad()
    def get_representation(self, clf_model, idx_set, dataset):
        if self.cfg.ACTIVE_LEARNING.FEATURE not in ['random', 'finetune']:
            batch_size = 1024
        else:
            batch_size = 128

        clf_model.to(self.device)
        tempIdxSetLoader = self.dataObj.getSequentialDataLoader(
            indexes=idx_set, batch_size=batch_size, data=dataset)
        print(f"len(dataLoader): {len(tempIdxSetLoader)}")

        features = []
        for i, (x, _) in enumerate(tqdm(tempIdxSetLoader, desc="Extracting Representations")):
            with torch.no_grad():
                x = x.to(self.device)
                temp_z = clf_model(x)['features']
                features.append(temp_z.cpu().numpy())

        features = np.concatenate(features, axis=0)
        return features


    def select_samples(self):
        start_time = time.time()

        norm_rel_features = F.normalize(self.relevant_features, dim=1)
        sample_ids = self.optimize_dist(norm_rel_features) # 0 <= elements < |uSet|

        activeSet = self.relevant_indices[sample_ids].reshape(-1)
        remainSet = np.array(sorted(list(set(self.total_uSet) - set(activeSet))))
        assert len(activeSet) == self.budgetSize, 'added a different number of samples'

        print(f'Finished the selection of {len(activeSet)} samples.')
        print(f'Active set is {activeSet}')
        print(f'Time: {np.round(time.time() - start_time, 4)}sec')

        return activeSet, remainSet


    def optimize_dist(self, norm_rel_features, slice=None):
        #  features: (|lSet| + |uSet|, c)
        lSet_num = len(self.lSet)
        lSet_features = norm_rel_features[:lSet_num].cuda()
        uSet_features = norm_rel_features[lSet_num:].cuda()

        sample_model = SampleModel(
            lSet_features, uSet_features, self.budgetSize).cuda()

        optimizer = optim.Adam(sample_model.parameters(), lr=self.lr)
        scheduler = None

        for i in range(self.max_iter):
            loss = sample_model.get_loss()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            lr = optimizer.param_groups[0]["lr"]
            print("Iter: %d, lr: %.6f, loss: %f" % (i, lr, loss.item()))
            sample_model.correct_centroids()

        centroids = sample_model.centroids.detach()
        centroids = F.normalize(centroids, dim=1)
        slice = sample_model.slice

        sample_slice_num = math.ceil(centroids.shape[0] / slice)
        sample_ids = set()
        for sid in range(sample_slice_num):
            start = sid * slice
            end = min((sid + 1) * slice, centroids.shape[0])

            dist = torch.matmul(centroids[start:end], uSet_features.transpose(1, 0)).cpu()  # (slice_num, |uSet|)
            _, ids_sort = torch.sort(dist, dim=1, descending=True)
            for i in range(lSet_num, ids_sort.shape[0], 1):
                for j in range(ids_sort.shape[1]):
                    if ids_sort[i, j].item() not in sample_ids:
                        sample_ids.add(ids_sort[i, j].item())
                        break

        sample_ids = list(sample_ids)
        return sample_ids
