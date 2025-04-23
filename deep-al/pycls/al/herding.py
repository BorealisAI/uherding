# Copyright (c) 2025-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import copy
import time
from tqdm import tqdm
import pycls.datasets.utils as ds_utils

from pycls.utils.metrics import compute_coverage
from pycls.utils.io import compute_cand_size


def compute_norm(x1, x2, device, batch_size=512):
    x1, x2 = x1.unsqueeze(0).to(device), x2.unsqueeze(0).to(device) # 1 x n x d, 1 x n' x d
    dist_matrix = []
    batch_round = x2.shape[1] // batch_size + int(x2.shape[1] % batch_size > 0)
    for i in range(batch_round):
        # distance comparisons are done in batches to reduce memory consumption
        x2_subset = x2[:, i * batch_size: (i + 1) * batch_size]
        dist = torch.cdist(x1, x2_subset, p=2.0)

        dist_matrix.append(dist.cpu())
        del dist

    dist_matrix = torch.cat(dist_matrix, dim=-1).squeeze(0)
    return dist_matrix


class NegNormKernel(object):
    def __init__(self, device):
        self.device = device

    def compute_kernel(self, x1, x2, h, batch_size=512):
        dist_matrix = compute_norm(x1, x2, self.device, batch_size=batch_size)
        return -dist_matrix

class TopHatKernel(object):
    def __init__(self, device):
        self.device = device

    def compute_kernel(self, x1, x2, h, batch_size=512):
        x1, x2 = x1.unsqueeze(0).to(self.device), x2.unsqueeze(0).to(self.device) # 1 x n x d, 1 x n' x d
        dist_matrix = []
        batch_round = x2.shape[1] // batch_size + int(x2.shape[1] % batch_size > 0)
        for i in range(batch_round):
            # distance comparisons are done in batches to reduce memory consumption
            x2_subset = x2[:, i * batch_size: (i + 1) * batch_size]
            dist = torch.cdist(x1, x2_subset)
            dist_matrix.append(dist.cpu())

        dist_matrix = torch.cat(dist_matrix, dim=-1).squeeze(0)
        k = (dist_matrix < h)
        return k

class RBFKernel(object):
    def __init__(self, device):
        self.device = device

    def compute_kernel(self, x1, x2, h=1.0, batch_size=512):
        norm = compute_norm(x1, x2, self.device, batch_size=batch_size)
        k = torch.exp(-1.0 * (norm / h) ** 2)
        return k

class StudentTKernel(object):
    def __init__(self, device):
        self.device = device

    def compute_kernel(self, x1, x2, h=1.0, batch_size=512, beta=0.5):
        norms = compute_norm(x1, x2, self.device, batch_size=batch_size)
        k = (1 + ((norms / h) ** 2) / beta) ** (-(beta+1)/2)
        return k

class LaplaceKernel(object):
    def __init__(self, device):
        self.device = device

    def compute_kernel(self, x1, x2, h=1.0, batch_size=512, beta=1):
        norms = compute_norm(x1, x2, self.device, batch_size=batch_size)
        k = torch.exp(-1 / h * (norms ** beta))
        return k

class CauchyKernel(object):
    def __init__(self, device):
        self.device = device

    def compute_kernel(self, x1, x2, h=1.0, batch_size=512):
        norms = compute_norm(x1, x2, self.device, batch_size=batch_size)
        k =  1 / (1 + norms**2)
        return k

class RationalQuadKernel(object):
    def __init__(self, device):
        self.device = device

    def compute_kernel(self, x1, x2, h=1.0, batch_size=512, alpha=1.0):
        norms = compute_norm(x1, x2, self.device, batch_size=batch_size)
        k = (1 + norms**2 / (2 * alpha))**(-alpha)
        return k


class Herding:
    def __init__(self, cfg, lSet, uSet, budgetSize, delta, clf_model,
                 dataObj=None, dataset=None, kernel="rbf", device="cuda",
                 batch_size=1024, permute=True):
        self.cfg = cfg
        self.ds_name = self.cfg['DATASET']['NAME']
        self.seed = self.cfg['RNG_SEED']
        self.dataset = dataset
        self.dataObj = dataObj
        self.device = device 
        assert self.dataset is not None
        if clf_model is not None:
            clf_model.eval()

        if self.cfg.ACTIVE_LEARNING.UNC_FEATURE == 'classifier' and self.cfg.ACTIVE_LEARNING.FEATURE == 'classifier':
            self.all_features = self.get_representation(
                clf_model, np.arange(len(dataset)), dataset)
            print(f'Obtained features from a classifier')
        else:
            feature_type = self.cfg.ACTIVE_LEARNING.UNC_FEATURE
            print('==================================')
            print(f'feature_type: {feature_type}')

            self.all_features = ds_utils.load_features(
                self.ds_name, self.seed, train=True, is_diffusion=False, feature_type=feature_type,
                dataset=dataset)
            print(f'Obtained features from {feature_type}')

        # normalize features
        self.all_features = self.all_features / np.linalg.norm(self.all_features, axis=-1, keepdims=True)

        self.batch_size = batch_size
        self.lSet = lSet
        self.total_uSet = copy.deepcopy(uSet)
        self.budgetSize = budgetSize
        self.delta = delta

        subset_size = compute_cand_size(len(self.lSet), self.budgetSize)
        print(f'Subset size: {subset_size}')
        if permute:
            self.uSet = np.random.permutation(self.total_uSet)[:subset_size]
        else:
            self.uSet = self.total_uSet

        self.relevant_indices = np.concatenate([self.lSet, self.uSet]).astype(int)
        self.relevant_features = torch.from_numpy(
            self.all_features[self.relevant_indices])

        self.kernel_fn = self.construct_kernel_fn(kernel_name=kernel)

        self.kernel_all = self.kernel_fn.compute_kernel(
            self.relevant_features, self.relevant_features, self.delta,
            batch_size=self.batch_size).to(self.device) # (l+u) x (l+u)
        print(f"Memory size of kernel: {self.kernel_all.element_size() * self.kernel_all.nelement()}")

        if len(self.lSet) > 0:
            self.kernel_la = self.kernel_fn.compute_kernel(
                self.relevant_features[:len(self.lSet)], self.relevant_features, self.delta,
                batch_size=self.batch_size).to(self.device)

        torch.cuda.empty_cache()

    @torch.no_grad()
    def get_representation(self, clf_model, idx_set, dataset):

        clf_model.to(self.device)
        tempIdxSetLoader = self.dataObj.getSequentialDataLoader(
            indexes=idx_set, batch_size=int(self.cfg.TRAIN.BATCH_SIZE/self.cfg.NUM_GPUS), data=dataset)
        features = []

        print(f"len(dataLoader): {len(tempIdxSetLoader)}")

        for i, (x, _) in enumerate(tqdm(tempIdxSetLoader, desc="Extracting Representations")):
            with torch.no_grad():
                x = x.to(self.device)
                temp_z = clf_model(x)['features']
                features.append(temp_z.cpu().numpy())

        features = np.concatenate(features, axis=0)
        return features


    def construct_kernel_fn(self, kernel_name):
        if kernel_name == "rbf":
            kernel = RBFKernel('cuda')
        elif kernel_name == "tophat":
            kernel = TopHatKernel('cuda')
        elif kernel_name == "student":
            kernel = StudentTKernel('cuda')
        elif kernel_name == 'negnorm':
            kernel = NegNormKernel('cuda')
        elif kernel_name == "laplace":
            kernel = LaplaceKernel('cuda')
        elif kernel_name == "cauchy":
            kernel = CauchyKernel('cuda')
        elif kernel_name == 'rational':
            kernel = RationalQuadKernel('cuda')
        else:
            raise NotImplementedError(f"{kernel_name} not implemented")
        print(f'Constructed kernel: {kernel_name}')
        return kernel

    def get_lSet(self, lSet, dataset):
        lSetLoader = self.dataObj.getSequentialDataLoader(
            indexes=lSet, batch_size=int(self.cfg.TRAIN.BATCH_SIZE), data=self.dataset)
        x_ls, y_ls = [], []
        for x_l, y_l in lSetLoader:
            x_ls.append(x_l)
            y_ls.append(y_l)
        x_ls = torch.cat(x_ls, dim=0)
        y_ls = torch.cat(y_ls, dim=0)
        print(f"Loaded training set: {x_ls.shape}")
        return x_ls, y_ls

    def select_samples(self):
        uncertainties = torch.ones(1, len(self.relevant_indices)).float().to(self.device)

        start_time = time.time()
        inner_lSet = torch.arange(len(self.lSet)).to(self.device)

        fixed_inner_uSet = torch.arange(len(self.relevant_indices))[len(inner_lSet):].to(self.device)
        inner_uSet_bool = torch.ones_like(fixed_inner_uSet).bool().to(self.device)
        inner_uSet = fixed_inner_uSet[inner_uSet_bool].to(self.device)

        if inner_lSet.shape[0] > 0:
            max_embedding = self.kernel_la.max(dim=0, keepdim=True).values # 1 x N
        else:
            max_embedding = torch.zeros(1, len(inner_lSet) + len(fixed_inner_uSet)).to(self.device) # 1 x N

        selected = []
        for i in range(self.budgetSize):
            num_lSet = len(inner_lSet)
            num_uSet = len(inner_uSet)

            updated_max_embedding = (self.kernel_all - max_embedding) # N x N
            updated_max_embedding[updated_max_embedding < 0] = 0.

            mean_max_embedding = (uncertainties * updated_max_embedding).mean(dim=-1) # N

            # select a point from u
            mean_max_embedding[inner_lSet] = -np.inf
            selected_index = torch.argmax(mean_max_embedding)

            # update lSet and uSet
            inner_lSet = torch.cat((inner_lSet, selected_index.view(-1)))
            inner_uSet_bool[selected_index - len(self.lSet)] = False
            inner_uSet = fixed_inner_uSet[inner_uSet_bool]

            max_embedding = updated_max_embedding[selected_index].unsqueeze(0) + max_embedding

            if len(set(inner_lSet.cpu().numpy())) != num_lSet + 1:
                print(f'inner_lSet: {len(set(inner_lSet.numpy()))} is not equal to {num_lSet+1}')
                import IPython; IPython.embed()
            if len(set(inner_uSet.cpu().numpy())) != num_uSet - 1:
                print(f'inner_lSet: {len(set(inner_uSet.numpy()))} is not equal to {num_uSet+1}')
                import IPython; IPython.embed()
            assert len(np.intersect1d(inner_lSet.cpu().numpy(), inner_uSet.cpu().numpy())) == 0

        selected = inner_lSet[len(self.lSet):].cpu()

        total_inner_lSet = torch.cat((torch.arange(len(self.lSet)), selected))
        total_lSet_features = self.relevant_features[total_inner_lSet].to(self.device)
        coverage = compute_coverage(total_lSet_features, self.relevant_features, self.kernel_fn)
        print(f'Mean coverage herding: {coverage}')

        assert len(selected) == self.budgetSize, 'added a different number of samples'
        activeSet = self.relevant_indices[selected].reshape(-1)
        remainSet = np.array(sorted(list(set(self.total_uSet) - set(activeSet))))

        print(f'Finished the selection of {len(activeSet)} samples.')
        print(f'Active set is {activeSet}')
        print(f'Time: {np.round(time.time() - start_time, 4)}sec')

        return activeSet, remainSet

