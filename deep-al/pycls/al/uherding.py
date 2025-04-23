# Copyright (c) 2025-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import copy
import time
import pickle
from tqdm import tqdm
import pycls.datasets.utils as ds_utils
import os
from pycls.utils.io import compute_cand_size

class KernelDataset(torch.utils.data.Dataset):
    def __init__(self, save_dir, batch_round, device='cuda'):
        self.save_dir = save_dir
        self.indices = torch.arange(batch_round)
        self.device = device

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        save_path = os.path.join(self.save_dir, f'batch_{idx}.pth')
        batch_kernel = torch.load(save_path)
        return batch_kernel


def compute_norm(x1, x2, device, batch_size=512, save_dir=None):
    x1, x2 = x1.unsqueeze(0).to(device), x2.unsqueeze(0).to(device) # 1 x n x d, 1 x n' x d
    dist_matrix = []
    batch_round = x2.shape[1] // batch_size + int(x2.shape[1] % batch_size > 0)
    for i in range(batch_round):
        # distance comparisons are done in batches to reduce memory consumption
        x2_subset = x2[:, i * batch_size: (i + 1) * batch_size]
        dist = torch.cdist(x1, x2_subset, p=2.0)

        if (i+1) % 100 == 0:
            print(f'{i+1}/{batch_round}')

        dist_matrix.append(dist.cpu())

    dist_matrix = torch.cat(dist_matrix, dim=-1).squeeze(0)
    if dist_matrix.shape[0] == dist_matrix.shape[1]:
        dist_matrix[torch.arange(dist_matrix.shape[0]),
                    torch.arange(dist_matrix.shape[0])] = torch.zeros(dist_matrix.shape[0]).to(dist_matrix.device)
    return dist_matrix


class NegNormKernel(object):

    def __init__(self, device):
        self.device = device

    def compute_kernel(self, x1, x2, h, batch_size=512, save_dir=None):
        dist_matrix = compute_norm(x1, x2, self.device, batch_size=batch_size)
        return -dist_matrix


class TopHatKernel(object):

    def __init__(self, device):
        self.device = device

    def compute_kernel(self, x1, x2, h, batch_size=512, save_dir=None):
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

    def compute_kernel(self, x1, x2, h=1.0, batch_size=512, save_dir=None):
        norms = compute_norm(x1, x2, self.device)
        #k = torch.exp(-1 / h * norms)
        k = torch.exp( -1.0 * (norms / h) ** 2 )
        return k

class StudentTKernel(object):
    def __init__(self, device):
        self.device = device

    def compute_kernel(self, x1, x2, h=1.0, batch_size=512, beta=0.5, save_dir=None):
        norms = compute_norm(x1, x2, self.device)
        k = (1 + ((norms / h) ** 2) / beta)**(-(beta+1)/2)
        return k

class LaplaceKernel(object):
    def __init__(self, device):
        self.device = device

    def compute_kernel(self, x1, x2, h=1.0, batch_size=512, beta=1, save_dir=None):
        norms = compute_norm(x1, x2, self.device)
        k = torch.exp(-1 / h * (norms ** beta))
        return k

class CauchyKernel(object):
    def __init__(self, device):
        self.device = device

    def compute_kernel(self, x1, x2, h=1.0, batch_size=512, save_dir=None):
        norms = compute_norm(x1, x2, self.device)
        k =  1 / (1 + norms**2)
        return k

class RationalQuadKernel(object):
    def __init__(self, device):
        self.device = device

    def compute_kernel(self, x1, x2, h=1.0, batch_size=512, alpha=1.0, save_dir=None):
        norms = compute_norm(x1, x2, self.device)
        k = (1 + norms**2 / (2 * alpha))**(-alpha)
        return k

def identity_fn(uncertainty):
    return uncertainty

def log_fn(uncertainty):
    return torch.log(uncertainty + 1e-8)


class UHerding:
    def __init__(self, cfg, lSet, uSet, budgetSize, delta, clf_model, weighted=False,
                 dataObj=None, dataset=None, kernel="rbf", device="cuda",
                 batch_size=128, permute=True, val_dataloader=None,
                 train_dataloader=None):
        self.cfg = cfg
        self.ds_name = self.cfg['DATASET']['NAME']
        self.seed = self.cfg['RNG_SEED']
        self.weighted = weighted
        self.clf_model = clf_model
        self.dataset = dataset
        self.dataObj = dataObj
        self.val_dataloader = val_dataloader
        self.train_dataloader = train_dataloader
        self.device = device 
        assert self.clf_model is not None
        assert self.dataset is not None
        self.clf_model.eval()
        self.unc_trans_fn = self.construct_uncertainty_transform_fn(self.cfg.ACTIVE_LEARNING.UNC_TRANS_FN)

        self.unc_measure = self.cfg.ACTIVE_LEARNING.SAMPLING_FN.lower().split('_')[1]

        self.normalize = self.cfg.ACTIVE_LEARNING.NORMALIZE
        self.adaptive_delta = self.cfg.ACTIVE_LEARNING.ADAPTIVE_DELTA

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
        self.save_dir = self.cfg.EXP_DIR
        self.lSet = lSet
        self.total_uSet = copy.deepcopy(uSet)
        self.budgetSize = budgetSize
        self.delta = delta

        subset_size = compute_cand_size(
            len(self.lSet), self.budgetSize, 35000)
        print(f'Subset size: {subset_size}')
        if permute:
            self.uSet = np.random.permutation(self.total_uSet)[:subset_size]
        else:
            self.uSet = self.total_uSet

        self.relevant_indices = np.concatenate([self.lSet, self.uSet]).astype(int)
        self.relevant_features = torch.from_numpy(
            self.all_features[self.relevant_indices])

        if self.adaptive_delta and len(self.lSet) > 0:
            self.lSet_features = self.relevant_features[:len(self.lSet)]
            dist_matrix = compute_norm(self.lSet_features, self.lSet_features, self.device, batch_size=512)
            dist_tril = torch.tril(dist_matrix, diagonal=-1)
            min_dist = dist_tril[dist_tril > 0].min().item()
            delta_scale = 1.0
            print(f'delta scale: {delta_scale}')
            self.delta = min_dist * delta_scale
            del dist_matrix
            print(f'delta: {self.delta}')

        self.kernel_fn = self.construct_kernel_fn(kernel_name=kernel)

        kernel_all = self.kernel_fn.compute_kernel(
            self.relevant_features, self.relevant_features, self.delta,
            batch_size=self.batch_size, save_dir=None)
        self.kernel_all = kernel_all.to(self.device) # (l+u) x (l+u)
        print(f"Memory size of kernel_all: {self.kernel_all.element_size() * self.kernel_all.nelement()}")


        if len(self.lSet) > 0:
            self.kernel_la = self.kernel_fn.compute_kernel(
                self.relevant_features[:len(self.lSet)], self.relevant_features, self.delta,
                batch_size=self.batch_size, save_dir=None).to(self.device)
            print(f"Memory size of kernel_la: {self.kernel_la.element_size() * self.kernel_la.nelement()}")

        torch.cuda.empty_cache()

        self.coverage_path = self.cfg.COVERAGE_PATH  
        self.uncertainty_path = self.cfg.UNCERTAINTY_PATH 
        self.indices_path = self.cfg.INDICES_PATH 
        self.logit_path = self.cfg.LOGIT_PATH 


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

    def save_results(self, path, arr):
        if os.path.exists(path):
            with open(path, "rb") as f:
                existing_list = pickle.load(f)
        else:
            existing_list = []
        existing_list.append(arr)

        with open(path, "wb") as f:
            pickle.dump(existing_list, f)

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

    def construct_uncertainty_transform_fn(self, fn_name):
        if fn_name == 'identity':
            fn = identity_fn
        elif fn_name == "log":
            fn = log_fn
        else:
            raise NotImplementedError(f"{fn_name} not implemented")
        print(f'Constructed transformation fn: {fn_name}')
        return fn

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
        start_time = time.time()
        self.clf_model.to(self.device)

        aSetLoader = self.dataObj.getSequentialDataLoader(
            indexes=self.relevant_indices, batch_size=128, data=self.dataset) # include all N = L + U
        aSetLoader.dataset.no_aug = True

        if len(self.lSet) <= 0:
            orig_uncertainties = torch.ones(1, len(self.relevant_indices)).float().to(self.device)
            print(f'Init lSet = 0 so init uncertainty with ones')
        else:
            best_temp = self.cfg.ACTIVE_LEARNING.TEMP

            orig_uncertainties = []
            logit_total = []
            n_aLoader = len(aSetLoader)
            print("len(uSetLoader): {}".format(n_aLoader))
            for i, (x_a, y_a) in enumerate(tqdm(aSetLoader, desc="aSet Activations")):
                with torch.no_grad():
                    x_a = x_a.cuda(0)
                    y_a = y_a.cuda(0)
                    logits = self.clf_model(x_a, y_a)['preds'] # (B, k)
                    logits = logits / best_temp

                    temp_u_rank = torch.nn.functional.softmax(logits, dim=1)

                    if self.unc_measure == 'entropy':
                        uncertainty = -1.0 * torch.sum(temp_u_rank * torch.log(temp_u_rank + 1e-8), dim=1)
                    elif self.unc_measure == 'margin':
                        batch_size = temp_u_rank.shape[0]
                        topk_indices = torch.topk(temp_u_rank, k=2, dim=-1, largest=True).indices
                        top1_probs = temp_u_rank[torch.arange(batch_size), topk_indices[:,0]]
                        top2_probs = temp_u_rank[torch.arange(batch_size), topk_indices[:,1]]
                        uncertainty = 1.0 - (top1_probs - top2_probs)
                    elif self.unc_measure == 'conf':
                        batch_size = temp_u_rank.shape[0]
                        topk_indices = torch.topk(temp_u_rank, k=2, dim=-1, largest=True).indices
                        top1_probs = temp_u_rank[torch.arange(batch_size), topk_indices[:,0]]
                        uncertainty = 1.0 - top1_probs
                    else:
                        raise NotImplementedError(f'Uncertainty measure {self.unc_measure} was not specified')

                    orig_uncertainties.append(uncertainty)
                    logit_total.append(logits)

            orig_uncertainties = torch.cat(orig_uncertainties, dim=0)
            orig_uncertainties[:len(self.lSet)] = 0.
            orig_uncertainties = orig_uncertainties.reshape(1, -1)

            logit_total = torch.cat(logit_total, dim=0)

            self.save_results(self.uncertainty_path, orig_uncertainties.reshape(-1).detach().cpu().numpy())
            self.save_results(self.indices_path, self.relevant_indices.reshape(-1))
            self.save_results(self.logit_path, logit_total.detach().cpu().numpy())

        aSetLoader.dataset.no_aug = False

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

            # compute mean of max embedding
            updated_max_embedding = (self.kernel_all - max_embedding) # N x N
            updated_max_embedding[updated_max_embedding < 0] = 0.

            uncertainties = orig_uncertainties
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
                print(f'inner_lSet: {len(set(inner_lSet.numpy()))} is not equal to {num_lSet+1}'); exit()
            if len(set(inner_uSet.cpu().numpy())) != num_uSet - 1:
                print(f'inner_lSet: {len(set(inner_uSet.numpy()))} is not equal to {num_uSet+1}'); exit()
            assert len(np.intersect1d(inner_lSet.cpu().numpy(), inner_uSet.cpu().numpy())) == 0

        selected = inner_lSet[len(self.lSet):].cpu()

        assert len(selected) == self.budgetSize, 'added a different number of samples'
        activeSet = self.relevant_indices[selected].reshape(-1)
        remainSet = np.array(sorted(list(set(self.total_uSet) - set(activeSet))))

        print(f'Finished the selection of {len(activeSet)} samples.')
        print(f'Active set is {activeSet}')
        print(f'Time: {np.round(time.time() - start_time, 4)}sec')

        return activeSet, remainSet
