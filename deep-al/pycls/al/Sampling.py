# Code is originally from Prateek Munjal et al. (https://arxiv.org/abs/2002.09564)
# from https://github.com/PrateekMunjal/TorchAL by Prateek Munjal which is licensed under MIT license
# You may obtain a copy of the License at
#
# https://github.com/PrateekMunjal/TorchAL/blob/master/LICENSE
#
####################################################################################

import numpy as np
import torch
import math
import time
from tqdm import tqdm
from scipy import stats
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn

import pycls.datasets.utils as ds_utils
from pycls.utils.io import compute_cand_size
from .vaal_util import train_vae_disc


def distance(X1, X2, mu):
    Y1, Y2 = mu
    X1_vec, X1_norm_square = X1
    X2_vec, X2_norm_square = X2
    Y1_vec, Y1_norm_square = Y1
    Y2_vec, Y2_norm_square = Y2
    dist = X1_norm_square * X2_norm_square + Y1_norm_square * Y2_norm_square - 2 * (X1_vec @ Y1_vec) * (X2_vec @ Y2_vec)
    # Numerical errors may cause the distance squared to be negative.
    assert np.min(dist) / np.max(dist) > -1e-4
    dist = np.sqrt(np.clip(dist, a_min=0, a_max=None))
    return dist


def init_centers(X1, X2, chosen, chosen_list,  mu, D2):
    if len(chosen) == 0:
        ind = np.argmax(X1[1] * X2[1])
        mu = [((X1[0][ind], X1[1][ind]), (X2[0][ind], X2[1][ind]))]
        D2 = distance(X1, X2, mu[0]).ravel().astype(float)
        D2[ind] = 0
    else:
        newD = distance(X1, X2, mu[-1]).ravel().astype(float)
        D2 = np.minimum(D2, newD)
        D2[chosen_list] = 0
        Ddist = (D2 ** 2) / sum(D2 ** 2)
        customDist = stats.rv_discrete(name='custm', values=(np.arange(len(Ddist)), Ddist))
        ind = customDist.rvs(size=1)[0]
        while ind in chosen: ind = customDist.rvs(size=1)[0]
        mu.append(((X1[0][ind], X1[1][ind]), (X2[0][ind], X2[1][ind])))
    chosen.add(ind)
    chosen_list.append(ind)
    return chosen, chosen_list, mu, D2


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
        norms = compute_norm(x1, x2, self.device)
        k = torch.exp( -1.0 * (norms / h) ** 2 )
        return k

class LinearKernel(object):
    def __init__(self, device):
        self.device = device

    def compute_kernel(self, x1, x2, h=1.0, batch_size=512):
        k = torch.matmul(x1, x2.T)
        return k

def compute_grad_embed_kernel(prob_kernel_fn, feat_kernel_fn,
                              x1_probs, x1_embeddings, x2_probs, x2_embeddings,
                              h=1.0, batch_size=512, device="cuda", init=False):
    x1_prob_kernel = prob_kernel_fn.compute_kernel(
        x1_probs, x1_probs, batch_size=batch_size)
    x1_prob_norm_sq = torch.diag(x1_prob_kernel) # n1

    x1_embed_kernel = feat_kernel_fn.compute_kernel(
        x1_embeddings, x1_embeddings, batch_size=batch_size)
    x1_embed_norm_sq = torch.diag(x1_embed_kernel) # n1

    if init:
        x1_norm_sq = x1_embed_norm_sq # n1
    else:
        x1_norm_sq = x1_prob_norm_sq * x1_embed_norm_sq # n1

    x2_prob_kernel = prob_kernel_fn.compute_kernel(
        x2_probs, x2_probs, batch_size=batch_size)
    x2_prob_norm_sq = torch.diag(x2_prob_kernel) # n2

    x2_embed_kernel = feat_kernel_fn.compute_kernel(
        x2_embeddings, x2_embeddings, batch_size=batch_size)
    x2_embed_norm_sq = torch.diag(x2_embed_kernel) # n2

    if init:
        x2_norm_sq = x2_embed_norm_sq # n2
    else:
        x2_norm_sq = x2_prob_norm_sq * x2_embed_norm_sq # n2

    x1_x2_prob_kernel = prob_kernel_fn.compute_kernel(
        x1_probs, x2_probs, batch_size=batch_size) # n1 x n2
    x1_x2_embed_kernel = feat_kernel_fn.compute_kernel(
        x1_embeddings, x2_embeddings, batch_size=batch_size) # n1 x n2

    if init:
        x1_x2_kernel = x1_x2_embed_kernel # n1 x n2
    else:
        x1_x2_kernel = x1_x2_prob_kernel * x1_x2_embed_kernel # n1 x n2

    x1_x2_kernel = 1.0 + x1_x2_kernel - 0.5 * x1_norm_sq.unsqueeze(-1) - 0.5 * x2_norm_sq.unsqueeze(0)
    return x1_x2_kernel


def construct_kernel_fn(kernel_name):
    if kernel_name == "linear":
        kernel = LinearKernel('cuda')
    elif kernel_name == "rbf":
        kernel = RBFKernel('cuda')
    elif kernel_name == "tophat":
        kernel = TopHatKernel('cuda')
    else:
        raise NotImplementedError(f"{kernel_name} not implemented")
    print(f'Constructed kernel: {kernel_name}')
    return kernel


def maxherding(u_probs, u_embeddings, budgetSize, l_probs=None, l_embeddings=None,
               kernel_name='linear', is_grad_embed=True, h=1.0, init=False, device='cuda'):
    if isinstance(u_probs, (np.ndarray, np.generic)):
        u_probs = torch.from_numpy(u_probs)
    if isinstance(u_embeddings, (np.ndarray, np.generic)):
        u_embeddings = torch.from_numpy(u_embeddings)
    if isinstance(l_probs, (np.ndarray, np.generic)):
        l_probs = torch.from_numpy(l_probs)
    if isinstance(l_embeddings, (np.ndarray, np.generic)):
        l_embeddings = torch.from_numpy(l_embeddings)

    prob_kernel_fn = construct_kernel_fn('linear')
    feat_kernel_fn = construct_kernel_fn(kernel_name)

    uSet_size = u_embeddings.shape[0]
    if l_embeddings is not None and l_probs is not None:
        lSet_size = l_embeddings.shape[0]
        a_probs = torch.cat((l_probs, u_probs), dim=0)
        a_embeddings = torch.cat((l_embeddings, u_embeddings), dim=0)
    else:
        lSet_size = 0
        a_probs = u_probs
        a_embeddings = u_embeddings

    if is_grad_embed:
        kernel_all = compute_grad_embed_kernel(
            prob_kernel_fn, feat_kernel_fn, a_probs, a_embeddings,
            a_probs, a_embeddings, device=device, init=init).to(device)
    else:
        kernel_all = feat_kernel_fn.compute_kernel(a_embeddings, a_embeddings).to(device)

    if l_embeddings is not None and l_probs is not None:
        kernel_la = kernel_all[:lSet_size].clone()

    inner_lSet = torch.arange(lSet_size).to(device)
    fixed_inner_uSet = torch.arange(uSet_size + lSet_size)[len(inner_lSet):].to(device)
    inner_uSet_bool = torch.ones_like(fixed_inner_uSet).bool().to(device)
    inner_uSet = fixed_inner_uSet[inner_uSet_bool].to(device)

    if inner_lSet.shape[0] > 0:
        max_embedding = kernel_la.max(dim=0, keepdim=True).values # 1 x N
        del kernel_la
    else:
        max_embedding = torch.zeros(1, len(inner_lSet) + len(fixed_inner_uSet)).to(device) # 1 x N

    selected = []
    for i in range(budgetSize):
        num_lSet = len(inner_lSet)
        num_uSet = len(inner_uSet)

        updated_max_embedding = (kernel_all - max_embedding) # N x N
        updated_max_embedding[updated_max_embedding < 0] = 0.

        mean_max_embedding = (updated_max_embedding).mean(dim=-1) # N

        mean_max_embedding[inner_lSet] = -np.inf
        selected_index = torch.argmax(mean_max_embedding)

        # update lSet and uSet
        inner_lSet = torch.cat((inner_lSet, selected_index.view(-1)))
        inner_uSet_bool[selected_index - lSet_size] = False
        inner_uSet = fixed_inner_uSet[inner_uSet_bool]

        max_embedding = updated_max_embedding[selected_index].unsqueeze(0) + max_embedding

        if len(set(inner_lSet.cpu().numpy())) != num_lSet + 1:
            print(f'inner_lSet: {len(set(inner_lSet.numpy()))} is not equal to {num_lSet+1}')
            import IPython; IPython.embed()
        if len(set(inner_uSet.cpu().numpy())) != num_uSet - 1:
            print(f'inner_lSet: {len(set(inner_uSet.numpy()))} is not equal to {num_uSet+1}')
            import IPython; IPython.embed()
        assert len(np.intersect1d(inner_lSet.cpu().numpy(), inner_uSet.cpu().numpy())) == 0

    selected = inner_lSet[lSet_size:].cpu() # [0, uSet_size + lSet_size)
    selected = selected - lSet_size # [0, uSet_size)
    return selected


class EntropyLoss(nn.Module):
    """
    This class contains the entropy function implemented.
    """
    def __init__(self):
        super(EntropyLoss, self).__init__()

    def forward(self, x, applySoftMax=True):
        #Assuming x : [BatchSize, ]

        if applySoftMax:
            entropy = torch.nn.functional.softmax(x, dim=1)*torch.nn.functional.log_softmax(x, dim=1)
        else:
            entropy = x * torch.log2(x)
        entropy = -1*entropy.sum(dim=1)
        return entropy


class CoreSetMIPSampling():
    """
    Implements coreset MIP sampling operation
    """
    def __init__(self, cfg, dataObj, lSet, uSet, dataset, budgetSize, clf_model=None, isMIP = False, device='cuda'):
        self.dataObj = dataObj
        self.cuda_id = torch.cuda.current_device()
        self.cfg = cfg
        self.isMIP = isMIP
        self.device = device
        self.budgetSize = budgetSize

        self.lSet = lSet
        self.uSet = uSet
        self.ds_name = self.cfg['DATASET']['NAME']
        self.seed = self.cfg['RNG_SEED']
        self.relevant_indices = np.concatenate([self.lSet, self.uSet]).astype(int)
        self.all_features = ds_utils.load_features(
                self.ds_name, self.seed, train=True, is_diffusion=False, dataset=dataset)
        self.relevant_features = torch.from_numpy(
            self.all_features[self.relevant_indices]).to(self.device)

        self.clf_model = clf_model

        self.lb_repr = self.relevant_features[:len(lSet)]
        self.ul_repr = self.relevant_features[len(lSet):]

    @torch.no_grad()
    def get_representation(self, clf_model, idx_set, dataset):

        clf_model.cuda(self.cuda_id)
        tempIdxSetLoader = self.dataObj.getSequentialDataLoader(indexes=idx_set, batch_size=int(self.cfg.TRAIN.BATCH_SIZE/self.cfg.NUM_GPUS), data=dataset)
        features = []

        print(f"len(dataLoader): {len(tempIdxSetLoader)}")

        for i, (x, _) in enumerate(tqdm(tempIdxSetLoader, desc="Extracting Representations")):
            with torch.no_grad():
                x = x.cuda(self.cuda_id)
                x = x.type(torch.cuda.FloatTensor)
                temp_z = clf_model(x)['features']
                features.append(temp_z.cpu().numpy())

        features = np.concatenate(features, axis=0)
        return features

    def gpu_compute_dists(self,M1,M2):
        """
        Computes L2 norm square on gpu
        Assume
        M1: M x D matrix
        M2: N x D matrix

        output: M x N matrix
        """
        M1_norm = (M1**2).sum(1).reshape(-1,1)

        M2_t = torch.transpose(M2, 0, 1)
        M2_norm = (M2**2).sum(1).reshape(1,-1)
        dists = M1_norm + M2_norm - 2.0 * torch.mm(M1, M2_t)
        return dists

    def compute_dists(self, X, X_train):
        dists = -2 * np.dot(X, X_train.T) + np.sum(X_train**2,axis=1) + np.sum(X**2, axis=1).reshape((-1,1))
        return dists

    def optimal_greedy_k_center(self, labeled, unlabeled):
        n_lSet = labeled.shape[0]
        lSetIds = np.arange(n_lSet)
        n_uSet = unlabeled.shape[0]

        #order is important
        features = np.vstack((labeled,unlabeled))
        print("Started computing distance matrix of {}x{}".format(features.shape[0], features.shape[0]))
        start = time.time()
        distance_mat = self.compute_dists(features, features)
        end = time.time()
        print("Distance matrix computed in {} seconds".format(end-start))
        greedy_indices = []
        for i in range(self.budgetSize):
            if i!=0 and i%500==0:
                print("Sampled {} samples".format(i))
            lab_temp_indexes = np.array(np.append(lSetIds, greedy_indices),dtype=int)
            min_dist = np.min(distance_mat[lab_temp_indexes, n_lSet:],axis=0)
            active_index = np.argmax(min_dist)
            greedy_indices.append(n_lSet + active_index)

        remainSet = set(np.arange(features.shape[0])) - set(greedy_indices) - set(lSetIds)
        remainSet = np.array(list(remainSet))
        return greedy_indices-n_lSet, remainSet

    def greedy_k_center(self, labeled, unlabeled):
        greedy_indices = [None for i in range(self.budgetSize)]
        greedy_indices_counter = 0

        print(f"[GPU] Labeled.shape: {labeled.shape}")
        print(f"[GPU] Unlabeled.shape: {unlabeled.shape}")
        # get the minimum distances between the labeled and unlabeled examples (iteratively, to avoid memory issues):
        st = time.time()
        min_dist,_ = torch.min(self.gpu_compute_dists(labeled[0,:].reshape((1,labeled.shape[1])), unlabeled), dim=0)
        min_dist = torch.reshape(min_dist, (1, min_dist.shape[0]))
        print(f"time taken: {time.time() - st} seconds")

        temp_range = 500
        dist = np.empty((temp_range, unlabeled.shape[0]))
        for j in tqdm(range(1, labeled.shape[0], temp_range), desc="Getting first farthest index"):
            if j + temp_range < labeled.shape[0]:
                dist = self.gpu_compute_dists(labeled[j:j+temp_range, :], unlabeled)
            else:
                dist = self.gpu_compute_dists(labeled[j:, :], unlabeled)

            min_dist = torch.cat((min_dist, torch.min(dist,dim=0)[0].reshape((1,min_dist.shape[1]))))

            min_dist = torch.min(min_dist, dim=0)[0]
            min_dist = torch.reshape(min_dist, (1, min_dist.shape[0]))

        # iteratively insert the farthest index and recalculate the minimum distances:
        _, farthest = torch.max(min_dist, dim=1)
        greedy_indices [greedy_indices_counter] = farthest.item()
        greedy_indices_counter += 1

        amount = self.budgetSize-1

        for i in tqdm(range(amount), desc = "Constructing Active set"):
            dist = self.gpu_compute_dists(unlabeled[greedy_indices[greedy_indices_counter-1], :].reshape((1,unlabeled.shape[1])), unlabeled)

            min_dist = torch.cat((min_dist, dist.reshape((1, min_dist.shape[1]))))

            min_dist, _ = torch.min(min_dist, dim=0)
            min_dist = min_dist.reshape((1, min_dist.shape[0]))
            _, farthest = torch.max(min_dist, dim=1)
            greedy_indices [greedy_indices_counter] = farthest.item()
            greedy_indices_counter += 1

        remainSet = set(np.arange(unlabeled.shape[0])) - set(greedy_indices)
        remainSet = np.array(list(remainSet))
        if self.isMIP:
            return greedy_indices,remainSet,math.sqrt(np.max(min_dist))
        else:
            return greedy_indices, remainSet

    def query(self):
        print("lb_repr.shape: ", self.lb_repr.shape)
        print("ul_repr.shape: ", self.ul_repr.shape)

        if self.isMIP == True:
            raise NotImplementedError
        else:
            print("Solving K Center Greedy Approach")
            start = time.time()
            greedy_indexes, remainSet = self.greedy_k_center(labeled=self.lb_repr, unlabeled=self.ul_repr)
            end = time.time()
            print("Time taken to solve K center: {} seconds".format(end-start))
            activeSet = self.uSet[greedy_indexes]
            remainSet = self.uSet[remainSet]
        return activeSet, remainSet


class Sampling:
    """
    Here we implement different sampling methods which are used to sample
    active learning points from unlabelled set.
    """

    def __init__(self, dataObj, cfg):
        self.cfg = cfg
        self.cuda_id = 0 if cfg.ACTIVE_LEARNING.SAMPLING_FN.startswith("ensemble") else torch.cuda.current_device()
        self.dataObj = dataObj

    def gpu_compute_dists(self,M1,M2):
        """
        Computes L2 norm square on gpu
        Assume
        M1: M x D matrix
        M2: N x D matrix

        output: M x N matrix
        """
        M1_norm = (M1**2).sum(1).reshape(-1,1)

        M2_t = torch.transpose(M2, 0, 1)
        M2_norm = (M2**2).sum(1).reshape(1,-1)
        dists = M1_norm + M2_norm - 2.0 * torch.mm(M1, M2_t)
        return dists

    def get_predictions(self, clf_model, idx_set, dataset):

        clf_model.cuda(self.cuda_id)
        tempIdxSetLoader = self.dataObj.getSequentialDataLoader(indexes=idx_set, batch_size=int(self.cfg.TRAIN.BATCH_SIZE/self.cfg.NUM_GPUS),data=dataset)
        tempIdxSetLoader.dataset.no_aug = True
        preds = []
        for i, (x, _) in enumerate(tqdm(tempIdxSetLoader, desc="Collecting predictions in get_predictions function")):
            with torch.no_grad():
                x = x.cuda(self.cuda_id)
                x = x.type(torch.cuda.FloatTensor)

                temp_pred = clf_model(x)['preds']

                #To get probabilities
                temp_pred = torch.nn.functional.softmax(temp_pred,dim=1)
                preds.append(temp_pred.cpu().numpy())

        preds = np.concatenate(preds, axis=0)
        tempIdxSetLoader.dataset.no_aug = False
        return preds

    def get_lSet(self, lSet, dataset):
        lSetLoader = self.dataObj.getSequentialDataLoader(indexes=lSet, batch_size=int(self.cfg.TRAIN.BATCH_SIZE), data=dataset)
        x_ls, y_ls = [], []
        for x_l, y_l in lSetLoader:
            x_ls.append(x_l)
            y_ls.append(y_l)
        x_ls = torch.cat(x_ls, dim=0)
        y_ls = torch.cat(y_ls, dim=0)
        print(f"Loaded training set: {x_ls.shape}")
        return x_ls, y_ls


    def random(self, uSet, budgetSize):
        """
        Chooses <budgetSize> number of data points randomly from uSet.

        NOTE: The returned uSet is modified such that it does not contain active datapoints.

        INPUT
        ------

        uSet: np.ndarray, It describes the index set of unlabelled set.

        budgetSize: int, The number of active data points to be chosen for active learning.

        OUTPUT
        -------

        Returns activeSet, uSet
        """

        np.random.seed(self.cfg.RNG_SEED)

        assert isinstance(uSet, np.ndarray), "Expected uSet of type np.ndarray whereas provided is dtype:{}".format(type(uSet))
        assert isinstance(budgetSize,int), "Expected budgetSize of type int whereas provided is dtype:{}".format(type(budgetSize))
        assert budgetSize > 0, "Expected a positive budgetSize"
        assert budgetSize < len(uSet), "BudgetSet cannot exceed length of unlabelled set. Length of unlabelled set: {} and budgetSize: {}"\
            .format(len(uSet), budgetSize)

        tempIdx = [i for i in range(len(uSet))]
        np.random.shuffle(tempIdx)
        activeSet = uSet[tempIdx[0:budgetSize]]
        uSet = uSet[tempIdx[budgetSize:]]
        return activeSet, uSet


    def badge(self, budgetSize, lSet, uSet, model, dataset, simclr=False,
              prime=False, herding=False, w_lset=False, normalize=True):
        init = False
        if len(lSet) <= 0:
            init = True
        print(f'init: {init}')

        ds_name = self.cfg['DATASET']['NAME']
        seed = self.cfg['RNG_SEED']
        feature_type = self.cfg.ACTIVE_LEARNING.UNC_FEATURE

        num_classes = self.cfg.MODEL.NUM_CLASSES
        assert not model.training, "Model expected in eval mode whereas currently it is in {}".format(model.training)

        if len(lSet) > 0:
            best_temp = self.cfg.ACTIVE_LEARNING.TEMP
        else:
            best_temp = 1.0

        clf = model.cuda()
        embDim = clf.fc_in_dim
        num_classes = clf.num_classes

        if herding:
            subset_size = compute_cand_size(len(lSet) * 1., budgetSize, max_size=35000)
        else:
            subset_size = 35000
        subset_uSet = np.random.permutation(uSet)[:subset_size]
        subset_size = len(subset_uSet)
        print(f'lSet size: {len(lSet)}')
        print(f'uSet size: {subset_size}')

        uSetLoader = self.dataObj.getSequentialDataLoader(indexes=subset_uSet, batch_size=256, data=dataset)
        uSetLoader.dataset.no_aug = True

        embeddings = torch.zeros([subset_size, embDim])
        probs = torch.zeros(subset_size, num_classes)
        idxs = 0
        for i, (x_u, _) in enumerate(tqdm(uSetLoader, desc="uSet Activations")):
            with torch.no_grad():
                x_u = Variable(x_u.cuda())
                output_dict = clf(x_u)
                out = output_dict['features'].cpu()
                cout = output_dict['preds'].cpu()

                batchProbs = F.softmax(cout / best_temp, dim=1)
                for j in range(len(x_u)):
                    probs[idxs] = batchProbs[j]
                    embeddings[idxs] = out[j]
                    idxs += 1

        if normalize:
            embeddings = F.normalize(embeddings, dim=-1)

        embeddings = embeddings.numpy()
        probs = probs.numpy()
        uSetLoader.dataset.no_aug = False

        if len(lSet) > 0:
            lSetLoader = self.dataObj.getSequentialDataLoader(indexes=lSet, batch_size=256, data=dataset)
            lSetLoader.dataset.no_aug = True

            l_embeddings = torch.zeros([len(lSet), embDim])
            l_probs = torch.zeros(len(lSet), num_classes)
            idxs = 0
            for i, (x_l, _) in enumerate(tqdm(lSetLoader, desc="uSet Activations")):
                with torch.no_grad():
                    x_l = Variable(x_l.cuda())
                    output_dict = clf(x_l)
                    out = output_dict['features'].cpu()
                    cout = output_dict['preds'].cpu()

                    batchProbs = F.softmax(cout / best_temp, dim=1)
                    for j in range(len(x_l)):
                        l_probs[idxs] = batchProbs[j]
                        l_embeddings[idxs] = out[j]
                        idxs += 1

            if normalize:
                l_embeddings = F.normalize(l_embeddings, dim=-1)

            l_embeddings = l_embeddings.numpy()
            l_probs = l_probs.numpy()
            lSetLoader.dataset.no_aug = False
        else:
            l_embeddings = None
            l_probs = None

        if herding or simclr:
            all_features = ds_utils.load_features(
                ds_name, seed, train=True, is_diffusion=False, feature_type=feature_type, dataset=dataset)

            embeddings = all_features[subset_uSet.astype(int)]
            l_embeddings = all_features[lSet.astype(int)]

        chosen_list = []
        if not prime:
            max_inds = np.argmax(probs, axis=-1)
            probs = -1 * probs
            probs[np.arange(subset_size), max_inds] += 1

        if self.cfg.ACTIVE_LEARNING.ADAPTIVE_DELTA and len(lSet) > 0:
            dist_matrix = compute_norm(
                torch.from_numpy(l_embeddings), torch.from_numpy(l_embeddings), device="cuda", batch_size=512)
            dist_tril = torch.tril(dist_matrix, diagonal=-1)
            h = dist_tril[dist_tril > 0].min().item()
        else:
            h = 1.0

        print(f'kernel delta: {h}')
        if herding:
            if not w_lset:
                l_probs, l_embeddings = None, None

            chosen_list = maxherding(
                probs, embeddings, budgetSize, l_probs=l_probs, l_embeddings=l_embeddings,
                kernel_name='rbf', h=h, init=init)
        else:
            mu = None
            D2 = None
            chosen = set()
            emb_norms_square = np.sum(embeddings ** 2, axis=-1)

            prob_norms_square = np.sum(probs ** 2, axis=-1)
            for _ in range(budgetSize):
                chosen, chosen_list, mu, D2 = init_centers((probs, prob_norms_square), (embeddings, emb_norms_square), chosen, chosen_list, mu, D2)


        activeSet = subset_uSet[chosen_list]
        remainSet = np.array(sorted(list(set(uSet) - set(activeSet))))
        assert len(activeSet.shape) == 1 and  len(activeSet) == budgetSize

        print(f'Finished the selection of {len(activeSet)} samples.')
        print(f'Active set is {activeSet}')

        return activeSet, remainSet


    def weighted_kmeans(self, budgetSize, lSet, total_uSet, model, dataset, herding=False):
        ds_name = self.cfg['DATASET']['NAME']
        seed = self.cfg['RNG_SEED']

        init = False
        if len(lSet) <= 0:
            init = True
        print(f'init: {init}')

        if 'DOMAINNET' in ds_name:
            beta = 50
            cand_size = 150000
        elif 'CIFAR100' in ds_name:
            beta = 100
            cand_size = 50000
        elif 'CIFAR10' in ds_name:
            beta = 1000
            cand_size = 50000
        elif 'TINYIMAGENET' in ds_name:
            beta = 50
            cand_size = 150000
        elif 'IMAGENET' in ds_name:
            beta = 20
            cand_size = 150000
        else:
            raise NotImplementedError(f'Dataset: {ds_name} is not recognized in weighted kmeans')

        if len(lSet) > 0:
            best_temp = self.cfg.ACTIVE_LEARNING.TEMP
        else:
            best_temp = 1.0

        num_classes = self.cfg.MODEL.NUM_CLASSES
        assert not model.training, "Model expected in eval mode whereas currently it is in {}".format(model.training)

        clf = model.cuda()
        embDim = clf.fc_in_dim
        num_classes = clf.num_classes
        subset_size = num_classes * beta

        uSet = np.random.permutation(total_uSet)[:cand_size]
        print(f'lSet size: {len(lSet)}')

        uSetLoader = self.dataObj.getSequentialDataLoader(indexes=uSet, batch_size=1024, data=dataset)
        uSetLoader.dataset.no_aug = True

        embeddings = torch.zeros([len(uSet), embDim])
        probs = torch.zeros(len(uSet), num_classes)
        idxs = 0
        for i, (x_u, _) in enumerate(tqdm(uSetLoader, desc="uSet Activations")):
            with torch.no_grad():
                x_u = Variable(x_u.cuda())
                output_dict = clf(x_u)
                out = output_dict['features'].cpu()
                cout = output_dict['preds'].cpu()

                batchProbs = F.softmax(cout / best_temp, dim=1)
                for j in range(len(x_u)):
                    probs[idxs] = batchProbs[j]
                    embeddings[idxs] = out[j]
                    idxs += 1

        uSetLoader.dataset.no_aug = False

        # Step 1: Use topk to find the top 2 probabilities for each sample
        top2_probs, _ = torch.topk(probs, 2, dim=1, largest=True)

        # Step 2: Compute the margin by subtracting the second highest probability from the highest probability
        margins = (1.0 - (top2_probs[:, 0] - top2_probs[:, 1]))

        # Prefilter to top beta*k informative examples
        topk_margins, topk_indices = torch.topk(
                margins, min(subset_size, len(margins)), largest=True)

        if len(lSet) > 0:
            lSetLoader = self.dataObj.getSequentialDataLoader(indexes=lSet, batch_size=1024, data=dataset)
            lSetLoader.dataset.no_aug = True

            l_embeddings = torch.zeros([len(lSet), embDim])
            l_probs = torch.zeros(len(lSet), num_classes)
            idxs = 0
            for i, (x_l, _) in enumerate(tqdm(lSetLoader, desc="lSet Activations")):
                with torch.no_grad():
                    x_l = Variable(x_l.cuda())
                    output_dict = clf(x_l)
                    out = output_dict['features'].cpu()
                    cout = output_dict['preds'].cpu()

                    batchProbs = F.softmax(cout / best_temp, dim=1)
                    for j in range(len(x_l)):
                        l_probs[idxs] = batchProbs[j]
                        l_embeddings[idxs] = out[j]
                        idxs += 1

            lSetLoader.dataset.no_aug = False
        else:
            l_embeddings = None

        feature_type = self.cfg.ACTIVE_LEARNING.UNC_FEATURE
        if herding:
            all_features = ds_utils.load_features(
                ds_name, seed, train=True, is_diffusion=False, feature_type=feature_type, dataset=dataset)

            embeddings = all_features[uSet.astype(int)][topk_indices]
            l_embeddings = all_features[lSet.astype(int)]

        if self.cfg.ACTIVE_LEARNING.ADAPTIVE_DELTA and len(lSet) > 0:
            dist_matrix = compute_norm(
                torch.from_numpy(l_embeddings), torch.from_numpy(l_embeddings), device="cuda", batch_size=512)
            dist_tril = torch.tril(dist_matrix, diagonal=-1)
            h = dist_tril[dist_tril > 0].min().item()
        else:
            h = 1.0

        total_start_time = time.time()
        if (subset_size == num_classes):
            chosen_list = topk_indices.numpy()
        else:
            start_time = time.time()
            if herding:
                chosen_list = maxherding(
                    probs, embeddings, budgetSize, l_probs=l_probs, l_embeddings=l_embeddings,
                    kernel_name='rbf', is_grad_embed=False, h=h, init=init)
            print(f'sampling took {time.time() - start_time}sec')

        activeSet = uSet[chosen_list]
        remainSet = np.array(sorted(list(set(total_uSet) - set(activeSet))))
        assert len(activeSet.shape) == 1 and  len(activeSet) == budgetSize

        print(f'Finished the selection of {len(activeSet)} samples.')
        print(f'Active set is {activeSet}')
        print(f'it took {time.time() - total_start_time}sec')

        return activeSet, remainSet


    def bald(self, budgetSize, uSet, clf_model, dataset):
        "Implements BALD acquisition function where we maximize information gain."

        clf_model.cuda(self.cuda_id)

        assert self.cfg.ACTIVE_LEARNING.DROPOUT_ITERATIONS != 0, "Expected dropout iterations > 0."

        #Set Batchnorm in eval mode whereas dropout in train mode
        clf_model.train()
        for m in clf_model.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.eval()

        uSetLoader = self.dataObj.getSequentialDataLoader(indexes=uSet, batch_size=int(self.cfg.TRAIN.BATCH_SIZE/self.cfg.NUM_GPUS),data=dataset)
        uSetLoader.dataset.no_aug = True
        n_uPts = len(uSet)

        # Heuristic: G_X - F_X
        score_All = np.zeros(shape=(n_uPts, self.cfg.MODEL.NUM_CLASSES))
        all_entropy_dropout = np.zeros(shape=(n_uPts))

        for d in tqdm(range(self.cfg.ACTIVE_LEARNING.DROPOUT_ITERATIONS), desc="Dropout Iterations"):
            dropout_score = self.get_predictions(clf_model=clf_model, idx_set=uSet, dataset=dataset)

            score_All += dropout_score

            #computing F_x
            dropout_score_log = np.log2(dropout_score+1e-6)
            Entropy_Compute = -np.multiply(dropout_score, dropout_score_log)
            Entropy_per_Dropout = np.sum(Entropy_Compute, axis=1)

            all_entropy_dropout += Entropy_per_Dropout

        Avg_Pi = np.divide(score_All, self.cfg.ACTIVE_LEARNING.DROPOUT_ITERATIONS)
        Log_Avg_Pi = np.log2(Avg_Pi+1e-6)
        Entropy_Avg_Pi = -np.multiply(Avg_Pi, Log_Avg_Pi)
        Entropy_Average_Pi = np.sum(Entropy_Avg_Pi, axis=1)
        G_X = Entropy_Average_Pi
        Average_Entropy = np.divide(all_entropy_dropout, self.cfg.ACTIVE_LEARNING.DROPOUT_ITERATIONS)
        F_X = Average_Entropy

        U_X = G_X - F_X
        print("U_X.shape: ",U_X.shape)
        sorted_idx = np.argsort(U_X)[::-1] # argsort helps to return the indices of u_scores such that their corresponding values are sorted.
        activeSet = sorted_idx[:budgetSize]

        activeSet = uSet[activeSet]
        remainSet = uSet[sorted_idx[budgetSize:]]
        # Setting task model in train mode for further learning
        clf_model.train()
        uSetLoader.dataset.no_aug = False
        return activeSet, remainSet

    def ensemble_var_R(self, budgetSize, uSet, clf_models, dataset):
        """
        Implements ensemble variance_ratio measured as the number of disagreement in committee
        with respect to the predicted class.
        If f_m is number of members agreeing to predicted class then
        variance ratio(var_r) is evaludated as follows:

            var_r = 1 - (f_m / T); where T is number of commitee members

        For more details refer equation 4 in
        http://openaccess.thecvf.com/content_cvpr_2018/papers/Beluch_The_Power_of_CVPR_2018_paper.pdf
        """
        from scipy import stats
        T = len(clf_models)

        for cmodel in clf_models:
            cmodel.cuda(self.cuda_id)
            cmodel.eval()

        uSetLoader = self.dataObj.getSequentialDataLoader(indexes=uSet, batch_size=int(self.cfg.TRAIN.BATCH_SIZE/self.cfg.NUM_GPUS),data=dataset)
        uSetLoader.dataset.no_aug = True
        print("len usetLoader: {}".format(len(uSetLoader)))

        temp_i=0
        var_r_scores = np.zeros((len(uSet),1), dtype=float)

        for k, (x_u, y_u) in enumerate(tqdm(uSetLoader, desc="uSet Forward Passes through "+str(T)+" models")):
            x_u = x_u.type(torch.cuda.FloatTensor)
            ens_preds = np.zeros((x_u.shape[0], T), dtype=float)
            for i in range(len(clf_models)):
               with torch.no_grad():
                    x_u = x_u.cuda(self.cuda_id)
                    y_u = y_u.cuda(self.cuda_id)
                    temp_op = clf_models[i](x_u, y_u)
                    _, temp_pred = torch.max(temp_op, 1)
                    temp_pred = temp_pred.cpu().numpy()
                    ens_preds[:,i] = temp_pred
            _, mode_cnt = stats.mode(ens_preds, 1)
            temp_varr = 1.0 - (mode_cnt / T*1.0)
            var_r_scores[temp_i : temp_i+x_u.shape[0]] = temp_varr

            temp_i = temp_i + x_u.shape[0]

        var_r_scores = np.squeeze(np.array(var_r_scores))
        print("var_r_scores: ")
        print(var_r_scores.shape)

        sorted_idx = np.argsort(var_r_scores)[::-1] #argsort helps to return the indices of u_scores such that their corresponding values are sorted.
        activeSet = sorted_idx[:budgetSize]

        activeSet = uSet[activeSet]
        remainSet = uSet[sorted_idx[budgetSize:]]
        uSetLoader.dataset.no_aug = False
        return activeSet, remainSet

    def confidence(self, budgetSize, lSet, uSet, model, dataset):

        """
        Implements the uncertainty principle as a acquisition function.
        """
        num_classes = self.cfg.MODEL.NUM_CLASSES
        assert model.training == False, "Model expected in eval mode whereas currently it is in {}".format(model.training)

        clf = model.cuda()
        subset_size = 10000
        subset_uSet = np.random.permutation(uSet)[:subset_size]

        u_ranks = []
        x_ls, y_ls = self.get_lSet(lSet, dataset)

        uSetLoader = self.dataObj.getSequentialDataLoader(indexes=subset_uSet, batch_size=256, data=dataset)
        uSetLoader.dataset.no_aug = True

        n_uLoader = len(uSetLoader)
        print("len(uSetLoader): {}".format(n_uLoader))
        for i, (x_u, y_u) in enumerate(tqdm(uSetLoader, desc="uSet Activations")):
            with torch.no_grad():
                x_u = x_u.cuda(0)
                y_u = y_u.cuda(0)

                try:
                    temp_u_rank = torch.nn.functional.softmax(clf(x_u, y_u)['preds'], dim=1)
                except:
                    temp_u_rank = torch.nn.functional.softmax(clf(x_ls, y_ls, x_u, return_logits=True)['preds'], dim=1)

                temp_u_rank, _ = torch.max(temp_u_rank, dim=1)
                temp_u_rank = 1 - temp_u_rank
                u_ranks.append(temp_u_rank.detach().cpu().numpy())

        u_ranks = np.concatenate(u_ranks, axis=0)
        # Now u_ranks has shape: [U_Size x 1]

        # index of u_ranks serve as key to refer in u_idx
        print(f"u_ranks.shape: {u_ranks.shape}")
        # we add -1 for reversing the sorted array
        sorted_idx = np.argsort(u_ranks)[::-1] # argsort helps to return the indices of u_ranks such that their corresponding values are sorted.
        activeSet = sorted_idx[:budgetSize]

        activeSet = subset_uSet[activeSet].astype(int)
        remainSet = np.array(sorted(list(set(uSet) - set(activeSet))))
        uSetLoader.dataset.no_aug = False
        assert len(activeSet.shape) == 1 and  len(activeSet) == budgetSize

        print(f'Finished the selection of {len(activeSet)} samples.')
        print(f'Active set is {activeSet}')


        return activeSet, remainSet


    def entropy(self, budgetSize, lSet, uSet, model, dataset, val_dataloader=None):

        """
        Implements the uncertainty principle as a acquisition function.
        """
        assert model.training == False, "Model expected in eval mode whereas currently it is in {}".format(model.training)

        clf = model.cuda()

        best_temp = 1.0
        subset_size = 10000
        subset_uSet = np.random.permutation(uSet)[:subset_size]

        u_ranks = []
        x_ls, y_ls = self.get_lSet(lSet, dataset)
        x_ls = x_ls.cuda(0)
        y_ls = y_ls.to(x_ls.device)

        uSetLoader = self.dataObj.getSequentialDataLoader(indexes=subset_uSet, batch_size=256, data=dataset)
        uSetLoader.dataset.no_aug = True

        n_uLoader = len(uSetLoader)
        print("len(uSetLoader): {}".format(n_uLoader))
        for i, (x_u, y_u) in enumerate(tqdm(uSetLoader, desc="uSet Activations")):
            with torch.no_grad():
                x_u = x_u.cuda(0)
                y_u = y_u.to(x_u.device)

                logits = clf(x_u, y_u)['preds']
                logits = logits / best_temp
                temp_u_rank = torch.nn.functional.softmax(logits, dim=1)

                temp_u_rank = temp_u_rank * torch.log(temp_u_rank + 1e-8)
                temp_u_rank = -1*torch.sum(temp_u_rank, dim=1)
                u_ranks.append(temp_u_rank.detach().cpu().numpy())

        u_ranks = np.concatenate(u_ranks, axis=0)
        # Now u_ranks has shape: [U_Size x 1]

        # index of u_ranks serve as key to refer in u_idx
        print(f"u_ranks.shape: {u_ranks.shape}")
        # we add -1 for reversing the sorted array
        sorted_idx = np.argsort(u_ranks)[::-1] # argsort helps to return the indices of u_ranks such that their corresponding values are sorted.
        activeSet = sorted_idx[:budgetSize]

        activeSet = subset_uSet[activeSet].astype(int)
        remainSet = np.array(sorted(list(set(uSet) - set(activeSet))))
        uSetLoader.dataset.no_aug = False
        assert len(activeSet.shape) == 1 and  len(activeSet) == budgetSize

        print(f'Finished the selection of {len(activeSet)} samples.')
        print(f'Active set is {activeSet}')

        return activeSet, remainSet


    def margin(self, budgetSize, lSet, uSet, model, dataset):

        """
        Implements the uncertainty principle as a acquisition function.
        """
        assert model.training == False, "Model expected in eval mode whereas currently it is in {}".format(model.training)

        clf = model.cuda()
        subset_size = 10000
        subset_uSet = np.random.permutation(uSet)[:subset_size]

        u_ranks = []
        x_ls, y_ls = self.get_lSet(lSet, dataset)

        uSetLoader = self.dataObj.getSequentialDataLoader(indexes=subset_uSet, batch_size=256, data=dataset)
        uSetLoader.dataset.no_aug = True

        n_uLoader = len(uSetLoader)
        print("len(uSetLoader): {}".format(n_uLoader))
        for i, (x_u, y_u) in enumerate(tqdm(uSetLoader, desc="uSet Activations")):
            with torch.no_grad():
                x_u = x_u.cuda(0)
                y_u = y_u.cuda(0)
                try:
                    temp_u_rank = torch.nn.functional.softmax(clf(x_u, y_u)['preds'], dim=1)
                except:
                    temp_u_rank = torch.nn.functional.softmax(clf(x_ls, y_ls, x_u,
                                                                  return_logits=True)['preds'], dim=1)

                temp_u_rank, _ = torch.sort(temp_u_rank, descending=True)
                difference = temp_u_rank[:, 0] - temp_u_rank[:, 1]
                # for code consistency across uncertainty, entropy methods i.e., picking datapoints with max value
                difference = 1.0 - difference
                u_ranks.append(difference.detach().cpu().numpy())
        u_ranks = np.concatenate(u_ranks, axis=0)
        # Now u_ranks has shape: [U_Size x 1]

        # index of u_ranks serve as key to refer in u_idx
        print(f"u_ranks.shape: {u_ranks.shape}")
        # we add -1 for reversing the sorted array
        sorted_idx = np.argsort(u_ranks)[::-1] # argsort helps to return the indices of u_ranks such that their corresponding values are sorted.
        activeSet = sorted_idx[:budgetSize]

        activeSet = subset_uSet[activeSet].astype(int)
        remainSet = np.array(sorted(list(set(uSet) - set(activeSet))))
        uSetLoader.dataset.no_aug = False
        assert len(activeSet.shape) == 1 and  len(activeSet) == budgetSize

        print(f'Finished the selection of {len(activeSet)} samples.')
        print(f'Active set is {activeSet}')

        return activeSet, remainSet


class AdversarySampler:

    def __init__(self, dataObj, cfg, budgetSize):
        self.cfg = cfg
        self.dataObj = dataObj
        self.budget = budgetSize
        self.cuda_id = torch.cuda.current_device()
        if cfg.DATASET.NAME == 'TINYIMAGENET':
            cfg.VAAL.Z_DIM = 64
            cfg.VAAL.IM_SIZE = 64
        else:
            cfg.VAAL.Z_DIM = 32
            cfg.VAAL.IM_SIZE = 32


    def compute_dists(self, X, X_train):
        dists = -2 * np.dot(X, X_train.T) + np.sum(X_train**2,axis=1) + np.sum(X**2, axis=1)[:, np.newaxis]
        return dists

    def vaal_perform_training(self, lSet, uSet, dataset, debug=False):
        oldmode = self.dataObj.eval_mode
        self.dataObj.eval_mode = True
        self.dataObj.eval_mode = oldmode

        # First train vae and disc
        vae, disc = train_vae_disc(self.cfg, lSet, uSet, dataset, self.dataObj, debug)
        uSetLoader = self.dataObj.getSequentialDataLoader(indexes=uSet, batch_size=int(self.cfg.TRAIN.BATCH_SIZE / self.cfg.NUM_GPUS) \
            ,data=dataset)

        # Do active sampling
        vae.eval()
        disc.eval()

        return vae, disc, uSetLoader

    def greedy_k_center(self, labeled, unlabeled):
        greedy_indices = []

        # get the minimum distances between the labeled and unlabeled examples (iteratively, to avoid memory issues):
        min_dist = np.min(self.compute_dists(labeled[0, :].reshape((1, labeled.shape[1])), unlabeled), axis=0)
        min_dist = min_dist.reshape((1, min_dist.shape[0]))
        temp_range = 1000
        for j in range(1, labeled.shape[0], temp_range):
            if j + temp_range < labeled.shape[0]:
                dist = self.compute_dists(labeled[j:j+temp_range, :], unlabeled)
            else:
                dist = self.compute_dists(labeled[j:, :], unlabeled)
            min_dist = np.vstack((min_dist, np.min(dist, axis=0).reshape((1, min_dist.shape[1]))))
            min_dist = np.min(min_dist, axis=0)
            min_dist = min_dist.reshape((1, min_dist.shape[0]))

        # iteratively insert the farthest index and recalculate the minimum distances:
        farthest = np.argmax(min_dist)
        greedy_indices.append(farthest)

        amount = self.cfg.ACTIVE_LEARNING.BUDGET_SIZE-1
        for i in range(amount):
            if i!=0 and i%500 == 0:
                print("{} Sampled out of {}".format(i, amount+1))
            dist = self.compute_dists(unlabeled[greedy_indices[-1], :].reshape((1,unlabeled.shape[1])), unlabeled)
            min_dist = np.vstack((min_dist, dist.reshape((1, min_dist.shape[1]))))
            min_dist = np.min(min_dist, axis=0)
            min_dist = min_dist.reshape((1, min_dist.shape[0]))
            farthest = np.argmax(min_dist)
            greedy_indices.append(farthest)

        remainSet = set(np.arange(unlabeled.shape[0])) - set(greedy_indices)
        remainSet = np.array(list(remainSet))
        return greedy_indices, remainSet


    def get_vae_activations(self, vae, dataLoader):
        acts = []
        vae.eval()

        temp_max_iter = len(dataLoader)
        print("len(dataloader): {}".format(temp_max_iter))
        temp_iter = 0
        for x,y in dataLoader:
            x = x.type(torch.cuda.FloatTensor)
            x = x.cuda(self.cuda_id)
            _, _, mu, _ = vae(x)
            acts.append(mu.cpu().numpy())
            if temp_iter%100 == 0:
                print(f"Iteration [{temp_iter}/{temp_max_iter}] Done!!")

            temp_iter += 1

        acts = np.concatenate(acts, axis=0)
        return acts


    def get_predictions(self, vae, discriminator, data, cuda):
        all_preds = []
        all_indices = []

        assert vae.training == False,"Expected vae model to be in eval mode"
        assert discriminator.training == False, "Expected discriminator model to be in eval mode"

        temp_idx = 0
        for images,_ in data:
            if cuda:
                images = images.cuda()

            with torch.no_grad():
                _, _, mu, _ = vae(images)
                preds = discriminator(mu)

            preds = preds.cpu().data
            all_preds.extend(preds)
            temp_idx += images.shape[0]

        all_indices = np.arange(temp_idx)
        all_preds = torch.stack(all_preds)
        all_preds = all_preds.view(-1)
        all_preds = all_preds.cpu().numpy()
        return all_preds


    def gpu_compute_dists(self, M1, M2):
        """
        Computes L2 norm square on gpu
        Assume
        M1: M x D matrix
        M2: N x D matrix

        output: M x N matrix
        """
        M1_norm = (M1**2).sum(1).reshape(-1,1)

        M2_t = torch.transpose(M2, 0, 1)
        M2_norm = (M2**2).sum(1).reshape(1,-1)
        dists = M1_norm + M2_norm - 2.0 * torch.mm(M1, M2_t)
        return dists


    def efficient_compute_dists(self, labeled, unlabeled):
        """
        """
        N_L = labeled.shape[0]
        N_U = unlabeled.shape[0]
        dist_matrix = None

        temp_range = 1000

        unlabeled = torch.from_numpy(unlabeled).cuda(self.cuda_id)
        temp_dist_matrix = np.empty((N_U, temp_range))
        for i in tqdm(range(0, N_L, temp_range), desc="Computing Distance Matrix"):
            end_index = i+temp_range if i+temp_range < N_L else N_L
            temp_labeled = labeled[i:end_index, :]
            temp_labeled = torch.from_numpy(temp_labeled).cuda(self.cuda_id)
            temp_dist_matrix = self.gpu_compute_dists(unlabeled, temp_labeled)
            temp_dist_matrix = torch.min(temp_dist_matrix, dim=1)[0]
            temp_dist_matrix = torch.reshape(temp_dist_matrix,(temp_dist_matrix.shape[0],1))
            if dist_matrix is None:
                dist_matrix = temp_dist_matrix
            else:
                dist_matrix = torch.cat((dist_matrix, temp_dist_matrix), dim=1)
                dist_matrix = torch.min(dist_matrix, dim=1)[0]
                dist_matrix = torch.reshape(dist_matrix,(dist_matrix.shape[0],1))

        return dist_matrix.cpu().numpy()


    @torch.no_grad()
    def vae_sample_for_labeling(self, vae, uSet, lSet, unlabeled_dataloader, lSetLoader):

        vae.eval()
        print("Computing activattions for uset....")
        u_scores = self.get_vae_activations(vae, unlabeled_dataloader)
        print("Computing activattions for lset....")
        l_scores = self.get_vae_activations(vae, lSetLoader)

        print("l_scores.shape: ",l_scores.shape)
        print("u_scores.shape: ",u_scores.shape)

        dist_matrix = self.efficient_compute_dists(l_scores, u_scores)
        print("Dist_matrix.shape: ",dist_matrix.shape)

        min_scores = np.min(dist_matrix, axis=1)
        sorted_idx = np.argsort(min_scores)[::-1]

        activeSet = uSet[sorted_idx[0:self.budget]]
        remainSet = uSet[sorted_idx[self.budget:]]

        return activeSet, remainSet


    def sample_vaal_plus(self, vae, disc_task, data, cuda):
        all_preds = []
        all_indices = []

        assert vae.training == False,"Expected vae model to be in eval mode"
        assert disc_task.training == False, "Expected disc_task model to be in eval mode"

        temp_idx = 0
        for images,_ in data:
            if cuda:
                images = images.cuda()

            with torch.no_grad():
                _, _, mu, _ = vae(images)
                preds,_ = disc_task(mu)

            preds = preds.cpu().data
            all_preds.extend(preds)
            temp_idx += images.shape[0]

        all_indices = np.arange(temp_idx)
        all_preds = torch.stack(all_preds)
        all_preds = all_preds.view(-1)
        # need to multiply by -1 to be able to use torch.topk
        all_preds *= -1

        # select the points which the discriminator things are the most likely to be unlabeled
        _, querry_indices = torch.topk(all_preds, int(self.budget))
        querry_indices = querry_indices.numpy()
        remain_indices = np.asarray(list(set(all_indices) - set(querry_indices)))
        assert len(remain_indices) + len(querry_indices) == len(all_indices)," Indices are overlapped between activeSet and uSet"
        activeSet = all_indices[querry_indices]
        uSet = all_indices[remain_indices]
        return activeSet,uSet


    def sample(self, vae, discriminator, data):
        all_preds = []
        all_indices = []

        assert vae.training == False,"Expected vae model to be in eval mode"
        assert discriminator.training == False, "Expected discriminator model to be in eval mode"

        vae.cuda(self.cuda_id)
        discriminator.cuda(self.cuda_id)

        temp_idx = 0
        for images,_ in data:
            images = images.type(torch.cuda.FloatTensor)
            images = images.cuda()

            with torch.no_grad():
                _, _, mu, _ = vae(images)
                preds = discriminator(mu)

            preds = preds.cpu().data
            all_preds.extend(preds)
            temp_idx += images.shape[0]

        all_indices = np.arange(temp_idx)
        all_preds = torch.stack(all_preds)
        all_preds = all_preds.view(-1)
        # need to multiply by -1 to be able to use torch.topk
        all_preds *= -1

        # select the points which the discriminator things are the most likely to be unlabeled
        _, querry_indices = torch.topk(all_preds, int(self.budget))
        querry_indices = querry_indices.numpy()
        remain_indices = np.asarray(list(set(all_indices) - set(querry_indices)))
        assert len(remain_indices) + len(querry_indices) == len(all_indices), " Indices are overlapped between activeSet and uSet"
        activeSet = all_indices[querry_indices]
        uSet = all_indices[remain_indices]
        return activeSet, uSet


    @torch.no_grad()
    def sample_for_labeling(self, vae, discriminator, unlabeled_dataloader, uSet):
        """
        Picks samples from uSet to form activeSet.

        INPUT
        ------
        vae: object of model VAE

        discriminator: object of model discriminator

        unlabeled_dataloader: Sequential dataloader iterating over uSet

        uSet: Collection of unlabelled datapoints

        NOTE: Please pass the unlabelled dataloader as sequential dataloader else the
        results won't be appropriate.

        OUTPUT
        -------

        Returns activeSet, [remaining]uSet
        """
        unlabeled_dataloader.dataset.no_aug = True
        activeSet, remainSet = self.sample(vae,
                                             discriminator,
                                             unlabeled_dataloader,
                                             )

        activeSet = uSet[activeSet]
        remainSet = uSet[remainSet]
        unlabeled_dataloader.dataset.no_aug = False
        return activeSet, remainSet
