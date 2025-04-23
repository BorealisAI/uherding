# Code is originally from https://github.com/facebookresearch/pycls
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license
# You may obtain a copy of the License at
#
# https://github.com/facebookresearch/pycls/blob/main/LICENSE
#
####################################################################################

"""Functions for computing metrics."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import MiniBatchKMeans

from pycls.core.config import cfg

# Number of bytes in a megabyte
_B_IN_MB = 1024 * 1024


def topks_correct(preds, labels, ks):
    """Computes the number of top-k correct predictions for each k."""
    assert preds.size(0) == labels.size(0), \
        'Batch dim of predictions and labels must match'
    # Find the top max_k predictions for each sample
    _top_max_k_vals, top_max_k_inds = torch.topk(
        preds, max(ks), dim=1, largest=True, sorted=True
    )
    # (batch_size, max_k) -> (max_k, batch_size)
    top_max_k_inds = top_max_k_inds.t()
    # (batch_size, ) -> (max_k, batch_size)
    rep_max_k_labels = labels.view(1, -1).expand_as(top_max_k_inds)
    # (i, j) = 1 if top i-th prediction for the j-th sample is correct
    top_max_k_correct = top_max_k_inds.eq(rep_max_k_labels)
    # Compute the number of topk correct predictions for each k
    topks_correct = [
        top_max_k_correct[:k, :].reshape(-1).float().sum() for k in ks
    ]
    return topks_correct


def topk_errors(preds, labels, ks):
    """Computes the top-k error for each k."""
    num_topks_correct = topks_correct(preds, labels, ks)
    return [(1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct]


def topk_accuracies(preds, labels, ks):
    """Computes the top-k accuracy for each k."""
    num_topks_correct = topks_correct(preds, labels, ks)
    return [(x / preds.size(0)) * 100.0 for x in num_topks_correct]


def params_count(model):
    """Computes the number of parameters."""
    return np.sum([p.numel() for p in model.parameters()]).item()


def flops_count(model):
    """Computes the number of flops statically."""
    h, w = cfg.TRAIN.IM_SIZE, cfg.TRAIN.IM_SIZE
    count = 0
    for n, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            if 'se.' in n:
                count += m.in_channels * m.out_channels + m.bias.numel()
                continue
            h_out = (h + 2 * m.padding[0] - m.kernel_size[0]) // m.stride[0] + 1
            w_out = (w + 2 * m.padding[1] - m.kernel_size[1]) // m.stride[1] + 1
            count += np.prod([
                m.weight.numel(),
                h_out, w_out
            ])
            if '.proj' not in n:
                h, w = h_out, w_out
        elif isinstance(m, nn.MaxPool2d):
            h = (h + 2 * m.padding - m.kernel_size) // m.stride + 1
            w = (w + 2 * m.padding - m.kernel_size) // m.stride + 1
        elif isinstance(m, nn.Linear):
            count += m.in_features * m.out_features
    return count.item()


def gpu_mem_usage():
    """Computes the GPU memory usage for the current device (MB)."""
    mem_usage_bytes = torch.cuda.max_memory_allocated()
    return mem_usage_bytes / _B_IN_MB


def compute_dists(features):
    dists = torch.cdist(features, features, p=2)
    return dists

def compute_norm(x1, x2, device='cuda', batch_size=512, operation=None, save_dir=None):
    x1, x2 = x1.unsqueeze(0).to(device), x2.unsqueeze(0).to(device) # 1 x n x d, 1 x n' x d
    dist_matrix = []
    batch_round = x2.shape[1] // batch_size + int(x2.shape[1] % batch_size > 0)
    for i in range(batch_round):
        x2_subset = x2[:, i * batch_size: (i + 1) * batch_size]
        dist = torch.cdist(x1, x2_subset, p=2.0)
        dist_matrix.append(dist.cpu())
        del dist

    dist_matrix = torch.cat(dist_matrix, dim=-1).squeeze(0)
    return dist_matrix


def blob_purity(dists, center_idx, radius, assignments):
    label = assignments[center_idx]
    dist = dists[center_idx] # n x n -> n
    total_count = torch.sum(dist < radius)
    cond = torch.logical_and(dist < radius, assignments == label)
    pure_count = torch.sum(cond)
    if total_count == pure_count:
        return 1
    else:
        return 0

def compute_purity(feature_path, num_classes, ratio=1.0, device="cuda", normalize=True,
                   purity_threshold=0.95):
    if not isinstance(feature_path, str):
        features = feature_path
        if not isinstance(features, torch.Tensor):
            features = torch.from_numpy(features).to(device)
    elif feature_path.endswith('.pth'):
        features = torch.load(feature_path).to(device)
    else:
        features = np.load(feature_path)
        features = torch.from_numpy(features).to(device)

    num_samples = features.shape[0]
    num_subset = int(num_samples * ratio)

    indices = np.random.choice(num_samples, size=num_subset, replace=False)
    subset_features = features[indices]
    num_subset_samples = subset_features.shape[0]
    if normalize:
        subset_features= F.normalize(subset_features, dim=1)

    km = MiniBatchKMeans(n_clusters=num_classes, batch_size=5000)
    km.fit_predict(subset_features.cpu().numpy())
    assignments = torch.from_numpy(km.labels_)

    dists = compute_norm(subset_features, subset_features)

    is_first = True
    best_purity_radius = 0
    purity_rates = []
    radiuses = np.linspace(0.05, 1.0, 20)
    for r, radius in enumerate(radiuses):
        purity_count = 0
        for i in range(num_subset_samples):
            purity_count += blob_purity(dists, i, radius, assignments)
        purity_rate = purity_count / num_subset_samples
        purity_rates.append(purity_rate)
        print(np.around(radius, 2), purity_rate)

        if is_first and purity_rate < purity_threshold:
            best_purity_radius = radiuses[r-1]
            is_first = False

    print(f'best radius: {best_purity_radius}')
    return best_purity_radius

def compute_coverage(selected_x, all_x, kernel_fn):
    kernel = kernel_fn.compute_kernel(all_x, selected_x) # N x K
    max_embedding = torch.max(kernel, dim=1).values
    coverage = torch.mean(max_embedding)
    return coverage
