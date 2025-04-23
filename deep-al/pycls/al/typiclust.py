# Code is originally from the Typiclust (https://arxiv.org/abs/2202.02794) and ProbCover (https://arxiv.org/abs/2205.11320) implementation
# from https://github.com/avihu111/TypiClust by Avihu Dekel and Guy Hacohen which is licensed under MIT license
# You may obtain a copy of the License at
#
# https://github.com/avihu111/TypiClust/blob/main/LICENSE
#
####################################################################################

import numpy as np
import time
import copy
import pandas as pd
import faiss
from sklearn.cluster import MiniBatchKMeans, KMeans
import pycls.datasets.utils as ds_utils
from sklearn.metrics import pairwise_distances
from sklearn_extra.cluster import KMedoids


class MiniBatchKMedoids:
    def __init__(self, n_clusters=3, max_iter=100, batch_size=100, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.random_state = random_state

    def fit(self, X):
        np.random.seed(self.random_state)
        n_samples = X.shape[0]

        # Step 1: Initialize medoids randomly
        initial_medoids_idx = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.medoids_ = X[initial_medoids_idx]

        for i in range(self.max_iter):
            # Step 2: Randomly sample a minibatch from the data
            minibatch_idx = np.random.choice(n_samples, self.batch_size, replace=False)
            minibatch = X[minibatch_idx]

            # Step 3: Compute distances between minibatch points and current medoids
            distances = pairwise_distances(minibatch, self.medoids_)

            # Step 4: Assign points in minibatch to closest medoid
            labels = np.argmin(distances, axis=1)

            # Step 5: Update medoids using minibatch data
            for medoid_idx in range(self.n_clusters):
                # Points assigned to the current medoid
                cluster_points = minibatch[labels == medoid_idx]

                if len(cluster_points) == 0:
                    continue  # No points assigned to this medoid in this minibatch

                # Compute distances within the cluster
                intra_cluster_distances = pairwise_distances(cluster_points)

                # Find the point with the minimum total distance to other points in the cluster
                new_medoid_idx = np.argmin(intra_cluster_distances.sum(axis=1))
                new_medoid = cluster_points[new_medoid_idx]

                # Update the medoid
                self.medoids_[medoid_idx] = new_medoid

        return self

    def predict(self, X):
        # Compute distances between points and medoids
        distances = pairwise_distances(X, self.medoids_)
        # Assign each point to the closest medoid
        return np.argmin(distances, axis=1)

    def fit_predict(self, X):
        # Perform fit and return the predicted labels
        self.fit(X)
        labels = self.predict(X)
        self.labels_ = labels
        return labels


def get_nn(features, num_neighbors):
    # calculates nearest neighbors on GPU
    d = features.shape[1]
    features = features.astype(np.float32)
    cpu_index = faiss.IndexFlatL2(d)
    gpu_index = faiss.index_cpu_to_all_gpus(cpu_index)
    gpu_index.add(features)  # add vectors to the index
    distances, indices = gpu_index.search(features, num_neighbors + 1)
    # 0 index is the same sample, dropping it
    return distances[:, 1:], indices[:, 1:]


def get_mean_nn_dist(features, num_neighbors, return_indices=False):
    distances, indices = get_nn(features, num_neighbors)
    mean_distance = distances.mean(axis=1)
    if return_indices:
        return mean_distance, indices
    return mean_distance


def calculate_typicality(features, num_neighbors):
    mean_distance = get_mean_nn_dist(features, num_neighbors)
    # low distance to NN is high density
    typicality = 1 / (mean_distance + 1e-5)
    return typicality

def kmeans(features, num_clusters):
    if num_clusters <= 50:
        km = KMeans(n_clusters=num_clusters)
        km.fit_predict(features)
    else:
        km = MiniBatchKMeans(n_clusters=num_clusters, batch_size=5000)
        km.fit_predict(features)
    return km.labels_


class TypiClust:
    MIN_CLUSTER_SIZE = 5
    MAX_NUM_CLUSTERS = 500
    K_NN = 20

    def __init__(self, cfg, lSet, uSet, budgetSize, dataset, is_scan=False, permute=True, remove_rate=0.0):
        self.cfg = cfg
        self.ds_name = self.cfg['DATASET']['NAME']
        self.seed = self.cfg['RNG_SEED']
        self.features = None
        self.clusters = None
        self.remove_rate = remove_rate

        self.lSet = lSet
        self.total_uSet = copy.deepcopy(uSet)
        self.num_classes = self.cfg.MODEL.NUM_CLASSES

        subset_size = 150000
        if permute:
            self.uSet = np.random.permutation(self.total_uSet)[:subset_size]
        else:
            self.uSet = self.total_uSet

        self.budgetSize = budgetSize
        self.dataset = dataset

        if self.ds_name in ['CIFAR100']:
            self.MAX_NUM_CLUSTERS = 500
        if self.ds_name in ['TINYIMAGENET']:
            self.MAX_NUM_CLUSTERS = 1000
        elif self.ds_name in ['IMBALANCED_CIFAR100']:
            self.MAX_NUM_CLUSTERS = 200
        elif self.ds_name in ['IMBALANCED_TINYIMAGENET']:
            self.MAX_NUM_CLUSTERS = 400
        elif self.ds_name in ['IMBALANCED_IMAGENET100']:
            self.MAX_NUM_CLUSTERS = 400
        elif self.ds_name in ['CIFAR100'] and self.cfg.ACTIVE_LEARNING.FEATURE in ['dino']:
            self.MAX_NUM_CLUSTERS = 100
        elif self.ds_name in ['CIFAR100'] and self.cfg.ACTIVE_LEARNING.FEATURE in ['scan']:
            self.MAX_NUM_CLUSTERS = 500

        self.relevant_indices = np.concatenate([self.lSet, self.uSet]).astype(int)

        if self.remove_rate > 0.0:
            budget_per_class = max(int(len(self.uSet) * self.remove_rate / self.num_classes), 1)
            self.budget_per_class_list = [budget_per_class] * self.num_classes
            self.budgetSize = np.sum(self.budget_per_class_list)
            self.MIN_CLUSTER_SIZE = 0

        is_diffusion = self.cfg.ACTIVE_LEARNING.FEATURE == 'diffusion'
        self.init_features_and_clusters(is_scan, is_diffusion)
        self.rel_features = self.features[self.relevant_indices]

        self.true_labels = np.array([self.dataset[idx][1] for idx in self.uSet])

        self.clusters_df, self.labels, self.existing_indices = self.preprocess()
        print(f'MIN_CLUSTER_SIZE: {self.MIN_CLUSTER_SIZE}')


    def init_features_and_clusters(self, is_scan, is_diffusion):
        num_clusters = min(len(self.lSet) + self.budgetSize, self.MAX_NUM_CLUSTERS)
        print(f'Clustering into {num_clusters} clustering. Scan clustering: {is_scan}')
        if is_scan:
            fname_dict = {'CIFAR10': f'../../scan/results/cifar-10/scan/features_seed{self.seed}_clusters{num_clusters}.npy',
                          'CIFAR100': f'../../scan/results/cifar-100/scan/features_seed{self.seed}_clusters{num_clusters}.npy',
                          'TINYIMAGENET': f'../../scan/results/tiny-imagenet/scan/features_seed{self.seed}_clusters{num_clusters}.npy',
                          }
            fname = fname_dict[self.ds_name]
            self.features = np.load(fname)
            self.clusters = np.load(fname.replace('features', 'probs')).argmax(axis=-1)
        elif self.cfg.ACTIVE_LEARNING.UNC_FEATURE == 'classifier':
            raise NotImplementedError(
                f"Unc features: {self.cfg.ACTIVE_LEARNING.UNC_FEATURE} for Typiclust is not implemented")
        else:
            feature_type = self.cfg.ACTIVE_LEARNING.UNC_FEATURE
            print('==================================')
            print(f'feature_type: {feature_type}')

            self.features = ds_utils.load_features(self.ds_name, self.seed,
                                                   is_diffusion=is_diffusion, dataset=self.dataset,
                                                   feature_type=feature_type)
            self.clusters = kmeans(self.features, num_clusters=num_clusters)
        print(f'Finished clustering into {num_clusters} clusters.')
        self.num_clusters = num_clusters

    def preprocess(self):
        # using only labeled+unlabeled indices, without validation set.
        labels = np.copy(self.clusters[self.relevant_indices])
        existing_indices = np.arange(len(self.lSet))
        # counting cluster sizes and number of labeled samples per cluster
        cluster_ids, cluster_sizes = np.unique(labels, return_counts=True)
        complete_cluster_sizes = np.zeros(self.num_clusters, dtype=int)
        complete_cluster_sizes[cluster_ids] = cluster_sizes
        expected_ids = np.arange(self.num_clusters)

        if self.remove_rate > 0.0:
            cluster_labeled_counts = np.bincount(labels[existing_indices], minlength=len(expected_ids))
        else:
            cluster_labeled_counts = np.bincount(labels[existing_indices], minlength=len(cluster_ids))

        clusters_df = pd.DataFrame({
                'cluster_id': expected_ids, 'cluster_size': complete_cluster_sizes,
                'existing_count': cluster_labeled_counts, 'neg_cluster_size': -1 * complete_cluster_sizes})

        # drop too small clusters
        clusters_df = clusters_df[clusters_df.cluster_size > self.MIN_CLUSTER_SIZE]
        # sort clusters by lowest number of existing samples, and then by cluster sizes (large to small)
        clusters_df = clusters_df.sort_values(['existing_count', 'neg_cluster_size'])
        labels[existing_indices] = -1
        return clusters_df, labels, existing_indices

    def select_samples(self):
        selected = []
        i = 0
        while len(selected) < self.budgetSize:
            # labels may run out of members. In that case, we move to the next index
            row_idx = i % len(self.clusters_df)
            indices = np.array([])

            while len(indices) <= 0:
                cluster = self.clusters_df.iloc[row_idx].cluster_id
                indices = (self.labels == cluster).nonzero()[0]
                if len(indices) <= 0:
                    row_idx = (row_idx + 1) % len(self.clusters_df)

            rel_feats = self.rel_features[indices]
            # in case we have too small cluster, calculate density among half of the cluster
            typicality = calculate_typicality(rel_feats, min(self.K_NN, len(indices) // 2))
            idx = indices[typicality.argmax()]

            is_valid = True
            if self.remove_rate > 0:
                true_label = self.true_labels[idx]
                is_valid = self.budget_per_class_list[true_label] > 0

            if is_valid:
                selected.append(idx)
                self.labels[idx] = -1
                if self.remove_rate > 0:
                    self.budget_per_class_list[true_label] -= 1
            else:
                self.labels[idx] = -1
            i += 1


        selected = np.array(selected)
        assert len(selected) == self.budgetSize, 'added a different number of samples'
        assert len(np.intersect1d(selected, self.existing_indices)) == 0, 'should be new samples'
        activeSet = self.relevant_indices[selected]
        remainSet = np.array(sorted(list(set(self.total_uSet) - set(activeSet))))

        print(f'Finished the selection of {len(activeSet)} samples.')
        print(f'Active set is {activeSet}')

        return activeSet, remainSet
