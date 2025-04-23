# Code is originally from Prateek Munjal et al. (https://arxiv.org/abs/2002.09564)
# from https://github.com/PrateekMunjal/TorchAL by Prateek Munjal which is licensed under MIT license

# You may obtain a copy of the License at
#
# https://github.com/PrateekMunjal/TorchAL/blob/master/LICENSE
#
####################################################################################

from torch.utils.data.sampler import Sampler
class IndexedSequentialSampler(Sampler):
    r"""Samples elements sequentially, always in the same order.

    Arguments:
        data_idxes (Dataset indexes): dataset indexes to sample from
    """

    def __init__(self, data_idxes, isDebug=False):
        if isDebug: print("========= my custom squential sampler =========")
        self.data_idxes = data_idxes

    def __iter__(self):
        return (self.data_idxes[int(i)] for i in range(len(self.data_idxes)))

    def __len__(self):
        return len(self.data_idxes)
