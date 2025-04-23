# Copyright (c) 2025-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

def project_into_probability_simplex(y, k=10):
    batch_size = y.shape[0]
    u = torch.sort(y, dim=-1, descending=True)[0]
    u_cumsum = torch.cumsum(u, dim=-1)
    rho_helper = u + (1. / (torch.arange(k).to(y.device) + 1)) * (1 - u_cumsum)
    rho = k - torch.argmax(torch.sort((rho_helper > 0).float(), dim=-1)[0], dim=-1)
    lmbda = (1. / rho) * (1 - u_cumsum[torch.arange(batch_size).to(y.device), rho-1])

    probs = torch.maximum(torch.zeros(batch_size, 1).to(y.device), y + lmbda.unsqueeze(-1))
    return probs
