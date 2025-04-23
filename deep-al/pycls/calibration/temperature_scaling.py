# Copyright (c) 2025-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from tqdm import tqdm
import numpy as np

from pycls.calibration.metrics import ECELoss

@ torch.no_grad()
def obtain_temperature(cfg, clf_model, val_dataloader, temps, temp_values=None, verbose=True):
    # Define temperature values to search
    if temp_values is None:
        temp_values = np.arange(1.0, 20, 0.1)

    use_softmax = cfg.MODEL.LOSS_FUN != 'mse' 
    ece_criterion = ECELoss(cfg.MODEL.NUM_CLASSES, use_softmax=use_softmax).cuda()

    val_ece_dict = {temp: {'logits': [], 'labels': []} for temp in temp_values}

    for cur_iter, (inputs, labels) in enumerate(tqdm(val_dataloader, desc="val for temp")):
        inputs, labels = inputs.cuda(), labels.cuda(non_blocking=True)
        output_dict = clf_model(inputs) # (B, k)
        logits = output_dict['preds']
        for temp in temp_values:
            scaled_logits = logits / temp

            val_ece_dict[temp]['logits'].append(scaled_logits)
            val_ece_dict[temp]['labels'].append(labels)

    ece_temp_1 = np.inf
    best_ece = np.inf
    best_temperature = 1.0
    for temp in val_ece_dict:
        logits = torch.cat(val_ece_dict[temp]['logits'])
        labels = torch.cat(val_ece_dict[temp]['labels'])

        ece = ece_criterion(logits, labels).item()
        if ece < best_ece:
            best_ece = ece
            best_temperature = temp
        if temp == 1.0:
            ece_temp_1 = ece

    temps.append(best_temperature)

    window_size = 1
    if len(temps) == 1:
        moving_temp = temps[0]
    else:
        window = min(len(temps), window_size)
        moving_temp = np.mean(temps[-window:])

    if verbose:
        print("window size:", window_size)
        print("Best temperature:", best_temperature)
        print("Moving temperature:", moving_temp)
        print("Best ECE:", best_ece)
        
        print("Without temperature scaling: ")
        print("ECE:", ece_temp_1)

    return moving_temp


