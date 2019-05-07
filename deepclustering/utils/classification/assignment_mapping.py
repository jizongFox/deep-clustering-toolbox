from typing import Dict, Tuple

import numpy as np
import torch
from sklearn.utils.linear_assignment_ import linear_assignment


def flat_acc(flat_preds, flat_targets):
    assert flat_preds.shape == flat_targets.shape
    mean = (flat_preds == flat_targets).sum().float() / flat_preds.numel()
    return mean.item()


def hungarian_match(flat_preds, flat_targets, preds_k, targets_k) -> Tuple[torch.Tensor, Dict[int, int]]:
    assert (isinstance(flat_preds, torch.Tensor) and
            isinstance(flat_targets, torch.Tensor) and
            flat_preds.is_cuda and flat_targets.is_cuda)

    assert flat_preds.shape == flat_targets.shape

    num_samples = flat_targets.shape[0]

    assert (preds_k == targets_k)  # one to one
    num_k = preds_k
    num_correct = np.zeros((num_k, num_k))

    for c1 in range(num_k):
        for c2 in range(num_k):
            # elementwise, so each sample contributes once
            votes = int(((flat_preds == c1) * (flat_targets == c2)).sum())
            num_correct[c1, c2] = votes

    # num_correct is small
    match = linear_assignment(num_samples - num_correct)

    # return as list of tuples, out_c to gt_c
    res = {}
    for out_c, gt_c in match:
        # res.append((out_c, gt_c))
        res[out_c] = gt_c

    flat_preds_reorder = torch.zeros_like(flat_preds)
    for k, v in res.items():
        flat_preds_reorder[flat_preds == k] = torch.Tensor([v])
    return flat_preds_reorder.to(flat_preds.device), res
