from typing import Callable

import numpy as np
import torch
import torch.nn as nn


def rejection_sampling(
        clf: Callable[[np.ndarray], np.ndarray],
        maj_dist: nn.Module,
        c: float,
        condition: torch.Tensor,
        post_map: Callable[[np.ndarray], np.ndarray]
) -> np.ndarray:
    need_sample = np.ones(len(condition), dtype=np.bool)
    samples = np.empty((len(condition), maj_dist.dim, ))
    np_cond = condition.detach().cpu().numpy()
    while True:
        idxs = np.where(need_sample)[0]
        if len(idxs) == 0:
            break
        samples_ = maj_dist.sample(condition[idxs], post_map=post_map)
        accept_log_prob = clf(np.hstack([samples_, np_cond[idxs]])) - np.log(c)
        is_accept = (accept_log_prob > np.log(np.random.uniform(0, 1, len(idxs))))
        need_sample[idxs[is_accept]] = False
        samples[idxs[is_accept]] = samples_[is_accept]
    return samples
