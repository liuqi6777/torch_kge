import torch

from typing import Dict, Tuple, Set


def filter_scores(label_ids: torch.Tensor, scores: torch.Tensor, inputs: torch.Tensor, filter_dict: Dict[Tuple[int, int], Set[int]]):
    filt_scores = scores.clone()
    for i in range(scores.shape[0]):
        true_targets = filter_dict.get((inputs[0][i].item(), inputs[1][i].item()), None)
        if true_targets is not None:
            true_targets = true_targets.copy()
            true_targets.remove(label_ids[i].item())
            if len(true_targets) > 0:
                true_targets = torch.Tensor(list(true_targets))
                filt_scores[i][true_targets] = float('-inf')
    return filt_scores