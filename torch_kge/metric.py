import numpy as np

from typing import Dict, Tuple, Set
from transformers import EvalPrediction


def metric(p: EvalPrediction) -> Dict[str, float]:
    """Compute metrics.
    """
    assert p.predictions.shape[0] == p.label_ids.shape[0]
    argsort = np.argsort(-p.predictions, axis=-1)  # desc
    ranks = np.array(argsort == p.label_ids).nonzero()[1]
    metrics = {
        'MRR': (1.0 / ranks).mean(),
        'MR': ranks.mean(),
        'Hits@1': (ranks < 1).mean(),
        'Hits@3': (ranks < 3).mean(),
        'Hits@10': (ranks < 10).mean()
    }
    return metrics
