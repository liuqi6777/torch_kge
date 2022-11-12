import torch
import torch.nn as nn
import torch.nn.functional as F


class KGELoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
    def forward(self, pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> torch.Tensor:
        """Compute loss.

        Args:
            pos_scores: (B, 1)
            neg_scores: (B, num_negs)
        """
        raise NotImplementedError


class MarginLoss(KGELoss):
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(self, pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> torch.Tensor:
        tmp = self.margin + neg_scores - pos_scores
        # print(tmp.size())
        return torch.max(tmp, torch.zeros_like(tmp)).mean()


class LogisticLoss(KGELoss):
    def __init__(self):
        super().__init__()

    def forward(self, pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> torch.Tensor:
        return super().forward(pos_scores, neg_scores)


class BinaryCrossEntropyLoss(KGELoss):
    def __init__(self):
        super().__init__()

    def forward(self, pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> torch.Tensor:
        return super().forward(pos_scores, neg_scores)