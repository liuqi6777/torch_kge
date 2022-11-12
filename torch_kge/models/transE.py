import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_kge.models.base_model import BaseModel


class TransE(BaseModel):
    
    def forward(self, h: torch.Tensor, r: torch.Tensor, t: torch.Tensor, mode: str = 'head_batch') -> torch.Tensor:
        if self.model_args.do_normalize:
            h = F.normalize(h, p=2, dim=-1)
            r = F.normalize(r, p=2, dim=-1)
            t = F.normalize(t, p=2, dim=-1)

        if mode == 'head_batch':
            score = h + (r - t)
        elif mode == 'tail_batch':
            score = (h + r) - t
        else:
            raise ValueError('Mode should be "head_batch" or "tail_batch"')

        if self.model_args.p_norm is None:
            self.model_args.p_norm = 1
            # raise ValueError(
            #     'You should give `p_norm` parameter when use TransE model')
        score = -torch.norm(score, p=self.model_args.p_norm, dim=-1)

        return score
    
    def regularization(self, head_emb: torch.Tensor, rel_emb: torch.Tensor, tail_emb: torch.Tensor) -> torch.Tensor:
        regular = (torch.mean(head_emb ** 2) + torch.mean(rel_emb ** 2) + torch.mean(tail_emb ** 2)) / 3
        return regular
