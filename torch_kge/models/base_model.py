import torch
import torch.nn as nn

from torch_kge.arguments import DataTrainingArguments, KGEArguments


class BaseModel(nn.Module):
    def __init__(self, data_args, model_args):
        super().__init__()

        self.data_args: DataTrainingArguments = data_args
        self.model_args: KGEArguments = model_args
        
    def forward(self, head: torch.Tensor, rel: torch.Tensor, tail: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    
    def regularization(self, head_emb: torch.Tensor, rel_emb: torch.Tensor, tail_emb: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
