import torch
import torch.nn as nn

from ..arguments import DataTrainingArguments, KGEArguments


class BaseModel(nn.Module):
    def __init__(self, data_args, model_args):
        super().__init__()

        self.data_args: DataTrainingArguments = data_args
        self.model_args: KGEArguments = model_args

        self.ent_embeddings = nn.Embedding(
            self.data_args.num_entities, self.model_args.dim_ent)
        self.rel_embeddings = nn.Embedding(
            self.data_args.num_relations, self.model_args.dim_rel)

        if self.model_args.margin == None or self.model_args.epsilon == None:
            nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
            nn.init.xavier_uniform_(self.rel_embeddings.weight.data)
        else:
            self.ent_embedding_range = nn.Parameter(
				torch.Tensor([(self.model_args.margin + self.model_args.epsilon) / self.model_args.dim_ent]),
                requires_grad=False
			)
            nn.init.uniform_(
                tensor=self.ent_embeddings.weight.data,
				a=-self.ent_embedding_range.item(),
				b=self.ent_embedding_range.item()
			)
            
            self.rel_embedding_range = nn.Parameter(
				torch.Tensor([(self.model_args.margin + self.model_args.epsilon) / self.model_args.dim_rel]),
                requires_grad=False
			)
            nn.init.uniform_(
				tensor=self.rel_embeddings.weight.data,
				a=-self.rel_embedding_range.item(),
				b=self.rel_embedding_range.item()
			)

        if self.model_args.margin != None:
            self.margin = nn.Parameter(torch.Tensor([self.model_args.margin]))
            self.margin.requires_grad = False
        
    def forward(self, head: torch.Tensor, rel: torch.Tensor, tail: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
