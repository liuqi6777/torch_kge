import torch
import torch.nn as nn

from typing import Dict

from torch_kge.arguments import DataTrainingArguments, KGEArguments
from torch_kge.models import BaseModel
from torch_kge.utils import filter_scores


class KGEModel(nn.Module):
    def __init__(self,
                 data_args: DataTrainingArguments,
                 model_args: KGEArguments,
                 score_fn: BaseModel,
                 loss_fn: nn.Module
                 ):
        super().__init__()

        self.data_args: DataTrainingArguments = data_args
        self.model_args: KGEArguments = model_args
        
        self.score_fn = score_fn
        self.loss_fn = loss_fn

        assert self.data_args.num_entities > 0
        self.ent_embeddings = nn.Embedding(
            self.data_args.num_entities, self.model_args.dim_ent)
        self.rel_embeddings = nn.Embedding(
            self.data_args.num_relations, self.model_args.dim_rel)

        if self.model_args.margin == None or self.model_args.epsilon == None:
            nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
            nn.init.xavier_uniform_(self.rel_embeddings.weight.data)
        else:
            self.ent_embedding_range = nn.Parameter(
                torch.Tensor(
                    [(self.model_args.margin + self.model_args.epsilon) / self.model_args.dim_ent]),
                requires_grad=False
            )
            nn.init.uniform_(
                tensor=self.ent_embeddings.weight.data,
                a=-self.ent_embedding_range.item(),
                b=self.ent_embedding_range.item()
            )

            self.rel_embedding_range = nn.Parameter(
                torch.Tensor(
                    [(self.model_args.margin + self.model_args.epsilon) / self.model_args.dim_rel]),
                requires_grad=False
            )
            nn.init.uniform_(
                tensor=self.rel_embeddings.weight.data,
                a=-self.rel_embedding_range.item(),
                b=self.rel_embedding_range.item()
            )

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        label_ids = torch.vstack([inputs['head'], inputs['tail']])  # (2 * B, 1)
        head = self.ent_embeddings(inputs['head'])  # (B, 1, h)
        rel = self.rel_embeddings(inputs['rel'])
        tail = self.ent_embeddings(inputs['tail'])
        if 'head_negs' in inputs.keys() or 'tail_negs' in inputs.keys():
            head_negs = self.ent_embeddings(inputs['head_negs'])  # (B, num_h_negs, h)
            tail_negs = self.ent_embeddings(inputs['tail_negs'])
            # print(head_negs.shape)
            pos_scores = self.score_fn(head, rel, tail)  # (B, 1)
            # input('{} continue?'.format(pos_scores.shape))
            head_neg_scores = self.score_fn(head_negs, rel, tail, mode='head_batch')
            tail_neg_scores = self.score_fn(head, rel, tail_negs, mode='tail_batch')
            neg_scores = torch.hstack([head_neg_scores, tail_neg_scores])  # (B, num_negs)
            # input('{} continue?'.format(neg_scores))
            loss = self.loss_fn(pos_scores, neg_scores)
            if self.model_args.do_regularization:
                loss_reg = self.score_fn.regularization(head, rel, tail)
                loss += loss_reg
            return {'loss': loss}
        else:
            with torch.no_grad():
                # print(self.ent_embeddings.weight.shape)
                all_entities = torch.stack([self.ent_embeddings.weight] * head.size(0), dim=0)
                # print(all_entities.shape)
                # print(head.shape)
                right_pred_scores = self.score_fn(head, rel, all_entities, mode='tail_batch')  # (B, N)
                left_pred_scores = self.score_fn(all_entities, rel, tail, mode='head_batch')
                if self.model_args.do_filter:
                    # TODO
                    pass
                scores = torch.vstack([right_pred_scores, left_pred_scores])  # (2 * B, N)
                result = {
                    'scores': scores,
                    'label_ids': label_ids
                }
            return result
