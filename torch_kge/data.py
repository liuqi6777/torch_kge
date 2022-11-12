import torch
import numpy as np

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict
from torch.utils.data import Dataset

from torch_kge.arguments import DataTrainingArguments, KGETrainingArguments


def load_data(data_args: DataTrainingArguments, split: str, return_triplets_only=True
              ) -> Tuple[List[List[int]], Optional[Dict[str, int]], Optional[Dict[str, int]]]:
    if split == 'train':
        datapath = data_args.train_path
    elif split == 'eval':
        datapath = data_args.eval_path
    elif split == 'test':
        datapath = data_args.test_path
    else:
        raise ValueError("Dataset split should be `train`, `eval` or `test`.")
    with open(datapath, 'r', encoding='utf-8') as f:
        f.readline()
        triplets = [[int(x) for x in line.split()] for line in f]
    ent2id, rel2id = None, None
    if not return_triplets_only:
        with open(data_args.ent2id_path, 'r', encoding='utf-8') as f:
            f.readline()
            ent2id = {x.split()[0]: int(x.split()[1]) for x in f}
        with open(data_args.rel2id_path, 'r', encoding='utf-8') as f:
            f.readline()
            rel2id = {x.split()[0]: int(x.split()[1]) for x in f}
    if return_triplets_only:
        return triplets
    return triplets, ent2id, rel2id


class KGTrainCollator:
    
    def __init__(self, train_args: KGETrainingArguments, sampler):
        self.train_args = train_args
        self.sampler = sampler
        
        strategies = ['uniform']
        if self.train_args.neg_sample_strategy not in strategies:
            raise ValueError('negtive sampling strategy must be in {}'.format(strategies))
    
    def __call__(self, example) -> Dict[str, torch.Tensor]:
        head = torch.Tensor([[item[0]] for item in example]).long()
        rel = torch.Tensor([[item[2]] for item in example]).long()
        tail = torch.Tensor([[item[1]] for item in example]).long()
        inputs = {
            'head': head,  # (B, 1)
            'rel': rel,    # (B, 1)
            'tail': tail,  # (B, 1)
        }
        head_negs, tail_negs = [], []
        for h, r, t in example:
            assert isinstance(h, int)
            h_negs, t_negs = self.sampler.sampling(h, r, t, num_negs=self.train_args.num_negs)
            head_negs.append(h_negs)
            tail_negs.append(t_negs)
        head_negs = torch.vstack(head_negs).long()
        tail_negs = torch.vstack(tail_negs).long()
        inputs['head_negs'] = head_negs  # (B, num_neg)
        inputs['tail_negs'] = tail_negs
        return inputs
    

class KGETestCollator:
    def __init__(self, train_args: KGETrainingArguments):
        self.train_args = train_args
        
    def __call__(self, example) -> Dict[str, torch.Tensor]:
        head = torch.Tensor([[item[0]] for item in example]).long()
        rel = torch.Tensor([[item[2]] for item in example]).long()
        tail = torch.Tensor([[item[1]] for item in example]).long()
        inputs = {
            'head': head,  # (B, 1)
            'rel': rel,    # (B, 1)
            'tail': tail,  # (B, 1)
        }
        return inputs


class KGDataset(Dataset):
    def __init__(self, dataset, data_args: DataTrainingArguments):
        super().__init__()
        self.dataset = dataset
        self.data_args = data_args
        self.num_ents = self.data_args.num_entities
        self.num_rels = self.data_args.num_relations
        self._count_htr()
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        head, rel, tail = self.dataset[index]
        return head, rel, tail
    
    def _count_htr(self):
        self.h_of_tr = defaultdict(set)
        self.r_of_ht = defaultdict(set)
        self.t_of_hr = defaultdict(set)
        
        for h, r, t in self.dataset:
            self.h_of_tr[(t, r)].add(h)
            self.r_of_ht[(h, t)].add(r)
            self.t_of_hr[(h, r)].add(r)
