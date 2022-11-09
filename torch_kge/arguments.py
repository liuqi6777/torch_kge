import os

from dataclasses import dataclass, field
from typing import Optional


datasets = ['FB15K237', 'WN18RR', 'YAGO3-10']


@dataclass
class DataTrainingArguments:
    data_dir: str = field(default=None)
    dataset_name: str = field(default=None)
    
    train_file: str = field(default='train2id.txt')
    eval_file: str = field(default='valid2id.txt')
    test_file: str = field(default='test2id.txt')
    
    def __post_init__(self):
        if self.data_dir is None or not os.path.exists(os.path.abspath(self.data_dir)):
            raise ValueError("Make sure there is a correct data path")
        assert self.dataset_name in datasets, "Dataset must be selected from {}".format(datasets)
        dataset_dir = os.path.join(self.data_dir, self.dataset_name)
        if not os.path.exists(dataset_dir) or not os.listdir(dataset_dir):
            raise ValueError("Dataset path must exist and be not empty")
        
        with open(os.path.join(dataset_dir, 'entity2id.txt'), 'r', encoding='utf-8') as f:
            self.num_entities = eval(f.readline().rstrip())
        with open(os.path.join(dataset_dir, 'relation2id.txt'), 'r', encoding='utf-8') as f:
            self.num_relations = eval(f.readline().rstrip())
            
        self.train_path = os.path.join(dataset_dir, self.train_file)
        self.eval_path = os.path.join(dataset_dir, self.eval_file)
        self.test_path = os.path.join(dataset_dir, self.test_file)
        
        
@dataclass
class KGEArguments:
    dim_ent: int = field(default=100)
    dim_rel: int = field(default=100)
    
    p_norm: Optional[int] = field(default=None)
    margin = Optional[float] = field(default=None)
    epsilon = Optional[float] = field(default=None)
    
    do_normalize: bool = field(default=False)


@dataclass
class TrainingArguments:
    pass
