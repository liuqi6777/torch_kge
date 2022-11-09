import torch

from torch.utils.data import Dataset

from arguments import DataTrainingArguments


def load_data(data_args: DataTrainingArguments):
    pass


class KGDataset(Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        return super().__getitem__(index)
