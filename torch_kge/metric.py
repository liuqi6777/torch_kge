import torch


class Metric:
    def __init__(self):
        pass
    
    def cal_all_metric(self):
        return {
            'MRR': 0.,
            'MR': 0.,
            'Hit@1': 0.,
            'Hit@3': 0.,
            'Hit@10': 0.
        }
