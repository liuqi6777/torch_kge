import torch
import torch.nn as nn

from typing import Any, Dict, List, Tuple, Optional, Union
from torch.utils.data import Dataset, DataLoader
from transformers.trainer import Trainer

from torch_kge.data import KGETestCollator


class KGETrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        return model(inputs)['loss']
    
    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        data_collator = KGETestCollator(self.args)
        data_collator = self._get_collator_with_removed_columns(data_collator, description="evaluation")
        eval_sampler = self._get_eval_sampler(eval_dataset)

        return DataLoader(
            eval_dataset,
            sampler=eval_sampler,
            batch_size=self.args.eval_batch_size,
            collate_fn=data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )
    
    def get_test_dataloader(self, test_dataset: Dataset) -> DataLoader:
        data_collator = KGETestCollator(self.args)
        data_collator = self._get_collator_with_removed_columns(data_collator, description="test")
        test_sampler = self._get_eval_sampler(test_dataset)

        # We use the same batch_size as for eval.
        return DataLoader(
            test_dataset,
            sampler=test_sampler,
            batch_size=self.args.eval_batch_size,
            collate_fn=data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )
    
    def prediction_step(self,
                        model: nn.Module,
                        inputs: Dict[str, Union[torch.Tensor, Any]],
                        prediction_loss_only: bool,
                        ignore_keys: Optional[List[str]] = None
                        ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        # _ = super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)
        outputs = model(inputs)
        return None, outputs['scores'], outputs['label_ids']
