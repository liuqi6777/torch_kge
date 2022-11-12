import logging
import os
import sys

from transformers import (
    HfArgumentParser,
    set_seed
)

from torch_kge.arguments import DataTrainingArguments, KGEArguments, KGETrainingArguments
from torch_kge.data import load_data, KGDataset, KGTrainCollator
from torch_kge.losses import MarginLoss
from torch_kge.metric import metric
from torch_kge.models import TransE
from torch_kge.modeling import KGEModel
from torch_kge.sampler import UniformNegativeSampler
from torch_kge.trainer import KGETrainer


logger = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser(
        (KGEArguments, DataTrainingArguments, KGETrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    model_args: KGEArguments
    data_args: DataTrainingArguments
    training_args: KGETrainingArguments

    set_seed(training_args.seed)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    
    train_set = KGDataset(load_data(data_args, split='train'), data_args)
    eval_set = KGDataset(load_data(data_args, split='eval'), data_args)
    
    sampler = UniformNegativeSampler(train_set)
    data_collator = KGTrainCollator(training_args, sampler=sampler)
    
    model = KGEModel(
        data_args=data_args,
        model_args=model_args,
        score_fn=TransE(data_args, model_args),
        loss_fn=MarginLoss(margin=6.0)
    )
    
    trainer = KGETrainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=eval_set,
        data_collator=data_collator,
        compute_metrics=metric
    )
    
    if training_args.do_train:
        trainer.train()
        trainer.save_model()
        
    if training_args.do_predict:
        test_set = KGDataset(load_data(data_args, split='test'), data_args)
        results = trainer.predict(test_set)
        trainer.log_metrics('test', results.metrics)
        trainer.save_metrics('test', results.metrics)


if __name__ == '__main__':
    main()
