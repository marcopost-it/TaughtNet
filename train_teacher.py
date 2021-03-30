import argparse
import os
from src.arguments import ModelArguments, DataTrainingArguments
from src.data_handling.DataHandlers import NERDataHandler

# General
import logging
import os
import dataclasses

# Transformers
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    Trainer,
    set_seed,
	EvalPrediction,
    TrainingArguments
)

import numpy as np
from typing import Dict, List, Tuple
from torch import nn
from transformers import EvalPrediction
from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score

def align_predictions(predictions: np.ndarray, label_ids: np.ndarray) -> Tuple[List[int], List[int]]:
	global label_map
	preds = np.argmax(predictions, axis=2)
	batch_size, seq_len = preds.shape
	out_label_list = [[] for _ in range(batch_size)]
	preds_list = [[] for _ in range(batch_size)]
	for i in range(batch_size):
		for j in range(seq_len):
			if label_ids[i, j] != nn.CrossEntropyLoss().ignore_index:
				out_label_list[i].append(label_map[label_ids[i][j]])
				preds_list[i].append(label_map[preds[i][j]])
	return preds_list, out_label_list

def compute_metrics(p: EvalPrediction) -> Dict:
    preds_list, out_label_list = align_predictions(p.predictions, p.label_ids)
    return {
        "accuracy_score": accuracy_score(out_label_list, preds_list),
        "precision": precision_score(out_label_list, preds_list),
        "recall": recall_score(out_label_list, preds_list),
        "f1": f1_score(out_label_list, preds_list),
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--max_seq_length', type=int, default=128)
    parser.add_argument('--model_name_or_path', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--logging_dir', type=str, default=None)
    parser.add_argument('--num_train_epochs', type=int, default=3)
    parser.add_argument('--per_device_train_batch_size', type=int, default=32)
    parser.add_argument('--logging_steps', type=int, default=100)
    parser.add_argument('--do_train', type=bool, default=True)
    parser.add_argument('--do_eval', type=bool, default=True)
    parser.add_argument('--do_predict', type=bool, default=True)
    parser.add_argument('--evaluation_strategy', type=str, default='epoch')
    parser.add_argument('--save_steps', type=int, default=None)
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()
    print(args)

    outputs = []
    dataclass_types = [ModelArguments, DataTrainingArguments, TrainingArguments]
    for dtype in dataclass_types:
        keys = {f.name for f in dataclasses.fields(dtype)}
        inputs = {k: v for k, v in vars(args).items() if k in keys}
        obj = dtype(**inputs)
        outputs.append(obj)

    model_args, data_args, training_args = outputs

    logger = logging.getLogger(__name__)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )

    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )

    logger.info("Training/evaluation parameters %s", training_args)

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast,
    )

    data_handler = NERDataHandler(tokenizer)

    labels = data_handler.get_labels(data_args.labels)
    global label_map
    label_map = {i: label for i, label in enumerate(labels)}
    num_labels = len(labels)

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        id2label=label_map,
        label2id={label: i for i, label in enumerate(labels)},
        cache_dir=model_args.cache_dir,
    )

    # setting training dataset
    data_handler.set_dataset(
        data_dir=data_args.data_dir,
        labels=labels,
        model_type=config.model_type,
        max_seq_length=data_args.max_seq_length,
        overwrite_cache=data_args.overwrite_cache,
        mode="train_dev"
    )

    # setting test dataset
    data_handler.set_dataset(
        data_dir=data_args.data_dir,
        labels=labels,
        model_type=config.model_type,
        max_seq_length=data_args.max_seq_length,
        overwrite_cache=data_args.overwrite_cache,
        mode="test"
    )

    model = AutoModelForTokenClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=data_handler.datasets['train_dev'],
        eval_dataset=data_handler.datasets['test'],
        compute_metrics=compute_metrics,
    )

    trainer.train(
        model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
    )

    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)

if __name__ == '__main__':
    main()