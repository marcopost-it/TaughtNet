import os
import argparse
import dataclasses
from transformers import TrainingArguments

import sys
sys.path.insert(0,"/content/drive/MyDrive/Multi-NER-KG")
sys.path.insert(0,"/content/drive/MyDrive/Multi-NER-KG/src")
from arguments import ModelArguments, DataTrainingArguments

from data_handling.DataHandlers import NERDataHandler

from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score

# General
import logging
import os
import sys
from importlib import import_module
from typing import Dict, List, Optional, Tuple
import numpy as np
import numpy as np
from functools import reduce
import operator
from scipy.special import softmax

# Transformers
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    Trainer,
    set_seed,
	EvalPrediction
)

from torch import nn

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


"""
  input 
    prob: matrice NxMAX_LENx(3k) organizzata come [B1,I1,O1,B2,I2,O2,...,Bk,Ik,Ok]
  output
    matrice NxMAX_LENx(2k+1) organizzata come [B1,I1,B2,I2,...,Bk,Ik,O]
"""

def aggregate_proba(prob):
  k = int(prob.shape[-1] / 3)
  B_ids = np.array(range(0,3*k,3))
  I_ids = np.array(range(1,3*k,3))
  O_ids = np.array(range(2,3*k,3))

  result = np.empty((prob.shape[0], prob.shape[1], 2*k+1))
  for entity in range(k):
    result[:,:,entity*2] = np.prod(
        np.array([
            prob[:,:,B_ids[entity]],
            np.prod(
                np.array([np.sum(prob[:,:,reduce(operator.concat, [[I_ids[i]], [O_ids[i]]])], axis = -1) for i in range(len(O_ids)) if i != entity]),
                axis = 0
            )
        ]), axis = 0
    )
    result[:,:,entity*2+1] = np.prod(
        np.array([
            prob[:,:,I_ids[entity]],
            np.prod(
                np.array([np.sum(prob[:,:,reduce(operator.concat, [[B_ids[i]], [O_ids[i]]])], axis = -1) for i in range(len(O_ids)) if i != entity]),
                axis = 0
            )
        ]), axis = 0
    )

  result[:,:,-1] = np.prod(prob[:,:,O_ids], axis=-1)
  result = softmax(result, axis=-1)
  return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--teachers_dir', type=str, default='models/Teachers')
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

    predictions = {} # it will be filled with teachers' predictions
    teachers_folders = os.listdir(args.teachers_dir)

    for teacher in teachers_folders:
        print("Obtaining predictions of teacher: ", teacher)

        global_data_handler = NERDataHandler(tokenizer)
        labels = global_data_handler.get_labels(data_args.labels)
        global label_map
        label_map = {i: label for i, label in enumerate(labels)}
        num_labels = len(labels)

        config = AutoConfig.from_pretrained(
            os.path.join(args.teachers_dir, teacher),
            num_labels=num_labels,
            id2label=label_map,
            label2id={label: i for i, label in enumerate(labels)},
            cache_dir=model_args.cache_dir,
        )
        model = AutoModelForTokenClassification.from_pretrained(
            os.path.join(args.teachers_dir, teacher),
            from_tf=bool(".ckpt" in os.path.join(args.teachers_dir, teacher)),
            config=config,
            cache_dir=model_args.cache_dir,
        )
        trainer = Trainer(
            model=model,
        )
        # setting test dataset
        global_data_handler.set_dataset(
            data_dir=os.path.join(data_args.data_dir, 'GLOBAL', teacher),
            labels=labels,
            model_type=config.model_type,
            max_seq_length=data_args.max_seq_length,
            overwrite_cache=data_args.overwrite_cache,
            mode="train_dev",
            save_cache=False
        )
        predictions[teacher], _, _ = trainer.predict(global_data_handler.datasets['train_dev'])
        predictions[teacher] = softmax(predictions[teacher], axis = 2)

    print("Aggregating distributions...")
    teachers_predictions = np.concatenate([predictions[teacher] for teacher in teachers_folders], axis = -1)
    teachers_predictions = aggregate_proba(teachers_predictions)

    np.save(os.path.join(args.data_dir, 'GLOBAL', 'Student', 'teachers_predictions.npy'), teachers_predictions)

if __name__ == '__main__':
    main()