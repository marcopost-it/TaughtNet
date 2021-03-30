import argparse
from src.arguments import ModelArguments, DataTrainingArguments

# General
import logging
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
from torch import nn
from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score

from transformers import TrainerCallback, TrainerState, TrainerControl
from src.data_handling.DataHandlers import MultiNERDataHandler
import os


class EvaluateCallback(TrainerCallback):

    def __init__(self, model_path, labels, data_dir):

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            cache_dir=None,
            use_fast=False,
        )

        teacher_data_handler = MultiNERDataHandler(self.tokenizer)

        self.teacher_sets = {}

        self.labels = labels

        teachers = os.listdir('data')
        teachers.remove("GLOBAL")

        for teacher in teachers:
            teacher_data_handler.set_dataset(
                data_dir= os.path.join('data', teacher),
                labels=self.labels,
                model_type='bert',
                max_seq_length=128,
                overwrite_cache=False,
                mode="test",
                save_cache=False
            )
            self.teacher_sets[teacher] = teacher_data_handler.datasets['test']

        self.best_f1 = 0.844281

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        model_path = args.output_dir + "/checkpoint-" + str(956 * int(state.epoch))

        global label_map
        label_map = {i: label for i, label in enumerate(self.labels)}
        num_labels = len(self.labels)

        config = AutoConfig.from_pretrained(
            model_path,
            num_labels=num_labels,
            id2label=label_map,
            label2id={label: i for i, label in enumerate(self.labels)},
            cache_dir=None,
        )
        model = AutoModelForTokenClassification.from_pretrained(
            model_path,
            from_tf=False,
            config=config,
            cache_dir=None,
        )

        trainer = Trainer(
            model=model,
        )

        m1, m2, m3 = student_performance(trainer, self.teacher_sets)

        results = str(m1['precision']) + ", " + str(m1['recall']) + ", " + str(m1['f1']) + ", " + str(
            m2['precision']) + ", " + str(m2['recall']) + ", " + str(m2['f1']) + ", " + str(
            m3['precision']) + ", " + str(m3['recall']) + ", " + str(m3['f1']) + "\n"
        f = open(args.output_dir + "/results.csv", "a")
        f.write(results)
        f.close()

        print(results)

        if (m1['f1'] + m3['f1']) / 2 <= self.best_f1:
            delete_filename = model_path + "/pytorch_model.bin"
            open(delete_filename, 'w').close()
            os.remove(delete_filename)

            delete_filename = model_path + "/optimizer.pt"
            open(delete_filename, 'w').close()
            os.remove(delete_filename)
            print("deleted")
        else:
            self.best_f1 = (m1['f1'] + m3['f1']) / 2


def student_metrics(predictions, label_ids, student_to_teacher_map):
    label_map = {i: label for i, label in enumerate(['B', 'I', 'O'])}
    preds = np.argmax(predictions, axis=2)
    batch_size, seq_len = preds.shape
    out_label_list = [[] for _ in range(batch_size)]
    preds_list = [[] for _ in range(batch_size)]
    for i in range(batch_size):
        for j in range(seq_len):
            if label_ids[i, j] != -100:
                out_label_list[i].append(label_map[label_ids[i][j]])
                preds_list[i].append(label_map[student_to_teacher_map[preds[i][j]]])

    metrics = {
        "accuracy_score": accuracy_score(out_label_list, preds_list),
        "precision": precision_score(out_label_list, preds_list),
        "recall": recall_score(out_label_list, preds_list),
        "f1": f1_score(out_label_list, preds_list),
    }
    return metrics


def student_performance(trainer, teacher_sets):
    predictions, label_ids, _ = trainer.predict(teacher_sets['NCBI-disease'])
    student_to_teacher_map = {0: 2, 1: 2, 2: 2, 3: 2, 4: 0, 5: 1, 6: 2}
    metrics_disease = student_metrics(predictions, label_ids, student_to_teacher_map)
    predictions, label_ids, _ = trainer.predict(teacher_sets['BC5CDR-chem'])
    student_to_teacher_map = {0: 2, 1: 2, 2: 0, 3: 1, 4: 2, 5: 2, 6: 2}
    metrics_drugchem = student_metrics(predictions, label_ids, student_to_teacher_map)
    predictions, label_ids, _ = trainer.predict(teacher_sets['BC2GM'])
    student_to_teacher_map = {0: 0, 1: 1, 2: 2, 3: 2, 4: 2, 5: 2, 6: 2}
    metrics_geneprot = student_metrics(predictions, label_ids, student_to_teacher_map)
    return metrics_disease, metrics_drugchem, metrics_geneprot


from transformers import Trainer
import torch
import torch.nn.functional as F


class KGTrainer(Trainer):
    def compute_loss(self, model, inputs):
        labels = inputs.pop("labels")
        teachers_proba = inputs.pop("teachers_proba")
        outputs = model(**inputs)
        logits = outputs['logits']
        return loss(logits.float(), labels, teachers_proba.float())


def loss(logits, labels, teachers_proba):
    lbd = 0.99

    mask_ner = (labels != -100) & (labels != 6)
    loss_ner = F.nll_loss(torch.log(F.softmax(logits[mask_ner], dim=-1)), labels[mask_ner])

    mask_kd = (labels != -100)
    loss_kd = nn.KLDivLoss()(torch.log(F.softmax(logits[mask_kd], dim=-1)), teachers_proba[mask_kd])

    return (1. - lbd) * loss_ner + lbd * loss_kd


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

    data_handler = MultiNERDataHandler(tokenizer)

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

    model = AutoModelForTokenClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
    )

    trainer = KGTrainer(
        model=model,
        args=training_args,
        train_dataset=data_handler.datasets['train_dev'],
    )

    trainer.add_callback(EvaluateCallback(model_args.model_name_or_path, labels, data_args.data_dir))

    trainer.train(model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None)
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)

if __name__ == '__main__':
    main()