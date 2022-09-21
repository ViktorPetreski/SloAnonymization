import pickle
# from NERDA.models import NERDA
# from sklearn.metrics import classification_report
# from sklearn.preprocessing import MultiLabelBinarizer
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer, IntervalStrategy
import numpy as np
from src import vp_ta_constants
from src.data_parser import parser

import utils
import torch
from transformers import AutoTokenizer
import warnings
# from datasets import load_metric
# import pandas as pd
from pathlib import Path
import logging
import os
import sys
from typing import List

warnings.filterwarnings("ignore")
# from seqeval.metrics import f1_score as seq_f1, classification_report as seq_cr
from src.ner import performance

# Metrics
# metric = load_metric("seqeval")
label_list = []
tag2id = {}
id2tag = {}

logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s'
)
logger = logging.getLogger('TrainL1OStrategy')


# def compute_metrics(p):
#     predictions, labels = p
#     predictions = np.argmax(predictions, axis=2)
#
#     # Remove ignored index (special tokens)
#     true_predictions = [
#         [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
#         for prediction, label in zip(predictions, labels)
#     ]
#     true_labels = [
#         [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
#         for prediction, label in zip(predictions, labels)
#     ]
#
#     results = metric.compute(predictions=true_predictions, references=true_labels)
#     # return results
#     # if data_args.return_entity_level_metrics:
#         # Unpack nested dictionaries
#     final_results = {}
#     for key, value in results.items():
#         if isinstance(value, dict):
#             for n, v in value.items():
#                 final_results[f"{key}_{n}"] = v
#         else:
#             final_results[key] = value
#     # final_results |= {
#     #     "precision": results["overall_precision"],
#     #     "recall": results["overall_recall"],
#     #     "f1": results["overall_f1"],
#     #     "accuracy": results["overall_accuracy"],
#     # }
#     return final_results


def eval_test(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    # labels = list(filter(lambda x: x != "O", label_list))
    removed = False
    if "O" in label_list:
        label_list.remove("O")
        removed = True
    results = performance.evaluate_performance(y_pred=true_predictions, y_true=true_labels, labels=label_list)
    if removed:
        label_list.append("O")
    results_dict = results.to_dict("records")
    print(results_dict)
    rr = {}
    # for item, value in results_dict.items():
    #     rr[]
    # return metric.compute(predictions=true_predictions, references=true_labels)
    return results


def compute_metrics(p) -> dict:
    results = eval_test(p)
    return {
        "Precision_MICRO": results["Precision"].values[-2],
        "Precision_MACRO": results["Precision"].values[-1],
        "Recall_MICRO": results["Recall"].values[-2],
        "Recall_MACRO": results["Recall"].values[-1],
        "F1_MICRO": results["F1-Score"].values[-2],
        "F1_MACRO": results["F1-Score"].values[-1],
    }


def get_configurations(task="ner") -> list[dict]:
    combined_train, combined_test, combined_dev = parser.get_combined(task)
    ssj500k_train, ssj500k_test, ssj500k_dev = parser.get_ssj500k(task)
    bsnlp_train, bsnlp_test, bsnlp_dev = parser.get_bsnlp(task)

    return [
       {
            "transformer": "../../models/sloberta-2.0",
            "name": "model_sloberta-combined-3e-32b-0ws-6e5lr",
            "data": "combined",
            "base_model_name": "sloberta",
            "hyperparameters": {
                'epochs': 3,
                'warmup_steps': 0,
                'train_batch_size': 32,
                'learning_rate': 6e-5
            },
            "train": combined_train,
            "test": combined_test,
            "dev": combined_dev,
            "tag_scheme": list(map(lambda x: x.upper(), vp_ta_constants.TAG_SCHEME_COMBINED))
        },
        {
            "transformer": "../../models/sloberta-2.0",
            "name": "model_sloberta-combined-5e_8b_500ws_5e-5lr",
            "data": "combined",
            "base_model_name": "sloberta",
            "hyperparameters": {
                'epochs': 5,
                'warmup_steps': 500,
                'train_batch_size': 8,
                'learning_rate': 5e-5
            },
            "train": combined_train,
            "test": combined_test,
            "dev": combined_dev,
            "tag_scheme": list(map(lambda x: x.upper(), vp_ta_constants.TAG_SCHEME_COMBINED))
        },
        {
            "transformer": "../../models/sloberta-2.0",
            "name": "model_sloberta-combined-8e_16b_500ws_6e-5lr",
            "data": "combined",
            "base_model_name": "sloberta",
            "hyperparameters": {
                'epochs': 8,
                'warmup_steps': 500,
                'train_batch_size': 32,
                'learning_rate': 6e-5
            },
            "train": combined_train,
            "test": combined_test,
            "dev": combined_dev,
            "tag_scheme": list(map(lambda x: x.upper(), vp_ta_constants.TAG_SCHEME_COMBINED))
        },
        # {
        #     "transformer": "../../models/sloberta-2.0",
        #     "name": "model_sloberta-combined-3e_8b_500ws_5e-5lr",
        #     "data": "combined",
        #     "base_model_name": "sloberta",
        #     "hyperparameters": {
        #         'epochs': 12,
        #         'warmup_steps': 500,
        #         'train_batch_size': 16,
        #         'learning_rate': 6e-5
        #     },
        #     "train": combined_train,
        #     "test": combined_test,
        #     "dev": combined_dev,
        #     "tag_scheme": list(map(lambda x: x.upper(), vp_ta_constants.TAG_SCHEME_COMBINED))
        # },
        # {
        #     "transformer": "../../models/sloberta-2.0",
        #     "name": "model_sloberta-combined-3e_8b_500ws_5e-5lr",
        #     "data": "combined",
        #     "base_model_name": "sloberta",
        #     "hyperparameters": {
        #         'epochs': 8,
        #         'warmup_steps': 500,
        #         'train_batch_size': 24,
        #         'learning_rate': 5e-5
        #     },
        #     "train": combined_train,
        #     "test": combined_test,
        #     "dev": combined_dev,
        #     "tag_scheme": list(map(lambda x: x.upper(), vp_ta_constants.TAG_SCHEME_COMBINED))
        # },
        # {
        #     "transformer": "../../models/sloberta-2.0",
        #     "name": "model_sloberta-combined-3e_8b_500ws_5e-5lr",
        #     "data": "combined",
        #     "base_model_name": "sloberta",
        #     "hyperparameters": {
        #         'epochs': 10,
        #         'warmup_steps': 200,
        #         'train_batch_size': 16,
        #         'learning_rate': 5e-5
        #     },
        #     "train": combined_train,
        #     "test": combined_test,
        #     "dev": combined_dev,
        #     "tag_scheme": list(map(lambda x: x.upper(), vp_ta_constants.TAG_SCHEME_COMBINED))
        # },

    ]

    # return [
    #     {
    #         "transformer": "../../models/sloberta-2.0",
    #         "name": "model_sloberta-combined-20e-32b-500ws-5e5lr",
    #         "data": "combined",
    #         "base_model_name": "sloberta",
    #         "hyperparameters": {
    #             'epochs' : 3,
    #             'warmup_steps' : 0,
    #             'train_batch_size': 8,
    #             'learning_rate': 5e-5
    #         },
    #         "train": combined_train,
    #         "test": combined_test,
    #         "dev": combined_dev,
    #         "tag_scheme": list(map(lambda x: x.upper(), vp_ta_constants.TAG_SCHEME_COMBINED))
    #     },
    #     {
    #         "transformer": "../../models/sloberta-2.0",
    #         "name": "model_sloberta-combined-20e-32b-500ws-5e5lr",
    #         "data": "combined",
    #         "base_model_name": "sloberta",
    #         "hyperparameters": {
    #             'epochs' : 5,
    #             'warmup_steps' : 0,
    #             'train_batch_size': 8,
    #             'learning_rate': 4e-5
    #         },
    #         "train": combined_train,
    #         "test": combined_test,
    #         "dev": combined_dev,
    #         "tag_scheme": list(map(lambda x: x.upper(), vp_ta_constants.TAG_SCHEME_COMBINED))
    #     },
    #     {
    #         "transformer": "../../models/sloberta-2.0",
    #         "name": "model_sloberta-combined-20e_32b_500ws_5e-5lr",
    #         "data": "combined",
    #         "base_model_name": "sloberta",
    #         "hyperparameters": {
    #             'epochs' : 8,
    #             'warmup_steps' : 0,
    #             'train_batch_size': 8,
    #             'learning_rate': 3e-5
    #         },
    #         "train": combined_train,
    #         "test": combined_test,
    #         "dev": combined_dev,
    #         "tag_scheme": list(map(lambda x: x.upper(), vp_ta_constants.TAG_SCHEME_COMBINED))
    #     },
    #     # {
    #     #     "transformer": "../../models/sloberta-2.0",
    #     #     "name": "model_sloberta-bsnlp-15e-13b-500ws-2e5lr",
    #     #     "data": "bsnlp",
    #     #     "base_model_name": "sloberta",
    #     #     "hyperparameters": {
    #     #         'epochs' : 5,
    #     #         'warmup_steps' : 0,
    #     #         'train_batch_size': 8,
    #     #         'learning_rate': 5e-5
    #     #     },
    #     #     "train": bsnlp_train,
    #     #     "test": bsnlp_test,
    #     #     "dev": bsnlp_dev,
    #     #     "tag_scheme": list(map(lambda x: x.upper(), vp_ta_constants.TAG_SCHEME_BSNLP))
    #     # },
    #     # {
    #     #     "transformer": "../../models/sloberta-2.0",
    #     #     "name": "model_sloberta-bsnlp-15e-13b-500ws-3e5lr",
    #     #     "data": "bsnlp",
    #     #     "base_model_name": "sloberta",
    #     #     "hyperparameters": {
    #     #         'epochs' : 5,
    #     #         'warmup_steps' : 0,
    #     #         'train_batch_size': 8,
    #     #         'learning_rate': 1e-5
    #     #     },
    #     #     "train": bsnlp_train,
    #     #     "test": bsnlp_test,
    #     #     "dev": bsnlp_dev,
    #     #     "tag_scheme": list(map(lambda x: x.upper(), vp_ta_constants.TAG_SCHEME_BSNLP))
    #     # },
    #     # {
    #     #     "transformer": "../../models/sloberta-2.0",
    #     #     "name": "model_sloberta-ssj500k-15e-13b-500ws-1e5lr",
    #     #     "data": "ssj500k",
    #     #     "base_model_name": "sloberta",
    #     #     "hyperparameters": {
    #     #         'epochs' : 3,
    #     #         'warmup_steps' : 0,
    #     #         'train_batch_size': 13,
    #     #         'learning_rate': 5e-5
    #     #     },
    #     #     "train": ssj500k_train,
    #     #     "test": ssj500k_test,
    #     #     "dev": ssj500k_dev,
    #     #     "tag_scheme": list(map(lambda x: x.upper(), vp_ta_constants.TAG_SCHEME_SSK))
    #     # },
    #     # {
    #     #     "transformer": "../../models/sloberta-2.0",
    #     #     "name": "model_sloberta-ssj500k-20e-32b-200ws-3e5lr",
    #     #     "data": "ssj500k",
    #     #     "base_model_name": "sloberta",
    #     #     "hyperparameters": {
    #     #         'epochs' : 5,
    #     #         'warmup_steps' : 200,
    #     #         'train_batch_size': 16,
    #     #         'learning_rate': 3e-5
    #     #     },
    #     #     "train": ssj500k_train,
    #     #     "test": ssj500k_test,
    #     #     "dev": ssj500k_dev,
    #     #     "tag_scheme": list(map(lambda x: x.upper(), vp_ta_constants.TAG_SCHEME_SSK))
    #     # },
    #     # {
    #     #     "transformer": "../../models/sloberta-2.0",
    #     #     "name": "model_sloberta-ssj500k-20e-32b-200ws-3e5lr",
    #     #     "data": "ssj500k",
    #     #     "base_model_name": "sloberta",
    #     #     "hyperparameters": {
    #     #         'epochs' : 5,
    #     #         'warmup_steps' : 500,
    #     #         'train_batch_size': 8,
    #     #         'learning_rate': 5e-5
    #     #     },
    #     #     "train": ssj500k_train,
    #     #     "test": ssj500k_test,
    #     #     "dev": ssj500k_dev,
    #     #     "tag_scheme": list(map(lambda x: x.upper(), vp_ta_constants.TAG_SCHEME_SSK))
    #     # }
    # ]


def start_fine_tune():
    for configuration in get_configurations():
        train = configuration["train"]
        test = configuration["test"]
        dev = configuration["dev"]
        transformer = configuration["transformer"]

        # model = NERDA(
        #     dataset_training=train,
        #     dataset_validation=dev,
        #     tag_scheme=configuration["tag_scheme"],
        #     tag_outside='O',
        #     transformer=transformer,
        #     dropout=0.1,
        #     max_len=150,
        #     hyperparameters=configuration["hyperparameters"],
        #     device="cuda"
        # )
        #
        # model.train()
        # model_name = configuration["name"]
        # torch.save(model, f'../../models/{model_name}.pt')
        # results = model.evaluate_performance(test)
        # print(results)
        # results.to_csv(f"../../results/ner/{model_name}.csv", index=False)


def start_pure_ft(task="ner"):
    for configuration in get_configurations(task):
        train_labels = configuration["train"]["sentences"]
        train_tags = configuration["train"]["tags"]

        test_labels = configuration["test"]["sentences"]
        test_tags = configuration["test"]["tags"]

        dev_labels = configuration["dev"]["sentences"]
        dev_tags = configuration["dev"]["tags"]
        transformer = configuration["transformer"]

        global label_list, tag2id, id2tag
        if task == "ner":
            tag_scheme = configuration["tag_scheme"]
            label_list = tag_scheme
            tag2id = {tag: str(id) for id, tag in enumerate(label_list)}
            id2tag = {int(id): tag for id, tag in enumerate(label_list)}
        else:
            all_tags = set()
            for x in train_tags:
                all_tags |= set(x)
            for x in test_tags:
                all_tags |= set(x)
            for x in dev_tags:
                all_tags |= set(x)
            # all_tags.add("-PAD-")
            tag2id = {tag: str(id) for id, tag in enumerate(all_tags)}
            id2tag = {int(id): tag for id, tag in enumerate(all_tags)}
            label_list = list(all_tags)
            # tag_scheme = list(all_tags)

        data_type = configuration["data"]

        train_encodings_path = Path(f"../../data/encodings/{data_type}/{task}/train.pickle")
        test_encodings_path = Path(f"../../data/encodings/{data_type}/{task}/test.pickle")
        dev_encodings_path = Path(f"../../data/encodings/{data_type}/{task}/dev.pickle")

        if train_encodings_path.is_file() and test_encodings_path.is_file() and dev_encodings_path.is_file():
            train_encodings = pickle.load(open(train_encodings_path, "rb"))
            # train_encodings = train_encodings[:100]
            test_encodings = pickle.load(open(test_encodings_path, "rb"))
            dev_encodings = pickle.load(open(dev_encodings_path, "rb"))
            # dev_encodings = dev_encodings[:20]
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                transformer,
                from_pt=True,
                do_lower_case=False,
            )
            logger.info("Started generating encodings for train")
            train_encodings = utils.generate_encodings(tokenizer, train_labels)
            logger.info("Started generating encodings for test")
            test_encodings = utils.generate_encodings(tokenizer, test_labels)
            logger.info("Started generating encodings for dev")
            dev_encodings = utils.generate_encodings(tokenizer, dev_labels)

            pickle.dump(train_encodings, open(train_encodings_path, "wb"))
            pickle.dump(test_encodings, open(test_encodings_path, "wb"))
            pickle.dump(dev_encodings, open(dev_encodings_path, "wb"))

        train_labels = utils.encode_tags(train_tags, train_encodings, label_list)
        test_labels = utils.encode_tags(test_tags, test_encodings, label_list)
        dev_labels = utils.encode_tags(dev_tags, dev_encodings, label_list)

        train_dataset = utils.generate_dataset(train_encodings, train_labels)
        test_dataset = utils.generate_dataset(test_encodings, test_labels)
        val_dataset = utils.generate_dataset(dev_encodings, dev_labels)

        model_name = f"custom_{task}_model_{configuration['base_model_name']}—{data_type}—{configuration['hyperparameters']['epochs']}e—" \
                     f"{configuration['hyperparameters']['train_batch_size']}b—{configuration['hyperparameters']['warmup_steps']}ws—" \
                     f"{configuration['hyperparameters']['learning_rate']}lr"

        logger.info("=====================%s====================", model_name)
        Path(f'../../logs/{task}/{model_name}').mkdir(parents=True, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=f'../../models/{task}',  # output directory
            num_train_epochs=configuration["hyperparameters"]["epochs"],  # total number of training epochs
            per_device_train_batch_size=configuration["hyperparameters"]["train_batch_size"],
            # batch size per device during training
            per_device_eval_batch_size=32,  # batch size for evaluation
            warmup_steps=configuration["hyperparameters"]["warmup_steps"],
            # number of warmup steps for learning rate scheduler
            weight_decay=0.01,  # strength of weight decay
            logging_first_step=True,
            logging_dir=f'../../logs/{task}/{model_name}_new_test',  # directory for storing logs
            logging_steps=10,
            logging_strategy=IntervalStrategy.STEPS,
            evaluation_strategy=IntervalStrategy.EPOCH,
            save_strategy=IntervalStrategy.EPOCH,
            learning_rate=configuration["hyperparameters"]["learning_rate"],
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            load_best_model_at_end=True,
            run_name=model_name,
            # dataloader_num_workers=4,
        )

        model = AutoModelForTokenClassification.from_pretrained(
            transformer,
            label2id=tag2id,
            id2label=id2tag,
            num_labels=len(label_list))

        trainer = Trainer(
            model=model,  # the instantiated 🤗 Transformers model to be trained
            args=training_args,  # training arguments, defined above
            train_dataset=train_dataset,  # training dataset
            eval_dataset=val_dataset,  # evaluation dataset
            compute_metrics=compute_metrics,
        )

        trainer.train()
        # predictions = trainer.predict(test_dataset, metric_key_prefix="", ignore_keys=["O"])
        # p = (predictions.predictions, predictions.label_ids)
        # results = eval_test(p)
        model_save_name = f"../../models/{task}/{model_name}_new_test"
        trainer.save_model(model_save_name)

        test(model_save_name, test_dataset)
        # predictions = trainer.predict(test_dataset, metric_key_prefix="", ignore_keys=["O"])
        # p = (predictions.predictions, test_dataset.labels)
        # results = eval_test(p)
        # print(results)
        # results.to_csv(f"../../results/{task}/{model_name}_new_test.csv", index=False)


def test(model_name, test_dataset):
    training_args = TrainingArguments(
        output_dir=f"../../models/ner",
        do_train=False,
        do_predict=True,
        do_eval=False,
        per_device_eval_batch_size=16,  # batch size for evaluation
        warmup_steps=0,
        weight_decay=0.01,  # strength of weight decay
        run_name=model_name,
        # dataloader_num_workers=4
    )

    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(label_list),
        output_attentions=False,
        output_hidden_states=False
    )

    trainer = Trainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
    )
    predictions = trainer.predict(test_dataset, metric_key_prefix="", ignore_keys=["O"])
    p = (predictions.predictions, test_dataset.labels)
    results = eval_test(p)
    results_path = "/".join(model_name.split("/")[-2:])

    results.to_csv(f"../../results/{results_path}.csv", index=False)
    print(results)


if __name__ == '__main__':
    start_pure_ft("ner")
    # test(model_name, test_dataset)
