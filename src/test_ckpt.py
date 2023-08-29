#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for sequence to sequence.
"""
# You can also adapt this script on your own sequence to sequence task. Pointers for this are left as comments.

import logging
import os
import sys
import json
import argparse

import datasets
import nltk  # Here to have a nice missing dependency error message early on
import numpy as np
from datasets import load_dataset

import transformers
from filelock import FileLock
from transformers import (
AutoTokenizer, AutoModelForSeq2SeqLM,
set_seed)
from transformers.file_utils import is_offline_mode

from uie_collator import DataCollatorForUIE
from uie_dataset import gen_cache_path

from trainer import UIETrainer, skip_instructions
from compute_metrics import compute_metrics, compute_grouped_metrics

# off wandb
os.environ['WANDB_DISABLED'] = "True"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
logger = logging.getLogger(__name__)
CURRENT_DIR = os.path.dirname(__file__)

try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    if is_offline_mode():
        raise LookupError(
            "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
        )
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.


    # Set seed before initializing model.
    seed = 123
    set_seed(seed)

    output_dir = 'output/t5-700M-ie-single'
    # Get the UIE dataset
    CURRENT_DIR = os.path.dirname(__file__)
    data_dir = '/Users/daishitong/Downloads'
    task_config_dir = 'configs/new_multi_configs'
    instruction_file = 'configs/instruction_config.json'
    instruction_strategy = 'single'
    max_num_instances_per_task = 100
    max_num_instances_per_eval_task = 20
    num_examples = 0
    over_sampling = False

    data_args = argparse.ArgumentParser()
    data_args.data_dir = data_dir
    data_args.task_config_dir = task_config_dir
    data_args.instruction_file = instruction_file
    data_args.instruction_strategy = instruction_strategy
    data_args.max_num_instances_per_task = max_num_instances_per_task
    data_args.max_num_instances_per_eval_task = max_num_instances_per_eval_task

    data_cache_dir = gen_cache_path(output_dir,data_args)
    
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained('ZWK/InstructUIE')
    # model =  AutoModel.from_config(config)
    # model_args = argparse.ArgumentParser()
    # model_args.model_name_or_path = '/Users/daishitong/Downloads/Instruct-IE/models--ZWK--InstructUIE/snapshots/48f45b25a01df1798f8c1c31751a973adb7e8647'
    # model_args.cache_dir = '/Users/daishitong/Downloads/Instruct-IE/models--ZWK--InstructUIE/snapshots/48f45b25a01df1798f8c1c31751a973adb7e8647'

    # model_path = '/Users/daishitong/Downloads/Instruct-IE/models--ZWK--InstructUIE/snapshots/48f45b25a01df1798f8c1c31751a973adb7e8647'
    tokenizer = AutoTokenizer.from_pretrained('ZWK/InstructUIE')
    # model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    model_class = AutoModelForSeq2SeqLM
    # model_class.to('cpu')
    model = model_class.from_pretrained(
        # 'ZWK/InstructUIE',
        # model_args.model_name_or_path,
        # from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        # cache_dir=model_args.cache_dir,
        # revision=model_args.model_revision,
        # use_auth_token=True if model_args.use_auth_token else None,
    )
    print('\n\n Model initialized\n\n')
    model.resize_token_embeddings(len(tokenizer))

    raw_datasets = load_dataset(
        os.path.join(CURRENT_DIR, "uie_dataset.py"),
        data_dir=data_dir,
        task_config_dir=task_config_dir,
        instruction_file=instruction_file,
        instruction_strategy=instruction_strategy,
        cache_dir=data_cache_dir,  # for debug, change dataset size, otherwise open it
        max_num_instances_per_task=max_num_instances_per_task,
        max_num_instances_per_eval_task=max_num_instances_per_eval_task,
        num_examples=num_examples,
        over_sampling=over_sampling
    )
    raw_datasets.cleanup_cache_files()
    # Data collator
    label_pad_token_id = -10
    data_collator = DataCollatorForUIE(
        tokenizer,
        model=model,
        padding="longest",
        max_source_length=512,
        max_target_length=50,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8,
        add_task_name=False,
        add_dataset_name=False,
        common_dataset_name=False,
        num_examples=0,
        input_record_file='flan-t5.record'
    )
    eval_dataset = raw_datasets["validation"]

    # if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id

    # we don't want to remove unused columns because we will prepare each batch during training,
    # and some of the information will also be used in evaluation.
    remove_unused_columns = False

    # Metric
    def compute_rouge_metrics(dataset, preds, save_prefix=None):
        decoded_preds = skip_instructions(model, preds, tokenizer)
        references = [e["Instance"]["label"] for e in dataset]
        result = compute_metrics(predictions=decoded_preds, references=references)
        result_per_task = compute_grouped_metrics(predictions=decoded_preds, references=references,
                                                  groups=dataset["Task"])
        result.update(result_per_task)
        categories = dataset["Dataset"]
        result_per_category = compute_grouped_metrics(predictions=decoded_preds, references=references,
                                                      groups=categories)
        result.update(result_per_category)
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        if save_prefix is not None:
            with open(os.path.join(output_dir, f"{save_prefix}_eval_predictions.jsonl"), "w") as fout:
                for example, pred in zip(dataset, decoded_preds):
                    fout.write(json.dumps({
                        "Task": example["Task"],
                        "Dataset": example["Dataset"],
                        "Instance": example["Instance"],
                        "Prediction": pred
                    }) + "\n")
        return result
    print('dataset success')


    # Training
    # 训练epoch数，按照 num_train_epochs 传入，在trainer中解析
    # TODO, train debug, bloomz, flan-t5
    trainer = UIETrainer(
        model=model,
        # args=training_args,
        train_dataset=None,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_rouge_metrics,
        callbacks= None
    )

    # Evaluation
    results = {}

    generation_max_length = 50

    # in case the batch is shorter than max length, the output should be padded
    max_new_tokens = (
        generation_max_length
    )

    num_beams = 1
    repetition_penalty = 1.0

    do_predict = True

    run_name= 't5-700M-mult-mi-experiment'
    all_metrics = {"run_name": run_name}
    if do_predict:
        print("*** Prediction ***")
        print("*** Loading CheckPoint ***")

        # if data_args.max_predict_samples is not None:
        predict_dataset = raw_datasets["test"]

        predict_results = trainer.predict(
            predict_dataset,
            metric_key_prefix="predict",
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            repetition_penalty=repetition_penalty,
            pad_token_id=tokenizer.pad_token_id
        )
        metrics = predict_results.metrics
        max_predict_samples = ( len(predict_dataset) )
        metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))

        trainer.log(metrics)
        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)
        all_metrics.update(metrics)

    # if training_args.do_demo:
    #     logger.info("Serving the model as a demo...")
    #     user_input = ''
    #     while True:
    #         user_input = input("Please enter your input to the model, or enter 'quit' to exit: ")
    #         if user_input.lower() == "quit":
    #             break
    #         inputs = tokenizer([user_input], return_tensors="pt")
    #         _, preds, _ = trainer.prediction_step(model, inputs=inputs, prediction_loss_only=False)
    #         print(f"Model generates: {tokenizer.decode(preds[0], skip_special_tokens=True)}\n\n")

    return results

if __name__ == "__main__":
    main()