# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# For dataset details visit: https://crfm.stanford.edu/2023/03/13/alpaca.html
import copy
import json
from typing import Dict

import torch
from torch.utils.data import Dataset


class ShareGPTDataset(Dataset):
    def __init__(self, dataset_config, tokenizer, partition="train"):
        if partition == "train":
            file_path = f"{dataset_config.data_path}/train.jsonl"
        else:
            file_path = f"{dataset_config.data_path}/validation.jsonl"
        self.file_path = file_path
        self.dataset_items = []
        self.tokenizer = tokenizer
        self.max_input_length = dataset_config.input_length
        self._load()

    def _load(self):
        for line in open(self.file_path):
            data = json.loads(line)
            input = data["input"] + " "
            encoded_input = self.tokenizer.encode(input)
            # skip instances that do not allow to compute loss
            if len(encoded_input) >= self.max_input_length or len(data["output"]) < 2:
                continue
            self.dataset_items.append(data)

    def __len__(self):
        return len(self.dataset_items)

    def __getitem__(self, index):
        item = self.dataset_items[index]
        encoded_input = torch.tensor(self.tokenizer.encode(item["input"]), dtype=torch.int64)
        encoded_example = torch.tensor(self.tokenizer.encode(
            item["input"] + " " + item["output"]
        ), dtype=torch.int64)
        labels = copy.deepcopy(encoded_example)
        labels[: len(encoded_input)] = 0
        return {
            "input_ids": encoded_example,
            "labels": labels,
        }


    def collate_fn(self, batch) -> Dict[str, torch.Tensor]:
        # already tokenized

        longer_seq_in_batch = max([sample['input_ids'].shape[0] for sample in batch])
        longer_seq_in_batch = min(longer_seq_in_batch, self.max_input_length)

        # set default value as padding token id for inputs
        batch_input_ids = torch.ones((len(batch), longer_seq_in_batch), dtype=torch.int64)
        padding_token_id = self.tokenizer.pad_token_id
        batch_input_ids *= padding_token_id

        # set default value -1 for labels
        batch_labels = torch.zeros((len(batch), longer_seq_in_batch), dtype=torch.int64)

        for sample_idx, sample in enumerate(batch):
            # update input ids
            source_len = min(sample['input_ids'].shape[0], longer_seq_in_batch)
            batch_input_ids[sample_idx, :source_len] = sample['input_ids'][:source_len]
            # update labels
            batch_labels[sample_idx, :source_len] = sample['labels'][:source_len]

        attention_mask = batch_labels.ge(1).float()

        return {
            "input_ids": batch_input_ids,
            "labels": batch_labels,
            "attention_mask": attention_mask
        }