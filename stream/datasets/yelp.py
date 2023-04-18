import logging
import os
from pathlib import Path
from typing import List

import h5py
import numpy as np
import pandas as pd
import scipy
from stream.feat_extractors.gpt2 import GPT2

from stream.dataset import Dataset
from stream.utils import extract, is_archive

import torch


class Yelp(Dataset):
    metadata_url = "https://huggingface.co/datasets/yelp_review_full"
    remote_urls = {
        "yelp_review_full_csv.tgz": "https://s3.amazonaws.com/fast-ai-nlp/yelp_review_full_csv.tgz",
    }
    name = "yelp"
    file_hash_map = {"yelp_review_full_csv.tgz": "a4acce1892d0f927165488f62860cabe"}
    dataset_type = "text"
    default_task_name ="none"
    default_feat_extractor = "gpt2"

    def _process(self, raw_data_dir: Path):
        for archive in self.remote_urls.keys():
            archive_path = raw_data_dir.joinpath(archive)
            extract(archive_path, raw_data_dir)

    def _make_metadata(self, raw_data_dir: Path):
        train_samples = pd.read_csv(
            list(raw_data_dir.rglob("train.csv"))[0], header=None
        )
        val_samples = pd.read_csv(list(raw_data_dir.rglob("test.csv"))[0], header=None)
        train_idx = np.arange(len(train_samples))
        val_idx = np.arange(len(val_samples))

        # to metadata
        file_names = {}
        file_names[self.default_task_name] = {}
        for split in ["train", "val"]:
            file_tuples = []
            indices = train_idx if split == "train" else val_idx
            data = train_samples if split == "train" else val_samples
            indices = indices[~data[1].isna()]
            data = data[~data[1].isna()]
            for idx in indices:
                label = data.iloc[idx][0]
                file_tuples.append((data.iloc[idx].name, label))
            file_names[self.default_task_name][split] = file_tuples
        # to class name
        class_names = self._make_class_names(file_names)
        # save
        metadata = dict(file_names=file_names, class_names=class_names)
        torch.save(metadata, self.metadata_path)

    task_names = ["none"]

    def _init_actions(self):
        super()._init_actions()
        if not self.feats_name:
            if self.split == "train":
                data = list(self.dataset_path.rglob("train.csv"))[0]
            else:
                data = list(self.dataset_path.rglob("test.csv"))[0]
            self.csv_dataset = pd.read_csv(data, header=None)

    def _data_loader(self, index: int):
        return self.csv_dataset.loc[index][1]
