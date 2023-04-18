import os
from pathlib import Path
from typing import List

import h5py
import numpy as np
import scipy
from sklearn.model_selection import train_test_split

from stream.dataset import Dataset
from stream.utils import extract, is_archive

import torch


class Ham10000(Dataset):
    metadata_url = "https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000"
    remote_urls = {
        "skin-cancer-mnist-ham10000.zip": "kaggle datasets download -d kmader/skin-cancer-mnist-ham10000",
    }
    file_hash_map = {'skin-cancer-mnist-ham10000.zip': 'fd84b842e863f3c82f2f4231da626014'}

    name = "ham10000"
    dataset_type = "image"
    default_task_name ="none"

    def _process(self, raw_data_dir: Path):
        for archive in self.remote_urls.keys():
            archive_path = raw_data_dir.joinpath(archive)
            folder_name = archive_path.stem.lower()
            save_path = raw_data_dir.joinpath(folder_name)
            extract(archive_path, save_path)

    def _make_metadata(self, raw_data_dir: Path):
        images = list(raw_data_dir.rglob("*.jpg"))
        # train test split
        train_idx, val_idx = train_test_split(np.arange(len(images)),
                                              test_size=self.test_size, random_state=self.test_split_random_state)
        # read labels
        image_to_label = {}
        labels = list(raw_data_dir.rglob("HAM10000_metadata.csv"))[0]
        with open(labels, 'r') as f:
            for line in f.readlines()[1:]:
                line = line.split('\n')[0]
                if len(line) == 0:
                    continue
                attr = line.split(',')
                image_to_label[attr[1]] = attr[2]
        # to metadata
        file_names = {}
        file_names[self.default_task_name] = {}
        for split in ["train", "val"]:
            file_tuples = []
            indices = train_idx if split == 'train' else val_idx
            for idx in indices:
                path = images[idx]
                label = image_to_label[path.stem]
                path = str(path.relative_to(raw_data_dir))
                file_tuples.append((path, label))
            file_names[self.default_task_name][split] = file_tuples
        # to class name
        class_names = self._make_class_names(file_names)
        # save
        metadata = dict(file_names=file_names, class_names=class_names)
        torch.save(metadata, self.metadata_path)

    task_names = ["none"]



