import os
from pathlib import Path
from typing import List

import h5py
import numpy as np
import scipy
from sklearn.model_selection import train_test_split

from autods.dataset import Dataset
from autods.utils import extract, is_archive

import torch


class Core50(Dataset):
    metadata_url = "https://vlomonaco.github.io/core50/"
    remote_urls = {
        "core50_128x128.zip": "http://bias.csr.unibo.it/maltoni/download/core50/core50_128x128.zip",
    }
    name = "core50"
    dataset_type = "image"
    default_task_name ="object"
    file_hash_map = {'core50_128x128.zip': '745f3373fed08d69343f1058ee559e13'}

    def _process(self, raw_data_dir: Path):
        for archive in self.remote_urls.keys():
            archive_path = raw_data_dir.joinpath(archive)
            folder_name = archive_path.stem.lower()
            save_path = raw_data_dir.joinpath(folder_name)
            extract(archive_path, save_path)

    def _make_metadata(self, raw_data_dir: Path):
        object_types = ["plug adapters", "mobile phones", "scissors", "light bulbs", "cans", "glasses", "balls", "markers", "cups", "remote controls"]
        images = list(raw_data_dir.rglob("*.png"))
        # train test split
        train_idx, val_idx = train_test_split(np.arange(len(images)),
                                              test_size=self.test_size, random_state=self.test_split_random_state)
        # to metadata
        file_names = {}
        for task_name in self.task_names:
            file_names[task_name] = {}
            for split in ["train", "val"]:
                file_tuples = []
                indices = train_idx if split == 'train' else val_idx
                for idx in indices:
                    path = images[idx]
                    obj_type = (int(path.parent.name[1:]) - 1) // 5
                    obj_idx = (int(path.parent.name[1:]) - 1) % 5
                    if task_name == "object":
                        label = object_types[obj_type] + ' ' + str(obj_idx)
                    else:
                        label = object_types[obj_type]
                    path = str(path.relative_to(raw_data_dir))
                    file_tuples.append((path, label))
                file_names[task_name][split] = file_tuples
        # to class name
        class_names = self._make_class_names(file_names)
        # save
        metadata = dict(file_names=file_names, class_names=class_names)
        torch.save(metadata, self.metadata_path)

    task_names = ["object", "category"]



