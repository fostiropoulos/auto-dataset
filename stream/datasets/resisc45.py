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


class Resisc45(Dataset):
    metadata_url = "https://www.tensorflow.org/datasets/catalog/resisc45"
    remote_urls = {
        "NWPU-RESISC45.rar": "1RoB26emW0_RAlosGR-4yfyS-9W8TgqLC",
    }
    name = "resisc45"
    file_hash_map = {'NWPU-RESISC45.rar': 'd824acb73957502b00efd559fc6cfbbb'}
    dataset_type = "image"
    default_task_name ="none"

    def _process(self, raw_data_dir: Path):
        archive_path = raw_data_dir.joinpath("NWPU-RESISC45.rar")
        folder_name = archive_path.stem.lower()
        save_path = raw_data_dir.joinpath(folder_name)
        extract(archive_path, save_path)

    def _make_metadata(self, raw_data_dir: Path):
        images = list(raw_data_dir.rglob("*.jpg"))
        # train test split
        train_idx, val_idx = train_test_split(np.arange(len(images)),
                                              test_size=self.test_size, random_state=self.test_split_random_state)
        # to metadata
        file_names = {}
        file_names[self.default_task_name] = {}
        for split in ["train", "val"]:
            file_tuples = []
            indices = train_idx if split == 'train' else val_idx
            for idx in indices:
                path = images[idx]
                label = path.parent.name
                path = str(path.relative_to(raw_data_dir))
                file_tuples.append((path, label))
            file_names[self.default_task_name][split] = file_tuples
        # to class name
        class_names = self._make_class_names(file_names)
        # save
        metadata = dict(file_names=file_names, class_names=class_names)
        torch.save(metadata, self.metadata_path)

    task_names = ["none"]



