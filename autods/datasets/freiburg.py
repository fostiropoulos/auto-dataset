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


class Freiburg(Dataset):
    metadata_url = "https://github.com/PhilJd/freiburg_groceries_dataset"
    remote_urls = {
        "freiburg_groceries_dataset.tar.gz": "http://aisdatasets.informatik.uni-freiburg.de/freiburg_groceries_dataset/freiburg_groceries_dataset.tar.gz",
    }
    name = "freiburg"
    file_hash_map = {'freiburg_groceries_dataset.tar.gz': '4d7a9d202da5f0d0f09e69eca4c28bf0'}
    dataset_type = "image"
    default_task_name ="none"

    def _process(self, raw_data_dir: Path):
        archive_path = raw_data_dir.joinpath("freiburg_groceries_dataset.tar.gz")
        folder_name = archive_path.stem.lower()
        save_path = raw_data_dir.joinpath(folder_name)
        extract(archive_path, save_path)

    def _make_metadata(self, raw_data_dir: Path):
        images_path = list(raw_data_dir.rglob("*.png"))
        # train test split
        train_idx, val_idx = train_test_split(np.arange(len(images_path)),
                                              test_size=self.test_size, random_state=self.test_split_random_state)
        # to metadata
        file_names = {}
        file_names[self.default_task_name] = {}
        for split in ["train", "val"]:
            file_tuples = []
            indices = train_idx if split == 'train' else val_idx
            for idx in indices:
                path = images_path[idx]
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



