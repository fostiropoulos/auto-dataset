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


class Fish(Dataset):
    metadata_url = "https://www.kaggle.com/datasets/crowww/a-large-scale-fish-dataset"
    remote_urls = {
        "a-large-scale-fish-dataset.zip": "kaggle datasets download -d crowww/a-large-scale-fish-dataset",
    }
    name = "fish"

    file_hash_map = {'a-large-scale-fish-dataset.zip': 'b9da047a59238d2f2eb7d8c2617ef4ed'}

    dataset_type = "image"
    default_task_name ="none"

    def _process(self, raw_data_dir: Path):
        for archive in self.remote_urls.keys():
            archive_path = raw_data_dir.joinpath(archive)
            folder_name = archive_path.stem.lower()
            save_path = raw_data_dir.joinpath(folder_name)
            extract(archive_path, save_path)

    def _make_metadata(self, raw_data_dir: Path):
        images = list(raw_data_dir.rglob("Fish_Dataset/*/*/*.png"))
        # exclude segmentation
        images = [img for img in images if " GT" not in img.as_posix()]
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



