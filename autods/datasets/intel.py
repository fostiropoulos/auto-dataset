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


class Intel(Dataset):
    metadata_url = "https://www.kaggle.com/datasets/puneet6060/intel-image-classification"
    remote_urls = {
        "intel-image-classification.zip": "kaggle datasets download -d puneet6060/intel-image-classification",
    }
    name = "intel"
    file_hash_map = {'intel-image-classification.zip': '7ca81e130042cf96ee43c0802da46daf'}
    dataset_type = "image"
    default_task_name ="none"

    def _process(self, raw_data_dir: Path):
        for archive in self.remote_urls.keys():
            archive_path = raw_data_dir.joinpath(archive)
            folder_name = archive_path.stem.lower()
            save_path = raw_data_dir.joinpath(folder_name)
            extract(archive_path, save_path)

    def _make_metadata(self, raw_data_dir: Path):
        train_images = list(raw_data_dir.rglob("seg_train/*/*.jpg"))
        val_images = list(raw_data_dir.rglob("seg_test/*/*.jpg"))
        # to metadata
        file_names = {}
        file_names[self.default_task_name] = {}
        for split in ["train", "val"]:
            file_tuples = []
            images = train_images if split == "train" else val_images
            for path in images:
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



