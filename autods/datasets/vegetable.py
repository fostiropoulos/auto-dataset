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


class Vegetable(Dataset):
    metadata_url = "https://www.kaggle.com/datasets/misrakahmed/vegetable-image-dataset"
    remote_urls = {
        "vegetable-image-dataset.zip": "kaggle datasets download -d misrakahmed/vegetable-image-dataset",
    }
    file_hash_map = {'vegetable-image-dataset.zip': 'b8a8d77324b79c4bce1ef6080d7c4f29'}
    name = "vegetable"
    dataset_type = "image"
    default_task_name ="none"

    def _process(self, raw_data_dir: Path):
        for archive in self.remote_urls.keys():
            archive_path = raw_data_dir.joinpath(archive)
            folder_name = archive_path.stem.lower()
            save_path = raw_data_dir.joinpath(folder_name)
            extract(archive_path, save_path)

    def _make_metadata(self, raw_data_dir: Path):
        train_images = list(raw_data_dir.rglob("train/*/*.jpg")) + list(raw_data_dir.rglob("validation/*/*.jpg"))
        val_images = list(raw_data_dir.rglob("test/*/*.jpg"))
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



