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


class Ip02(Dataset):
    metadata_url = "https://www.kaggle.com/datasets/rtlmhjbn/ip02-dataset"
    remote_urls = {
        "ip02-dataset.zip": "kaggle datasets download -d rtlmhjbn/ip02-dataset",
    }
    file_hash_map = {'ip02-dataset.zip': 'cd3a1fa40c8524485bd430850daed9b2'}

    name = "ip02"
    dataset_type = "image"
    default_task_name ="none"

    def _process(self, raw_data_dir: Path):
        for archive in self.remote_urls.keys():
            archive_path = raw_data_dir.joinpath(archive)
            folder_name = archive_path.stem.lower()
            save_path = raw_data_dir.joinpath(folder_name)
            extract(archive_path, save_path)

    def _make_metadata(self, raw_data_dir: Path):
        train_images = list(raw_data_dir.rglob("train/*/*.jpg")) + list(raw_data_dir.rglob("val/*/*.jpg"))
        val_images = list(raw_data_dir.rglob("test/*/*.jpg"))
        # read class names
        label_to_name = {}
        class_names = list(raw_data_dir.rglob("classes.txt"))[0]
        with open(class_names, 'r') as f:
            for line in f.readlines():
                line = line.split('\n')[0].strip()
                if len(line) == 0:
                    continue
                attr = line.split(' ')
                label = int(attr[0])
                name = " ".join(attr[1:]).strip()
                label_to_name[label] = name

        # to metadata
        file_names = {}
        file_names[self.default_task_name] = {}
        for split in ["train", "val"]:
            file_tuples = []
            images = train_images if split == "train" else val_images
            for path in images:
                label = label_to_name[int(path.parent.name) + 1]
                path = str(path.relative_to(raw_data_dir))
                file_tuples.append((path, label))
            file_names[self.default_task_name][split] = file_tuples
        # to class name
        class_names = self._make_class_names(file_names)
        # save
        metadata = dict(file_names=file_names, class_names=class_names)
        torch.save(metadata, self.metadata_path)

    task_names = ["none"]



