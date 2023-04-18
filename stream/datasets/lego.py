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


class Lego(Dataset):
    metadata_url = "https://www.kaggle.com/datasets/joosthazelzet/lego-brick-images"
    remote_urls = {
        "lego-brick-images.zip": "kaggle datasets download -d joosthazelzet/lego-brick-images",
    }
    name = "lego"
    file_hash_map = {'lego-brick-images.zip': 'a20d1f6b96368ade0cb13f8aeb02c685'}
    dataset_type = "image"
    default_task_name ="none"

    def _process(self, raw_data_dir: Path):
        for archive in self.remote_urls.keys():
            archive_path = raw_data_dir.joinpath(archive)
            folder_name = archive_path.stem.lower()
            save_path = raw_data_dir.joinpath(folder_name)
            extract(archive_path, save_path)

    def _make_metadata(self, raw_data_dir: Path):
        images = list(raw_data_dir.rglob("dataset/*.png"))
        # get validation
        val_file = list(raw_data_dir.rglob("validation.txt"))[0]
        val_images = []
        with open(val_file, 'r') as f:
            for line in f.readlines():
                line = line.split('\n')[0]
                if len(line) == 0:
                    continue
                val_images.append(line)
        # to metadata
        file_names = {}
        file_names[self.default_task_name] = {}
        train_set = []
        val_set = []
        for path in images:
            image = path.name
            label = ' '.join(image.split(' ')[1:-1])
            path = str(path.relative_to(raw_data_dir))
            if image in val_images:
                val_set.append((path, label))
            else:
                train_set.append((path, label))
        file_names[self.default_task_name]["train"] = train_set
        file_names[self.default_task_name]["val"] = val_set
        # to class name
        class_names = self._make_class_names(file_names)
        # save
        metadata = dict(file_names=file_names, class_names=class_names)
        torch.save(metadata, self.metadata_path)

    task_names = ["none"]



