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


class Food101(Dataset):
    metadata_url = "https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/"
    remote_urls = {
        "food-101.tar.gz": "http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz",
    }
    file_hash_map = {'food-101.tar.gz': '85eeb15f3717b99a5da872d97d918f87'}

    name = "food101"
    dataset_type = "image"
    default_task_name ="none"

    def _process(self, raw_data_dir: Path):
        for archive in self.remote_urls.keys():
            archive_path = raw_data_dir.joinpath(archive)
            folder_name = archive_path.stem.lower()
            save_path = raw_data_dir.joinpath(folder_name)
            extract(archive_path, save_path)

    def _make_metadata(self, raw_data_dir: Path):
        train_list = list(raw_data_dir.rglob("train.txt"))[0]
        val_list = list(raw_data_dir.rglob("test.txt"))[0]
        image_folder = list(raw_data_dir.rglob("images"))[0]
        # to metadata
        file_names = {}
        file_names[self.default_task_name] = {}
        for split in ["train", "val"]:
            file_tuples = []
            file_list = train_list if split == 'train' else val_list
            with open(file_list, 'r') as f:
                for line in f.readlines():
                    line = line.split('\n')[0]
                    if len(line) == 0:
                        continue
                    path = str(Path(image_folder.joinpath(line).as_posix() + '.jpg').relative_to(raw_data_dir))
                    label = line.split('/')[0]
                    file_tuples.append((path, label))
            file_names[self.default_task_name][split] = file_tuples
        # to class name
        class_names = self._make_class_names(file_names)
        # save
        metadata = dict(file_names=file_names, class_names=class_names)
        torch.save(metadata, self.metadata_path)

    task_names = ["none"]



