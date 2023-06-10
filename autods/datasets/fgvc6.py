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


class Fgvc6(Dataset):
    metadata_url = "https://www.kaggle.com/competitions/ifood-2019-fgvc6"
    remote_urls = {
        "ifood-2019-fgvc6.zip": "kaggle competitions download -c ifood-2019-fgvc6",
    }
    file_hash_map = {'ifood-2019-fgvc6.zip': '91ef06b243304f07e496208b726c8640'}
    name = "fgvc6"
    dataset_type = "image"
    default_task_name ="none"
    label_to_name = {}

    def _process(self, raw_data_dir: Path):
        for archive in self.remote_urls.keys():
            archive_path = raw_data_dir.joinpath(archive)
            folder_name = archive_path.stem.lower()
            save_path = raw_data_dir.joinpath(folder_name)
            extract(archive_path, save_path)
        # extract train.zip, val.zip
        for archive in ["train_set.zip", "val_set.zip"]:
            archive_path = list(raw_data_dir.rglob(archive))[0]
            folder_name = archive_path.stem.lower()
            save_path = raw_data_dir.joinpath(folder_name)
            extract(archive_path, save_path)

    def _make_metadata(self, raw_data_dir: Path):
        train_folder = list(raw_data_dir.rglob("train_set/*.jpg"))[0].parent
        val_folder = list(raw_data_dir.rglob("val_set/*.jpg"))[0].parent
        class_list = list(raw_data_dir.rglob("class_list.txt"))[0]
        # read class labels
        with open(class_list, 'r') as f:
            for line in f.readlines():
                line = line.split('\n')[0]
                if len(line) == 0:
                    continue
                k, v = line.split(' ')
                self.label_to_name[k] = v
        # to metadata
        file_names = {}
        file_names[self.default_task_name] = {}
        for split in ["train", "val"]:
            file_tuples = []
            label_file = list(raw_data_dir.rglob(split + "_labels.csv"))[0]
            folder = train_folder if split == 'train' else val_folder
            with open(label_file, 'r') as f:
                for line in f.readlines()[1:]:
                    line = line.split('\n')[0]
                    if len(line) == 0:
                        continue
                    image, label = line.split(',')
                    path = str(folder.joinpath(image).relative_to(raw_data_dir))
                    file_tuples.append((path, label))
            file_names[self.default_task_name][split] = file_tuples
        # to class name
        class_names = self._make_class_names(file_names)
        # save
        metadata = dict(file_names=file_names, class_names=class_names)
        torch.save(metadata, self.metadata_path)

    task_names = ["none"]



