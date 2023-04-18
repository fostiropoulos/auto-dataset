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


class Indoor(Dataset):
    metadata_url = "https://paperswithcode.com/dataset/mit-indoors-scenes"
    remote_urls = {
        "indoor-scenes-cvpr-2019.zip": "kaggle datasets download -d itsahmad/indoor-scenes-cvpr-2019",
    }
    file_hash_map = {'indoor-scenes-cvpr-2019.zip': 'b5a8ee875edc974ab49f4cad3b8607da'}
    name = "indoor"
    dataset_type = "image"
    default_task_name ="none"

    def _process(self, raw_data_dir: Path):
        for archive in self.remote_urls.keys():
            archive_path = raw_data_dir.joinpath(archive)
            folder_name = archive_path.stem.lower()
            save_path = raw_data_dir.joinpath(folder_name)
            extract(archive_path, save_path)

    def _make_metadata(self, raw_data_dir: Path):
        train_list = list(raw_data_dir.rglob("TrainImages.txt"))[0]
        val_list = list(raw_data_dir.rglob("TestImages.txt"))[0]
        image_folder = list(raw_data_dir.rglob("Images"))[0]
        # to metadata
        file_names = {}
        file_names[self.default_task_name] = {}
        for split in ["train", "val"]:
            file_tuples = []
            image_list = train_list if split == 'train' else val_list
            with open(image_list, 'r') as f:
                for line in f.readlines():
                    line = line.split('\n')[0]
                    if len(line) == 0:
                        continue
                    path = str(image_folder.joinpath(line).relative_to(raw_data_dir))
                    label = line.split('/')[0]
                    file_tuples.append((path, label))
            file_names[self.default_task_name][split] = file_tuples
        # to class name
        class_names = self._make_class_names(file_names)
        # save
        metadata = dict(file_names=file_names, class_names=class_names)
        torch.save(metadata, self.metadata_path)

    task_names = ["none"]



