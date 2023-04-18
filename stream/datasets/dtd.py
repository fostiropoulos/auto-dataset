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


class Dtd(Dataset):
    metadata_url = "https://www.robots.ox.ac.uk/~vgg/data/dtd/"
    remote_urls = {
        "dtd-r1.0.1.tar.gz": "https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz",
    }
    name = "dtd"
    file_hash_map = {'dtd-r1.0.1.tar.gz': 'fff73e5086ae6bdbea199a49dfb8a4c1'}
    dataset_type = "image"
    default_task_name ="split_1"

    def _process(self, raw_data_dir: Path):
        archive_path = raw_data_dir.joinpath("dtd-r1.0.1.tar.gz")
        folder_name = archive_path.stem.lower()
        save_path = raw_data_dir.joinpath(folder_name)
        extract(archive_path, save_path)

    def _make_metadata(self, raw_data_dir: Path):
        file_names = {}
        image_folder = list(raw_data_dir.rglob("images"))[0]
        for task_name in self.task_names:
            file_idx = task_name.split('_')[-1]
            train_file = list(raw_data_dir.rglob("train" + file_idx + ".txt"))[0]
            val_file = list(raw_data_dir.rglob("test" + file_idx + ".txt"))[0]
            # to metadata
            file_names[task_name] = {}
            for split in ["train", "val"]:
                file_tuples = []
                img_list = train_file if split == 'train' else val_file
                with open(img_list, 'r') as f:
                    for line in f.readlines():
                        line = line.split('\n')[0]
                        if len(line) == 0:
                            continue
                        label = line.split('/')[0]
                        path = str(image_folder.joinpath(line).relative_to(raw_data_dir))
                        file_tuples.append((path, label))
                file_names[task_name][split] = file_tuples
        # to class name
        class_names = self._make_class_names(file_names)
        # save
        metadata = dict(file_names=file_names, class_names=class_names)
        torch.save(metadata, self.metadata_path)

    task_names = ["split_1", "split_2", "split_3", "split_4", "split_5",
                "split_6", "split_7", "split_8", "split_9", "split_10"]



