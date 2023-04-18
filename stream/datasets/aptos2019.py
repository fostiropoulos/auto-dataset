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


class Aptos2019(Dataset):
    metadata_url = "https://www.kaggle.com/competitions/aptos2019-blindness-detection"
    remote_urls = {
        "aptos2019-blindness-detection.zip": "kaggle competitions download -c aptos2019-blindness-detection",
    }
    file_hash_map = {'aptos2019-blindness-detection.zip': 'fff5440648ab9ee707bc56c97c40dc2f'}

    name = "aptos2019"
    dataset_type = "image"
    default_task_name ="none"

    def _process(self, raw_data_dir: Path):
        for archive in self.remote_urls.keys():
            archive_path = raw_data_dir.joinpath(archive)
            folder_name = archive_path.stem.lower()
            save_path = raw_data_dir.joinpath(folder_name)
            extract(archive_path, save_path)

    def _make_metadata(self, raw_data_dir: Path):
        label_to_name = {
            '0': 'No DR',
            '1': 'Mild',
            '2': 'Moderate',
            '3': 'Severe',
            '4': 'Proliferative DR',
        }
        image_list = list(raw_data_dir.rglob("train.csv"))[0]
        image_folder = list(raw_data_dir.rglob("train_images"))[0]
        images = []
        with open(image_list, 'r') as f:
            for line in f.readlines()[1:]:
                line = line.split('\n')[0]
                if len(line) == 0:
                    continue
                name, label = line.split(',')
                label = label_to_name[label]
                path = image_folder.joinpath(name).as_posix() + '.png'
                path = str(Path(path).relative_to(raw_data_dir))
                images.append((path, label))
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
                file_tuples.append(images[idx])
            file_names[self.default_task_name][split] = file_tuples
        # to class name
        class_names = self._make_class_names(file_names)
        # save
        metadata = dict(file_names=file_names, class_names=class_names)
        torch.save(metadata, self.metadata_path)

    task_names = ["none"]



