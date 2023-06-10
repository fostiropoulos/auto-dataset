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


class Deepweedsx(Dataset):
    metadata_url = "https://www.kaggle.com/datasets/coreylammie/deepweedsx"
    remote_urls = {
        "deepweedsx.zip": "kaggle datasets download -d coreylammie/deepweedsx",
    }
    name = "deepweedsx"
    file_hash_map = {'deepweedsx.zip': 'fd11132ef28fd1f041be9cc8ccdc6634'}
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
            '0': 'Chinee Apple',
            '1': 'Lantana',
            '2': 'Parkinsonia',
            '3': 'Parthenium',
            '4': 'Prickly Acacia',
            '5': 'Rubber Vine',
            '6': 'Siam Weed',
            '7': 'Snake Weed',
            '8': 'Other',
        }
        train_list = list(raw_data_dir.rglob("train_set_labels.csv"))[0]
        val_list = list(raw_data_dir.rglob("test_set_labels.csv"))[0]
        image_folder = list(raw_data_dir.rglob("DeepWeeds_Images_256"))[0].as_posix()
        # to metadata
        file_names = {}
        file_names[self.default_task_name] = {}
        for split in ["train", "val"]:
            file_tuples = []
            img_list = train_list if split == 'train' else val_list
            with open(img_list, 'r') as f:
                for line in f.readlines()[1:]:
                    line = line.split('\n')[0]
                    if len(line) == 0:
                        continue
                    img, label = line.split(',')
                    label = label_to_name[label]
                    path = os.path.join(image_folder, img)
                    path = str(Path(path).relative_to(raw_data_dir))
                    file_tuples.append((path, label))
            file_names[self.default_task_name][split] = file_tuples
        # to class name
        class_names = self._make_class_names(file_names)
        # save
        metadata = dict(file_names=file_names, class_names=class_names)
        torch.save(metadata, self.metadata_path)

    task_names = ["none"]



