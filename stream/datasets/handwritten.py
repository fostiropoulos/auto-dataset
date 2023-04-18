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


class Handwritten(Dataset):
    metadata_url = "https://www.kaggle.com/datasets/olgabelitskaya/classification-of-handwritten-letters"
    remote_urls = {
        "classification-of-handwritten-letters.zip": "kaggle datasets download -d olgabelitskaya/classification-of-handwritten-letters",
    }
    name = "handwritten"
    file_hash_map = {'classification-of-handwritten-letters.zip': '393bb5ba533a05238b09c888916550fd'}
    dataset_type = "image"
    default_task_name ="letters"

    def _process(self, raw_data_dir: Path):
        for archive in self.remote_urls.keys():
            archive_path = raw_data_dir.joinpath(archive)
            folder_name = archive_path.stem.lower().split(".")[0]
            save_path = raw_data_dir.joinpath(folder_name)
            extract(archive_path, save_path)

    def _make_metadata(self, raw_data_dir: Path):
        base_folder = list(raw_data_dir.rglob("letters.txt"))[0].parent
        # to metadata
        file_names = {}
        for task_name in self.task_names:
            file_names[task_name] = {}
            # read images
            image_file = list(raw_data_dir.rglob(task_name + ".txt"))[0]
            images = []
            with open(image_file, 'r') as f:
                for line in f.readlines()[1:]:
                    line = line.split('\n')[0]
                    if len(line) == 0:
                        continue
                    label, _, img, _ = line.split(',')
                    path = str(base_folder.joinpath(task_name + '/' + img).relative_to(raw_data_dir))
                    images.append((path, label))
            # train test split
            train_idx, val_idx = train_test_split(np.arange(len(images)),
                                                  test_size=self.test_size, random_state=self.test_split_random_state)
            for split in ["train", "val"]:
                file_tuples = []
                indices = train_idx if split == 'train' else val_idx
                for idx in indices:
                    file_tuples.append(images[idx])
                file_names[task_name][split] = file_tuples
        # to class name
        class_names = self._make_class_names(file_names)
        # save
        metadata = dict(file_names=file_names, class_names=class_names)
        torch.save(metadata, self.metadata_path)

    task_names = ["letters", "letters2", "letters3"]



