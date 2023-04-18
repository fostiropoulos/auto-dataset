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


class Office(Dataset):
    metadata_url = "https://paperswithcode.com/dataset/office-home"
    remote_urls = {
        "OfficeHomeDataset_10072016.zip": "1o2ieb-WCn-d4UBpJ_Fg5n1uIil60nJz2",
    }
    file_hash_map = {'OfficeHomeDataset_10072016.zip': 'b1c14819770c4448fd5b6d931031c91c'}
    name = "office"
    dataset_type = "image"
    default_task_name ="Art"

    def _process(self, raw_data_dir: Path):
        for archive in self.remote_urls.keys():
            archive_path = raw_data_dir.joinpath(archive)
            folder_name = archive_path.stem.lower()
            save_path = raw_data_dir.joinpath(folder_name)
            extract(archive_path, save_path)

    def _make_metadata(self, raw_data_dir: Path):
        # to metadata
        file_names = {}
        for task_name in self.task_names:
            file_names[task_name] = {}
            subset_folder = list(raw_data_dir.rglob(task_name))[0]
            images = list(subset_folder.rglob("*.jpg"))
            # train test split
            train_idx, val_idx = train_test_split(np.arange(len(images)),
                                                  test_size=self.test_size, random_state=self.test_split_random_state)
            for split in ["train", "val"]:
                file_tuples = []
                indices = train_idx if split == 'train' else val_idx
                for idx in indices:
                    path = images[idx]
                    label = path.parent.name
                    path = str(path.relative_to(raw_data_dir))
                    file_tuples.append((path, label))
                file_names[task_name][split] = file_tuples
        # to class name
        class_names = self._make_class_names(file_names)
        # save
        metadata = dict(file_names=file_names, class_names=class_names)
        torch.save(metadata, self.metadata_path)


    task_names = ["Art", "Clipart", "Product", "Real World"]



