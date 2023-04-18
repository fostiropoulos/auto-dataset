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


class Textures(Dataset):
    metadata_url = "https://github.com/abin24/Textures-Dataset"
    remote_urls = {
        "Splited.zip": "13LBYN6eTfV9G9xdgZtdpNHrXSA8mpv-2",
    }
    name = "textures"
    file_hash_map = {'Splited.zip': '1fe9d94d6e73594591b46032d22b46c6'}
    dataset_type = "image"
    default_task_name ="none"

    def _process(self, raw_data_dir: Path):
        for archive in self.remote_urls.keys():
            archive_path = raw_data_dir.joinpath(archive)
            folder_name = archive_path.stem.lower()
            save_path = raw_data_dir.joinpath(folder_name)
            extract(archive_path, save_path)

    def _make_metadata(self, raw_data_dir: Path):
        train_folder = list(raw_data_dir.rglob("train"))[0]
        val_folder = list(raw_data_dir.rglob("valid"))[0]
        # to metadata
        file_names = {}
        file_names[self.default_task_name] = {}
        for split in ["train", "val"]:
            file_tuples = []
            image_folder = train_folder if split == 'train' else val_folder
            images = list(image_folder.rglob("*.jpg"))
            for path in images:
                label = path.parent.name
                path = str(path.relative_to(raw_data_dir))
                file_tuples.append((path, label))
            file_names[self.default_task_name][split] = file_tuples
        # to class name
        class_names = self._make_class_names(file_names)
        # save
        metadata = dict(file_names=file_names, class_names=class_names)
        torch.save(metadata, self.metadata_path)

    task_names = ["none"]



