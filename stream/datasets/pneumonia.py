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


class Pneumonia(Dataset):
    metadata_url = (
        "https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia"
    )
    remote_urls = {
        "chest-xray-pneumonia.zip": "kaggle datasets download -d paultimothymooney/chest-xray-pneumonia",
    }
    file_hash_map = {"chest-xray-pneumonia.zip": "930763e3580e76de9c2c849ec933b5d6"}

    name = "pneumonia"
    dataset_type = "image"
    default_task_name ="none"

    def _process(self, raw_data_dir: Path):
        for archive in self.remote_urls.keys():
            archive_path = raw_data_dir.joinpath(archive)
            folder_name = archive_path.stem.lower()
            save_path = raw_data_dir.joinpath(folder_name)
            extract(archive_path, save_path)

    def _make_metadata(self, raw_data_dir: Path):
        base_folder = "/".join(
            list(raw_data_dir.rglob("chest_xray"))[0].as_posix().split("/")[-2:]
        )
        train_images = list(raw_data_dir.rglob(base_folder + "/train/*/*.jpeg")) + list(
            raw_data_dir.rglob(base_folder + "/val/*/*.jpeg")
        )
        val_images = list(raw_data_dir.rglob(base_folder + "/test/*/*.jpeg"))
        # to metadata
        file_names = {}
        file_names[self.default_task_name] = {}
        for split in ["train", "val"]:
            file_tuples = []
            images = train_images if split == "train" else val_images
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



