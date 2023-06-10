from pathlib import Path
from typing import List

import numpy as np
from sklearn.model_selection import train_test_split

from autods.dataset import Dataset
from autods.utils import extract

import torch


class Apparel(Dataset):
    metadata_url = "https://www.kaggle.com/datasets/trolukovich/apparel-images-dataset"
    remote_urls = {
        "apparel-images-dataset.zip": "kaggle datasets download -d trolukovich/apparel-images-dataset",
    }
    file_hash_map = {"apparel-images-dataset.zip": "34c949e107d3e9ad75480d3d17d162f6"}
    name = "apparel"
    dataset_type = "image"
    default_task_name ="color"

    def _process(self, raw_data_dir: Path):
        for archive in self.remote_urls.keys():
            archive_path = raw_data_dir.joinpath(archive)
            folder_name = archive_path.stem.lower()
            save_path = raw_data_dir.joinpath(folder_name)
            extract(archive_path, save_path)

    def _make_metadata(self, raw_data_dir: Path):
        images = list(raw_data_dir.rglob("*.jpg")) + list(raw_data_dir.rglob("*.png"))
        # train test split
        train_idx, val_idx = train_test_split(
            np.arange(len(images)),
            test_size=self.test_size,
            random_state=self.test_split_random_state,
        )
        # to metadata
        file_names = {}
        for task_name in self.task_names:
            file_names[task_name] = {}
            for split in ["train", "val"]:
                file_tuples = []
                indices = train_idx if split == "train" else val_idx
                for idx in indices:
                    path = images[idx]
                    label = path.parent.name
                    if task_name == "color":
                        label = label.split("_")[0]
                    else:
                        label = label.split("_")[1]
                    path = str(path.relative_to(raw_data_dir))
                    file_tuples.append((path, label))
                file_names[task_name][split] = file_tuples
        # to class name
        class_names = self._make_class_names(file_names)
        # save
        metadata = dict(file_names=file_names, class_names=class_names)
        torch.save(metadata, self.metadata_path)

    task_names = ["color", "type"]
