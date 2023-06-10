import logging
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


class Oriset(Dataset):
    metadata_url = "https://github.com/multimedia-berkeley/OriSet"
    remote_urls = {
        "oriset": "https://github.com/multimedia-berkeley/OriSet.git",
    }
    name = "oriset"
    file_hash_map = {"oriset": "b0cd2196ed1aa96ce9c61fa5db0e346d53af8d55"}
    dataset_type = "image"
    default_task_name ="origami"

    def _process(self, raw_data_dir: Path):
        pass

    def _make_metadata(self, raw_data_dir: Path):
        file_names = {}
        corrupted = []
        for task_name in self.task_names:
            file_names[task_name] = {}
            subset_folder = (
                task_name + "_classification_data" if task_name == "origami" else task_name
            )
            images = list(raw_data_dir.rglob(subset_folder + "/*/*.jpg")) + list(
                raw_data_dir.rglob(subset_folder + "/*/*.JPEG")
            )
            train_idx, val_idx = train_test_split(
                np.arange(len(images)),
                test_size=self.test_size,
                random_state=self.test_split_random_state,
            )
            for split in ["train", "val"]:
                file_tuples = []
                indices = train_idx if split == "train" else val_idx
                for idx in indices:
                    path = images[idx]
                    label = path.parent.name
                    path = str(path.relative_to(raw_data_dir))
                    # handle corrupted image
                    try:
                        self._loader(path)
                        file_tuples.append((path, label))
                    except Exception as e:
                        corrupted.append(raw_data_dir.joinpath(path))
                        logging.warning(f"{self.name}: Skipping corrupt image file {path}")
                    finally:
                        pass
                file_names[task_name][split] = file_tuples
        # to class name
        class_names = self._make_class_names(file_names)
        # save
        metadata = dict(
            file_names=file_names, class_names=class_names, corrupt_files=corrupted
        )
        torch.save(metadata, self.metadata_path)

    task_names = ["origami", "giladori", "oriwiki"]

