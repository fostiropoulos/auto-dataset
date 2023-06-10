import logging
import os
from pathlib import Path
from typing import List

import h5py
import numpy as np
import pandas as pd
import scipy
from sklearn.model_selection import train_test_split

from autods.dataset import Dataset
from autods.utils import extract, is_archive

import torch


class Fashion(Dataset):
    metadata_url = (
        "https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset"
    )
    remote_urls = {
        "fashion-product-images-dataset.zip": "kaggle datasets download -d paramaggarwal/fashion-product-images-dataset",
    }
    name = "fashion"
    file_hash_map = {
        "fashion-product-images-dataset.zip": "1e146312734ba7bf88195bd00ed8bc64"
    }

    dataset_type = "image"
    default_task_name ="gender"

    def _process(self, raw_data_dir: Path):
        for archive in self.remote_urls.keys():
            archive_path = raw_data_dir.joinpath(archive)
            extract(archive_path, raw_data_dir)

    def _make_metadata(self, raw_data_dir: Path):
        folder = list(raw_data_dir.rglob("fashion-dataset/images"))[0]
        images = []
        labels_file = list(raw_data_dir.rglob("styles.csv"))[0]

        with open(labels_file, "r") as f:
            for line in f.readlines()[1:]:
                line = line.split("\n")[0]
                if len(line) == 0:
                    continue
                # mark empty attr as other
                attr = [
                    label if len(label) > 0 else "Others" for label in line.split(",")
                ]
                images.append((attr[0], attr[1:]))

        # train test split
        train_idx, val_idx = train_test_split(
            np.arange(len(images)),
            test_size=self.test_size,
            random_state=self.test_split_random_state,
        )
        # to metadata
        file_names = {}
        num_images_do_not_exist = 0
        for i, subset in enumerate(self.task_names):
            file_names[subset] = {}
            for split in ["train", "val"]:
                file_tuples = []
                indices = train_idx if split == "train" else val_idx
                for idx in indices:
                    img, attrs = images[idx]
                    label = attrs[i]
                    path = Path(folder.joinpath(f"{img}.jpg"))

                    if not path.exists():
                        num_images_do_not_exist+=1
                    else:
                        relative_path = path.relative_to(raw_data_dir).as_posix()
                        file_tuples.append((relative_path, label))
                file_names[subset][split] = file_tuples

        if num_images_do_not_exist:
            logging.warning(f"{self.name}: Skipped {num_images_do_not_exist} out of {len(images)} images. Since they do not exist.")
        # to class name
        class_names = self._make_class_names(file_names)
        # save
        metadata = dict(file_names=file_names, class_names=class_names)
        torch.save(metadata, self.metadata_path)

    task_names = [
            "gender",
            "masterCategory",
            "subCategory",
            "articleType",
            "baseColour",
            "season",
            "year",
            "usage",
        ]
