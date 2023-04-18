import os
from pathlib import Path
from typing import List
from PIL import Image

import h5py
import numpy as np
import scipy
from sklearn.model_selection import train_test_split

from stream.dataset import Dataset
from stream.utils import extract, is_archive

import torch


class Pcam(Dataset):
    metadata_url = "https://github.com/basveeling/pcam#details"
    remote_urls = {
        "camelyonpatch_level_2_split_train_x.h5.gz": "1g7PyE99UVPe6uZJtCqXM6wLB9Ma6Dy8q",  # Re-uploaded
        "camelyonpatch_level_2_split_train_y.h5.gz": "1269yhu3pZDP8UYFQs-NYs3FPwuK-nGSG",
        "camelyonpatch_level_2_split_valid_x.h5.gz": "1hgshYGWK8V-eGRy8LToWJJgDU_rXWVJ3",
        "camelyonpatch_level_2_split_valid_y.h5.gz": "1bH8ZRbhSVAhScTS0p9-ZzGnX91cHT3uO",
    }
    file_hash_map = {
        "camelyonpatch_level_2_split_train_x.h5.gz": "1571f514728f59376b705fc836ff4b63",
        "camelyonpatch_level_2_split_train_y.h5.gz": "35c2d7259d906cfc8143347bb8e05be7",
        "camelyonpatch_level_2_split_valid_x.h5.gz": "d5b63470df7cfa627aeec8b9dc0c066e",
        "camelyonpatch_level_2_split_valid_y.h5.gz": "2b85f58b927af9964a4c15b8f7e8f179",
    }

    name = "pcam"
    dataset_type = "image"
    default_task_name ="none"

    def _process(self, raw_data_dir: Path):
        for archive in self.remote_urls.keys():
            archive_path = raw_data_dir.joinpath(archive)
            folder_name = archive_path.stem.lower()
            save_path = raw_data_dir.joinpath(folder_name)
            extract(archive_path, save_path)

        # extract h5
        train_x_h5 = list(raw_data_dir.rglob("camelyonpatch_level_2_split_train_x.h5"))[
            -1
        ]
        with h5py.File(train_x_h5, "r") as F:
            train_x = np.array(F["x"])
        train_y_h5 = list(raw_data_dir.rglob("camelyonpatch_level_2_split_train_y.h5"))[
            -1
        ]
        with h5py.File(train_y_h5, "r") as F:
            train_y = np.array(F["y"])
        train_y = train_y.reshape((train_y.shape[0], 1))
        val_x_h5 = list(raw_data_dir.rglob("camelyonpatch_level_2_split_valid_x.h5"))[
            -1
        ]
        with h5py.File(val_x_h5, "r") as F:
            val_x = np.array(F["x"])
        val_y_h5 = list(raw_data_dir.rglob("camelyonpatch_level_2_split_valid_y.h5"))[
            -1
        ]
        with h5py.File(val_y_h5, "r") as F:
            val_y = np.array(F["y"])
        val_y = val_y.reshape((val_y.shape[0], 1))
        self._nparray_to_image(train_x, train_y, split="train")
        self._nparray_to_image(val_x, val_y, split="val")

    def _make_metadata(self, raw_data_dir: Path):
        label_to_name = {"0": "tumor", "1": "normal"}
        train_images = list(raw_data_dir.rglob("train/*.png"))
        val_images = list(raw_data_dir.rglob("val/*.png"))
        # to metadata
        file_names = {}
        file_names[self.default_task_name] = {}
        for split in ["train", "val"]:
            file_tuples = []
            images = train_images if split == "train" else val_images
            for path in images:
                label = label_to_name[path.stem.split("_")[1]]
                path = str(path.relative_to(raw_data_dir))
                file_tuples.append((path, label))
            file_names[self.default_task_name][split] = file_tuples
        # to class name
        class_names = self._make_class_names(file_names)
        # save
        metadata = dict(file_names=file_names, class_names=class_names)
        torch.save(metadata, self.metadata_path)

    task_names = ["none"]

