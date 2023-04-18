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


class Gtsrb(Dataset):
    metadata_url = "https://benchmark.ini.rub.de/gtsrb_news.html"
    remote_urls = {
        "GTSRB_Final_Training_Images.zip": "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Training_Images.zip",
        "GTSRB_Final_Test_Images.zip": "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_Images.zip",
        "GTSRB_Final_Test_GT.zip": "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_GT.zip",
    }
    name = "gtsrb"
    file_hash_map = {
        "GTSRB_Final_Training_Images.zip": "f33fd80ac59bff73c82d25ab499e03a3",
        "GTSRB_Final_Test_Images.zip":"c7e4e6327067d32654124b0fe9e82185",
        "GTSRB_Final_Test_GT.zip":"fe31e9c9270bbcd7b84b7f21a9d9d9e5"
    }
    dataset_type = "image"
    default_task_name ="none"

    def _process(self, raw_data_dir: Path):
        for archive in self.remote_urls.keys():
            archive_path = raw_data_dir.joinpath(archive)
            folder_name = archive_path.stem.lower()
            save_path = raw_data_dir.joinpath(folder_name)
            extract(archive_path, save_path)

    def _make_metadata(self, raw_data_dir: Path):
        train_images = list(raw_data_dir.rglob("Final_Training/*/*/*.ppm"))
        val_images = list(raw_data_dir.rglob("Final_Test/**/*.ppm"))
        # val labels
        val_labels = list(raw_data_dir.rglob("GT-final_test.csv"))[0]
        val_label_map = {}
        with open(val_labels, "r") as f:
            for line in f.readlines()[1:]:
                line = line.split("\n")[0]
                if len(line) == 0:
                    continue
                attr = line.split(";")
                val_label_map[attr[0]] = attr[-1]
        # to metadata
        file_names = {}
        file_names[self.default_task_name] = {}
        for split in ["train", "val"]:
            file_tuples = []
            images = train_images if split == "train" else val_images
            for path in images:
                if split == "train":
                    label = path.parent.name.lstrip("0")
                    label = "0" if label == "" else label
                else:
                    label = val_label_map[path.name]
                path = str(path.relative_to(raw_data_dir))
                file_tuples.append((path, label))
            file_names[self.default_task_name][split] = file_tuples
        # to class name
        class_names = self._make_class_names(file_names)
        # save
        metadata = dict(file_names=file_names, class_names=class_names)
        torch.save(metadata, self.metadata_path)

    task_names = ["none"]
