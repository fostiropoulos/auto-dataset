import os
from pathlib import Path
from typing import List

import h5py
import numpy as np
from scipy.io import loadmat
from autods.dataset import Dataset
from autods.utils import extract, is_archive

import torch


class Cars(Dataset):
    metadata_url = "http://ai.stanford.edu/~jkrause/cars/car_dataset.html#:~:text=The%20Cars%20dataset%20contains%2016%2C185,or%202012%20BMW%20M3%20coupe"
    remote_urls = {
        "cars_train.tgz": "http://ai.stanford.edu/~jkrause/car196/cars_train.tgz",
        "cars_test.tgz": "http://ai.stanford.edu/~jkrause/car196/cars_test.tgz",
        "car_devkit.tgz": "http://ai.stanford.edu/~jkrause/cars/car_devkit.tgz",
        "cars_test_annos.mat": "http://ai.stanford.edu/~jkrause/car196/cars_test_annos_withlabels.mat",
    }
    name = "cars"
    file_hash_map = {
        "cars_train.tgz": "065e5b463ae28d29e77c1b4b166cfe61",
        "cars_test.tgz": "4ce7ebf6a94d07f1952d94dd34c4d501",
        "car_devkit.tgz": "c3b158d763b6e2245038c8ad08e45376",
        "cars_test_annos.mat": "b0a2b23655a3edd16d84508592a98d10",
    }
    dataset_type = "image"
    default_task_name ="none"

    def _process(self, raw_data_dir: Path):
        for archive in ["cars_train.tgz", "cars_test.tgz", "car_devkit.tgz"]:
            archive_path = raw_data_dir.joinpath(archive)
            folder_name = archive_path.stem.lower()
            save_path = raw_data_dir.joinpath(folder_name)
            extract(archive_path, save_path)

    def _make_metadata(self, raw_data_dir: Path):
        train_image_folder = list(raw_data_dir.rglob("cars_train/*.jpg"))[0].parent
        val_image_folder = list(raw_data_dir.rglob("cars_test/*.jpg"))[0].parent
        train_labels = list(raw_data_dir.rglob("cars_train_annos.mat"))[0]
        val_labels = list(raw_data_dir.rglob("cars_test_annos.mat"))[0]
        # get label names
        label_names = list(raw_data_dir.rglob("cars_meta.mat"))[0]
        f = loadmat(label_names)
        label_to_name = [name[0] for name in f['class_names'][0]]
        # to metadata
        file_names = {}
        file_names[self.default_task_name] = {}
        for split in ["train", "val"]:
            file_tuples = []
            image_folder = train_image_folder if split == 'train' else val_image_folder
            label_file = train_labels if split == 'train' else val_labels
            f = loadmat(label_file)
            for anno in f['annotations'][0]:
                label = label_to_name[anno[-2][0][0] - 1]
                path = str(Path(os.path.join(image_folder, anno[-1][0])).relative_to(raw_data_dir))
                file_tuples.append((path, label))
            file_names[self.default_task_name][split] = file_tuples
        # to class name
        class_names = self._make_class_names(file_names)
        # save
        metadata = dict(file_names=file_names, class_names=class_names)
        torch.save(metadata, self.metadata_path)

    task_names = ["none"]



