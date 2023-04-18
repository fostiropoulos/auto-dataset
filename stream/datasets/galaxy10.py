import os
from pathlib import Path
from typing import List
from PIL import Image

import h5py
import numpy as np
from sklearn.model_selection import train_test_split

from stream.dataset import Dataset
from stream.utils import extract, is_archive

import torch


class Galaxy10(Dataset):
    metadata_url = "https://astronn.readthedocs.io/en/latest/galaxy10.html"
    remote_urls = {
        "Galaxy10_DECals.h5": "https://astro.utoronto.ca/~hleung/shared/Galaxy10/Galaxy10_DECals.h5"
    }
    file_hash_map = {'Galaxy10_DECals.h5': 'c6b7b4db82b3a5d63d6a7e3e5249b51c'}
    name = "galaxy10"
    dataset_type = "image"
    default_task_name ="none"

    def _process(self, raw_data_dir: Path):
        archive_path = raw_data_dir.joinpath('Galaxy10_DECals.h5')
        # To get the images and labels from file
        with h5py.File(archive_path, 'r') as F:
            images = np.array(F['images'])
            labels = np.array(F['ans'])
        labels = labels.reshape(labels.shape[0], 1)
        # train test split
        images_train, images_test, labels_train, labels_test = train_test_split(
            images, labels, test_size=self.test_size, random_state=self.test_split_random_state)
        # save as separate images
        self._nparray_to_image(images_train, labels_train, split='train')
        self._nparray_to_image(images_test, labels_test, split='val')

    def _make_metadata(self, raw_data_dir: Path):
        label_to_name = {
            '0': 'Disturbed Galaxies',
            '1': 'Merging Galaxies',
            '2': 'Round Smooth Galaxies',
            '3': 'In-between Round Smooth Galaxies',
            '4': 'Cigar Shaped Smooth Galaxies',
            '5': 'Barred Spiral Galaxies',
            '6': 'Unbarred Tight Spiral Galaxies',
            '7': 'Unbarred Loose Spiral Galaxies',
            '8': 'Edge-on Galaxies without Bulge',
            '9': 'Edge-on Galaxies with Bulge',
        }
        train_images = list(raw_data_dir.rglob("train/*.png"))
        val_images = list(raw_data_dir.rglob("val/*.png"))
        # to metadata
        file_names = {}
        file_names[self.default_task_name] = {}
        for split in ["train", "val"]:
            file_tuples = []
            images = train_images if split == 'train' else val_images
            for path in images:
                label = label_to_name[path.stem.split('_')[1]]
                path = str(path.relative_to(raw_data_dir))
                file_tuples.append((path, label))
            file_names[self.default_task_name][split] = file_tuples
        # to class name
        class_names = self._make_class_names(file_names)
        # save
        metadata = dict(file_names=file_names, class_names=class_names)
        torch.save(metadata, self.metadata_path)

    task_names = ["none"]



