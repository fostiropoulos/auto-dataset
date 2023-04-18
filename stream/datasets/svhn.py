from pathlib import Path
from typing import List
import os
from PIL import Image

import numpy as np
import scipy

from stream.dataset import Dataset
from stream.utils import extract, is_archive
from scipy.io import loadmat
import torch


class Svhn(Dataset):
    metadata_url = "http://ufldl.stanford.edu/housenumbers/"
    remote_urls = {
        "train_32x32.mat": "http://ufldl.stanford.edu/housenumbers/train_32x32.mat",
        "test_32x32.mat": "http://ufldl.stanford.edu/housenumbers/test_32x32.mat",
    }
    name = "svhn"
    file_hash_map = {
        "train_32x32.mat": "e26dedcc434d2e4c54c9b2d4a06d8373",
        "test_32x32.mat": "eb5a983be6a315427106f1b164d9cef3",
    }
    dataset_type = "image"
    default_task_name ="none"

    def _process(self, raw_data_dir: Path):
        train_data = list(raw_data_dir.rglob("train_32x32.mat"))[0]
        val_data = list(raw_data_dir.rglob("test_32x32.mat"))[0]
        f = loadmat(train_data)
        train_x, train_y = np.einsum("klij->jkli", np.array(f["X"])), np.squeeze(
            np.array(f["y"])
        )
        f = loadmat(val_data)
        val_x, val_y = np.einsum('klij->jkli', np.array(f['X'])), np.squeeze(np.array(f['y']))
        train_y, val_y = train_y.reshape(train_y.shape[0], 1), val_y.reshape(val_y.shape[0], 1)
        # save as images
        self._nparray_to_image(train_x, train_y, split='train')
        self._nparray_to_image(val_x, val_y, split='val')

    def _make_metadata(self, raw_data_dir: Path):
        train_images = list(raw_data_dir.rglob("train/*.png"))
        val_images = list(raw_data_dir.rglob("val/*.png"))
        # to metadata
        file_names = {}
        file_names[self.default_task_name] = {}
        for split in ["train", "val"]:
            file_tuples = []
            images = train_images if split == 'train' else val_images
            for path in images:
                label = path.stem.split('_')[1]
                path = str(path.relative_to(raw_data_dir))
                file_tuples.append((path, label))
            file_names[self.default_task_name][split] = file_tuples
        # to class name
        class_names = self._make_class_names(file_names)
        # save
        metadata = dict(file_names=file_names, class_names=class_names)
        torch.save(metadata, self.metadata_path)

    task_names = ["none"]



