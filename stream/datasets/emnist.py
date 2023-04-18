import os
from pathlib import Path
from typing import List

import h5py
import numpy as np
import scipy
import idx2numpy
from PIL import Image

from stream.dataset import Dataset
from stream.utils import extract, is_archive

import torch


class Emnist(Dataset):
    metadata_url = "https://www.nist.gov/itl/products-and-services/emnist-dataset"
    remote_urls = {
        "gzip.zip": "http://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/gzip.zip",
    }
    name = "emnist"
    file_hash_map = {'gzip.zip': '58c8d27c78d21e728a6bc7b3cc06412e'}
    dataset_type = "image"
    default_task_name ="balanced"

    def _process(self, raw_data_dir: Path):
        def nparray_to_image(array, targets, subset, split):
            base_path = raw_data_dir.joinpath(f'{subset}/{split}')
            if not os.path.exists(base_path):
                os.makedirs(base_path)
            for i in range(len(targets)):
                path = base_path.joinpath(f'{targets[i]}_{i}.png')
                img = Image.fromarray(array[i])
                img.save(path)

        archive_path = raw_data_dir.joinpath("gzip.zip")
        folder_name = archive_path.stem.lower()
        save_path = raw_data_dir.joinpath(folder_name)
        extract(archive_path, save_path)
        # extract all subsets ubyte
        for task_name in self.task_names:
            archives = list(raw_data_dir.rglob("emnist-" + task_name + "*.gz"))
            for archive in archives:
                archive_path = raw_data_dir.joinpath(archive)
                if is_archive(archive_path):
                    folder_name = archive_path.name.lower()
                    save_path = raw_data_dir.joinpath(folder_name)
                    extract(archive_path, save_path)
        # read ubyte files
        for task_name in self.task_names:
            train_x_path = list(raw_data_dir.rglob("emnist-" + task_name + "-train-images-idx3-ubyte"))[-1].as_posix()
            test_x_path = list(raw_data_dir.rglob("emnist-" + task_name + "-test-images-idx3-ubyte"))[-1].as_posix()
            train_y_path = list(raw_data_dir.rglob("emnist-" + task_name + "-train-labels-idx1-ubyte"))[-1].as_posix()
            test_y_path = list(raw_data_dir.rglob("emnist-" + task_name + "-test-labels-idx1-ubyte"))[-1].as_posix()
            train_x = idx2numpy.convert_from_file(train_x_path)
            train_y = idx2numpy.convert_from_file(train_y_path)
            test_x = idx2numpy.convert_from_file(test_x_path)
            test_y = idx2numpy.convert_from_file(test_y_path)
            # save as image
            nparray_to_image(train_x, train_y, subset=task_name, split='train')
            nparray_to_image(test_x, test_y, subset=task_name, split='val')

    def _make_metadata(self, raw_data_dir: Path):
        # to metadata
        file_names = {}
        for task_name in self.task_names:
            file_names[task_name] = {}
            for split in ["train", "val"]:
                file_tuples = []
                images = list(raw_data_dir.rglob(task_name + '/' + split + "/*.png"))
                for path in images:
                    label = path.name.split('_')[0]
                    path = str(path.relative_to(raw_data_dir))
                    file_tuples.append((path, label))
                file_names[task_name][split] = file_tuples
        # to class name
        class_names = self._make_class_names(file_names)
        # save
        metadata = dict(file_names=file_names, class_names=class_names)
        torch.save(metadata, self.metadata_path)

    task_names = ["balanced", "byclass", "bymerge", "digits", "letters", "mnist"]



