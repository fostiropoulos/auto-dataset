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


class Histaerial(Dataset):
    metadata_url = "http://eidolon.univ-lyon2.fr/~remi1/HistAerialDataset/"
    remote_urls = {
        "HistAerialDataset.zip": "http://eidolon.univ-lyon2.fr/~remi1/HistAerialDataset/dataset/HistAerialDataset.zip",
    }
    name = "histaerial"
    file_hash_map = {'HistAerialDataset.zip': 'cff202e58d8905cf108ae0cc9cee9feb'}

    dataset_type = "image"
    default_task_name ="small"

    def _process(self, raw_data_dir: Path):
        archive_path = raw_data_dir.joinpath("HistAerialDataset.zip")
        folder_name = archive_path.stem.lower()
        save_path = raw_data_dir.joinpath(folder_name)
        extract(archive_path, save_path)

    def _make_metadata(self, raw_data_dir: Path):
        small_path = list(raw_data_dir.rglob("25x25*"))[0]
        medium_path = list(raw_data_dir.rglob("50x50*"))[0]
        large_path = list(raw_data_dir.rglob("100x100*"))[0]
        # to metadata
        file_names = {}
        for task_name in self.task_names:
            file_names[task_name] = {}
            if task_name == "small":
                subset_path = small_path
            elif task_name == "medium":
                subset_path = medium_path
            else:
                subset_path = large_path
            for split in ["train", "val"]:
                file_tuples = []
                filepath = os.path.join(subset_path, split + '6k.txt')
                with open(filepath, 'r') as f:
                    for line in f.readlines():
                        line = line.split('\n')[0]
                        if len(line) == 0:
                            continue
                        path = line.split(' ')[0]
                        label = path.split('/')[0]
                        path = str(subset_path.joinpath(path).relative_to(raw_data_dir))
                        file_tuples.append((path, label))
                file_names[task_name][split] = file_tuples
        # to class name
        class_names = self._make_class_names(file_names)
        # save
        metadata = dict(file_names=file_names, class_names=class_names)
        torch.save(metadata, self.metadata_path)

    task_names = ["small", "medium", "large"]



