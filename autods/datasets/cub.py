import os
from pathlib import Path
from typing import List

import h5py
import numpy as np
import scipy

from autods.dataset import Dataset
from autods.utils import extract, is_archive

import torch


class Cub(Dataset):
    metadata_url = "https://paperswithcode.com/dataset/cub-200-2011"
    remote_urls = {
        "CUB_200_2011.tgz": "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz",
    }
    name = "cub"
    file_hash_map = {'CUB_200_2011.tgz': '97eceeb196236b17998738112f37df78'}
    dataset_type = "image"
    default_task_name ="none"

    def _process(self, raw_data_dir: Path):
        archive_path = raw_data_dir.joinpath("CUB_200_2011.tgz")
        folder_name = archive_path.stem.lower()
        save_path = raw_data_dir.joinpath(folder_name)
        extract(archive_path, save_path)

    def _make_metadata(self, raw_data_dir: Path):
        image_folder = list(raw_data_dir.rglob("images"))[0]
        datasplits_path = list(raw_data_dir.rglob("train_test_split.txt"))[0]
        imageid_path = list(raw_data_dir.rglob("images.txt"))[0]
        # Get train test sets
        datasplit = {}
        with open(datasplits_path, 'r') as f:
            for line in f.readlines():
                line = line.split('\n')[0]
                if len(line) == 0:
                    continue
                image_id, split = line.split(' ')
                datasplit[image_id] = split == '1'
        # Get image id
        image_ids = {}
        with open(imageid_path, 'r') as f:
            for line in f.readlines():
                line = line.split('\n')[0]
                if len(line) == 0:
                    continue
                image_id, path = line.split(' ')
                image_ids[image_id] = path
        # to metadata
        file_names = {}
        file_names[self.default_task_name] = {}
        train_set, test_set = [], []
        for image_id, image_path in image_ids.items():
            path = image_folder.joinpath(image_path)
            path = str(path.relative_to(raw_data_dir))
            label = image_path.split('.')[1].split('/')[0]
            if datasplit[image_id]:
                train_set.append((path, label))
            else:
                test_set.append((path, label))
        file_names[self.default_task_name]['train'] = train_set
        file_names[self.default_task_name]['val'] = test_set
        # to class name
        class_names = self._make_class_names(file_names)
        # save
        metadata = dict(file_names=file_names, class_names=class_names)
        torch.save(metadata, self.metadata_path)

    task_names = ["none"]



