import os
from pathlib import Path
from typing import List

import h5py
import numpy as np
import scipy

from autods.dataset import Dataset
from autods.utils import extract, is_archive

import torch


class Products(Dataset):
    metadata_url = "https://cvgl.stanford.edu/projects/lifted_struct/"
    remote_urls = {
        "Stanford_Online_Products.zip": "ftp://cs.stanford.edu/cs/cvgl/Stanford_Online_Products.zip",
    }
    file_hash_map = {'Stanford_Online_Products.zip': '7f73d41a2f44250d4779881525aea32e'}

    name = "products"
    dataset_type = "image"
    default_task_name ="none"

    def _process(self, raw_data_dir: Path):
        archive_path = raw_data_dir.joinpath("Stanford_Online_Products.zip")
        folder_name = archive_path.stem.lower()
        save_path = raw_data_dir.joinpath(folder_name)
        extract(archive_path, save_path)

    def _make_metadata(self, raw_data_dir: Path):
        train_set = list(raw_data_dir.rglob("Ebay_train.txt"))[0]
        val_set = list(raw_data_dir.rglob("Ebay_test.txt"))[0]
        data_folder = '/'.join(train_set.as_posix().split('/')[:-1])
        # to metadata
        file_names = {}
        file_names[self.default_task_name] = {}
        for split in ["train", "val"]:
            file_tuples = []
            image_list = train_set if split == 'train' else val_set
            with open(image_list, 'r') as F:
                for line in F.readlines()[1:]:
                    line = line.split('\n')[0]
                    if len(line) == 0:
                        continue
                    path = Path(os.path.join(data_folder, line.split(' ')[-1]))
                    label = path.parent.name.split('_')[0]
                    path = str(path.relative_to(raw_data_dir))
                    file_tuples.append((path, label))
            file_names[self.default_task_name][split] = file_tuples
        # to class name
        class_names = self._make_class_names(file_names)
        # save
        metadata = dict(file_names=file_names, class_names=class_names)
        torch.save(metadata, self.metadata_path)

    task_names = ["none"]



