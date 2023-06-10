from pathlib import Path
from typing import List

import h5py
import numpy as np
from scipy.io import loadmat

from autods.dataset import Dataset
from autods.utils import extract, is_archive

import torch


class Flowers(Dataset):
    metadata_url = "https://www.robots.ox.ac.uk/~vgg/data/flowers/17/index.html"
    remote_urls = {
        "17flowers.tgz": "https://www.robots.ox.ac.uk/~vgg/data/flowers/17/17flowers.tgz",
        "datasplits.mat": "https://www.robots.ox.ac.uk/~vgg/data/flowers/17/datasplits.mat"
    }
    file_hash_map = {'17flowers.tgz': 'b59a65d8d1a99cd66944d474e1289eab', 'datasplits.mat': '4828cddfd0d803c5abbdebcb1e148a1e'}
    name = "flowers"
    dataset_type = "image"
    default_task_name ="1"

    def _process(self, raw_data_dir: Path):
        archive_path = raw_data_dir.joinpath("17flowers.tgz")
        folder_name = archive_path.stem.lower()
        save_path = raw_data_dir.joinpath(folder_name)
        extract(archive_path, save_path)

    def _make_metadata(self, raw_data_dir: Path):
        class_names = ["Daffodil", "Snowdrop", "LilyValley", "Bluebell", "Crocus", "Iris", "Tigerlily", "Tulip", "Fritillary", "Sunflower", "Daisy", "Corts'Foot", "Dandelion", "Cowslip", "Buttercup", "Windflower", "Pansy"]
        image_folder = list(raw_data_dir.rglob("jpg"))[0]
        datasplits_path = raw_data_dir.joinpath('datasplits.mat')
        # Get train test sets
        f = loadmat(datasplits_path)
        # to metadata
        file_names = {}
        for task_name in self.task_names:
            file_names[task_name] = {}
            train_ids = np.concatenate((f['trn' + task_name], f['val' + task_name]), axis=-1).reshape(-1)
            val_ids = np.array(f['tst' + task_name]).reshape(-1)
            for split in ["train", "val"]:
                file_tuples = []
                indices = train_ids if split == 'train' else val_ids
                for sample_id in indices:
                    path = str(image_folder.joinpath('image_' + str(sample_id).zfill(4) + '.jpg').relative_to(raw_data_dir))
                    label = class_names[(sample_id - 1) // 80]
                    file_tuples.append((path, label))
                file_names[task_name][split] = file_tuples
        # to class name
        class_names = self._make_class_names(file_names)
        # save
        metadata = dict(file_names=file_names, class_names=class_names)
        torch.save(metadata, self.metadata_path)

    task_names = ["1", "2", "3"]



