from pathlib import Path
import pickle
import torch
import numpy as np
from torchvision.transforms import transforms

import os
from pathlib import Path
from typing import List

from autods.dataset import Dataset
from autods.utils import extract, is_archive

# TODO Clean-up
class Cifar100(Dataset):
    metadata_url = "https://www.cs.toronto.edu/~kriz/cifar.html"
    remote_urls = {
        "cifar-100-python.tar.gz": "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz",
    }
    name = "cifar100"
    file_hash_map = {'cifar-100-python.tar.gz': 'eb9058c3a382ffc7106e4002c42a8d85'}
    dataset_type = "image"
    default_task_name ="none"

    default_feat_extractor = "clip"
    def _process(self, raw_data_dir: Path):
        for archive in self.remote_urls.keys():
            archive_path = raw_data_dir.joinpath(archive)
            folder_name = archive_path.stem.lower()
            save_path = raw_data_dir.joinpath(folder_name)
            extract(archive_path, save_path)

        train_set = self.unpickle(list(raw_data_dir.rglob('train'))[0])
        train_labels = np.array(train_set[b'fine_labels']).reshape(len(train_set[b'fine_labels']), 1)
        self._nparray_to_image(train_set[b'data'].reshape(-1,3,32,32).transpose(0,2,3,1), train_labels, 'train')

        test_set = self.unpickle(list(raw_data_dir.rglob('test'))[0])
        test_labels = np.array(test_set[b'fine_labels']).reshape(len(test_set[b'fine_labels']), 1)
        self._nparray_to_image(test_set[b'data'].reshape(-1,3,32,32).transpose(0,2,3,1), test_labels, 'val')

    def _make_metadata(self, raw_data_dir: Path):
        train_images = list(raw_data_dir.rglob("train/*.png"))
        val_images = list(raw_data_dir.rglob("val/*.png"))
        # read class name
        label_names = self.unpickle(list(raw_data_dir.rglob('meta'))[0])[b'fine_label_names']
        label_names = [b.decode('utf-8') for b in label_names]
        # to metadata
        file_names = {}
        file_names[self.default_task_name] = {}
        # to metadata
        file_names = {}
        file_names[self.default_task_name] = {}
        for split in ["train", "val"]:
            file_tuples = []
            images = train_images if split == 'train' else val_images
            for path in images:
                label = label_names[int(path.stem.split('_')[1])]
                path = str(path.relative_to(raw_data_dir))
                file_tuples.append((path, label))
            file_names[self.default_task_name][split] = file_tuples
        # to class name
        class_names = self._make_class_names(file_names)
        # save
        metadata = dict(file_names=file_names, class_names=class_names)
        torch.save(metadata, self.metadata_path)

    task_names = ["none"]

    @classmethod
    def make_transform(cls):
        base_transform = [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
        train_transform = transforms.Compose(
            [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
            + base_transform
        )
        test_transform = transforms.Compose(base_transform)
        return train_transform, test_transform

    def unpickle(self, file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

if __name__ == "__main__":
    dataset_path = Path.home().joinpath("stream_ds")
    ds = Cifar100(dataset_path, process=True)
    df = ds.make_metadata(ds.dataset_path)
    df
