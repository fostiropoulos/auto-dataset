from pathlib import Path
import pickle
import torch
import numpy as np
from torchvision.transforms import transforms

import os
from pathlib import Path
from typing import List

from tqdm import tqdm

from stream.dataset import Dataset
from stream.utils import extract, is_archive

# TODO Clean-up
class CImageNet(Dataset):
    metadata_url = ""
    remote_urls = {
        "cinic10.zip": "kaggle datasets download -d mengcius/cinic10",
    }
    name = "cimagenet"
    file_hash_map = {"cinic10.zip": "2fb34ba48517aff6e886d01faa80cc65"}
    dataset_type = "image"
    default_task_name ="none"

    default_feat_extractor = "vit"

    def _process(self, raw_data_dir: Path):
        for archive in self.remote_urls.keys():
            archive_path = raw_data_dir.joinpath(archive)
            folder_name = archive_path.stem.lower()
            save_path = raw_data_dir.joinpath(folder_name)
            extract(archive_path, save_path)

    def _make_metadata(self, raw_data_dir: Path):
        train_images = list(raw_data_dir.rglob("train/*/*.png"))
        val_images = list(raw_data_dir.rglob("valid/*/*.png"))
        # read class name
        labels = sorted(set([p.parent.name for p in train_images]))
        label_names = dict(zip(labels, range(10)))
        file_names = {}
        file_names[self.default_task_name] = {}
        for split in ["train", "val"]:
            file_tuples = []
            images = train_images if split == "train" else val_images
            for path in images:
                label = label_names[path.parent.name]
                if "cifar" not in path.name:
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
        with open(file, "rb") as fo:
            dict = pickle.load(fo, encoding="bytes")
        return dict


if __name__ == "__main__":
    dataset_path = Path.home().joinpath("stream_ds")
    ds = CImageNet(dataset_path)
    for train in [True, False]:
        dataset = CImageNet(
            root_path=dataset_path,
            train=train,
        )
        for s in tqdm(dataset):
            pass


    pass
