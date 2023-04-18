from pathlib import Path

import torch
from torchvision.transforms import transforms

import os
from pathlib import Path
from typing import List

from stream.dataset import Dataset
from stream.utils import extract, is_archive

import torch


class TinyImagenet(Dataset):
    metadata_url = "http://cs231n.stanford.edu/reports/2017/pdfs/930.pdf"
    remote_urls = {
        "tiny-imagenet-200.zip": "http://cs231n.stanford.edu/tiny-imagenet-200.zip",
    }
    name = "tinyimagenet"
    file_hash_map = {"tiny-imagenet-200.zip": "90528d7ca1a48142e341f4ef8d21d0de"}
    dataset_type = "image"
    default_task_name ="none"
    task_names = ["none"]

    def _process(self, raw_data_dir: Path):
        for archive in self.remote_urls.keys():
            archive_path = raw_data_dir.joinpath(archive)
            folder_name = archive_path.stem.lower()
            save_path = raw_data_dir.joinpath(folder_name)
            extract(archive_path, save_path)

        dataset_folder = list(raw_data_dir.rglob("wnids.txt"))[0].parent
        assert dataset_folder.exists(), "Folder does not exist"

        val_folder = dataset_folder.joinpath("val")
        wnids = dataset_folder.joinpath("wnids.txt").read_text().split("\n")
        for folder_name in wnids:
            os.makedirs(val_folder.joinpath(folder_name), exist_ok=True)

        val_dict = {}
        val_ann = val_folder.joinpath("val_annotations.txt").read_text()
        for line in val_ann.split("\n"):
            split_line = line.split("\t")
            if len(split_line) > 2:
                val_dict[split_line[0]] = split_line[1]
        assert len(set(val_dict.values()).difference(wnids)) == 0
        images = val_folder.joinpath("images").glob("*")
        for image_path in images:
            image_name = image_path.name
            class_name = val_dict[image_name]
            image_path.rename(val_folder.joinpath(class_name, image_name))

        val_folder.joinpath("images").rmdir()

    def _make_metadata(self, raw_data_dir: Path):
        train_folder = list(raw_data_dir.rglob("train"))[0]
        val_folder = list(raw_data_dir.rglob("val"))[0]
        # read class name
        class_names = list(raw_data_dir.rglob("words.txt"))[0].read_text().split("\n")
        label_to_name = {}
        for line in class_names:
            key, name = line.split("\t")
            label_to_name[key] = name
        # to metadata
        file_names = {}
        file_names[self.default_task_name] = {}
        for split in ["train", "val"]:
            file_tuples = []
            image_folder = train_folder if split == "train" else val_folder
            images = list(image_folder.rglob("*.JPEG"))
            for path in images:
                if split == "train":
                    label = label_to_name[path.parent.parent.name]
                else:
                    label = label_to_name[path.parent.name]
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
        augment_transform = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(64, padding=8),
        ] + base_transform
        train_transform = transforms.Compose(augment_transform)
        test_transform = transforms.Compose(base_transform)
        return train_transform, test_transform
