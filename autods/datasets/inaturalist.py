from functools import cached_property
from operator import itemgetter
import os
import json
from pathlib import Path
from typing import List

import pandas as pd
import torch

from autods.dataset import Dataset
from autods.utils import extract, is_archive


class Inaturalist(Dataset):
    metadata_url = "https://www.kaggle.com/datasets/clorichel/boat-types-recognition"
    remote_urls = {
        "train.tar.gz": "https://ml-inat-competition-datasets.s3.amazonaws.com/2021/train.tar.gz",
        "val.tar.gz": "https://ml-inat-competition-datasets.s3.amazonaws.com/2021/val.tar.gz",
        "train.json.tar.gz": "https://ml-inat-competition-datasets.s3.amazonaws.com/2021/train.json.tar.gz",
        "val.json.tar.gz": "https://ml-inat-competition-datasets.s3.amazonaws.com/2021/val.json.tar.gz",
    }
    name = "inaturalist"
    dataset_type = "image"
    default_task_name = "species"
    task_names = ["species"]
    file_hash_map = {
        "train.tar.gz": "e0526d53c7f7b2e3167b2b43bb2690ed",
        "val.tar.gz": "f6f6e0e242e3d4c9569ba56400938afc",
        "train.json.tar.gz": "38a7bb733f7a09214d44293460ec0021",
        "val.json.tar.gz": "4d761e0f6a86cc63e8f7afc91f6a8f0b",
    }

    subset_names = [
        "amphibians",
        "animalia",
        "arachnids",
        "birds",
        "fungi",
        "insects",
        "mammals",
        "mollusks",
        "plants",
        "ray-finned fishes",
        "reptiles",
    ]

    def _process(self, raw_data_dir: Path):
        for archive in self.remote_urls.keys():
            archive_path = raw_data_dir.joinpath(archive)
            folder_name = archive_path.stem.lower()
            save_path = raw_data_dir.joinpath(folder_name)
            extract(archive_path, save_path)

    def _make_metadata(self, raw_data_dir: Path):
        file_names = {}
        subset_indices = {}
        for split in ["train", "val"]:
            base_folder = list(raw_data_dir.rglob(split))[-1].parent
            annotation_file = list(raw_data_dir.rglob(f"{split}.json"))[0]
            # read from annotation file
            with open(annotation_file, "r") as f:
                annos = json.load(f)
            img_to_path = {}
            for image in annos["images"]:
                img_to_path[image["id"]] = image["file_name"]
            cate_to_name = {}
            for c in annos["categories"]:
                # NOTE: below is the original dataset implementation of using the common name as a label.
                # We use the class as a label due to disproportionally large number of classes in the dataset
                # compared to others.
                cate_to_name[c["id"]] = (c["supercategory"], c["class"])
            img_to_cate = {}
            for a in annos["annotations"]:
                img_to_cate[a["image_id"]] = a["category_id"]
            # to metadata
            for k, path in img_to_path.items():
                subset, label = cate_to_name[img_to_cate[k]]
                path = str(base_folder.joinpath(path).relative_to(raw_data_dir))
                if self.default_task_name not in file_names:
                    file_names[self.default_task_name] = {}
                if split not in file_names[self.default_task_name]:
                    file_names[self.default_task_name][split] = []
                # append index of file
                subset = subset.lower()
                if subset not in subset_indices:
                    subset_indices[subset] = {}
                if split not in subset_indices[subset]:
                    subset_indices[subset][split] = []
                subset_indices[subset][split].append(
                    len(file_names[self.default_task_name][split])
                )
                file_names[self.default_task_name][split].append((path, label))

        # to class name
        class_names = self._make_class_names(file_names)
        # save
        metadata = dict(
            file_names=file_names,
            class_names=class_names,
            subset_indices=subset_indices,
        )
        torch.save(metadata, self.metadata_path)

    def _init_actions(self, subset_name: str | None = None, **kwargs):
        super()._init_actions(**kwargs)
        if subset_name is not None:
            assert (
                subset_name.lower() in self.subset_names
            ), f"Could not find subset `{subset_name}` of {self.class_name} in {self.subset_names}."
            subset_indices = self.metadata["subset_indices"][subset_name][
                self.split
            ]
            if len(set(subset_indices)) != len(subset_indices):
                raise ValueError("Duplicate indices in subset_indices")
            if (
                max(subset_indices)
                >= len(self.metadata["file_names"][self.current_task_name][self.split])
                or min(subset_indices) < 0
            ):
                raise ValueError(
                    f"Invalid subset_indices value range. {min(subset_indices)} - {min(subset_indices)}"
                )
            self.dataset = itemgetter(*subset_indices)(self.dataset)
