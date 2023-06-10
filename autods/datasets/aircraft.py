from pathlib import Path
from typing import List

import numpy as np

from autods.dataset import Dataset
from autods.utils import extract, is_archive

import torch


class Aircraft(Dataset):
    metadata_url = "https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/"
    remote_urls = {
        "fgvc-aircraft-2013b.tar.gz": "https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz"
    }
    name = "aircraft"
    file_hash_map = {"fgvc-aircraft-2013b.tar.gz": "d4acdd33327262359767eeaa97a4f732"}
    dataset_type = "image"
    default_task_name ="family"

    task_names = ["family", "manufacturer", "variant"]

    def _process(self, raw_data_dir: Path):
        archive_path = raw_data_dir.joinpath("fgvc-aircraft-2013b.tar.gz")
        folder_name = archive_path.stem.lower()
        save_path = raw_data_dir.joinpath(folder_name)
        extract(archive_path, save_path)

    def _make_metadata(self, raw_data_dir: Path):
        image_paths = list(list(raw_data_dir.rglob("images"))[0].rglob("*"))

        def get_file_path(filename):
            for image_path in image_paths:
                if filename in image_path.as_posix():
                    return image_path.relative_to(self.dataset_path).as_posix()
            raise ValueError

        file_names = {}
        # NOTE if a dataset does not have subset variants. Simply add None as key
        for task_name in self.task_names:
            file_names[task_name] = {}
            for split in ["train", "val"]:
                devkit = list(raw_data_dir.rglob(f"images_{task_name}_{split}.txt"))[
                    0
                ].read_text()

                file_tuples = []
                for line in devkit.split("\n"):
                    # this is because EOF has an empty line
                    if len(line) == 0:
                        continue
                    name, *label = line.split(" ")
                    file_tuples.append((get_file_path(name), "_".join(label)))
                file_names[task_name][split] = file_tuples
        # to class name
        class_names = self._make_class_names(file_names)
        # save
        metadata = dict(file_names=file_names, class_names=class_names)
        torch.save(metadata, self.metadata_path)
