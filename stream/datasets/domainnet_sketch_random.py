import itertools
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd

import torch

from stream.dataset import Dataset
from stream.feat_extractors.vit import ViT
from stream.utils import extract, is_archive
from stream.datasets.domainnet_real import labels_csv


class DomainNetSketchRandom(Dataset):
    metadata_url = "http://ai.bu.edu/M3SDA/"
    remote_urls = {
        # different from the sketch in sketch.py
        "sketch.zip": "http://csr.bu.edu/ftp/visda/2019/multi-source/sketch.zip",
        "sketch_train.txt": "http://csr.bu.edu/ftp/visda/2019/multi-source/txt/sketch_train.txt",
        "sketch_test.txt": "http://csr.bu.edu/ftp/visda/2019/multi-source/txt/sketch_test.txt",
    }
    name = "domainnet_sketch_random"
    file_hash_map = {
        "sketch.zip": "658d8009644040ff7ce30bb2e820850f",
        "sketch_train.txt": "9460727d2604b53da710a80abb3ed9f4",
        "sketch_test.txt": "52cedfdb58b47974b6f5f1adb9d46f3d",
    }
    dataset_type = "image"
    default_task_name = "none"
    default_feat_extractor = "vit"

    def _process(self, raw_data_dir: Path):
        for archive in self.remote_urls.keys():
            archive_path = raw_data_dir.joinpath(archive)
            if is_archive(archive_path):
                folder_name = archive_path.stem.lower()
                save_path = raw_data_dir.joinpath(folder_name)
                extract(archive_path, save_path)

    def _make_metadata(self, raw_data_dir: Path):
        # to metadata
        file_names = {}
        label_df = pd.read_csv(labels_csv)
        label_df["label_id"] = np.random.permutation(label_df["label_id"])
        label_map = dict(zip(label_df["label_name"], label_df["label_id"]))
        for task_name in self.task_names:
            file_names[task_name] = {}
            for split in ["train", "val"]:
                split_file = list(
                    raw_data_dir.rglob(
                        f"sketch_{split if split == 'train' else 'test'}.txt"
                    )
                )[0]
                image_paths = [
                    path.split(" ")[0]
                    for path in split_file.read_text().split("\n")
                    if len(path) > 0
                ]
                file_tuples = []
                for path in image_paths:
                    path = raw_data_dir.joinpath(f"sketch/sketch").joinpath(path)
                    label = " ".join(path.parent.name.split("_")).lower()
                    label_remapped = label_map[label]
                    path = str(path.relative_to(raw_data_dir))
                    file_tuples.append((path, label_remapped))
                file_names[task_name][split] = file_tuples
        # to class name
        class_names = self._make_class_names(file_names)
        # save
        metadata = dict(file_names=file_names, class_names=class_names)
        torch.save(metadata, self.metadata_path)

    task_names = ["none"]


if __name__ == "__main__":
    root_path = Path.home().joinpath("stream_ds")
    ds = DomainNetSketchRandom(root_path)
    df = ds.process(clean=True)
    df
