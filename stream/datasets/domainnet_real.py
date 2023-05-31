import itertools
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd

import torch

from stream.dataset import Dataset
from stream.feat_extractors.clip import ClipModel
from stream.feat_extractors.vit import ViT
from stream.utils import extract, is_archive

labels_csv = "https://gist.githubusercontent.com/fostiropoulos/4f65c7b44f0827af1ea5ebc7b1197505/raw/536b2cd654a76a06d02204f6db3b8e2ad2984ecc/domainnet-labels.csv"

class DomainNetReal(Dataset):
    metadata_url = "http://ai.bu.edu/M3SDA/"
    remote_urls = {
        "real.zip": "http://csr.bu.edu/ftp/visda/2019/multi-source/real.zip",
        "real_train.txt": "http://csr.bu.edu/ftp/visda/2019/multi-source/txt/real_train.txt",
        "real_test.txt": "http://csr.bu.edu/ftp/visda/2019/multi-source/txt/real_test.txt",
    }
    name = "domainnet_real"
    file_hash_map = {
        "real.zip": "dcc47055e8935767784b7162e7c7cca6",
        "real_train.txt": "99c32cef0a3fe8060b47f7b5436e0197",
        "real_test.txt": "a7db28187438a946880c00783dd22f5e",
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
        label_map = dict(zip(label_df["label_name"], label_df["label_id"]))
        for task_name in self.task_names:
            file_names[task_name] = {}
            for split in ["train", "val"]:
                split_file = list(
                    raw_data_dir.rglob(
                        f"real_{split if split == 'train' else 'test'}.txt"
                    )
                )[0]
                image_paths = [
                    path.split(" ")[0]
                    for path in split_file.read_text().split("\n")
                    if len(path) > 0
                ]
                file_tuples = []
                for path in image_paths:
                    path = raw_data_dir.joinpath(f"real/real").joinpath(path)
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

    task_names =  ["none"]

if __name__ == "__main__":
    root_path = Path.home().joinpath("stream_ds")
    ds = DomainNetReal(root_path)
    ds.make_label_splits()
    # df = ds.make_metadata(ds.root_path)
    # df
