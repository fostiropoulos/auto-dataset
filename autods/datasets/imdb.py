from pathlib import Path
from typing import Any, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from autods.dataset import Dataset
from autods.feat_extractors.gpt2 import GPT2
from autods.utils import extract


class Imdb(Dataset):
    metadata_url = "https://ai.stanford.edu/~amaas/data/sentiment/"
    remote_urls = {
        "aclImdb_v1.tar.gz": "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz",
    }
    file_hash_map = {"aclImdb_v1.tar.gz": "7c2ac02c03563afcf9b574c7e56c153a"}
    name = "imdb"
    dataset_type = "image"
    default_task_name ="none"
    default_feat_extractor = "gpt2"

    def _process(self, raw_data_dir: Path):
        for archive in self.remote_urls.keys():
            archive_path = raw_data_dir.joinpath(archive)
            folder_name = archive_path.stem.lower()
            save_path = raw_data_dir.joinpath(folder_name)
            extract(archive_path, save_path)

    def _make_metadata(self, raw_data_dir: Path):
        main_folder = list(raw_data_dir.rglob("aclImdb"))[0]
        # train test split
        # to metadata
        ds_paths = np.array(list(main_folder.rglob("pos/*.txt")) + list(
            main_folder.rglob("neg/*.txt")
        ))
        train_idx, val_idx = train_test_split(
            np.arange(len(ds_paths)),
            test_size=self.test_size,
            random_state=self.test_split_random_state,
        )
        file_names = {}
        file_names[self.default_task_name] = {}
        for idxs, split in zip([train_idx, val_idx], ["train", "val"]):
            file_tuples = []
            for p in ds_paths[idxs]:
                label = int(p.stem.split("_")[-1])
                file_tuples.append((p.relative_to(raw_data_dir), label))
            file_names[self.default_task_name][split] = file_tuples
        # to class name
        class_names = self._make_class_names(file_names)
        # save
        metadata = dict(file_names=file_names, class_names=class_names)
        torch.save(metadata, self.metadata_path)

    task_names = ["none"]


    def get_feats(self, model: GPT2, paths):
        x = [self._data_loader(path) for path in paths]
        feats = model.get_text_feats(x).cpu().numpy()
        return feats

    def _data_loader(self, index: str):
        path = self.dataset_path.joinpath(index)
        return path.read_text()
