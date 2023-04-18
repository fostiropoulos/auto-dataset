import os
from pathlib import Path
from typing import List

import h5py
import numpy as np
import pandas as pd
import scipy
from sklearn.model_selection import train_test_split

from stream.dataset import Dataset
from stream.utils import extract, is_archive

import torch


class Celeba(Dataset):
    metadata_url = "https://www.kaggle.com/datasets/jessicali9530/celeba-dataset"
    remote_urls = {
        "celeba-dataset.zip": "kaggle datasets download -d jessicali9530/celeba-dataset",
    }
    name = "celeba"
    file_hash_map = {'celeba-dataset.zip': '19d14e50c7754acf2ed37e789d66d525'}
    dataset_type = "image"
    default_task_name ="5_o_Clock_Shadow"

    def _process(self, raw_data_dir: Path):
        for archive in self.remote_urls.keys():
            archive_path = raw_data_dir.joinpath(archive)
            folder_name = archive_path.stem.lower()
            save_path = raw_data_dir.joinpath(folder_name)
            extract(archive_path, save_path)

    def _make_metadata(self, raw_data_dir: Path):
        label_to_name = {}
        for task_name in self.task_names:
            label_to_name[task_name + '_-1'] = 'not ' + task_name
            label_to_name[task_name + '_1'] = task_name
        # get images
        images = list(raw_data_dir.rglob("*.jpg"))
        labels = pd.read_csv(list(raw_data_dir.rglob("list_attr_celeba.csv"))[0])
        labels = labels.set_index('image_id')
        # train test split
        train_idx, val_idx = train_test_split(np.arange(len(images)),
                                              test_size=self.test_size, random_state=self.test_split_random_state)
        # to metadata
        file_names = {}
        for task_name in self.task_names:
            file_names[task_name] = {}
            for split in ["train", "val"]:
                file_tuples = []
                indices = train_idx if split == 'train' else val_idx
                for idx in indices:
                    path = images[idx]
                    label = label_to_name[task_name + '_' + str(labels.loc[path.name, task_name])]
                    path = str(path.relative_to(raw_data_dir))
                    file_tuples.append((path, label))
                file_names[task_name][split] = file_tuples
        # to class name
        class_names = self._make_class_names(file_names)
        # save
        metadata = dict(file_names=file_names, class_names=class_names)
        torch.save(metadata, self.metadata_path)

    task_names = ["5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes", "Bald", "Bangs", "Big_Lips",
                "Big_Nose", "Black_Hair", "Blond_Hair", "Blurry", "Brown_Hair", "Bushy_Eyebrows", "Chubby",
                "Double_Chin", "Eyeglasses", "Goatee", "Gray_Hair", "Heavy_Makeup", "High_Cheekbones", "Male",
                "Mouth_Slightly_Open", "Mustache", "Narrow_Eyes", "No_Beard", "Oval_Face", "Pale_Skin", "Pointy_Nose",
                "Receding_Hairline", "Rosy_Cheeks", "Sideburns", "Smiling", "Straight_Hair", "Wavy_Hair",
                "Wearing_Earrings", "Wearing_Hat", "Wearing_Lipstick", "Wearing_Necklace", "Wearing_Necktie", "Young"]



