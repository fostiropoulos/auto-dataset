from pathlib import Path
from typing import List

from stream.dataset import Dataset
from stream.utils import extract, is_archive

import torch


class Rvl(Dataset):
    metadata_url = "https://huggingface.co/datasets/rvl_cdip"
    remote_urls = {
        "rvl-cdip.tar.gz": "https://huggingface.co/datasets/rvl_cdip/resolve/main/data/rvl-cdip.tar.gz",
        "train.txt": "https://huggingface.co/datasets/rvl_cdip/resolve/main/data/train.txt",
        "test.txt": "https://huggingface.co/datasets/rvl_cdip/resolve/main/data/test.txt",
        "val.txt": "https://huggingface.co/datasets/rvl_cdip/resolve/main/data/val.txt",
    }
    name = "rvl"
    file_hash_map = {
        "rvl-cdip.tar.gz": "d641dd4866145316a1ed628b420d8b6c",
        "train.txt": "7198b53f8950428b98927af03ade094e",
        "test.txt": "a74563497782b6b8fef5921c2de1c4c5",
        "val.txt": "efb16fad294f10251ca1de050a99785e",
    }

    dataset_type = "image"
    default_task_name ="none"
    bad_files = ["imagese/e/j/e/eje42e00/2500126531_2500126536.tif"]

    def _process(self, raw_data_dir: Path):
        for archive in self.remote_urls.keys():
            archive_path = raw_data_dir.joinpath(archive)
            if is_archive(archive_path):
                folder_name = archive_path.stem.lower()
                save_path = raw_data_dir.joinpath(folder_name)
                extract(archive_path, save_path)

    def _make_metadata(self, raw_data_dir: Path):
        label_to_name = {
            "0": "letter",
            "1": "form",
            "2": "email",
            "3": "handwritten",
            "4": "advertisement",
            "5": "scientific report",
            "6": "scientific publication",
            "7": "specification",
            "8": "file folder",
            "9": "news article",
            "10": "budget",
            "11": "invoice",
            "12": "presentation",
            "13": "questionnaire",
            "14": "resume",
            "15": "memo",
        }
        # TODO rglob is really slow for large dataset. The paths should be fixed.
        train_images = list(raw_data_dir.rglob("train.txt"))[0]
        val_images = list(raw_data_dir.rglob("test.txt"))[0]
        image_folder = list(raw_data_dir.rglob("images"))[0]
        # to metadata
        file_names = {}
        file_names[self.default_task_name] = {}
        for split in ["train", "val"]:
            file_tuples = []
            images = train_images if split == "train" else val_images
            with open(images, "r") as f:
                for line in f.readlines():
                    line = line.split("\n")[0]
                    if len(line) == 0:
                        continue
                    path, label = line.split(" ")
                    if path in self.bad_files:
                        continue
                    label = label_to_name[label]
                    path = str(image_folder.joinpath(path).relative_to(raw_data_dir))
                    file_tuples.append((path, label))
            file_names[self.default_task_name][split] = file_tuples
        # to class name
        class_names = self._make_class_names(file_names)
        # save
        metadata = dict(file_names=file_names, class_names=class_names)
        torch.save(metadata, self.metadata_path)

    task_names = ["none"]
