import logging
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
from stream.feat_extractors.gpt2 import GPT2

import torch


class Amazon(Dataset):
    name = "amazon"
    metadata_url = "https://huggingface.co/datasets/amazon_us_reviews"
    remote_urls = {
        # 'Wireless_v1_00.tsv.gz': 'https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Wireless_v1_00.tsv.gz',
        # 'Watches_v1_00.tsv.gz': 'https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Watches_v1_00.tsv.gz',
        # 'Video_Games_v1_00.tsv.gz': 'https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Video_Games_v1_00.tsv.gz',
        # 'Video_DVD_v1_00.tsv.gz': 'https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Video_DVD_v1_00.tsv.gz',
        # 'Video_v1_00.tsv.gz': 'https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Video_v1_00.tsv.gz',
        # 'Toys_v1_00.tsv.gz': 'https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Toys_v1_00.tsv.gz',
        # 'Tools_v1_00.tsv.gz': 'https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Tools_v1_00.tsv.gz',
        # 'Sports_v1_00.tsv.gz': 'https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Sports_v1_00.tsv.gz',
        # 'Software_v1_00.tsv.gz': 'https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Software_v1_00.tsv.gz',
        # 'Shoes_v1_00.tsv.gz': 'https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Shoes_v1_00.tsv.gz',
        # 'Pet_Products_v1_00.tsv.gz': 'https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Pet_Products_v1_00.tsv.gz',
        # 'Personal_Care_Appliances_v1_00.tsv.gz': 'https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Personal_Care_Appliances_v1_00.tsv.gz',
        # 'PC_v1_00.tsv.gz': 'https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_PC_v1_00.tsv.gz',
        # 'Outdoors_v1_00.tsv.gz': 'https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Outdoors_v1_00.tsv.gz',
        # 'Office_Products_v1_00.tsv.gz': 'https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Office_Products_v1_00.tsv.gz',
        # 'Musical_Instruments_v1_00.tsv.gz': 'https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Musical_Instruments_v1_00.tsv.gz',
        # 'Music_v1_00.tsv.gz': 'https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Music_v1_00.tsv.gz',
        "Mobile_Electronics_v1_00.tsv.gz": "https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Mobile_Electronics_v1_00.tsv.gz",
        # 'Mobile_Apps_v1_00.tsv.gz': 'https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Mobile_Apps_v1_00.tsv.gz',
        # 'Major_Appliances_v1_00.tsv.gz': 'https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Major_Appliances_v1_00.tsv.gz',
        # 'Luggage_v1_00.tsv.gz': 'https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Luggage_v1_00.tsv.gz',
        # 'Lawn_and_Garden_v1_00.tsv.gz': 'https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Lawn_and_Garden_v1_00.tsv.gz',
        # 'Kitchen_v1_00.tsv.gz': 'https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Kitchen_v1_00.tsv.gz',
        # 'Jewelry_v1_00.tsv.gz': 'https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Jewelry_v1_00.tsv.gz',
        # 'Home_Improvement_v1_00.tsv.gz': 'https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Home_Improvement_v1_00.tsv.gz',
        # 'Home_Entertainment_v1_00.tsv.gz': 'https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Home_Entertainment_v1_00.tsv.gz',
        # 'Home_v1_00.tsv.gz': 'https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Home_v1_00.tsv.gz',
        # 'Health_Personal_Care_v1_00.tsv.gz': 'https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Health_Personal_Care_v1_00.tsv.gz',
        # 'Grocery_v1_00.tsv.gz': 'https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Grocery_v1_00.tsv.gz',
        # 'Gift_Card_v1_00.tsv.gz': 'https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Gift_Card_v1_00.tsv.gz',
        # 'Furniture_v1_00.tsv.gz': 'https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Furniture_v1_00.tsv.gz',
        # 'Electronics_v1_00.tsv.gz': 'https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Electronics_v1_00.tsv.gz',
        # 'Digital_Video_Games_v1_00.tsv.gz': 'https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Digital_Video_Games_v1_00.tsv.gz',
        # 'Digital_Video_Download_v1_00.tsv.gz': 'https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Digital_Video_Download_v1_00.tsv.gz',
        # 'Digital_Software_v1_00.tsv.gz': 'https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Digital_Software_v1_00.tsv.gz',
        # 'Digital_Music_Purchase_v1_00.tsv.gz': 'https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Digital_Music_Purchase_v1_00.tsv.gz',
        # 'Digital_Ebook_Purchase_v1_00.tsv.gz': 'https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Digital_Ebook_Purchase_v1_00.tsv.gz',
        # 'Digital_Ebook_Purchase_v1_01.tsv.gz': 'https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Digital_Ebook_Purchase_v1_01.tsv.gz',
        # 'Camera_v1_00.tsv.gz': 'https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Camera_v1_00.tsv.gz',
        # 'Books_v1_00.tsv.gz': 'https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Books_v1_00.tsv.gz',
        # 'Books_v1_01.tsv.gz': 'https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Books_v1_01.tsv.gz',
        # 'Books_v1_02.tsv.gz': 'https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Books_v1_02.tsv.gz',
        # 'Beauty_v1_00.tsv.gz': 'https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Beauty_v1_00.tsv.gz',
        # 'Baby_v1_00.tsv.gz': 'https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Baby_v1_00.tsv.gz',
        # 'Automotive_v1_00.tsv.gz': 'https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Automotive_v1_00.tsv.gz',
        # 'Apparel_v1_00.tsv.gz': 'https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Apparel_v1_00.tsv.gz',
    }

    file_hash_map = {
        # 'Wireless_v1_00.tsv.gz': '2eca39340abcb615b21a5b7eec692fbe',
        # 'Watches_v1_00.tsv.gz': '7e32a656298e36bb19d67db9a489d9bf',
        # 'Video_Games_v1_00.tsv.gz': 'd16fc02870916370a25ff2d30cca9922',
        # 'Video_DVD_v1_00.tsv.gz': '4277de55db176f21866ec09c48eada7e',
        # 'Video_v1_00.tsv.gz': '310c703d180e588c251a8ad92c72322d',
        # 'Toys_v1_00.tsv.gz': '9e8b0cd140347ebc4fc19c61964b32fb',
        # 'Tools_v1_00.tsv.gz': '5f63885e7681c6c9b1a161f3f93c9cfc',
        # 'Sports_v1_00.tsv.gz': 'c651d20f67f2d6a89b6fe80a05ba825c',
        # 'Software_v1_00.tsv.gz': 'af84cba5eaf6c0ac7f6044832e9cc239',
        # 'Shoes_v1_00.tsv.gz': 'b51e298d07efc27a13cd9482fef33347',
        # 'Pet_Products_v1_00.tsv.gz': 'd6593da9a117d6ffba7bd05a4c3e5633',
        # 'Personal_Care_Appliances_v1_00.tsv.gz': '995bc4deb000e754719a4ef0267ce339',
        # 'PC_v1_00.tsv.gz': 'ed5c4b472c45ddb49559b0ecd6b9a01d',
        # 'Outdoors_v1_00.tsv.gz': '95a8b6a5d4cd30b7c3a79dbafb88ea78',
        # 'Office_Products_v1_00.tsv.gz': '9f00d191da6ba3c875adfba0de90a973',
        # 'Musical_Instruments_v1_00.tsv.gz': '19304f2bdc45b45a1408718fe57efa98',
        # 'Music_v1_00.tsv.gz': 'fb6762423e5bf2406bf96b60b4fe7dd6',
        "Mobile_Electronics_v1_00.tsv.gz": "913852bd42a08dacd6988a7917f9cbec",
        # 'Mobile_Apps_v1_00.tsv.gz': 'acf22ab56ce325d95076b6072b595bf8',
        # 'Major_Appliances_v1_00.tsv.gz': '6e7562b40e468be207573c3b8060da2e',
        # 'Luggage_v1_00.tsv.gz': 'ad4dcc623f1aed4b8eb6f4422cd83371',
        # 'Lawn_and_Garden_v1_00.tsv.gz': '00fb0aa5ef0c103f1705a01ddd0699c5',
        # 'Kitchen_v1_00.tsv.gz': 'b49b60a236486781558e5432047ab1b1',
        # 'Jewelry_v1_00.tsv.gz': 'd1cb306e78177c5f480e18de31d1a384',
        # 'Home_Improvement_v1_00.tsv.gz': '95e38f6e6261e4472a40d57af9ddee64',
        # 'Home_Entertainment_v1_00.tsv.gz': '1752a72d6cc123687cb9b7ae53f57281',
        # 'Home_v1_00.tsv.gz': 'f0115d5ff57b3e224b146a25b49937fc',
        # 'Health_Personal_Care_v1_00.tsv.gz': 'e796eb8d8c4c57ab4c13026439a86963',
        # 'Grocery_v1_00.tsv.gz': '56c47e250e0e3c735807a9cd429762d5',
        # 'Gift_Card_v1_00.tsv.gz': '1441d8f673cb88eba008e3b40844100c',
        # 'Furniture_v1_00.tsv.gz': '3308d168df9baec2523a057f3a7ec91f',
        # 'Electronics_v1_00.tsv.gz': '8a56fd53b8dab0ae64edd8e5a83e4332',
        # 'Digital_Video_Games_v1_00.tsv.gz': 'd796b44fc215986f1595226ae92e2818',
        # 'Digital_Video_Download_v1_00.tsv.gz': '308fa528e666b84547cfe415c9cf5832',
        # 'Digital_Software_v1_00.tsv.gz': '402fe7c01fb4febdc4e456258a1f838c',
        # 'Digital_Music_Purchase_v1_00.tsv.gz': 'a6478174df7744c984445aa4838c0010',
        # 'Digital_Ebook_Purchase_v1_00.tsv.gz': '7c21348036ea445f9080b87063ac59c0',
        # 'Digital_Ebook_Purchase_v1_01.tsv.gz': 'e73e5b5c368eaebd561c72694d7a58a1',
        # 'Camera_v1_00.tsv.gz': '7457488619e2b20ba396038f2964af8b',
        # 'Books_v1_00.tsv.gz': '6816f098f03932aade286ae178b15aa2',
        # 'Books_v1_01.tsv.gz': '0ceaada8274bae722319537c20bef12a',
        # 'Books_v1_02.tsv.gz': '527cfd9575f0801dc89e433752bb485d',
        # 'Beauty_v1_00.tsv.gz': '03ec0629915d0858cc5236155a4ee544',
        # 'Baby_v1_00.tsv.gz': '54372983ec70eff09770eb2028669e1d',
        # 'Automotive_v1_00.tsv.gz': '7a81d4cecd486a53304f84c1e93412bd',
        # 'Apparel_v1_00.tsv.gz': 'e2976fdc2992a772aae9f135ebf6f9c7',
    }
    dataset_type = "text"
    default_task_name ="none"
    default_feat_extractor = "gpt2"

    def _process(self, raw_data_dir: Path):
        for archive in self.remote_urls.keys():
            archive_path = raw_data_dir.joinpath(archive)
            extract(archive_path, raw_data_dir)

    def _make_metadata(self, raw_data_dir: Path):
        data = list(raw_data_dir.rglob("Mobile_Electronics_v1_00.tsv"))[0]
        data = pd.read_csv(
            data, sep="\t", header=0, low_memory=False, on_bad_lines="skip"
        )
        data = data[~data["review_body"].isna()]
        # train test split
        train_idx, val_idx = train_test_split(
            np.arange(len(data)),
            test_size=self.test_size,
            random_state=self.test_split_random_state,
        )

        # to metadata
        file_names = {}
        file_names[self.default_task_name] = {}
        for split in ["train", "val"]:
            file_tuples = []
            indices = train_idx if split == "train" else val_idx
            for idx in indices:
                label = data.iloc[idx]["star_rating"]
                file_tuples.append((data.iloc[idx].name, label))
            file_names[self.default_task_name][split] = file_tuples

        # to class name
        class_names = self._make_class_names(file_names)
        # save
        metadata = dict(file_names=file_names, class_names=class_names)
        torch.save(metadata, self.metadata_path)

    task_names = ["none"]

    def _init_actions(self):
        super()._init_actions()
        if not self.feats_name:
            data = list(self.dataset_path.rglob("Mobile_Electronics_v1_00.tsv"))[0]
            self.csv_dataset = pd.read_csv(
                data, sep="\t", header=0, low_memory=False, on_bad_lines="skip"
            )

    def _data_loader(self, index: int):
        return self.csv_dataset.loc[index]["review_body"]
