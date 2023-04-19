import hashlib
import logging
import os
import pickle
import shutil
from abc import ABC
from inspect import signature
from pathlib import Path
import tempfile
from typing import Any, Callable, Literal, Optional, Tuple, final

import lmdb
import numpy as np
import torch
from git import Repo
from PIL import Image, ImageFile
from tqdm import tqdm
from stream._base_dataset import BaseDataset
from stream.feat_extractors import FeatExtractor
from stream.feat_extractors.clip import ClipModel
from stream.feat_extractors.gpt2 import GPT2
from stream.feat_extractors.resnet import ResnetModel
from stream.feat_extractors.vit import ViT
from stream.utils import download_file


# NOTE: this property should be set really high.
# does not really allocate 1TB but it is the limit
# of the databse size.
LMDB_MAP_SIZE_1TB = 1099511627776
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

logger = logging.getLogger(__name__)


class Dataset(BaseDataset, ABC):
    splits = ["train", "val"]

    models = {
        "gpt2": GPT2,
        "vit": ViT,
        "clip": ClipModel,
        "resnet": ResnetModel,
    }
    default_feat_extractor = "clip"

    @final
    def __init__(
        self,
        root_path: str | Path,
        train: bool = True,
        task_name: Optional[str] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        test_size: Optional[int] = None,
        test_split_random_state: Optional[int] = None,
        feats_name: Optional[str] | Literal["default"] = None,
        subset_name: Optional[str] = None,
        action: Literal["download", "process", "verify", None] = None,
        clean: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.root_path = Path(root_path)
        self.class_name: str = self.__class__.__name__
        self.dataset_path = Path(root_path).joinpath(self.class_name.lower())
        self.metadata_path = self.dataset_path.joinpath("metadata.pickle")
        self.feats_path = self.dataset_path.joinpath("feats")
        self.test_size = test_size
        self.test_split_random_state = test_split_random_state
        self.subset_name = subset_name
        self.transform = transform
        self.target_transform = target_transform
        self.current_task_name = task_name
        self.split = "train" if train else "val"
        self._is_init = False

        if feats_name == "default":
            self.feats_name = getattr(
                self, "default_feat_extractor", list(self.models.keys())[0]
            )
        else:
            self.feats_name = feats_name

        self.lmdb_env = None
        self.read_txn = None
        if action == "download":
            self.download(clean=clean)
        elif action == "process":
            self.process(clean=clean)
        elif action == "verify":
            self.assert_downloaded()
        elif action is None:
            pass
        else:
            raise NotImplementedError(f"Unrecognized action {action}")

        if not self.dataset_path.exists():
            self._is_init = False
            logger.warn(
                f"Could not find a dataset in {self.dataset_path}. You will need to use `{self.class_name}.download(path)` or initialize with action=`download`"
            )
        elif not self.metadata_path.exists():
            self._is_init = False
            logger.warn(
                f"Could not find a metadata file in {self.dataset_path}. You will need to use `{self.class_name}.process(path)` or initialize with action=`process`"
            )
        else:
            self._is_init = True

        if self.feats_name is not None:
            self._load_lmdb()

        if self._is_init:
            self._init_actions(**kwargs)
        else:
            logger.warn(f"Initialized empty dataset {self.class_name}.")

    def _init_actions(self, **kwargs):
        self.metadata = torch.load(self.metadata_path)
        self.dataset = self.metadata["file_names"][self.current_task_name][self.split]
        assert (
            len(self.dataset) > 0
        ), f"No files found in the dataset {self.class_name}."
        # filepath, label list
        # filepath must be a relative path to raw_data_dir
        self.class_names = self.metadata["class_names"]
        unique_labels = np.unique([label for filename, label in self.dataset])
        self.class_to_idx = dict(zip(unique_labels, np.arange(len(unique_labels))))

    def __clean_processed(self):
        base_files = list(self.remote_urls.keys())
        self.__clean_files(self.dataset_path, base_files)

    def __clean_files(self, directory: Path, exclude_file_names: list[str] | None):
        # base_files = set(self.remote_urls.keys())
        files = set([p.name for p in directory.glob("*")])
        # NOTE this is because some of the remote_urls can be malformed or be for example "../something"
        exclude_file_names = [] if exclude_file_names is None else exclude_file_names
        rm_files = files.difference(exclude_file_names)
        if len(rm_files) > 0:
            logging.warning(f"{self.name}: Cleaning-up {', '.join(rm_files)}")
            for rm_file in rm_files:
                _f = self.dataset_path.joinpath(rm_file)
                shutil.rmtree(_f, ignore_errors=True)
                _f.unlink(missing_ok=True)

    def _load_lmdb(self):
        assert (
            self.feats_name in self.models
        ), f"Could not find {self.feats_name} in {self.models}"

        feat_path = self.feats_path.joinpath(self.feats_name, self.split)
        if not feat_path.exists():
            fn_signature = f"make_features{signature(self.make_features)}"
            logger.warn(
                f"Could not find a features file in {self.dataset_path}. You will need to use `{self.class_name}.{fn_signature}` or initialize with action=`make_features`"
            )
            self._is_init = False
            return

        self.lmdb_env = lmdb.open(
            feat_path.as_posix(),
            max_readers=1000,
            readonly=True,
            lock=False,
            readahead=True,
            meminit=False,
        )
        self.read_txn = self.lmdb_env.begin(write=False)
        try:
            self._feat_loader("file_names")
        except Exception as e:
            raise RuntimeError(f"Corrupt feats dataset {self.class_name}.") from e
        self._is_init = True

    @final
    def assert_downloaded(self) -> bool:
        class_name: str = self.__class__.__name__
        dataset_path = Path(self.root_path).joinpath(class_name.lower())
        for file, url in self.remote_urls.items():
            file_path = dataset_path.joinpath(file)
            assert file_path.exists(), f"{file_path} is missing."
            if url is not None and url.endswith(".git"):
                repo = Repo(file_path)
                file_hash = repo.head.object.hexsha
            else:
                file_hash = self.file_hash(file_path)

            assert (
                self.file_hash_map[file] == file_hash
            ), f"Integrity check failed for {self.name} and file {file}."

        return True

    def verify(self):
        if self.feats_name is not None:
            return self.verify_feature_vectors(
                self.root_path, feats_name=self.feats_name
            )
        return self.verify_processed(self.root_path)

    @final
    @classmethod
    def verify_downloaded(cls, root_path) -> bool:
        class_name: str = cls.__name__
        dataset_path = Path(root_path).joinpath(class_name.lower())
        for file, url in cls.remote_urls.items():
            file_path = dataset_path.joinpath(file)
            if not file_path.exists():
                return False
            if url is not None and url.endswith(".git"):
                repo = Repo(file_path)
                file_hash = repo.head.object.hexsha
            else:
                file_hash = cls.file_hash(file_path)

            if not cls.file_hash_map[file] == file_hash:
                return False

        return True

    @classmethod
    def verify_processed(cls, root_path) -> bool:
        class_name: str = cls.__name__
        dataset_path = Path(root_path).joinpath(class_name.lower())
        return dataset_path.joinpath("metadata.pickle").exists()

    @classmethod
    def verify_feature_vectors(cls, root_path, feats_name) -> bool:
        if feats_name == "default":
            feats_name = getattr(
                cls, "default_feat_extractor", list(cls.models.keys())[0]
            )

        class_name: str = cls.__name__
        dataset_path = Path(root_path).joinpath(class_name.lower())
        for split in ["train", "val"]:
            feat_path = dataset_path.joinpath("feats", feats_name, split)
            if not feat_path.exists():
                return False

            lmdb_env = lmdb.open(
                feat_path.as_posix(),
                max_readers=1000,
                readonly=True,
                lock=False,
                readahead=True,
                meminit=False,
            )
            read_txn = lmdb_env.begin(write=False)
            if read_txn.get(str("file_names").encode("utf-8")) is None:
                return False
        return True

    @classmethod
    def file_hash(cls, file_path) -> str:
        # 100_MB
        BUF_SIZE = 100_000_000

        md5 = hashlib.md5()

        with open(file_path, "rb") as f:
            while True:
                data = f.read(BUF_SIZE)
                if not data:
                    break
                md5.update(data)
        return md5.hexdigest()

    def download(self, clean=False):
        only_local_files = []
        if clean:
            only_local_files = [k for k, v in self.remote_urls.items() if v is None]
            logging.warn(f"Not removing local files {only_local_files}.")
            self.__clean_files(self.dataset_path, only_local_files)
        all_files = {f.name for f in self.dataset_path.glob("*")}
        local_files = {k for k, v in self.remote_urls.items() if v is None}

        if len(all_files.difference(local_files).difference(only_local_files)) > 0:
            raise FileExistsError(
                f"{self.dataset_path} is not empty. You must use with flag clean `{self.class_name}.download(path, clean=True)` that will remove all files and re-process the dataset"
            )
        logger.info("Downloading: %s" % self.class_name)
        for filename, url in self.remote_urls.items():
            download_file(url, self.dataset_path, filename)
        self.assert_downloaded()
        self.process()

    def _make_class_names(
        self, metadata: dict[str, dict[str, list[tuple]]]
    ) -> dict[str, list[str]]:
        class_names = {}
        for subset in self.task_names:
            unique_labels = list(
                np.unique([label for _, label in metadata[subset]["train"]])
            )
            class_names[subset] = unique_labels
        return class_names

    def _nparray_to_image(self, array, targets, split):
        base_path = self.dataset_path.joinpath(split)
        if not os.path.exists(base_path):
            os.makedirs(base_path)
        for i in range(len(targets)):
            label_string = "_".join([str(t) for t in targets[i]])
            path = base_path.joinpath(f"{i}_{label_string}.png")
            img = Image.fromarray(array[i])
            img.save(path)

    def process(self, clean=False):
        processed_files = {
            p.name for p in self.dataset_path.glob("*")
        } - self.remote_urls.keys()

        if clean:
            self.__clean_processed()
        elif len(processed_files) > 0:
            raise FileExistsError(
                f"Proccessed files {processed_files} already exist. You will need to pass argument `clean=True` that will remove the files and re-process the dataset."
            )
        assert self.verify_downloaded(
            self.root_path
        ), "Corrupt dataset. You will need to use `download(clean=True)` before processing."
        logger.info("Processing: %s" % self.class_name)
        self._process(self.dataset_path)
        self._make_metadata(self.dataset_path)
        if self.feats_name is None:
            self._is_init = True

    def _make_features_split(
        self,
        save_path: Path,
        batch_size: int,
        file_names: list[str],
        feat_model: torch.nn.Module,
        verbose=True,
    ):
        if save_path.exists():
            raise FileExistsError(
                f"feat directory {save_path} exists. Use `clean=True` or delete the directory."
            )
        save_path.mkdir(parents=True, exist_ok=True)
        env = lmdb.open(
            save_path.as_posix(),
            map_size=LMDB_MAP_SIZE_1TB,
            writemap=True,
            map_async=True,
            lock=False,
        )
        with env.begin(write=True) as txn:
            n_batches = int(np.ceil(len(file_names) / batch_size))
            iterator = range(n_batches) if not verbose else tqdm(range(n_batches))
            for i in iterator:
                paths = file_names[i * batch_size : (i + 1) * batch_size]
                feats = self.get_feats(feat_model, paths)

                for path, feat in zip(paths, feats):
                    txn.put(str(path).encode("utf-8"), pickle.dumps(feat))
            txn.put("file_names".encode("utf-8"), pickle.dumps(file_names))
        env.close()
        with tempfile.TemporaryDirectory() as fp:
            db = lmdb.open(save_path.as_posix())
            db.copy(fp)
            shutil.rmtree(save_path, ignore_errors=True)
            save_path.mkdir()
            db = lmdb.open(fp)
            db.copy(save_path.as_posix())

    def make_features(
        self,
        batch_size,
        device,
        feats_name: str | None = None,
        clean=False,
        split: Literal["train", "val"] | None = None,
        verbose: bool = True,
    ):
        if feats_name is None and self.feats_name is None:
            raise ValueError("Must specify `feats_name` or feature_extractor argument.")
        elif feats_name is None:
            feats_name = self.feats_name
        logger.info("Making Feats: %s" % self.class_name)
        if not self.metadata_path.exists():
            raise RuntimeError(
                f"Missing {self.metadata_path}. Please run `.process()` before making features."
            )
        metadata = torch.load(self.metadata_path)

        if split is None:
            splits = self.splits
            assert (
                not self.feats_path.joinpath(feats_name).exists() or clean
            ), f"Directory exists {self.feats_path.joinpath(feats_name)}. Use `clean=True`"
        else:
            splits = [split]
        for _split in splits:
            feat_model = self.models[feats_name](device)

            file_names = []

            for subset_name in self.task_names:
                file_name = [
                    path for path, label in metadata["file_names"][subset_name][_split]
                ]
                file_names += file_name
            file_names = sorted(list(set(file_names)))

            save_path = self.feats_path.joinpath(feats_name).joinpath(_split)
            if clean:
                shutil.rmtree(save_path, ignore_errors=True)
            self._make_features_split(
                save_path=save_path,
                batch_size=batch_size,
                file_names=file_names,
                feat_model=feat_model,
                verbose=verbose,
            )

    def get_feats(self, model: FeatExtractor, paths):
        x = [self._data_loader(path) for path in paths]
        model_name = model.__class__.__name__
        if self.dataset_type == "image":
            assert hasattr(
                model, "get_image_feats"
            ), f"{model_name} extractor must be able to process image input."
            feats = model.get_image_feats(x).cpu().numpy()
        elif self.dataset_type == "text":
            assert hasattr(
                model, "get_text_feats"
            ), f"{model_name} must be able to process text input."
            feats = model.get_text_feats(x).cpu().numpy()
        else:
            raise NotImplementedError(f"Unsupported dataset type: {self.dataset_type}")
        return feats

    def _data_loader(self, path):
        path = self.dataset_path.joinpath(path)
        with open(path, "rb") as f:
            img = Image.open(f)
            return img.convert("RGB")

    def _feat_loader(self, key):
        obj = self.read_txn.get(str(key).encode("utf-8"))
        if obj:
            return pickle.loads(obj)
        raise ValueError(f"Could not load {key} from the ldmb.")

    def _loader(self, path):
        if self.feats_name is not None:
            return self._feat_loader(path)
        return self._data_loader(path)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        filepath, target = self.dataset[index]
        sample = self._loader(filepath)
        label = self.class_to_idx[target]
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            label = self.target_transform(label)

        return sample, label
