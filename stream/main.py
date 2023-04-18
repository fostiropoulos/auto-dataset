import collections
import copy
import logging
import multiprocessing as mp
import shutil
import traceback
from functools import cached_property
from importlib import import_module
from inspect import isclass
from pathlib import Path
from pkgutil import iter_modules
from typing import Any, List, Type

import lmdb
import numpy as np
import setproctitle
import torch
import torch.utils.data as td
import tqdm
from torch.utils.data import ConcatDataset

from stream.dataset import Dataset

PACKAGE_DATASET_DIR = Path(__file__).parent.joinpath("datasets").resolve().as_posix()


def make_dataset_features(
    ds: Type[Dataset],
    dataset_args,
    batch_size,
    clean=False,
    feats_name=None,
):
    try:
        setproctitle.setproctitle(f"Features - {ds.name}")
        dataset = make_ds(ds, dataset_args)
        return dataset.make_features(
            batch_size, "cuda", clean=clean, feats_name=feats_name, verbose=False
        )

    except Exception as e:
        logging.error(f"Dataset - {ds.__name__} Error\n{traceback.format_exc()}")


def make_ds(ds_class, args):
    args = copy.deepcopy(args)
    if "action" in args:
        del args["action"]

    return ds_class(**args)


class Stream(td.Dataset):
    def __init__(
        self,
        root_path: Path,
        datasets: List[str] | None = None,
        dataset_kwargs: dict[str, dict[str, Any]] | None = None,
        task_id=None,
        make=False,
        clean=False,
        num_gpus=None,
        batch_size=None,
        **kwargs,
    ) -> None:
        """
        If clean it re-downloads the dataset if `feats_name` are not provided or removes and remakes the feats if `feats_name` argument is provided.
        """
        super().__init__()

        self.root_path = Path(root_path)

        dataset_classes = self.supported_datasets()

        dataset_names: List[str] = sorted([ds.name.lower() for ds in dataset_classes])
        self.dataset_map = collections.OrderedDict(
            (dict(zip(dataset_names, dataset_classes)).items())
        )
        if datasets is not None:
            datasets = [d.lower() for d in datasets]
            assert all(
                [d in self.dataset_map for d in datasets]
            ), f"not all datasets {datasets} were found in {dataset_names}"
            dataset_names = datasets
            self.dataset_map = collections.OrderedDict(
                {k: v for k, v in self.dataset_map.items() if k in dataset_names}
            )
        assert len(set(dataset_names)) == len(
            dataset_names
        ), f"Duplicate dataset names found {dataset_names}"

        self.task_names: List[str] = list(self.dataset_map.keys())
        self.dataset_classes: list[Type[Dataset]] = list(self.dataset_map.values())
        self.dataset = None
        self.dataset_kwargs = {}
        for name in self.task_names:
            args = {"root_path": root_path}
            args.update(copy.deepcopy(kwargs))
            if dataset_kwargs is not None and name in dataset_kwargs:
                aux_args = copy.deepcopy(dataset_kwargs[name])
            else:
                aux_args = {}
            args.update(aux_args)
            self.dataset_kwargs[name] = args
        if dataset_kwargs is not None:
            for ds_name in dataset_kwargs:
                if ds_name not in self.task_names:
                    raise ValueError(
                        f"{ds_name} not found in loaded tasks {self.task_names}."
                    )
        if make and "feats_name" not in kwargs:
            self._make(clean=clean)
        elif make and "feats_name" in kwargs and kwargs["feats_name"] is not None:
            self._make(clean=False)
            self.make_features(batch_size=batch_size, num_gpus=num_gpus, clean=clean)

        self.task_id = task_id

    def __getitem__(self, index):
        if self.task_id is None:
            if index < 0:
                index = len(self.dataset) + index
            return self.dataset[index], index
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)

    def export_feats(self, dest_folder: Path, export_all=True, clean_make=False):
        if export_all:
            s = self.root_path.glob("*/feats/")
        else:
            s = [self.root_path.joinpath(p, "feats") for p in self.dataset_map]

        for ds_feats_folder in tqdm.tqdm(s, desc="Exporting features"):
            metadata = ds_feats_folder.parent.joinpath("metadata.pickle")
            metadata_dest = dest_folder.joinpath(metadata.relative_to(self.root_path))
            metadata_dest.parent.mkdir(exist_ok=True, parents=True)
            shutil.copy(metadata, metadata_dest)
            for mdb_path in ds_feats_folder.rglob("data.mdb"):
                source_path = mdb_path.parent
                save_path = dest_folder.joinpath(
                    source_path.relative_to(self.root_path)
                )
                if clean_make:
                    shutil.rmtree(save_path, ignore_errors=True)
                elif save_path.exists():
                    raise RuntimeError(
                        f"export directory {save_path} already exists. Either run with arguments `clean_make=True` or remove the directory."
                    )
                save_path.mkdir(exist_ok=True, parents=True)

                db = lmdb.open(source_path.as_posix())
                db.copy(save_path.as_posix())

    @cached_property
    def task_end_idxs(self):
        end_idxs = []
        for ds_cls in self.dataset_classes:
            ds = make_ds(ds_cls, self.dataset_kwargs[ds_cls.name])
            end_idxs.append(len(ds.labels))

        return np.cumsum(end_idxs)

    @property
    def task_id(self):
        return self._task_id

    @property
    def task_name(self):
        if self.task_id is not None:
            return self.task_names[self.task_id]
        else:
            return "all"

    @task_name.setter
    def task_name(self, value):
        if value is not None:
            self.task_id = self.task_names.index(value.lower())
        else:
            self.task_id = None

    @task_id.setter
    def task_id(self, value):
        if value is not None and not (value >= 0 and value < len(self.dataset_classes)):
            raise ValueError(
                f"Invalid task_id {value}. Must be >=0 and < {len(self.dataset_classes)}"
            )
        if hasattr(self, "_task_id"):
            raise RuntimeError("Can not change task_id after initialization.")
        self._task_id = value
        if "dataset" in self.__dict__:
            del self.__dict__["dataset"]
        self.dataset

    @cached_property
    def dataset(self):
        if self.task_id is not None:
            return make_ds(
                self.dataset_classes[self.task_id], self.dataset_kwargs[self.task_name]
            )
        else:
            return ConcatDataset(
                [
                    make_ds(self.dataset_classes[self.task_names.index(k)], v)
                    for k, v in self.dataset_kwargs.items()
                ]
            )

    def _make(self, clean=False):
        procs: List[mp.Process] = []
        for i, ds in enumerate(self.dataset_classes):
            args = copy.deepcopy(self.dataset_kwargs[self.task_names[i]])
            if "feats_name" in args:
                del args["feats_name"]
            p = mp.Process(
                target=self._make_dataset, args=(ds, args, clean), name=ds.__name__
            )
            logging.info(f"Processing {ds.__name__}")
            p.start()
            procs.append(p)
        for p in procs:
            p.join()

        return procs

    def make_features(self, batch_size, num_gpus: float, clean=False, feats_name=None):
        ray_init = False
        try:
            import ray

            ray_init = True
            assert ray.is_initialized(), "Must run ray.init()."
            assert torch.cuda.is_available(), "Cuda is not available."
        except ImportError as e:
            assert (
                ray_init
            ), "Currently does not support making features without ray or cuda."
            raise e

        remotes = []
        for i, ds in enumerate(self.dataset_classes):
            args = copy.deepcopy(self.dataset_kwargs[self.task_names[i]])
            if "action" in args:
                del args["action"]
            remotes.append(
                ray.remote(num_gpus=num_gpus, max_calls=1)(
                    make_dataset_features
                ).remote(ds, args, batch_size, clean, feats_name)
            )
        return ray.get(remotes)

    @classmethod
    def _make_dataset(cls, ds: Type[Dataset], kwargs=None, clean=False, debug=True):
        args = copy.deepcopy(kwargs) if kwargs is not None else {}

        # is dataset downloaded? .verify
        # is the dataset processed?
        # do the feature vectors exist?
        root_path = args["root_path"]
        is_downloaded = ds.verify_downloaded(root_path)
        is_processed = ds.verify_processed(root_path)
        download_ds = False
        process_ds = False

        if (not is_downloaded or not is_processed) and not clean:
            download_ds = not is_downloaded
            process_ds = not is_processed
        elif clean:
            download_ds = True
            process_ds = True
        try:
            if download_ds:
                args["action"] = "download"
                args["clean"] = True
            elif process_ds:
                args["action"] = "process"
                args["clean"] = True
            dataset = ds(**args)
            dataset.verify()

            logging.info(f"Dataset - {ds.__name__} Processed")
        except Exception as e:
            logging.error(f"Dataset - {ds.__name__} Error\n{traceback.format_exc()}")
            if debug:
                raise e

    @classmethod
    def supported_datasets(cls) -> List[Type[Dataset]]:
        # iterate through the modules in the current package
        dataset_classes: List[Type[Dataset]] = []
        for _, module_name, _ in iter_modules([PACKAGE_DATASET_DIR]):
            # import the module and iterate through its attributes
            module = import_module(f"stream.datasets.{module_name}")
            attribute = None
            for attribute_name in dir(module):
                attribute = getattr(module, attribute_name)

                if (
                    isclass(attribute)
                    and issubclass(attribute, Dataset)
                    and attribute != Dataset
                ):
                    # Add the class to this package's variables
                    globals()[attribute_name] = attribute
                    dataset_classes.append(attribute)
                    break
            if attribute is None:
                raise ImportError(f"No valid dataset was found in {module_name}")

        return dataset_classes

    def verify(self):
        return {
            k: make_ds(self.dataset_classes[self.task_names.index(k)], v).verify()
            for k, v in self.dataset_kwargs.items()
        }
