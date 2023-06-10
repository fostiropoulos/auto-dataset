import copy
import io
import logging
import multiprocessing as mp
import shutil
import tempfile
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import ray
import torch
from PIL import Image

from autods.dataset import Dataset
from autods.main import AutoDS
from autods.utils import extract

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 12

def assert_error_msg(fn, error_msg: str | None = None):
    try:
        fn()
        assert False
    except Exception as excp:
        if error_msg is not None and not error_msg == str(excp):
            raise excp
        else:
            return str(excp)


def capture_output(fn, caplog=None):
    if caplog is not None:
        caplog.clear()
    out = io.StringIO()

    err = io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        fn()
    if caplog is not None:
        err = "\n".join([r.msg for r in caplog.records])
    else:
        err = err.getvalue()

    return out.getvalue(), err


class MockDataset(Dataset):
    metadata_url = "https://iordanis.me/"
    remote_urls = {"mock.tar": None}
    name = "mock"
    file_hash_map = {"mock.tar": "blahblah"}
    dataset_type = "image"
    default_task_name = "task1"

    task_names = ["task1", "task2", "task3"]

    def __init__(
        self, *args, mock_download=False, mock_process=True, size=100, **kwargs
    ) -> None:
        if mock_download:
            file_name = list(self.remote_urls)[0]
            archive_name, *_ = file_name.split(".")
            if len(args) > 0:
                root_path = args[0]
            else:
                root_path = kwargs["root_path"]
            archive_path = Path(root_path).joinpath(
                self.__class__.__name__.lower(), archive_name
            )
            rng = np.random.default_rng(seed=42)

            with tempfile.TemporaryDirectory() as fp:
                for split in ["train", "val"]:
                    for i in range(size):
                        img = Image.fromarray(
                            rng.integers(0, 255, (150, 150, 3)).astype(np.uint8)
                        )
                        name = f"{i}_{split}.png"
                        img.save(Path(fp).joinpath(name))
                shutil.make_archive(archive_path.as_posix(), "tar", root_dir=fp)
                self.file_hash_map[file_name] = self.file_hash(
                    str(archive_path) + ".tar"
                )
            if mock_process:
                kwargs["action"] = "process"
        super().__init__(*args, **kwargs)

    def _process(self, raw_data_dir: Path):
        archive_path = raw_data_dir.joinpath("mock.tar")
        extract(archive_path, raw_data_dir)

    def _make_metadata(self, raw_data_dir: Path):
        file_names = {}
        # NOTE if a dataset does not have subset variants. Simply add None as key
        for task_name in self.task_names:
            file_names[task_name] = {}
            for split in ["train", "val"]:
                file_tuples = []
                for f in raw_data_dir.joinpath("mock").glob(f"*_{split}.png"):
                    file_tuples.append(
                        (f.relative_to(raw_data_dir), np.random.randint(10))
                    )
                file_names[task_name][split] = file_tuples
        # to class name
        class_names = self._make_class_names(file_names)
        # save
        metadata = dict(file_names=file_names, class_names=class_names)
        torch.save(metadata, self.metadata_path)


class MockDataset2(MockDataset):
    name = "mock2"
    pass


def test_dataset(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, ds_class=MockDataset
):
    class_name = ds_class.__name__
    class_directory = class_name.lower()

    out, err = capture_output(lambda: ds_class(tmp_path), caplog)

    assert (
        err.strip()
        == f"Could not find a dataset in {tmp_path}/{class_directory}. You will need to use `{class_name}.download(path)` or initialize with action=`download`\nInitialized empty dataset {class_name}."
    )
    with tempfile.TemporaryDirectory() as fp:
        ds = ds_class(fp)
        assert assert_error_msg(lambda: ds.assert_downloaded()).endswith(" is missing.")
        assert assert_error_msg(lambda: ds.process()).endswith(
            " to use `download(clean=True)` before processing."
        )
        ds.dataset_path.mkdir()
        ds.dataset_path.joinpath("test.txt").write_text("a")
        assert_error_msg(
            lambda: ds.download(),
            f"{fp}/{class_directory} is not empty. You must use with flag clean `{class_name}.download(path, clean=True)` that will remove all files and re-process the dataset",
        )

    ds = ds_class(tmp_path, action="download", mock_download=True)
    assert len(ds) > 0
    out, err = capture_output(lambda: ds_class(tmp_path), caplog=caplog)
    assert len(out) == 0 and len(err) == 0

    ds = ds_class(tmp_path)
    assert_error_msg(
        lambda: ds.download(),
        f"{tmp_path}/{class_directory} is not empty. You must use with flag clean `{class_name}.download(path, clean=True)` that will remove all files and re-process the dataset",
    )
    assert ds.assert_downloaded()
    assert len(ds) == len(ds.metadata["file_names"][ds.default_task_name]["train"])
    for f in iter(ds):
        assert f is not None
    ds = ds_class(root_path=tmp_path, train=False)
    assert len(ds) == len(ds.metadata["file_names"][ds.default_task_name]["val"])
    for f in iter(ds):
        assert f is not None

    ds.process(clean=True)


def test_make_features(tmp_path: Path, ds_class=MockDataset):
    ds = ds_class(tmp_path, mock_download=True, mock_process=False)
    assert_error_msg(
        lambda: ds.make_features(1000, DEVICE, feats_name="clip"),
        f"Missing {ds.metadata_path}. Please run `.process()` before making features.",
    )

    ds.process()
    ds.make_features(1000, DEVICE, feats_name="clip")

    assert_error_msg(
        lambda: ds.make_features(1000, DEVICE, feats_name="clip"),
        f"Directory exists {ds.feats_path.joinpath('clip')}. Use `clean=True`",
    )
    ds.make_features(1000, DEVICE, feats_name="clip", clean=True)


def test_stream(tmp_path: Path):
    datasets = [MockDataset, MockDataset2]
    sizes = (np.arange(len(datasets)) + 1) * 100
    with mock.patch(
        "autods.main.AutoDS.supported_datasets",
        return_value=datasets,
    ):
        dataset_kwargs = {}
        for ds, size in zip(datasets, sizes):
            dataset_kwargs[ds.__name__.lower()] = {"size": size}

        # Passing global argument
        ds = AutoDS(tmp_path, dataset_kwargs=dataset_kwargs, mock_download=True)

        for p in ds:
            pass
        assert len(ds) == sizes.sum()

        def _change_task():
            ds.task_id = 0

        assert_error_msg(
            _change_task,
            f"Can not change task_id after initialization.",
        )
        ds = AutoDS(tmp_path, task_id=0)
        assert len(ds) == sizes[0]
        ds = AutoDS(tmp_path, task_id=1)
        assert len(ds) == sizes[1]


def test_stream_make(tmp_path: Path):
    datasets = [MockDataset, MockDataset2]
    sizes = (np.arange(len(datasets)) + 1) * 100
    with mock.patch(
        "autods.main.AutoDS.supported_datasets",
        return_value=datasets,
    ):
        for ds, size in zip(datasets, sizes):
            ds(tmp_path, size=size, mock_download=True)

        with mock.patch(
            "autods.dataset.Dataset.assert_downloaded", return_value=True
        ), mock.patch("autods.dataset.Dataset.verify_downloaded", return_value=True):
            # Passing global argument
            import ray

            ray.init()
            ds = AutoDS(
                tmp_path,
                clean=True,
                make=True,
            )
            ds = AutoDS(
                tmp_path,
                feats_name="clip",
                make=True,
                clean=True,
                batch_size=BATCH_SIZE,
                num_gpus=0.2,
            )
            ds = AutoDS(
                tmp_path,
                feats_name="clip",
            )
            ds.verify()
        return


if __name__ == "__main__":

    # with tempfile.TemporaryDirectory() as fp:
    #     test_dataset(Path(fp), caplog=None)

    # with tempfile.TemporaryDirectory() as fp:
    #     test_make_features(Path(fp))

    # with tempfile.TemporaryDirectory() as fp:
    #     test_stream(Path(fp))

    with tempfile.TemporaryDirectory() as fp:
        test_stream_make(Path(fp))
