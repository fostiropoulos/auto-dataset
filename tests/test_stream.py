import copy
import logging
from pathlib import Path

import ray
import torch
from torchvision import transforms
from tqdm import tqdm
from transformers import CLIPTokenizer

from autods.dataset import Dataset
from autods.main import AutoDS


vision_transform = transforms.Compose(
    [
        transforms.Resize(32),
        transforms.CenterCrop(32),
        transforms.ToTensor(),
    ]
)
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")


def run_ds(ds: type[Dataset], args, verbose: bool = False):
    try:
        _ds = ds(**args)
        if verbose:
            _ds = tqdm(_ds)
            _ds.set_description(str(ds.__name__))
        for i in _ds:
            pass
    except Exception as e:
        logging.error(f"Error with {ds.__name__}")
        # if verbose:
        #     raise e
    pass


def text_transform(text):
    return (
        tokenizer(
            text=text,
            padding="max_length",
            max_length=32**2 * 3,
            truncation=True,
            return_tensors="pt",
        )["input_ids"]
        .reshape(3, 32, 32)
        .type(torch.float)
    )


def test_all_dataset(
    root_path: Path,
    feats_name: str | None = None,
    verbose: bool = False,
    datasets: list[str] | None = None,
):
    kwargs = {}
    if feats_name is None:
        kwargs["dataset_kwargs"] = {
            "amazon": {"transform": text_transform},
            "yelp": {"transform": text_transform},
            "imdb": {"transform": text_transform},
        }
        kwargs["transform"] = (vision_transform,)
    s = AutoDS(root_path, datasets=datasets, feats_name=feats_name, **kwargs)
    import multiprocessing as mp

    procs: list[mp.Process] = []
    for train in [False, True]:
        for i, ds in enumerate(s.dataset_classes):
            args = copy.deepcopy(s.dataset_kwargs[s.task_names[i]])
            args["train"] = train

            p = mp.Process(target=run_ds, args=(ds, args, verbose), name=ds.__name__)
            p.start()
            procs.append(p)
    for p in procs:
        p.join()

    breakpoint()
    return
    pass


def test_features_vision(root_path: Path, verbose: bool = False):
    datasets = [
        d.name
        for d in AutoDS.supported_datasets()
        if d.name not in {"amazon", "yelp", "imdb"}
    ]
    test_all_dataset(root_path, feats_name="clip", datasets=datasets, verbose=verbose)
    return
    pass


def test_features_text(root_path: Path, verbose: bool = False):
    datasets = ["amazon", "yelp", "imdb"]
    test_all_dataset(root_path, feats_name="gpt2", datasets=datasets, verbose=verbose)
    breakpoint()
    return
    pass


def test_make_features_vision(root_path: Path):
    datasets = [
        d.name
        for d in AutoDS.supported_datasets()
        if d.name not in {"amazon", "yelp", "imdb"}
    ]
    s = AutoDS(
        root_path,
        transform=vision_transform,
        datasets=datasets,
    )
    ray.init()
    s.make_features(128, 6 / 40, feats_name="clip")
    breakpoint()
    return
    pass


def test_make_features_text(root_path: Path):
    datasets = ["amazon", "yelp", "imdb"]
    s = AutoDS(
        root_path,
        transform=text_transform,
        datasets=datasets,
    )
    ray.init()
    s.make_features(128, 1, feats_name="gpt2")
    breakpoint()
    return
    pass


def test_export_feats(root_path: Path, tmp_path: Path):
    s = AutoDS(
        root_path,
    )
    s.export_feats(tmp_path)

    return