import copy
from pathlib import Path
from typing import Literal

import requests
from PIL import Image
from autods.feat_extractors import FeatExtractor

from autods.feat_extractors.clip import ClipModel
from autods.feat_extractors.gpt2 import GPT2
from autods.feat_extractors.resnet import ResnetModel
from autods.feat_extractors.vit import ViT
import torch

lorem = "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum."

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ASSETS_FOLDER = Path(__file__).parent.joinpath("assets")
FEATS_FOLDER = ASSETS_FOLDER.joinpath("feats")
IMAGES_FOLDER = ASSETS_FOLDER.joinpath("images")


def make_image() -> list[Image.Image]:
    imgs = sorted(list(IMAGES_FOLDER.glob("*")))
    images = [Image.open(ipath) for ipath in imgs]
    return images


def make_text() -> list[str]:
    text = [lorem[int(100 * i) : int(100 * (i + 1))] for i in range(2)]
    return text


def _test_backbone_helper(model: FeatExtractor, task: Literal["text", "image"]):
    data = globals()[f"make_{task}"]()
    feats = getattr(model, f"get_{task}_feats")(data).cpu()

    assert feats.shape[1] == model.OUTPUT_SHAPE
    model_name = model.__class__.__name__
    stored_feats = torch.load(FEATS_FOLDER.joinpath(f"{model_name}_{task}.pt"))
    assert torch.isclose(feats, stored_feats).all()


def test_clip():
    model = ClipModel(device=DEVICE)
    _test_backbone_helper(model, task="text")
    _test_backbone_helper(model, task="image")


def test_gpt():
    model = GPT2(device=DEVICE)
    _test_backbone_helper(model, task="text")


def test_resnet():
    model = ResnetModel(device=DEVICE)
    _test_backbone_helper(model, task="image")


def test_vit():
    model = ViT(device=DEVICE)
    _test_backbone_helper(model, task="image")


if __name__ == "__main__":
    test_clip()
