from typing import List

import numpy as np
import torch
from PIL import Image
from torchvision.models import (
    ResNet50_Weights,
    resnet50,
)
from transformers import logging

from stream.feat_extractors import FeatExtractor
from stream.utils import isImageType


class ResnetModel(FeatExtractor):
    OUTPUT_SHAPE: int = 2048

    def __init__(self, device) -> None:
        logging.set_verbosity_error()
        weights = ResNet50_Weights.IMAGENET1K_V1

        self.model = resnet50(weights=weights)
        self.processor = weights.transforms()
        self.model.to(device)
        self.model.eval()
        self.device = device

    @torch.no_grad()
    def get_image_feats(self, images: List[Image.Image]):
        assert all([isImageType(image) for image in images]), "invalid input type"
        image_tensors = torch.stack(
            [
                self.processor(torch.from_numpy(np.array(img)).permute(2, 0, 1))
                for img in images
            ]
        )
        inputs = image_tensors.to(self.device)

        image_features = self._feature_forward(inputs)
        return image_features

    def get_text_feats(self, text: list[str]):
        raise NotImplementedError

    def _feature_forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        return x
