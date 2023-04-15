from typing import List
from PIL import Image
from transformers import (
    ViTModel,
    ViTImageProcessor,
)
import torch
from transformers import logging

from stream.feat_extractors import FeatExtractor
from stream.utils import isImageType


class ViT(FeatExtractor):
    OUTPUT_SHAPE:int = 768
    def __init__(self, device) -> None:

        logging.set_verbosity_error()
        model_name = "google/vit-base-patch16-224"
        self.model = ViTModel.from_pretrained(model_name, output_loading_info=False)
        self.processor = ViTImageProcessor.from_pretrained(
            model_name, output_loading_info=False
        )
        self.model.to(device)
        self.model.eval()
        self.device = device

    @torch.no_grad()
    def get_image_feats(self, images: List[Image.Image]):
        assert all([isImageType(image) for image in images]), "invalid input type"
        inputs = self.processor(images=images, return_tensors="pt")
        inputs["pixel_values"] = inputs["pixel_values"].to(self.device)
        image_features = self.model(**inputs)
        # From Hugging face:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token. We can use the raw hidden state instead
        return image_features["last_hidden_state"][:, 0]
        # Or
        # return image_features["last_hidden_state"].mean(1)

    def get_text_feats(self, text: list[str]):
        raise NotImplementedError