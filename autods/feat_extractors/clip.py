import copy
from typing import List

import requests
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer, logging

from autods.feat_extractors import FeatExtractor
from autods.utils import isImageType


class ClipModel(FeatExtractor):
    OUTPUT_SHAPE = 768

    def __init__(self, device) -> None:
        logging.set_verbosity_error()
        self.model: CLIPModel = CLIPModel.from_pretrained(
            "openai/clip-vit-large-patch14", output_loading_info=False
        )
        self.processor: CLIPProcessor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-large-patch14", output_loading_info=False
        )
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        self.model.to(device)
        self.model.eval()
        self.device = device

    @torch.no_grad()
    def get_image_feats(self, images: List[Image.Image]):
        assert all([isImageType(image) for image in images]), "invalid input type"
        inputs = self.processor(images=images, return_tensors="pt")
        inputs["pixel_values"] = inputs["pixel_values"].to(self.device)
        image_features = self.model.get_image_features(**inputs)
        return image_features

    @torch.no_grad()
    def get_text_feats(self, text: List[str]):
        inputs = self.tokenizer(
            text=text, padding="max_length", truncation=True, return_tensors="pt"
        )
        inputs["input_ids"] = inputs["input_ids"].to(self.device)
        inputs["attention_mask"] = inputs["attention_mask"].to(self.device)
        text_features = self.model.get_text_features(**inputs)
        return text_features
