from typing import List

import torch
from transformers import GPT2Model, GPT2Tokenizer, logging

from autods.feat_extractors import FeatExtractor

from PIL import Image


class GPT2(FeatExtractor):
    OUTPUT_SHAPE = 768
    def __init__(self, device) -> None:

        logging.set_verbosity_error()
        self.model: GPT2Model = GPT2Model.from_pretrained(
            "gpt2", output_loading_info=False
        )
        self.tokenizer: GPT2Tokenizer = GPT2Tokenizer.from_pretrained(
            "gpt2", output_loading_info=False
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.to(device)
        self.model.eval()
        self.device = device

    @torch.no_grad()
    def get_text_feats(self, text: List[str]):
        inputs = self.tokenizer(
            text=text,
            padding=True,
            max_length=1024,
            truncation=True,
            return_tensors="pt",
        )
        inputs["input_ids"] = inputs["input_ids"].to(self.device)
        inputs["attention_mask"] = inputs["attention_mask"].to(self.device)
        text_features = self.model(**inputs)["last_hidden_state"].detach().cpu()
        out = [
            feats[mask.type(torch.bool)].mean(0)
            for feats, mask in zip(text_features, inputs["attention_mask"].cpu())
        ]

        return torch.stack(out)

    def get_image_feats(self, images: list[Image.Image]):
        raise NotImplementedError
