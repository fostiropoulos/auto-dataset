from abc import ABC, abstractmethod
from PIL import Image


class FeatExtractor(ABC):
    @property
    @abstractmethod
    def OUTPUT_SHAPE(self) -> int:
        pass

    @abstractmethod
    def __init__(self, device) -> None:
        pass

    @abstractmethod
    def get_image_feats(self, images: list[Image.Image]):
        pass

    @abstractmethod
    def get_text_feats(self, text: list[str]):
        pass
