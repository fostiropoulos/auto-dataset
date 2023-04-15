from abc import ABC, abstractmethod
from functools import cached_property
from operator import itemgetter
from typing import Dict, List, Literal, Optional, Tuple, final

import torch


class BaseDataset(torch.utils.data.Dataset, ABC):
    def __init__(self) -> None:
        super().__init__()
        self.metadata: dict
        self.class_names: List[str]
        self.split: Literal["train", "val"]
        self.dataset: List[Tuple[str, str]]
        self._is_init: bool

    @abstractmethod
    def _process(self, raw_data_dir):
        raise NotImplementedError

    @abstractmethod
    def _make_metadata(self, raw_data_dir):
        raise NotImplementedError

    @property
    @abstractmethod
    def metadata_url(self) -> str:
        pass

    @property
    @abstractmethod
    def default_task_name(self) -> str:
        pass

    @property
    @abstractmethod
    def task_names(self) -> List[str]:
        raise NotImplementedError

    @property
    @abstractmethod
    def dataset_type(self) -> str:
        pass

    @cached_property
    def labels(self) -> List[str]:
        return self.class_names[self.current_task_name]

    @property
    def current_task_name(self):
        return self._task_name

    @current_task_name.setter
    def current_task_name(self, value):
        if value is None:
            value = self.default_task_name
        if value not in self.task_names:
            raise ValueError(f"{value} not found in tasks {self.task_names}")

        self._task_name = value

    @property
    def subset_names(self) -> Optional[List[str]]:
        return None

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def remote_urls(self) -> Dict[str, str]:
        raise NotImplementedError

    @property
    @abstractmethod
    def file_hash_map(self) -> Dict[str, str]:
        raise NotImplementedError

    def __len__(self) -> int:
        if self._is_init:
            return len(self.dataset)
        raise RuntimeError(
            f"{self.__class__.__name__} is uninitialized. Download and process the dataset."
        )
