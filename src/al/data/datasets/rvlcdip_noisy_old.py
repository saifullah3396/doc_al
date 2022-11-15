"""
Defines the RVLCDIP dataset with noisy labels.
"""


from typing import Callable, Optional

import pandas as pd
from xai_torch.core.constants import DataKeys
from xai_torch.core.data.args.data_args import DataArguments
from xai_torch.core.data.datasets.image_dataset_base import ImageDatasetBase


class RVLCDIPNoisyDataset(ImageDatasetBase):
    """RVLCDIP Noisy Labels dataset from https://www.cs.cmu.edu/~aharley/rvl-cdip/."""

    _is_downloadable = False
    _supported_splits = ["train", "test", "val"]

    def __init__(
        self,
        data_args: DataArguments,
        split: str,
        transforms: Optional[Callable] = None,
        prepare_only: bool = False,
        indices: list = [],
        quiet=False,
        load_images: bool = True,
        noise_percentage=10,
    ):
        super().__init__(
            data_args=data_args,
            split=split,
            transforms=transforms,
            prepare_only=prepare_only,
            indices=indices,
            quiet=quiet,
            load_images=load_images,
        )

        self._noise_percentage = noise_percentage

    def _load_dataset_properties(self):
        super()._load_dataset_properties()

        self._labels = [
            "letter",
            "form",
            "email",
            "handwritten",
            "advertisement",
            "scientific report",
            "scientific publication",
            "specification",
            "file folder",
            "news article",
            "budgetv",
            "invoice",
            "presentation",
            "questionnaire",
            "resume",
            "memo",
        ]

    def _load_dataset(self):
        import torch

        # load the annotations
        data_columns = [DataKeys.IMAGE_FILE_PATH, DataKeys.LABEL]
        data = pd.read_csv(
            self.dataset_dir / f"labels/{self.split}.txt",
            names=data_columns,
            delim_whitespace=True,
        )
        data[DataKeys.IMAGE_FILE_PATH] = [f"{self.dataset_dir}/images/{x}" for x in data[DataKeys.IMAGE_FILE_PATH]]

        # generate noisy labels
        if self._split == "train":
            size = len(data)
            indices = torch.randperm(size)[: size * self._noise_percentage // 100]
            data.loc[indices, DataKeys.LABEL] = torch.randint(0, len(self._labels), indices.shape).tolist()

        return data
