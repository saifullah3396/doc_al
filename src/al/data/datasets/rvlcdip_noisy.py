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

        self._noisy_label_weights = [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.42590782046318054,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.8061826825141907,
                0.0,
                0.4106968641281128,
                0.0,
                0.0,
            ],
            None,
            None,
            None,
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.8457979559898376,
                0.0,
                0.0,
                0.0,
                0.5335033535957336,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            None,
            None,
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.5963240265846252,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.8027438521385193,
                0.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.5243628621101379,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.4462662935256958,
                0.7251827120780945,
                0.0,
                0.0,
                0.0,
            ],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [
                0.0,
                0.7175472974777222,
                0.0,
                0.40182650089263916,
                0.0,
                0.42096108198165894,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.3826919198036194,
                0.0,
                0.0,
                0.0,
            ],
            None,
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]

    def _load_dataset(self):
        import numpy as np
        import torch

        # load the annotations
        data_columns = [DataKeys.IMAGE_FILE_PATH, DataKeys.LABEL]
        data = pd.read_csv(
            self.dataset_dir / f"labels/{self.split}.txt",
            names=data_columns,
            delim_whitespace=True,
        )
        data[DataKeys.IMAGE_FILE_PATH] = [f"{self.dataset_dir}/images/{x}" for x in data[DataKeys.IMAGE_FILE_PATH]]

        # generate noisy labels based on weights of similar class
        if self._split == "train":
            data = data.sample(frac=1).reset_index(drop=True)
            # s1 = torch.tensor(data["label"].tolist())
            for idx, label in enumerate(self._labels):
                if self._noisy_label_weights[idx] is None:
                    continue
                per_label_df = data[data[DataKeys.LABEL] == idx]
                size = len(per_label_df) * self._noise_percentage // 100
                data.loc[per_label_df.head(size).index, DataKeys.LABEL] = np.random.choice(
                    len(self._labels),
                    size=size,
                    p=self._noisy_label_weights[idx] / np.sum(self._noisy_label_weights[idx]),
                )
        # s2 = torch.tensor(data["label"].tolist())
        # from sklearn.metrics import confusion_matrix

        # m = confusion_matrix(s1, s2, normalize="true")
        # print(m)

        return data
