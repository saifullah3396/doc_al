"""
Defines the RVLCDIP dataset.
"""


import pandas as pd
from xai_torch.core.constants import DataKeys
from xai_torch.core.data.datasets.image_dataset_base import ImageDatasetBase


class RVLCDIPDataset(ImageDatasetBase):
    """RVLCDIP dataset from https://www.cs.cmu.edu/~aharley/rvl-cdip/."""

    _is_downloadable = False
    _supported_splits = ["train", "test", "val"]

    def __init__(self, *args, version=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._version = version

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
        # load the annotations
        data_columns = [DataKeys.IMAGE_FILE_PATH, DataKeys.LABEL]
        data = pd.read_csv(
            self.dataset_dir / f"labels/{self.split}.txt",
            names=data_columns,
            delim_whitespace=True,
        )
        if self._version is not None:
            data[DataKeys.IMAGE_FILE_PATH] = [
                f"{self.dataset_dir}/images/{x[:-4]}{self._version}{x[-4:]}" for x in data[DataKeys.IMAGE_FILE_PATH]
            ]
        else:
            data[DataKeys.IMAGE_FILE_PATH] = [f"{self.dataset_dir}/images/{x}" for x in data[DataKeys.IMAGE_FILE_PATH]]
        return data
