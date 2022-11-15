"""
Defines the Tobacco3482 dataset.
"""

from pathlib import Path

import pandas as pd
import tqdm
from xai_torch.core.constants import DataKeys
from xai_torch.core.data.datasets.image_dataset_base import ImageDatasetBase


class Tobacco3482Dataset(ImageDatasetBase):
    """Tobacco3482 dataset from https://www.kaggle.com/patrickaudriaz/tobacco3482jpg."""

    _is_downloadable = False
    _supported_splits = ["train", "test"]

    def _load_dataset_properties(self):
        super()._load_dataset_properties()
        self._labels = [
            "Letter",
            "Resume",
            "Scientific",
            "ADVE",
            "Email",
            "Report",
            "News",
            "Memo",
            "Form",
            "Note",
        ]

    def _load_dataset(self):
        import numpy as np

        # load all the data into a list
        files = []
        with open(f"{self.dataset_dir}/{self.split}.txt", "r") as f:
            files = f.readlines()
        files = [f.strip() for f in files]

        data = []
        for file in tqdm.tqdm(files):
            sample = []

            # generate the filepath
            fp = Path(self.dataset_dir) / Path(file)

            # add image path
            sample.append(str(fp))

            # add label
            label_str = str(fp.parent.name)
            label_idx = self._labels.index(label_str)
            sample.append(label_idx)

            # add sample to data
            data.append(sample)

        np.random.seed(0)
        np.random.shuffle(data)

        # convert data list to df
        data_columns = [DataKeys.IMAGE_FILE_PATH, DataKeys.LABEL]
        return pd.DataFrame(data, columns=data_columns)
