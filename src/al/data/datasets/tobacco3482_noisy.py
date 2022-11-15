"""
Defines the Tobacco3482 dataset.
"""

from pathlib import Path
from typing import Callable, Optional

import pandas as pd
import tqdm
from xai_torch.core.constants import DataKeys
from xai_torch.core.data.args.data_args import DataArguments
from xai_torch.core.data.datasets.image_dataset_base import ImageDatasetBase


class Tobacco3482NoisyDataset(ImageDatasetBase):
    """Tobacco3482 noisy dataset from https://www.kaggle.com/patrickaudriaz/tobacco3482jpg."""

    _is_downloadable = False
    _supported_splits = ["train", "test"]

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

        self._noisy_label_weights = [
            list([0.0, 0.0, 0.0, 0.0, 0.0, 0.7071067690849304, 0.0, 0.7071067690849304, 0.0, 0.0]),
            None,
            list([0.0, 0.0, 0.0, 0.0, 0.0, 0.7035264372825623, 0.0, 0.502518892288208, 0.502518892288208, 0.0]),
            None,
            None,
            list([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            None,
            None,
            list([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
            list([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
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
        data = pd.DataFrame(data, columns=data_columns)

        # generate noisy labels based on weights of similar class
        if self._split == "train":
            import torch

            data = data.sample(frac=1).reset_index(drop=True)
            s1 = torch.tensor(data["label"].tolist())
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
            s2 = torch.tensor(data["label"].tolist())
            from sklearn.metrics import confusion_matrix

            m = confusion_matrix(s1, s2, normalize="true")
            print(m)

        return data
