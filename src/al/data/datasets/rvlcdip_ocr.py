"""
Defines the RVLCDIP dataset with its OCR data.
"""


import pandas as pd
import tqdm
from al.data.datasets.rvlcdip import RVLCDIPDataset
from xai_torch.core.constants import DataKeys
from xai_torch.core.data.datasets.utilities import read_ocr_data


class RVLCDIPOcrDataset(RVLCDIPDataset):
    """RVLCDIP dataset with OCR from https://www.cs.cmu.edu/~aharley/rvl-cdip/."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _load_dataset(self):
        if self.split not in self.supported_splits:
            raise ValueError(f"Split argument '{self.split}' not supported.")

        # load the annotations
        data_columns = [DataKeys.IMAGE_FILE_PATH, DataKeys.LABEL]
        data = pd.read_csv(
            self.root_dir / f"labels/{self.split}.txt",
            names=data_columns,
            delim_whitespace=True,
        )
        data[DataKeys.IMAGE_FILE_PATH] = [f"{self.root_dir}/images/{x}" for x in data[DataKeys.IMAGE_FILE_PATH]]
        data[DataKeys.OCR_FILE_PATH] = [x[:-3] + "hocr.lstm" for x in data[DataKeys.IMAGE_FILE_PATH]]
        if not self.tokenize_per_sample:
            words_list = []
            word_bboxes_list = []
            word_angles_list = []
            for _, row in tqdm.tqdm(data.iterrows()):
                words, word_bboxes, word_angles = read_ocr_data(row[DataKeys.OCR_FILE_PATH])
                words_list.append(words)
                word_bboxes_list.append(word_bboxes)
                word_angles_list.append(word_angles)

            data[DataKeys.WORDS] = words_list
            data[DataKeys.WORD_BBOXES] = word_bboxes_list
            data[DataKeys.WORD_ANGLES] = word_angles_list
        return data

    def get_sample(self, idx):
        sample = super().get_sample(idx, load_image=self.data_args.extras["load_image"])
        if self.tokenize_per_sample:
            words, word_bboxes, word_angles = read_ocr_data(sample[DataKeys.OCR_FILE_PATH])
            sample[DataKeys.WORDS] = words
            sample[DataKeys.WORD_BBOXES] = word_bboxes
            sample[DataKeys.WORD_ANGLES] = word_angles
        return sample

    def _tokenize_sample(self, sample):
        return self.tokenizer.tokenize_sample(sample)
