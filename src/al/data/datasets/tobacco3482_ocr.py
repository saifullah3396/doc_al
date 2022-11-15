"""
Defines the Tobacco3482 dataset with its OCR data.
"""

from pathlib import Path

import pandas as pd
import tqdm
from al.data.datasets.tobacco3482 import Tobacco3482Dataset
from xai_torch.core.constants import DataKeys
from xai_torch.core.data.utilities.ocr import TesseractOCRReader


class Tobacco3482OcrDataset(Tobacco3482Dataset):
    """
    Tobacco3482 dataset with OCR from
    https://www.kaggle.com/patrickaudriaz/tobacco3482jpg.
    """

    def _load_dataset(self):
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

            # get hocr of file
            ocr = fp.with_suffix(".hocr.lstm")
            sample.append(str(ocr))
            if self._data_args.data_tokenizer_args is not None and not self.tokenize_per_sample:
                (
                    words,
                    word_bboxes,
                    word_angles,
                ) = self._read_ocr_data(ocr)
                sample.append(words)
                sample.append(word_bboxes)
                sample.append(word_angles)

            # add sample to data
            data.append(sample)

        # convert data list to df
        if self._data_args.data_tokenizer_args is not None and not self.tokenize_per_sample:
            data_columns = [
                DataKeys.IMAGE_FILE_PATH,
                DataKeys.LABEL,
                DataKeys.OCR_FILE_PATH,
                DataKeys.WORDS,
                DataKeys.WORD_BBOXES,
                DataKeys.WORD_ANGLES,
            ]
        else:
            data_columns = [
                DataKeys.IMAGE_FILE_PATH,
                DataKeys.LABEL,
                DataKeys.OCR_FILE_PATH,
            ]
        return pd.DataFrame(data, columns=data_columns)

    def get_sample(self, idx):
        sample = super().get_sample(idx)
        if self._data_args.data_tokenizer_args is not None and self.tokenize_per_sample:
            (
                words,
                word_bboxes,
                word_angles,
            ) = self._read_ocr_data(sample[DataKeys.OCR_FILE_PATH])
            sample[DataKeys.WORDS] = words
            sample[DataKeys.WORD_BBOXES] = word_bboxes
            sample[DataKeys.WORD_ANGLES] = word_angles
        return sample

    def _tokenize_sample(self, sample):
        return self.tokenizer.tokenize_sample(sample)

    def _read_ocr_data(self, ocr_file_path: str):
        return TesseractOCRReader(ocr_file_path).parse()
