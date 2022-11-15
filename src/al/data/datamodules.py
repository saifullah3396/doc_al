"""
Defines the DataModule for CIFAR10 dataset.
"""

from functools import cached_property

from xai_torch.core.data.data_modules.decorators import register_datamodule
from xai_torch.core.data.data_modules.image_datamodule import ImageDataModule

from al.data.datasets.rvlcdip import RVLCDIPDataset
from al.data.datasets.rvlcdip_noisy import RVLCDIPNoisyDataset
from al.data.datasets.rvlcdip_ocr import RVLCDIPOcrDataset
from al.data.datasets.tobacco3482 import Tobacco3482Dataset
from al.data.datasets.tobacco3482_noisy import Tobacco3482NoisyDataset
from al.data.datasets.tobacco3482_ocr import Tobacco3482OcrDataset


@register_datamodule(reg_name="rvlcdip_ocr")
class RVLCDIPOcred(ImageDataModule):
    @cached_property
    def dataset_class(self):
        return RVLCDIPOcrDataset


@register_datamodule(reg_name="rvlcdip")
class RVLCDIP(ImageDataModule):
    @cached_property
    def dataset_class(self):
        return RVLCDIPDataset


@register_datamodule(reg_name="rvlcdip_noisy")
class RVLCDIPNoisy(ImageDataModule):
    @cached_property
    def dataset_class(self):
        return RVLCDIPNoisyDataset


@register_datamodule(reg_name="tobacco3482")
class Tobacco3482(ImageDataModule):
    @cached_property
    def dataset_class(self):
        return Tobacco3482Dataset


@register_datamodule(reg_name="tobacco3482_noisy")
class Tobacco3482Noisy(ImageDataModule):
    @cached_property
    def dataset_class(self):
        return Tobacco3482NoisyDataset


@register_datamodule(reg_name="tobacco3482_ocr")
class Tobacco3482Ocred(ImageDataModule):
    @cached_property
    def dataset_class(self):
        return Tobacco3482OcrDataset
