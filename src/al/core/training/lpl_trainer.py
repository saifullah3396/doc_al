"""
The main script that serves as the entry-point for all kinds of training experiments.
"""


from __future__ import annotations

import logging
from functools import partial
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from al.core.models.lpl.xai_model import LPLXAIModel
from al.core.training.trainer import DALTrainer
from ignite.contrib.handlers import TensorboardLogger
from xai_torch.core.args import Arguments
from xai_torch.core.constants import DataKeys
from xai_torch.core.models.xai_model import XAIModel

if TYPE_CHECKING:
    from xai_torch.core.args import Arguments
    from xai_torch.core.data.data_modules.base import BaseDataModule
    from ignite.engine import Engine

from xai_torch.core.training.constants import TrainingStage

logging.basicConfig(level=logging.INFO)


class LPLTrainer(DALTrainer):
    @classmethod
    def setup_model(
        cls,
        args: Arguments,
        datamodule: BaseDataModule,
        tb_logger: TensorboardLogger,
        summarize: bool = False,
        stage: TrainingStage = TrainingStage.train,
    ) -> XAIModel:
        """
        Initializes the model for training.
        """
        from xai_torch.core.models.factory import ModelFactory

        # setup model
        model = ModelFactory.create(args, datamodule, tb_logger=tb_logger, wrapper_class=LPLXAIModel)
        model.setup(stage=stage)

        # generate model summary
        if summarize:
            model.summarize()

        return model
