"""
LPL base module for active learning.
"""


from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from ignite.contrib.handlers import TensorboardLogger
from torch import nn
from xai_torch.core.args import Arguments
from xai_torch.core.constants import DataKeys
from xai_torch.core.models.base import BaseModule
from xai_torch.core.training.constants import TrainingStage

from xai_torch.core.models.utilities.checkpoints import prepend_keys

if TYPE_CHECKING:
    from xai_torch.core.args import Arguments

logging.basicConfig(level=logging.INFO)


class LossNet(nn.Module):
    def __init__(self, feature_sizes=[28, 14, 7, 4], num_channels=[64, 128, 256, 512], interm_dim=128):
        super(LossNet, self).__init__()

        self.GAP1 = nn.AvgPool2d(feature_sizes[0])
        self.GAP2 = nn.AvgPool2d(feature_sizes[1])
        self.GAP3 = nn.AvgPool2d(feature_sizes[2])
        self.GAP4 = nn.AvgPool2d(feature_sizes[3])

        self.FC1 = nn.Linear(num_channels[0], interm_dim)
        self.FC2 = nn.Linear(num_channels[1], interm_dim)
        self.FC3 = nn.Linear(num_channels[2], interm_dim)
        self.FC4 = nn.Linear(num_channels[3], interm_dim)

        self.linear = nn.Linear(4 * interm_dim, 1)

    def forward(self, features):
        out1 = self.GAP1(features[0])
        out1 = out1.view(out1.size(0), -1)
        out1 = F.relu(self.FC1(out1))

        out2 = self.GAP2(features[1])
        out2 = out2.view(out2.size(0), -1)
        out2 = F.relu(self.FC2(out2))

        out3 = self.GAP3(features[2])
        out3 = out3.view(out3.size(0), -1)
        out3 = F.relu(self.FC3(out3))

        out4 = self.GAP4(features[3])
        out4 = out4.view(out4.size(0), -1)
        out4 = F.relu(self.FC4(out4))

        out = self.linear(torch.cat((out1, out2, out3, out4), 1))
        return out


class LPLBaseModule(BaseModule):
    def __init__(
        self,
        args: Arguments,
        tb_logger: TensorboardLogger,
        task_model: BaseModule,
    ):
        super().__init__(args, tb_logger)

        # add task_model
        self._task_model = task_model

    def _init_metrics(self):
        return self._task_model._init_metrics()

    @property
    def lpl_training_args(self):
        return self._args.al_args.training_args

    def _build_model(self):
        # setup variational auto encoder for training
        if self.lpl_training_args.loss_model == "cifar10":
            self._loss_model = LossNet(feature_sizes=[32, 16, 8, 4], num_channels=[64, 128, 256, 512], interm_dim=128)
        elif self.lpl_training_args.loss_model == "resnet50":
            self._loss_model = LossNet(
                feature_sizes=[56, 28, 14, 7], num_channels=[256, 512, 1024, 2048], interm_dim=128
            )

    def get_param_groups(self):
        return {
            "task": list(self._task_model.parameters()),
            "loss": list(self._loss_model.parameters()),
        }

    def training_step(self, batch) -> None:
        return self(**batch, stage=TrainingStage.train)

    def evaluation_step(self, batch, stage: TrainingStage = TrainingStage.test) -> None:
        assert stage in [TrainingStage.train, TrainingStage.val, TrainingStage.test]
        return self(**batch, stage=stage)

    def predict_step(self, batch) -> None:
        return self(**batch, stage=TrainingStage.predict)

    def loss_pred_loss(self, input, target, margin=1.0, reduction="mean"):
        assert len(input) % 2 == 0, "the batch size is not even."
        assert input.shape == input.flip(0).shape

        input = (input - input.flip(0))[
            : len(input) // 2
        ]  # [l_1 - l_2B, l_2 - l_2B-1, ... , l_B - l_B+1], batch size = 2B
        target = (target - target.flip(0))[: len(target) // 2]
        target = target.detach()

        one = 2 * torch.sign(torch.clamp(target, min=0)) - 1  # 1 operation which is defined by the authors

        if reduction == "mean":
            loss = torch.sum(torch.clamp(margin - one * input, min=0))
            loss = loss / input.size(0)  # Note that the size of input is already haved
        elif reduction == "none":
            loss = torch.clamp(margin - one * input, min=0)
        else:
            NotImplementedError()

        return loss

    def forward(self, image, label, stage=TrainingStage.predict):
        if stage == TrainingStage.train:
            output = self._task_model(image=image, label=label, stage=stage)
            task_loss = output[DataKeys.LOSS]
            embeddings = output[DataKeys.EMBEDDING]

            if self._training_engine.state.epoch >= self.lpl_training_args.loss_backprop_epochs:
                embeddings[0] = embeddings[0].detach()
                embeddings[1] = embeddings[1].detach()
                embeddings[2] = embeddings[2].detach()
                embeddings[3] = embeddings[3].detach()
            pred_loss = self._loss_model(embeddings)
            pred_loss = pred_loss.view(pred_loss.size(0))

            task_loss_avg = torch.sum(task_loss) / task_loss.size(0)
            lpl = self.loss_pred_loss(pred_loss, task_loss, self.lpl_training_args.margin)
            loss = task_loss_avg + self.lpl_training_args.weight * lpl
            return {
                DataKeys.LOSS: loss,
                DataKeys.LOGITS: output[DataKeys.LOGITS],
            }
        else:
            return self._task_model(image=image, label=label, stage=stage)

    def on_load_checkpoint(self, checkpoint: Dict[str, Any], strict: bool = True):
        state_dict_key = "state_dict"
        checkpoint = prepend_keys(checkpoint, state_dict_key, ["_task_model.model."])
        return super().on_load_checkpoint(checkpoint, strict)