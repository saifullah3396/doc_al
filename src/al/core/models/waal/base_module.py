"""
WAAL base module for active learning.
"""


from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
import torch.nn.init as init
from al.core.training.trainer import DALTrainer
from ignite.contrib.handlers import TensorboardLogger
from torch import nn
from torch.autograd import Variable, grad
from xai_torch.core.args import Arguments
from xai_torch.core.constants import DataKeys, MetricKeys
from xai_torch.core.models.base import BaseModule
from xai_torch.core.training.constants import TrainingStage
from xai_torch.core.models.utilities.checkpoints import prepend_keys

if TYPE_CHECKING:
    from xai_torch.core.args import Arguments

logging.basicConfig(level=logging.INFO)

# class Discriminator(nn.Module):
#     """Adversary architecture(Discriminator) for WAE-GAN."""

#     def __init__(self, dim=32):
#         super(Discriminator, self).__init__()
#         import numpy as np

#         self.dim = np.prod(dim)
#         self.net = nn.Sequential(
#             nn.Linear(self.dim, 512),
#             nn.ReLU(True),
#             nn.Linear(512, 512),
#             nn.ReLU(True),
#             nn.Linear(512, 1),
#             nn.Sigmoid(),
#         )
#         self.weight_init()

#     def weight_init(self):
#         for block in self._modules:
#             for m in self._modules[block]:
#                 kaiming_init(m)

#     def forward(self, z):
#         return self.net(z).reshape(-1)


class Discriminator(nn.Module):
    def __init__(self, dim):
        super(Discriminator, self).__init__()

        self.fc1 = nn.Linear(dim, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        x = torch.sigmoid(x)

        return x


def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


def normal_init(m, mean, std):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        m.weight.data.normal_(mean, std)
        if m.bias.data is not None:
            m.bias.data.zero_()
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        m.weight.data.fill_(1)
        if m.bias.data is not None:
            m.bias.data.zero_()


class WAALBaseModule(BaseModule):
    def __init__(
        self,
        args: Arguments,
        tb_logger: TensorboardLogger,
        task_model: BaseModule,
    ):
        super().__init__(args, tb_logger)

        # add task_model
        self._task_model = task_model

    @property
    def waal_training_args(self):
        return self._args.al_args.training_args

    def _init_metrics(self):
        return self._task_model._init_metrics()

    def _build_model(self):
        # setup variational auto encoder for training
        self._dsc = Discriminator(dim=self._task_model.get_embedding_dim())

    def get_param_groups(self):
        return {
            "task": list(self._task_model.parameters()),
            "dsc": list(self._dsc.parameters()),
        }

    def training_step(self, batch, step="task") -> None:
        return self(**batch, step=step, stage=TrainingStage.train)

    def evaluation_step(self, batch, step="task", stage: TrainingStage = TrainingStage.test) -> None:
        assert stage in [TrainingStage.train, TrainingStage.val, TrainingStage.test]
        return self(**{"labeled": {**batch}}, step=step, stage=stage)

    def predict_step(self, batch, step="task") -> None:
        return self(**{"labeled": {**batch}}, step=step, stage=TrainingStage.predict)

    def forward(self, labeled, unlabeled=None, step="task", stage=TrainingStage.predict):
        if stage == TrainingStage.train:
            if step == "task":
                self.set_requires_grad(self._task_model, requires_grad=True)
                self.set_requires_grad(self._dsc, requires_grad=False)

                lb_z = self._task_model(labeled[DataKeys.IMAGE])
                unlb_z = self._task_model(unlabeled[DataKeys.IMAGE])
                lb_out = self._task_model(lb_z, step="clf")

                # prediction loss (deafult we use F.cross_entropy)
                pred_loss = torch.mean(F.cross_entropy(lb_out, labeled[DataKeys.LABEL]))

                # Wasserstein loss (unbalanced loss, used the redundant trick)
                wassertein_distance = (
                    self._dsc(unlb_z).mean() - self.waal_training_args.gamma_ratio * self._dsc(lb_z).mean()
                )

                with torch.no_grad():
                    lb_z = self._task_model(labeled[DataKeys.IMAGE])
                    unlb_z = self._task_model(unlabeled[DataKeys.IMAGE])

                gp = self.gradient_penalty(self._dsc, unlb_z, lb_z, labeled[DataKeys.IMAGE].get_device())

                loss = (
                    pred_loss
                    + self.waal_training_args.alpha * wassertein_distance
                    + self.waal_training_args.alpha * gp * 5
                )
                # for CIFAR10 the gradient penality is 5
                # for SVHN the gradient penality is 2
                return {
                    DataKeys.LOGITS: lb_out,
                    DataKeys.LABEL: labeled[DataKeys.LABEL],
                    DataKeys.LOSS: loss,
                }
            elif step == "dsc":
                # Then the second step, training discriminator
                self.set_requires_grad(self._task_model, requires_grad=False)
                self.set_requires_grad(self._dsc, requires_grad=True)

                with torch.no_grad():
                    lb_z = self._task_model(labeled[DataKeys.IMAGE])
                    unlb_z = self._task_model(unlabeled[DataKeys.IMAGE])

                # gradient ascent for multiple times like GANS training
                gp = self.gradient_penalty(self._dsc, unlb_z, lb_z, labeled[DataKeys.IMAGE].get_device())
                wassertein_distance = (
                    self._dsc(unlb_z).mean() - self.waal_training_args.gamma_ratio * self._dsc(lb_z).mean()
                )

                dsc_loss = (
                    -1 * self.waal_training_args.alpha * wassertein_distance - self.waal_training_args.alpha * gp * 2
                )
                return {
                    DataKeys.LOSS: dsc_loss,
                }
        else:
            lb_z = self._task_model(labeled[DataKeys.IMAGE])
            lb_out = self._task_model(lb_z, step="clf")

            # prediction loss (deafult we use F.cross_entropy)
            pred_loss = torch.mean(F.cross_entropy(lb_out, labeled[DataKeys.LABEL]))

            return {
                DataKeys.LOGITS: lb_out,
                DataKeys.LABEL: labeled[DataKeys.LABEL],
                DataKeys.LOSS: pred_loss,
            }

    def set_requires_grad(self, model, requires_grad=True):
        for param in model.parameters():
            param.requires_grad = requires_grad

    def gradient_penalty(self, critic, h_s, h_t, device):
        alpha = torch.rand(h_s.size(0), 1, device=device)
        differences = h_t - h_s
        interpolates = h_s + (alpha * differences)
        interpolates = torch.cat([interpolates, h_s, h_t]).requires_grad_()
        preds = critic(interpolates)
        gradients = grad(
            preds, interpolates, grad_outputs=torch.ones_like(preds), retain_graph=True, create_graph=True
        )[0]
        gradient_norm = gradients.norm(2, dim=1)
        gradient_penalty = ((gradient_norm - 1) ** 2).mean()

        return gradient_penalty

    def on_load_checkpoint(self, checkpoint: Dict[str, Any], strict: bool = True):
        state_dict_key = "state_dict"
        checkpoint = prepend_keys(checkpoint, state_dict_key, ["_task_model."])
        return super().on_load_checkpoint(checkpoint, strict)