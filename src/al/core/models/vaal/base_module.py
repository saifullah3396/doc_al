"""
VAAL base module for active learning.
"""


from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
from al.core.models.vaal.vae_dsc_models import CIFAR10VAE, CIFAR10Discriminator, DefaultDiscriminator, DefaultVAE
from ignite.contrib.handlers import TensorboardLogger
from torch import nn
from xai_torch.core.args import Arguments
from xai_torch.core.constants import DataKeys
from xai_torch.core.models.base import BaseModule
from xai_torch.core.training.constants import TrainingStage

if TYPE_CHECKING:
    from xai_torch.core.args import Arguments

logging.basicConfig(level=logging.INFO)


class VAALBaseModule(BaseModule):
    def __init__(
        self,
        args: Arguments,
        tb_logger: TensorboardLogger,
        task_model: BaseModule,
    ):
        super().__init__(args, tb_logger)

        # add task_model
        self._task_model = task_model

        # update metrics from wrapped_model
        self._metrics = self._task_model._metrics

    @property
    def vaal_training_args(self):
        return self._args.al_args.training_args

    def _build_model(self):
        # setup variational auto encoder for training
        if self.vaal_training_args.vae == "default":
            self._vae = DefaultVAE(z_dim=self.vaal_training_args.latent_dim)
        elif self.vaal_training_args.vae == "cifar10":
            self._vae = CIFAR10VAE(z_dim=self.vaal_training_args.latent_dim)

        # setup discriminator for training
        if self.vaal_training_args.dsc == "default":
            self._dsc = DefaultDiscriminator(z_dim=self.vaal_training_args.latent_dim)
        elif self.vaal_training_args.dsc == "cifar10":
            self._dsc = CIFAR10Discriminator(z_dim=self.vaal_training_args.latent_dim)

        # define loss functions
        self.bce_loss = nn.BCELoss()
        self.mse_loss = nn.MSELoss()

    def get_param_groups(self):
        return {
            "task": list(self._task_model.parameters()),
            "vae": list(self._vae.parameters()),
            "dsc": list(self._dsc.parameters()),
        }

    def _vae_loss(self, x, recon, mu, logvar, beta):
        MSE = self.mse_loss(recon, x)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        KLD = KLD * beta
        return MSE + KLD

    def training_step(self, batch, step="task") -> None:
        return self(**batch, step=step, stage=TrainingStage.train)

    def evaluation_step(self, batch, step="task", stage: TrainingStage = TrainingStage.test) -> None:
        assert stage in [TrainingStage.train, TrainingStage.val, TrainingStage.test]
        return self(**batch, step=step, stage=stage)

    def predict_step(self, batch, step="task") -> None:
        return self(**batch, step=step, stage=TrainingStage.predict)

    def forward(self, unlabeled, labeled, step="task", stage=TrainingStage.predict):
        if stage == TrainingStage.train:
            if step == "task":
                return self._task_model(**labeled, stage=stage)
            elif step == "vae":
                recon, z, mu, logvar = self._vae(labeled[DataKeys.IMAGE])
                unsup_loss = self._vae_loss(labeled[DataKeys.IMAGE], recon, mu, logvar, self.vaal_training_args.beta)
                unlab_recon, unlab_z, unlab_mu, unlab_logvar = self._vae(unlabeled[DataKeys.IMAGE])
                transductive_loss = self._vae_loss(
                    unlabeled[DataKeys.IMAGE],
                    unlab_recon,
                    unlab_mu,
                    unlab_logvar,
                    self.vaal_training_args.beta,
                )

                labeled_preds = self._dsc(mu).squeeze()
                unlabeled_preds = self._dsc(unlab_mu).squeeze()

                lab_real_preds = torch.ones(
                    labeled[DataKeys.IMAGE].size(0), device=unlabeled[DataKeys.IMAGE].get_device()
                )
                unlab_real_preds = torch.ones(
                    unlabeled[DataKeys.IMAGE].size(0), device=unlabeled[DataKeys.IMAGE].get_device()
                )

                lab_real_preds = lab_real_preds.cuda()
                unlab_real_preds = unlab_real_preds.cuda()

                dsc_loss = self.bce_loss(labeled_preds, lab_real_preds) + self.bce_loss(
                    unlabeled_preds, unlab_real_preds
                )
                total_vae_loss = unsup_loss + transductive_loss + self.vaal_training_args.adv_param * dsc_loss
                return {"loss": total_vae_loss}
            elif step == "dsc":
                with torch.no_grad():
                    _, _, mu, _ = self._vae(labeled[DataKeys.IMAGE])
                    _, _, unlab_mu, _ = self._vae(unlabeled[DataKeys.IMAGE])

                labeled_preds = self._dsc(mu).squeeze()
                unlabeled_preds = self._dsc(unlab_mu).squeeze()

                lab_real_preds = torch.ones(
                    labeled[DataKeys.IMAGE].size(0), device=unlabeled[DataKeys.IMAGE].get_device()
                )
                unlab_fake_preds = torch.zeros(
                    unlabeled[DataKeys.IMAGE].size(0), device=unlabeled[DataKeys.IMAGE].get_device()
                )

                dsc_loss = self.bce_loss(labeled_preds, lab_real_preds) + self.bce_loss(
                    unlabeled_preds, unlab_fake_preds
                )
                return {"loss": dsc_loss}
        else:
            return self._task_model(**labeled, stage=stage)
