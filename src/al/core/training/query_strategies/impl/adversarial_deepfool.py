from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from xai_torch.core.constants import DataKeys

from al.core.training.query_strategies.base import QueryStrategy
from al.core.training.query_strategies.decorators import register_strategy

if TYPE_CHECKING:
    import torch
    from al.core.data.active_learning_datamodule import ActiveLearningDataModule
    from xai_torch.core.models.xai_model import XAIModel

import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, Tuple, Union

import eagerpy as ep
from foolbox.attacks.base import MinimizationAttack, T, get_criterion, raise_if_kwargs, verify_input_bounds
from foolbox.attacks.deepfool import L2DeepFoolAttack
from foolbox.criteria import Criterion
from foolbox.devutils import atleast_kd, flatten
from foolbox.distances import l2, linf
from foolbox.models import Model
from typing_extensions import Literal


class DeepFoolAttackForAL(L2DeepFoolAttack):
    def __call__(  # type: ignore
        self,
        model: Model,
        inputs: T,
        criterion: Any,
        **kwargs: Any,
    ) -> Union[Tuple[List[T], List[T], T], Tuple[T, T, T]]:
        x, restore_type = ep.astensor_(inputs)
        del inputs

        verify_input_bounds(x, model)

        criterion = get_criterion(criterion)

        # run the actual attack
        return self.run(model, x, criterion, **kwargs)

    def run(
        self,
        model: Model,
        inputs: T,
        criterion: Union[Criterion, T],
        *,
        early_stop: Optional[float] = None,
        **kwargs: Any,
    ) -> T:
        raise_if_kwargs(kwargs)
        x, restore_type = ep.astensor_(inputs)
        del inputs, kwargs

        verify_input_bounds(x, model)

        criterion = get_criterion(criterion)

        min_, max_ = model.bounds

        logits = model(x)
        classes = logits.argsort(axis=-1).flip(axis=-1)
        if self.candidates is None:
            candidates = logits.shape[-1]  # pragma: no cover
        else:
            candidates = min(self.candidates, logits.shape[-1])
            if not candidates >= 2:
                raise ValueError(  # pragma: no cover
                    f"expected the model output to have atleast 2 classes, got {logits.shape[-1]}"
                )
            logging.info(f"Only testing the top-{candidates} classes")
            classes = classes[:, :candidates]

        N = len(x)
        rows = range(N)

        loss_fun = self._get_loss_fn(model, classes)
        loss_aux_and_grad = ep.value_and_grad_fn(x, loss_fun, has_aux=True)

        x0 = x
        p_total = ep.zeros_like(x)
        criterion = get_criterion(logits.raw.max(1)[1])
        for _ in range(self.steps):
            # let's first get the logits using k = 1 to see if we are done
            diffs = [loss_aux_and_grad(x, 1)]
            _, (_, logits), _ = diffs[0]

            is_adv = criterion(x, logits)
            if is_adv.all():
                break

            # then run all the other k's as well
            # we could avoid repeated forward passes and only repeat
            # the backward pass, but this cannot currently be done in eagerpy
            diffs += [loss_aux_and_grad(x, k) for k in range(2, candidates)]

            # we don't need the logits
            diffs_ = [(losses, grad) for _, (losses, _), grad in diffs]
            losses = ep.stack([lo for lo, _ in diffs_], axis=1)
            grads = ep.stack([g for _, g in diffs_], axis=1)

            assert losses.shape == (N, candidates - 1)
            assert grads.shape == (N, candidates - 1) + x0.shape[1:]

            # calculate the distances
            distances = self.get_distances(losses, grads)
            assert distances.shape == (N, candidates - 1)

            # determine the best directions
            best = distances.argmin(axis=1)
            distances = distances[rows, best]
            losses = losses[rows, best]
            grads = grads[rows, best]
            assert distances.shape == (N,)
            assert losses.shape == (N,)
            assert grads.shape == x0.shape

            # apply perturbation
            distances = distances + 1e-4  # for numerical stability
            p_step = self.get_perturbations(distances, grads)
            assert p_step.shape == x0.shape

            p_total += p_step
            # don't do anything for those that are already adversarial
            x = ep.where(atleast_kd(is_adv, x.ndim), x, x0 + (1.0 + self.overshoot) * p_total)
            x = ep.clip(x, min_, max_)

        return (p_total * p_total).sum((1, 2, 3)).raw


def initialize_adv_deepfool_engine(
    model: XAIModel,
    device: Optional[Union[str, torch.device]] = torch.device("cpu"),
    max_iter: int = 50,
) -> Callable:
    from ignite.engine import Engine

    def step(engine: Engine, batch: Sequence[torch.Tensor]) -> Union[Any, Tuple[torch.Tensor]]:
        """
        Define the step per batch for deep fool
        """

        import foolbox as fb
        from ignite.utils import convert_tensor

        # ready model for evaluation
        model.torch_model.eval()

        # put batch to device
        batch = convert_tensor(batch, device=device)

        # forward pass
        model.torch_model.config.return_dict = False
        fmodel = fb.models.pytorch.PyTorchModel(model.torch_model, bounds=(-10, 10))
        attack = DeepFoolAttackForAL(loss="logits")
        output = {"adv_dis": attack(fmodel, batch[DataKeys.IMAGE], batch[DataKeys.LABEL])}
        model.torch_model.config.return_dict = True
        return output

    return Engine(step)


@register_strategy("adv_deepfool")
class AdversarialDeepFool(QueryStrategy):
    def __init__(
        self,
        datamodule: ActiveLearningDataModule,
        model: XAIModel,
        device: torch.device,
        embedding_layer: str = None,
        max_iter: int = 10,
        batch_size: int = 64,
        unlabeled_subset_size: int = None,
    ):
        super().__init__(datamodule, model, device, embedding_layer)
        self._max_iter = max_iter
        self._batch_size = batch_size
        self._unlabeled_subset_size = unlabeled_subset_size

        self.setup_adv_engine()

    def reset(self, model):
        super().reset(model)

        self.setup_adv_engine()

    def setup_adv_engine(self):
        from ignite.contrib.handlers import ProgressBar

        from al.core.training.metrics.output_gatherer import OutputGatherer

        # setup the adversarial engine to apply adversarial attacks on the unlabelled dataset
        self._adversarial_engine = initialize_adv_deepfool_engine(
            model=self._model, device=self._device, max_iter=self._max_iter
        )

        # attach output accumulator
        epoch_metric = OutputGatherer(output_transform=lambda output: output["adv_dis"])
        epoch_metric.attach(self._adversarial_engine, "adv_dis")

        # attach progress bar
        pbar = ProgressBar()
        pbar.attach(self._adversarial_engine)

    def get_adv_dis(self):
        if self._unlabeled_subset_size is not None:
            dataloader, self.unlabeled_subset = self._datamodule.unlabeled_dataloader(
                batch_size=self._batch_size, unlabeled_subset_size=self._unlabeled_subset_size
            )
        else:
            dataloader = self._datamodule.unlabeled_dataloader(batch_size=self._batch_size)
        return self._adversarial_engine.run(dataloader).metrics["adv_dis"]

    def query(self, n_samples: int):
        adv_dis = self.get_adv_dis()
        if self._unlabeled_subset_size is not None:
            return torch.tensor(self.unlabeled_subset.indices)[adv_dis.argsort()[:n_samples]]
        else:
            return self._datamodule.unlabeled_indices[adv_dis.argsort()[:n_samples]]
