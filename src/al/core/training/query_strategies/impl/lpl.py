from __future__ import annotations

from typing import Any, Callable, Optional, Sequence, Tuple, Union

import torch
from al.core.data.active_learning_datamodule import ActiveLearningDataModule
from al.core.models.lpl.xai_model import LPLXAIModel
from al.core.training.query_strategies.base import QueryStrategy
from al.core.training.query_strategies.decorators import register_strategy
from xai_torch.core.constants import DataKeys
from xai_torch.core.training.constants import TrainingStage


def initialize_lpl_prediction_engine(
    model: LPLXAIModel,
    device: Optional[Union[str, torch.device]] = torch.device("cpu"),
) -> Callable:
    from ignite.engine import Engine

    def step(engine: Engine, batch: Sequence[torch.Tensor]) -> Union[Any, Tuple[torch.Tensor]]:
        """
        Define the step per batch for deep fool
        """

        from ignite.utils import convert_tensor

        # ready model for evaluation
        model.torch_model.eval()

        # put batch to device
        batch = convert_tensor(batch, device=device)

        # forward pass
        with torch.no_grad():
            outputs = model.torch_model._task_model(**batch, stage=TrainingStage.predict)
            pred_loss = model.torch_model._loss_model(outputs[DataKeys.EMBEDDING])
            return {"loss": pred_loss.view(pred_loss.size(0))}

    return Engine(step)


@register_strategy("lpl")
class LossPredictionLoss(QueryStrategy):
    def __init__(
        self,
        datamodule: ActiveLearningDataModule,
        model: LPLXAIModel,
        device: torch.device,
        embedding_layer: str = None,
    ):
        super().__init__(datamodule, model, device, embedding_layer)
        self.setup_lpl_prediction_engines()

    def reset(self, model):
        super().reset(model)
        self.setup_lpl_prediction_engines()

    def query(self, n_samples: int):
        uncertainties = self.get_uncertainty_lpl()
        return self._datamodule.unlabeled_indices[uncertainties.sort()[1][:n_samples]]

    def setup_lpl_prediction_engines(self):
        from al.core.training.metrics.output_gatherer import OutputGatherer
        from ignite.contrib.handlers import ProgressBar

        # setup the adversarial engine to apply adversarial attacks on the unlabelled dataset
        self._lpl_prediction_engine = initialize_lpl_prediction_engine(model=self._model, device=self._device)

        # attach output accumulator
        epoch_metric = OutputGatherer(output_transform=lambda output: output["loss"])
        epoch_metric.attach(self._lpl_prediction_engine, "loss")

        # attach progress bar
        pbar = ProgressBar()
        pbar.attach(self._lpl_prediction_engine)

    def get_lpl_unc(self):
        dataloader = self._datamodule.unlabeled_dataloader()
        return self._lpl_prediction_engine.run(dataloader).metrics["loss"]

    def query(self, n_samples: int):
        unc = self.get_lpl_unc()
        return self._datamodule.unlabeled_indices[unc.sort(descending=True)[1][:n_samples]]
