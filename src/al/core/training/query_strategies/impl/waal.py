from __future__ import annotations

from typing import Any, Callable, Optional, Sequence, Tuple, Union

import torch
from al.core.data.active_learning_datamodule import ActiveLearningDataModule
from al.core.models.waal.xai_model import WAALXAIModel
from al.core.training.query_strategies.base import QueryStrategy
from al.core.training.query_strategies.decorators import register_strategy
from xai_torch.core.constants import DataKeys
from xai_torch.core.training.constants import TrainingStage


def initialize_waal_prediction_engine(
    model: WAALXAIModel,
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
            features = model.torch_model._task_model(batch[DataKeys.IMAGE])
            scores = model.torch_model._dsc(features)
            return {"scores": scores}

    return Engine(step)


@register_strategy("waal")
class LossPredictionLoss(QueryStrategy):
    def __init__(
        self,
        datamodule: ActiveLearningDataModule,
        model: WAALXAIModel,
        device: torch.device,
        embedding_layer: str = None,
        selection: float = 10.0,
    ):
        super().__init__(datamodule, model, device, embedding_layer)
        self._selection = selection
        self.setup_waal_prediction_engines()

    def reset(self, model):
        super().reset(model)
        self.setup_waal_prediction_engines()

    def query(self, n_samples: int):
        uncertainties = self.get_uncertainty_waal()
        return self._datamodule.unlabeled_indices[uncertainties.sort()[1][:n_samples]]

    def setup_waal_prediction_engines(self):
        from al.core.training.metrics.output_gatherer import OutputGatherer
        from ignite.contrib.handlers import ProgressBar

        # setup the adversarial engine to apply adversarial attacks on the unlabelled dataset
        self._waal_prediction_engine = initialize_waal_prediction_engine(model=self._model, device=self._device)

        # attach output accumulator
        epoch_metric = OutputGatherer(output_transform=lambda output: output["scores"])
        epoch_metric.attach(self._waal_prediction_engine, "scores")

        # attach progress bar
        pbar = ProgressBar()
        pbar.attach(self._waal_prediction_engine)

    def get_waal_dsc_scores(self):
        dataloader = self._datamodule.unlabeled_dataloader()
        return self._waal_prediction_engine.run(dataloader).metrics["scores"]

    def query(self, n_samples: int):
        probs = self.get_probs_on_unlabeled()
        uncertainty_score = 0.5 * self.l2_upper(probs) + 0.5 * self.l1_upper(probs)

        # prediction output discriminative score
        dsc_scores = self.get_waal_dsc_scores().squeeze()

        # computing the decision score
        total_score = uncertainty_score - self._selection * dsc_scores

        return self._datamodule.unlabeled_indices[total_score.sort()[1][:n_samples]]

    def l2_upper(self, probs):
        value = torch.norm(torch.log(probs), dim=1)
        return value

    def l1_upper(self, probs):
        value = torch.sum(-1 * torch.log(probs), dim=1)
        return value
