from __future__ import annotations

from typing import TYPE_CHECKING

from al.core.training.query_strategies.base import QueryStrategy
from al.core.training.query_strategies.decorators import register_strategy

if TYPE_CHECKING:
    import torch
    from al.core.data.active_learning_datamodule import ActiveLearningDataModule
    from xai_torch.core.models.xai_model import XAIModel


@register_strategy("mean_std")
class MeanSTD(QueryStrategy):
    def __init__(
        self,
        datamodule: ActiveLearningDataModule,
        model: XAIModel,
        device: torch.device,
        embedding_layer: str = None,
        dropout_runs: int = 10,
    ):
        super().__init__(datamodule, model, device, embedding_layer)
        self._dropout_runs = dropout_runs

    def query(self, n_samples: int):
        import torch
        import numpy as np

        probs = self.get_probs_on_unlabeled_with_dropout(dropout_runs=self._dropout_runs, take_mean=False)
        sigma_c = np.std(probs, axis=0)
        uncertainties = torch.from_numpy(np.mean(sigma_c, axis=-1))
        return self._datamodule.unlabeled_indices[uncertainties.sort(descending=True)[1][:n_samples]]
